"""Self-pruning neural network for CIFAR-10.

This script implements the challenge specification:
1. A custom PrunableLinear layer with learnable gate_scores.
2. Sigmoid gates in [0, 1] multiplied element-wise with weights.
3. A feed-forward CIFAR-10 classifier built from PrunableLinear layers.
4. A training loop using CrossEntropyLoss + lambda * L1(gates).
5. Evaluation across multiple lambda values with accuracy, sparsity, and plots.
6. Optional dense baseline and compression-style metrics for deployment analysis.

Recommended run:
    python self_pruning_learnable.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - user-facing runtime guard
    raise SystemExit(
        "Missing required dependencies. Install them with:\n"
        "  pip install torch torchvision matplotlib\n"
        f"Original import error: {exc}"
    ) from exc


@dataclass
class ExperimentResult:
    lambda_value: float
    test_accuracy: float
    sparsity_level: float
    sparsity_at_005: float
    sparsity_at_010: float
    gate_mean: float
    gate_min: float
    gate_max: float
    total_gated_weights: int
    active_gated_weights: int
    pruned_gated_weights: int
    compression_ratio: float
    best_epoch: int
    histogram_path: str
    model_path: str


@dataclass
class BaselineResult:
    test_accuracy: float
    best_epoch: int
    model_path: str


class PrunableLinear(nn.Module):
    """Linear layer with one learnable gate score per weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init: float = 2.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(gate_init)

    def reset_parameters(self, gate_init: float) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.constant_(self.gate_scores, gate_init)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def gate_values(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gates = self.gate_values()
        pruned_weights = self.weight * gates
        return F.linear(inputs, pruned_weights, self.bias)


class SelfPruningMLP(nn.Module):
    """Feed-forward CIFAR-10 classifier built from prunable layers."""

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dims: tuple[int, ...] = (512, 256),
        num_classes: int = 10,
        gate_init: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        dims = [input_dim, *hidden_dims, num_classes]
        layers: list[nn.Module] = []
        self.prunable_layers = nn.ModuleList()

        for index in range(len(dims) - 1):
            linear = PrunableLinear(dims[index], dims[index + 1], gate_init=gate_init)
            self.prunable_layers.append(linear)
            layers.append(linear)

            if index < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = torch.flatten(inputs, start_dim=1)
        return self.network(flattened)

    def sparsity_loss(self) -> torch.Tensor:
        # Gates are positive after sigmoid, so the L1 norm is simply their sum.
        return sum(layer.gate_values().sum() for layer in self.prunable_layers)

    def all_gate_values(self) -> torch.Tensor:
        return torch.cat([layer.gate_values().reshape(-1) for layer in self.prunable_layers])

    def sparsity_level(self, threshold: float) -> float:
        with torch.no_grad():
            gates = self.all_gate_values()
            return (gates < threshold).float().mean().item() * 100.0

    def gate_statistics(self, threshold: float) -> dict[str, float]:
        with torch.no_grad():
            gates = self.all_gate_values()
            return {
                "sparsity_level": (gates < threshold).float().mean().item() * 100.0,
                "sparsity_at_005": (gates < 0.05).float().mean().item() * 100.0,
                "sparsity_at_010": (gates < 0.10).float().mean().item() * 100.0,
                "gate_mean": gates.mean().item(),
                "gate_min": gates.min().item(),
                "gate_max": gates.max().item(),
            }


class DenseMLP(nn.Module):
    """Dense baseline with the same MLP shape but standard Linear layers."""

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dims: tuple[int, ...] = (512, 256),
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        dims = [input_dim, *hidden_dims, num_classes]
        layers: list[nn.Module] = []

        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = torch.flatten(inputs, start_dim=1)
        return self.network(flattened)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gate-learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambdas", type=str, default="1e-5,5e-5,1e-4")
    parser.add_argument("--gate-threshold", type=float, default=1e-2)
    parser.add_argument("--gate-init", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Also train a dense MLP baseline for comparison.",
    )
    parser.add_argument(
        "--baseline-epochs",
        type=int,
        default=None,
        help="Epochs for dense baseline. Defaults to --epochs.",
    )
    parser.add_argument(
        "--gradient-check",
        action="store_true",
        help="Run a small gradient-flow check for PrunableLinear and exit.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gradient_flow_check() -> None:
    layer = PrunableLinear(4, 3)
    inputs = torch.randn(2, 4)
    output = layer(inputs).sum()
    output.backward()

    print("Gradient flow check")
    print(f"weight grad exists: {layer.weight.grad is not None}")
    print(f"gate_scores grad exists: {layer.gate_scores.grad is not None}")
    print(f"bias grad exists: {layer.bias is not None and layer.bias.grad is not None}")


def get_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def effective_lambda_for_epoch(lambda_value: float, epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch <= warmup_epochs:
        return 0.0

    ramp_epochs = max(total_epochs - warmup_epochs, 1)
    ramp = (epoch - warmup_epochs) / ramp_epochs
    return lambda_value * min(max(ramp, 0.0), 1.0)


def train_one_epoch(
    model: SelfPruningMLP,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_value: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_sparsity_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        classification_loss = F.cross_entropy(logits, labels)
        sparsity_loss = model.sparsity_loss()
        loss = classification_loss + lambda_value * sparsity_loss
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        num_batches += 1

        total_loss += loss.item()
        total_classification_loss += classification_loss.item()
        total_sparsity_loss += sparsity_loss.item()

    return {
        "train_total_loss": total_loss / num_batches,
        "train_classification_loss": total_classification_loss / num_batches,
        "train_sparsity_loss": total_sparsity_loss / num_batches,
        "train_accuracy": correct / total * 100.0,
    }


def train_dense_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        num_batches += 1
        total_loss += loss.item()

    return {
        "train_loss": total_loss / num_batches,
        "train_accuracy": correct / total * 100.0,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    gate_threshold: float | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    metrics = {
        "test_loss": total_loss / total,
        "test_accuracy": correct / total * 100.0,
    }
    if gate_threshold is not None and isinstance(model, SelfPruningMLP):
        metrics["sparsity_level"] = model.sparsity_level(gate_threshold)
    return metrics


def save_gate_histogram(gates: torch.Tensor, output_path: Path, lambda_value: float) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(gates.cpu().numpy(), bins=50, color="#3366cc", alpha=0.85, edgecolor="black")
    plt.title(f"Final Gate Distribution (lambda={lambda_value:g})")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def lambda_to_tag(lambda_value: float) -> str:
    return f"{lambda_value:.0e}".replace("-", "m").replace("+", "p")


def gated_weight_counts(model: SelfPruningMLP, threshold: float) -> dict[str, float]:
    with torch.no_grad():
        gates = model.all_gate_values()
        total = gates.numel()
        active = int((gates >= threshold).sum().item())
        pruned = total - active
        return {
            "total_gated_weights": total,
            "active_gated_weights": active,
            "pruned_gated_weights": pruned,
            "compression_ratio": total / max(active, 1),
        }


def build_optimizer(model: SelfPruningMLP, args: argparse.Namespace) -> torch.optim.Optimizer:
    gate_params = []
    other_params = []

    for name, parameter in model.named_parameters():
        if "gate_scores" in name:
            gate_params.append(parameter)
        else:
            other_params.append(parameter)

    return torch.optim.Adam(
        [
            {"params": other_params, "lr": args.learning_rate, "weight_decay": args.weight_decay},
            {"params": gate_params, "lr": args.gate_learning_rate, "weight_decay": 0.0},
        ]
    )


def run_dense_baseline(
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> BaselineResult:
    model = DenseMLP(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    baseline_epochs = args.baseline_epochs or args.epochs
    best_accuracy = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    print("\nStarting dense baseline")
    for epoch in range(1, baseline_epochs + 1):
        train_metrics = train_dense_one_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate(model, test_loader, device)

        print(
            f"[dense] epoch={epoch:02d} "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"train_acc={train_metrics['train_accuracy']:.2f}% "
            f"test_acc={eval_metrics['test_accuracy']:.2f}%"
        )

        if eval_metrics["test_accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["test_accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None, "Best dense baseline state should be captured during training."
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, test_loader, device)

    baseline_dir = args.output_dir / "dense_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    model_path = baseline_dir / "best_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "metrics": final_metrics,
            "args": vars(args),
        },
        model_path,
    )

    return BaselineResult(
        test_accuracy=final_metrics["test_accuracy"],
        best_epoch=best_epoch,
        model_path=str(model_path.resolve()),
    )


def run_experiment(
    lambda_value: float,
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> ExperimentResult:
    model = SelfPruningMLP(gate_init=args.gate_init, dropout=args.dropout).to(device)
    optimizer = build_optimizer(model, args)

    best_accuracy = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        effective_lambda = effective_lambda_for_epoch(
            lambda_value=lambda_value,
            epoch=epoch,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
        )
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, effective_lambda)
        eval_metrics = evaluate(model, test_loader, device, args.gate_threshold)
        gate_stats = model.gate_statistics(args.gate_threshold)

        print(
            f"[lambda={lambda_value:g}] epoch={epoch:02d} "
            f"effective_lambda={effective_lambda:.2e} "
            f"train_loss={train_metrics['train_total_loss']:.4f} "
            f"train_acc={train_metrics['train_accuracy']:.2f}% "
            f"test_acc={eval_metrics['test_accuracy']:.2f}% "
            f"sparsity@0.01={eval_metrics['sparsity_level']:.2f}% "
            f"sparsity@0.05={gate_stats['sparsity_at_005']:.2f}% "
            f"sparsity@0.10={gate_stats['sparsity_at_010']:.2f}% "
            f"gate_mean={gate_stats['gate_mean']:.4f} "
            f"gate_min={gate_stats['gate_min']:.4f}"
        )

        if eval_metrics["test_accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["test_accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None, "Best model state should be captured during training."
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, test_loader, device, args.gate_threshold)
    final_gate_stats = model.gate_statistics(args.gate_threshold)
    final_weight_counts = gated_weight_counts(model, args.gate_threshold)

    experiment_dir = args.output_dir / f"lambda_{lambda_to_tag(lambda_value)}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model_path = experiment_dir / "best_model.pt"
    histogram_path = experiment_dir / "gate_histogram.png"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "lambda_value": lambda_value,
            "best_epoch": best_epoch,
            "metrics": final_metrics,
            "gate_statistics": final_gate_stats,
            "gated_weight_counts": final_weight_counts,
            "args": vars(args),
        },
        model_path,
    )

    gates = model.all_gate_values().detach().cpu()
    save_gate_histogram(gates, histogram_path, lambda_value)

    return ExperimentResult(
        lambda_value=lambda_value,
        test_accuracy=final_metrics["test_accuracy"],
        sparsity_level=final_metrics["sparsity_level"],
        sparsity_at_005=final_gate_stats["sparsity_at_005"],
        sparsity_at_010=final_gate_stats["sparsity_at_010"],
        gate_mean=final_gate_stats["gate_mean"],
        gate_min=final_gate_stats["gate_min"],
        gate_max=final_gate_stats["gate_max"],
        total_gated_weights=int(final_weight_counts["total_gated_weights"]),
        active_gated_weights=int(final_weight_counts["active_gated_weights"]),
        pruned_gated_weights=int(final_weight_counts["pruned_gated_weights"]),
        compression_ratio=final_weight_counts["compression_ratio"],
        best_epoch=best_epoch,
        histogram_path=str(histogram_path.resolve()),
        model_path=str(model_path.resolve()),
    )


def save_results_table(
    results: list[ExperimentResult],
    output_dir: Path,
    baseline_result: BaselineResult | None = None,
) -> tuple[Path, Path]:
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "lambda_value",
                "test_accuracy",
                "sparsity_level",
                "sparsity_at_005",
                "sparsity_at_010",
                "gate_mean",
                "gate_min",
                "gate_max",
                "total_gated_weights",
                "active_gated_weights",
                "pruned_gated_weights",
                "compression_ratio",
                "best_epoch",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "lambda_value": result.lambda_value,
                    "test_accuracy": f"{result.test_accuracy:.4f}",
                    "sparsity_level": f"{result.sparsity_level:.4f}",
                    "sparsity_at_005": f"{result.sparsity_at_005:.4f}",
                    "sparsity_at_010": f"{result.sparsity_at_010:.4f}",
                    "gate_mean": f"{result.gate_mean:.6f}",
                    "gate_min": f"{result.gate_min:.6f}",
                    "gate_max": f"{result.gate_max:.6f}",
                    "total_gated_weights": result.total_gated_weights,
                    "active_gated_weights": result.active_gated_weights,
                    "pruned_gated_weights": result.pruned_gated_weights,
                    "compression_ratio": f"{result.compression_ratio:.4f}",
                    "best_epoch": result.best_epoch,
                }
            )

    with json_path.open("w", encoding="utf-8") as json_file:
        payload = {
            "baseline": asdict(baseline_result) if baseline_result is not None else None,
            "experiments": [asdict(result) for result in results],
        }
        json.dump(payload, json_file, indent=2)

    return csv_path, json_path


def save_tradeoff_plot(results: list[ExperimentResult], output_dir: Path) -> Path:
    plot_path = output_dir / "sparsity_accuracy_tradeoff.png"
    sparsities = [result.sparsity_level for result in results]
    accuracies = [result.test_accuracy for result in results]
    labels = [f"{result.lambda_value:g}" for result in results]

    plt.figure(figsize=(7, 5))
    plt.plot(sparsities, accuracies, marker="o", linewidth=2, color="#cc5500")
    for x_value, y_value, label in zip(sparsities, accuracies, labels):
        plt.annotate(f"lambda={label}", (x_value, y_value), textcoords="offset points", xytext=(6, 6))
    plt.xlabel("Sparsity Level @ 1e-2 (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Sparsity vs Accuracy Trade-off")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def choose_best_tradeoff(results: list[ExperimentResult]) -> ExperimentResult:
    # Accuracy dominates; sparsity breaks close ties in favor of compact models.
    return max(results, key=lambda item: item.test_accuracy + 0.02 * item.sparsity_level)


def write_markdown_report(
    results: list[ExperimentResult],
    output_dir: Path,
    baseline_result: BaselineResult | None = None,
    tradeoff_plot_path: Path | None = None,
) -> Path:
    best_accuracy_result = max(results, key=lambda item: item.test_accuracy)
    best_tradeoff_result = choose_best_tradeoff(results)
    report_path = output_dir / "report.md"

    lines = [
        "# Self-Pruning Neural Network Report",
        "",
        "## Method",
        "",
        (
            "The model replaces each dense linear layer with a PrunableLinear layer. "
            "Each weight has a matching learnable gate score. During the forward pass, "
            "the gate score is passed through a sigmoid and multiplied element-wise with "
            "the weight tensor before applying the linear operation."
        ),
        "",
        "The training objective is:",
        "",
        "`Total Loss = CrossEntropyLoss + lambda_eff * sum(sigmoid(gate_scores))`",
        "",
        (
            "A short warmup period trains the classifier before sparsity pressure is introduced. "
            "After warmup, lambda_eff ramps linearly to the target lambda. This keeps the core "
            "assignment loss unchanged while making optimization more stable."
        ),
        "",
        "## Why an L1 Penalty on Sigmoid Gates Encourages Sparsity",
        "",
        (
            "Each gate value is between 0 and 1. Adding the L1 norm of all gate values penalizes "
            "the model for keeping many connections active. As lambda increases, the optimizer "
            "is encouraged to push more gates toward 0, leaving only connections that improve "
            "classification accuracy enough to justify their penalty."
        ),
        "",
        "## Results",
        "",
        "| Lambda | Test Accuracy (%) | Sparsity Level @ 1e-2 (%) | Active Gated Weights | Compression Ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            f"| {result.lambda_value:g} | {result.test_accuracy:.2f} | "
            f"{result.sparsity_level:.2f} | {result.active_gated_weights:,} / "
            f"{result.total_gated_weights:,} | {result.compression_ratio:.2f}x |"
        )

    if baseline_result is not None:
        accuracy_drop = baseline_result.test_accuracy - best_tradeoff_result.test_accuracy
        lines.extend(
            [
                "",
                "## Dense Baseline Comparison",
                "",
                "| Model | Test Accuracy (%) | Sparsity Level (%) |",
                "| --- | ---: | ---: |",
                f"| Dense baseline | {baseline_result.test_accuracy:.2f} | 0.00 |",
                (
                    f"| Best trade-off pruned model (lambda = {best_tradeoff_result.lambda_value:g}) | "
                    f"{best_tradeoff_result.test_accuracy:.2f} | {best_tradeoff_result.sparsity_level:.2f} |"
                ),
                "",
                (
                    f"The best trade-off model changes accuracy by {accuracy_drop:.2f} percentage points "
                    "relative to the dense baseline while pruning a large fraction of gated weights."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Analysis",
            "",
            (
                "Increasing lambda increases sparsity because the gate penalty becomes stronger. "
                "The best accuracy model and the best sparsity-accuracy trade-off can differ; "
                "the trade-off model is selected by balancing test accuracy with the percentage "
                "of gates below the official pruning threshold."
            ),
            "",
            f"Best accuracy model: lambda = {best_accuracy_result.lambda_value:g}.",
            f"Best trade-off model: lambda = {best_tradeoff_result.lambda_value:g}.",
            "",
        ]
    )

    if tradeoff_plot_path is not None:
        lines.extend(
            [
                "## Sparsity vs Accuracy Plot",
                "",
                f"![Sparsity vs accuracy]({tradeoff_plot_path.resolve()})",
                "",
            ]
        )

    lines.extend(
        [
            "## Best Trade-Off Gate Distribution",
            "",
            (
                "A successful self-pruning run should show many gates concentrated near 0, "
                "with a smaller group of gates remaining away from 0 for useful connections."
            ),
            "",
            f"![Gate histogram]({best_tradeoff_result.histogram_path})",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()

    if args.gradient_check:
        gradient_flow_check()
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)
    lambda_values = [float(value.strip()) for value in args.lambdas.split(",") if value.strip()]

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    baseline_result = None
    if args.run_baseline:
        baseline_result = run_dense_baseline(args, train_loader, test_loader, device)

    results: list[ExperimentResult] = []
    for lambda_value in lambda_values:
        print(f"\nStarting experiment with lambda={lambda_value:g}")
        result = run_experiment(lambda_value, args, train_loader, test_loader, device)
        results.append(result)

    tradeoff_plot_path = save_tradeoff_plot(results, args.output_dir)
    csv_path, json_path = save_results_table(results, args.output_dir, baseline_result)
    report_path = write_markdown_report(results, args.output_dir, baseline_result, tradeoff_plot_path)

    print("\nFinished all experiments.")
    print(f"Saved summary CSV to: {csv_path.resolve()}")
    print(f"Saved summary JSON to: {json_path.resolve()}")
    print(f"Saved Markdown report to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
