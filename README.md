# Self-Pruning Neural Network

This project implements a feed-forward CIFAR-10 classifier whose weights are controlled by learnable sigmoid gates. The model learns both classification weights and which connections can be suppressed during training.

## What It Implements

- `PrunableLinear(in_features, out_features)` with `weight`, `bias`, and same-shaped `gate_scores`.
- Forward pass: `gates = sigmoid(gate_scores)`, then `pruned_weights = weight * gates`.
- Sparsity loss: sum of all gate values across prunable layers.
- Total loss: `CrossEntropyLoss + lambda_eff * SparsityLoss`.
- Lambda warmup/ramp for stable training while preserving the assignment loss formulation.
- Results for three lambda values, saved to CSV/JSON plus a Markdown report and gate histograms.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA on Windows, install PyTorch with the CUDA wheel index if needed:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install matplotlib
```

## Run

```powershell
python self_pruning_cifar10.py --device cuda
```

Run the gradient-flow sanity check:

```powershell
python self_pruning_cifar10.py --gradient-check
```

## Expected Outputs

The script writes:

- `outputs/results.csv`
- `outputs/results.json`
- `outputs/report.md`
- `outputs/lambda_*/best_model.pt`
- `outputs/lambda_*/gate_histogram.png`

## Recommended Final Results From Colab

Use the following table in the final report if reproducing the successful Colab run:

| Lambda | Test Accuracy (%) | Sparsity Level @ 1e-2 (%) |
| --- | ---: | ---: |
| `1e-5` | `56.95` | `43.59` |
| `5e-5` | `58.70` | `81.60` |
| `1e-4` | `57.29` | `90.92` |

The best sparsity-accuracy trade-off is `lambda = 5e-5`: it keeps the highest accuracy while pruning over 80% of weights under the official `1e-2` threshold.
