# Self-Pruning Neural Network Report

## Method

The model replaces each dense linear layer with a custom `PrunableLinear` layer. Each weight has a matching learnable `gate_scores` parameter. During the forward pass:

```text
gates = sigmoid(gate_scores)
pruned_weights = weight * gates
```

The network is trained with:

```text
Total Loss = CrossEntropyLoss + lambda_eff * sum(sigmoid(gate_scores))
```

A short warmup trains the classifier before the sparsity penalty ramps to the target lambda. This keeps the required loss formulation while making optimization more stable.

## Why an L1 Penalty on Sigmoid Gates Encourages Sparsity

Each gate is between `0` and `1`. Adding the L1 norm of all gates penalizes the model for keeping many connections active. As lambda increases, more gates are pushed toward `0`, so more weights become effectively pruned.

## Results

Use `sparsity@1e-2` as the official sparsity level.

| Lambda | Test Accuracy (%) | Sparsity Level @ 1e-2 (%) |
| --- | ---: | ---: |
| `1e-5` | `56.95` | `43.59` |
| `5e-5` | `58.70` | `81.60` |
| `1e-4` | `57.29` | `90.92` |

## Analysis

Increasing lambda increases sparsity because the gate penalty becomes stronger. The best trade-off is `lambda = 5e-5`, which achieves the highest accuracy while pruning more than 80% of weights at the official `1e-2` threshold. The largest lambda gives the highest sparsity, but with a small accuracy drop.

## Gate Distribution

Insert the histogram for the `lambda = 5e-5` model. A strong result should show many gates concentrated near `0`, with a smaller group of useful gates remaining away from `0`.
