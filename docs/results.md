# Iterated-learning transfer results

These are saved outputs from the original experiment, re-read for this source snapshot. They are not fresh model runs.

Each condition trains a fresh reader on 20 quantities for 1,000 steps at batch size 32, then evaluates 8 examples for each of 68 untaught quantities. The reported measure is the per-example mean accuracy across parity, remainder, and small-addition tasks. The uncertainty below is the saved standard error in percentage points, not a confidence interval or variation across training seeds.

| Generation | Learned code | Matched random code |
| --- | --- | --- |
| [1](../results/iterated-learning/generation-1.json) | 17.71% ± 0.85 pp | 22.55% ± 0.89 pp |
| [2](../results/iterated-learning/generation-2.json) | 19.18% ± 0.82 pp | 24.94% ± 0.91 pp |
| [3](../results/iterated-learning/generation-3.json) | 18.63% ± 0.83 pp | 24.82% ± 0.95 pp |
| [4](../results/iterated-learning/generation-4.json) | 20.22% ± 0.84 pp | 21.57% ± 0.91 pp |
| [5](../results/iterated-learning/generation-5.json) | 20.16% ± 0.86 pp | 22.12% ± 0.84 pp |
| [6](../results/iterated-learning/generation-6.json) | 20.16% ± 0.89 pp | 24.02% ± 1.01 pp |
| [7](../results/iterated-learning/generation-7.json) | 20.28% ± 0.87 pp | 24.39% ± 0.90 pp |
| [8](../results/iterated-learning/generation-8.json) | 18.93% ± 0.87 pp | 22.24% ± 0.90 pp |

All generations use seed 0 and belong to one successive lineage. The learned code scored below its matched random control in all eight generations. The original criterion required beating the random control by three combined standard errors and maintaining that result in the last two generations; this run failed that criterion. This does not mean that every individual difference is statistically significant.

The raw files also contain unary and positional-code controls, per-task results, quantity-band results, and error-tolerance rates. The broader global held-out ranges differ from this reader-transfer split. See `outputs/eval_relearn_gauge.py` for how the measurements are computed.

Checkpoints are not included. Reproducing this experiment requires the corresponding checkpoints, GPU environment, and configuration; reading the saved results does not require them.
