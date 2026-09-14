# Learned Number Representations

Independent PyTorch experiments on whether a learned visual number representation can be understood by a new reader.

This repository focuses on the number-code and iterated-learning part of my SymbolEmergence project. The main question is whether a representation transfers beyond the quantities used to teach the reader. It includes the model and evaluator, relevant tests, and saved results from one eight-generation experiment.

## Experiment

1. A model writes a visual representation of a quantity.
2. A fresh reader is taught 20 quantities with a fixed training budget.
3. The reader is evaluated on 68 quantities it was not taught: 8 examples per quantity, or 544 examples per condition.
4. Learned representations are compared with random, unary, and positional codes under the same teaching budget.

The reader's unseen quantities are drawn from the broader training domain. This transfer test is distinct from the separate global interpolation and extrapolation splits.

## Result

In the saved run, the learned code scored below the matched random-code baseline in all eight generations. For the final generation, mean accuracy over the three transfer tasks was **18.93%** for the learned code and **22.24%** for the random code.

These are eight successive generations from one seed, not eight independent replications. The run did not meet its transfer criterion. It does not establish that useful symbolic representations cannot emerge under other settings.

[Results and protocol](docs/results.md) include the comparison table, uncertainty, and links to the saved JSON files.

## Where to start

| File | What to read it for |
| --- | --- |
| `symemerge/numcode/data.py` | Quantity generation and held-out ranges |
| `symemerge/numcode/pred/model.py` | Visual reader and task heads |
| `symemerge/numcode/pred/relearn.py` | Fresh-reader training and exposure selection |
| `outputs/eval_relearn_gauge.py` | Matched comparison and transfer evaluation |
| `tests/test_nc_relearn.py` | Exposure boundaries, gradient routing, and saved state |

## Environment

Python 3.11+ and PyTorch. Training and model tests use a Linux CUDA environment. After installing a PyTorch build compatible with that environment, install the project with `python -m pip install -e ".[dev]"`.

The saved JSON results can be read without PyTorch. Re-running model evaluation also requires the original checkpoint, which is not included in this source snapshot. No GPU results are claimed from a fresh run of this snapshot.

```sh
# On the configured Linux CUDA environment, with the checkpoint available:
python outputs/eval_relearn_gauge.py --ckpt /path/to/ckpt_last.pt --out outputs/transfer-check --budgets 1000 --seed 0
```

See [design notes](docs/design.md) for the scope and evaluation choices. Earlier experiment branches and machine-specific launch scripts are outside this focused repository.
