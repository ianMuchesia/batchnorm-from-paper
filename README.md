# batchnorm-from-paper

Minimal, from-scratch NumPy implementation of Batch Normalization while walking through the original paper.

## Status
- First commit snapshot.
- Forward pass only (training-time style normalization).
- Backward pass and full paper-complete implementation are still in progress.

## What is implemented
- `BatchNorm` class in `src/batchnorm.py`
- Per-feature batch mean and variance computation
- Epsilon-stabilized normalization
- Learnable scale/shift parameters (`gamma`, `beta`) initialized on first forward call

## Project structure
```text
src/
  batchnorm.py      # core BatchNorm forward implementation
  playground.py     # simple 1D normalization sanity script
notebooks/
  paper_walkthrough.ipynb  # interactive walkthrough
math-notes/         # reserved for derivations (currently empty)
experiments/        # reserved for experiments (currently empty)
```

## Quick start
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/playground.py
```

## Notebook
Open `notebooks/paper_walkthrough.ipynb` in VS Code or Jupyter and run cells top-to-bottom.

## Next planned work
- Backward pass derivation + implementation
- Running statistics for inference
- Numerical checks and tests against a reference implementation