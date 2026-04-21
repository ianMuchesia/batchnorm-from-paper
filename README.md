# batchnorm-from-paper

Minimal, from-scratch NumPy implementation of Batch Normalization while walking through the original paper.

## Status
- Forward pass implementation is working.
- Notebook now includes neural-network integration and visualization examples.
- Mathematical derivation notes have started in `math-notes/`.
- Backward pass and full paper-complete implementation are still in progress.

## What is implemented
- `BatchNorm` class in `src/batchnorm.py`
- Per-feature batch mean and variance computation
- Epsilon-stabilized normalization
- Learnable scale/shift parameters (`gamma`, `beta`) initialized on first forward call
- `NeuralNetwork` forward demo in `src/neural_network.py` showing linear → batchnorm → ReLU
- Histogram-based before/after normalization visualization in the notebook

## Project structure
```text
src/
  batchnorm.py      # core BatchNorm forward implementation
  neural_network.py # minimal NN block using BatchNorm
  playground.py     # simple 1D normalization sanity script
notebooks/
  paper_walkthrough.ipynb  # walkthrough, NN demo, and plots
math-notes/
  batchnorm_derivation.md  # derivation notes
  images/page1.jpeg        # handwritten derivation page
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