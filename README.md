# batchnorm-from-paper

Minimal, from-scratch NumPy implementation of Batch Normalization while walking through the original paper.

## Status
- Forward pass implementation is working.
- Notebook now includes neural-network integration and visualization examples.
- Mathematical derivation notes have started in `math-notes/`.
- Backward pass derivation, gradient checks, and train/eval behavior demos are now included in the notebook.

## What is implemented
- `BatchNorm` class in `src/batchnorm.py`
- Per-feature batch mean and variance computation
- Epsilon-stabilized normalization
- Learnable scale/shift parameters (`gamma`, `beta`) initialized on first forward call
- `NeuralNetwork` forward demo in `src/neural_network.py` showing linear → batchnorm → ReLU
- Histogram-based before/after normalization visualization in the notebook
- Numerical and analytical gradient checks for `BatchNorm.backward`
- Train/eval behavior validation for different input batches

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
  batchnorm_backprop_derivation.md  # backprop derivation notes
  images/page1.jpeg        # handwritten derivation page
  images/backprop-*.jpeg   # handwritten backprop pages
experiments/        # reserved for experiments (currently empty)
  activation_distributions.png
  grad_check_results.txt
notebooks/experiments/ # notebook-local experiment outputs
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