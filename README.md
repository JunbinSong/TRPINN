# TRPINN

A Trace Regularity Physics-Informed Neural Network (TRPINN) for solving a Poisson equation on the 2D unit disk ("ball") domain.

- `trpinn.py` — main entry point / training script  
- `trpinn_functions.py` — helper module (MLP model, sampling utilities, loss functions, relative errors, visualization, etc.)

## Quick Start
```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib, tensorflow, h5py
python trpinn.py                  # outputs saved to ./results (configurable via Save_dir)
