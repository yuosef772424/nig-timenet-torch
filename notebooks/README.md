# Notebooks

[`colab_nig_timenet_torch.ipynb`](colab_nig_timenet_torch.ipynb) runs the PyTorch port on Google Colab
(`docs/BUILD_PLAN.md` P45). Steps:

1. Clones the repo. A token is asked for at runtime and is never stored.
2. Installs the package over Colab's own torch (`--no-deps`).
3. Runs the full test suite.
4. Shows the TF parity budget for every ported model.
5. Builds all variants.
6. Trains on the fixture or on a `.dataset` folder from Drive.
7. Writes a PredictionTable and reports metrics before and after isotonic calibration.

Open it with [Open in Colab](https://colab.research.google.com/github/yuosef772424/nig-timenet-torch/blob/main/notebooks/colab_nig_timenet_torch.ipynb).
For a private repo, Colab asks to authorize GitHub. Then choose Runtime, then Run all.

To keep the notebook valid, commit it without outputs. `tests/notebooks/test_colab_notebook.py` checks this, and
runs the non-Colab cells headless.
