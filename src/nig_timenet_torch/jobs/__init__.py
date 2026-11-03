"""Job entries of the Colab runner (``colab/runner.py``): each module is run as
``python -m nig_timenet_torch.jobs.<entry> --config <json> --out <dir> --max-minutes <n>`` and follows the run contract
of ``colab/contract.py`` (device.json + manifest.json + the job's expected outputs)."""
