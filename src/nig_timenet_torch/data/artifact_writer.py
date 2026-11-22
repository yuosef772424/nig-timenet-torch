"""Old path of the contract artifact writer

PURPOSE: Re-export of ``data_io.artifact_writer`` (moved there by HB5 so ``data_io`` no longer imports ``data``).
TAGS: artifact writer, re-export, moved module, shim
PITFALLS: temporary: ``workflow/gate_run.py`` still imports this path (another package's work in flight); switch it
    to ``nig_timenet_torch.data_io.artifact_writer`` and delete this file.
"""

from nig_timenet_torch.data_io.artifact_writer import git_head, restrict_to_time_seal, write_artifact

__all__ = ["write_artifact", "git_head", "restrict_to_time_seal"]
