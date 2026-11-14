"""One small E1 context, built once and reused by every doctor test (the run is the slow part)."""

import pytest

from nig_timenet_torch.data_io.synthetic import SyntheticSettings, write_synthetic
from nig_timenet_torch.doctor.common import make_context


@pytest.fixture(scope="session")
def ctx(tmp_path_factory):
    root = tmp_path_factory.mktemp("doctor")
    art = root / "e1"
    write_synthetic(art, SyntheticSettings(n_coins=8, n_bars=1200), overwrite=True)
    return make_context(str(art), root / "out", epochs=1, batch_size=128)
