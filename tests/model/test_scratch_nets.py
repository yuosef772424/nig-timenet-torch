"""conv1d_net / no_dense_net (model/scratch_nets.py): the output contract, no Linear layer in no_dense_net, bounded
inputs under an outlier, row independence, and a backward pass that reaches every weight."""

import pytest
import torch

from nig_timenet_torch.model.architectures import ModelSettings, architecture_contract, build_model
from nig_timenet_torch.model.contract import check_contract
from nig_timenet_torch.model.scratch_nets import squash

ARCHS = ("conv1d_net", "no_dense_net")


@pytest.mark.parametrize("arch", ARCHS)
def test_contract_and_row_independence(arch):
    torch.manual_seed(0)
    m = build_model(ModelSettings(arch), 32, 3).eval()
    x = torch.randn(8, 32, 3)
    check_contract(m, architecture_contract(ModelSettings("tiny_mlp"), 32, 3), x)
    with torch.no_grad():
        full, one = m(x), m(x[5:6])
    for k in full:
        torch.testing.assert_close(full[k][5:6], one[k], rtol=1e-4, atol=1e-6)


def test_no_dense_net_has_no_linear_and_no_pointwise_conv():
    m = build_model(ModelSettings("no_dense_net"), 32, 3)
    assert not any(isinstance(mod, torch.nn.Linear) for mod in m.modules())
    assert all(c.kernel_size[0] > 1 for c in m.modules() if isinstance(c, torch.nn.Conv1d))


def test_squash_bounds_an_outlier():
    assert float(squash(torch.tensor(1e6))) < 14.0 and float(squash(torch.tensor(-1e6))) > -14.0
    torch.testing.assert_close(squash(torch.tensor(1e-4)), torch.tensor(1e-4), rtol=1e-3, atol=0)


@pytest.mark.parametrize("arch", ARCHS)
def test_a_backward_pass_reaches_every_weight(arch):
    m = build_model(ModelSettings(arch), 32, 3).train()
    out = m(torch.randn(8, 32, 3))
    sum(v.sum() for k, v in out.items() if not k.endswith("_confidence")).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None and not n.startswith("conf_out")]
    assert not missing
