"""chart_cnn (model/chart_cnn.py): the drawing, the output contract of the other architectures, robustness of the
image to an outlier bar, row independence, and a training step that reaches every weight."""

import torch

from nig_timenet_torch.model.architectures import ModelSettings, architecture_contract, build_model
from nig_timenet_torch.model.chart_cnn import draw_charts
from nig_timenet_torch.model.contract import check_contract


def test_drawing_is_binary_and_scale_free():
    torch.manual_seed(0)
    x = torch.randn(4, 32, 3) * 0.02
    img, log_rng = draw_charts(x, 32, (0, 1, 2))
    assert img.shape == (4, 2, 32, 32) and set(img.unique().tolist()) <= {0.0, 1.0}
    assert (img[:, 1].sum(dim=1) == 1).all(), "one close pixel per column"
    img2, log_rng2 = draw_charts(x * 10, 32, (0, 1, 2))
    assert torch.equal(img, img2), "the picture does not depend on the scale"
    torch.testing.assert_close(log_rng2 - log_rng, torch.full_like(log_rng, float(torch.tensor(10.0).log())))


def test_an_outlier_bar_keeps_the_input_bounded():
    x = torch.randn(1, 32, 3) * 0.02
    x[0, 5, 0] = 50.0                                   # one absurd high
    img, _ = draw_charts(x, 32, (0, 1, 2))
    assert img.abs().max() <= 1.0


def test_same_contract_as_the_other_architectures_and_rows_are_independent():
    torch.manual_seed(0)
    m = build_model(ModelSettings("chart_cnn"), 32, 3).eval()
    x = torch.randn(8, 32, 3) * 0.02
    check_contract(m, architecture_contract(ModelSettings("tiny_mlp"), 32, 3), x)
    with torch.no_grad():
        full, one = m(x), m(x[2:3])
    for k in full:
        torch.testing.assert_close(full[k][2:3], one[k], rtol=1e-4, atol=1e-6)


def test_a_backward_pass_reaches_every_weight():
    m = build_model(ModelSettings("chart_cnn"), 32, 3).train()
    out = m(torch.randn(8, 32, 3) * 0.02)
    sum(v.sum() for k, v in out.items() if not k.endswith("_confidence")).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None and not n.startswith("conf_out")]
    assert not missing
