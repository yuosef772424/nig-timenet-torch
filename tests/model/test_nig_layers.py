"""P14: NIG heads, OrderedMeans, NIGUncertainty, ConfidenceHead, classification head vs sublayers nig_*/ and
layer_outputs 018-036. Parametrized over the three price targets; inputs are oracle tensors (layers.json 'inputs')."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.model.heads import classification_head_forward
from nig_timenet_torch.model.layers import keras_dense_weight_to_torch
from nig_timenet_torch.model.nig_layers import ConfidenceHead, NIGHead, NIGUncertainty, OrderedMeans
from tests.oracle import oracle_loader as ol

D = 16
TARGETS = ["close", "high", "low"]


@pytest.fixture(scope="module")
def w():
    return ol.load_weights()


@pytest.fixture(scope="module")
def lo():
    return ol.load_layer_outputs()


@pytest.fixture(scope="module")
def sub():
    return ol.load_sublayers()


@pytest.fixture(scope="module")
def meta():
    return {m["name"]: m for m in ol.load_layers_meta()}


def out_keys(meta, name):
    return meta[name]["output_keys"]


def copy_linear(lin: nn.Linear, w, prefix: str) -> nn.Linear:
    with torch.no_grad():
        lin.weight.copy_(keras_dense_weight_to_torch(w[f"{prefix}.kernel"]))
        lin.bias.copy_(w[f"{prefix}.bias"])
    return lin


def nig_head(w, t) -> NIGHead:
    head = NIGHead(D, hidden=128, dropout=0.1, nu_min=0.1, alpha_min=2.0, beta_min=0.01)
    copy_linear(head.fc, w, f"nig_{t}.fc")
    copy_linear(head.out, w, f"nig_{t}.out")
    return head.eval()


@pytest.mark.parametrize("t", TARGETS)
@torch.no_grad()
def test_fc_out_matches_oracle(w, sub, t):
    head = nig_head(w, t)
    ol.assert_close(F.silu(head.fc(sub[f"nig_{t}/00_head_input"])), sub[f"nig_{t}/01_fc_out"], "forward", t)


@pytest.mark.parametrize("t", TARGETS)
@torch.no_grad()
def test_raw_params_matches_oracle(w, sub, t):
    p = nig_head(w, t).raw_params(sub[f"nig_{t}/00_head_input"])
    ol.assert_close(p, sub[f"nig_{t}/02_raw_params"], "forward", t)
    for i, name in enumerate(["mu", "nu", "alpha", "beta"]):
        ol.assert_close(p[:, i:i + 1], sub[f"nig_{t}/03_raw_{name}"], "forward", f"{t} raw {name}")
    mu = nig_head(w, t)(sub[f"nig_{t}/00_head_input"])[0]
    ol.assert_close(mu, p[:, 0:1], "weights", "mu is raw column 0 (linear, no floor)")


@pytest.mark.parametrize("t", TARGETS)
@torch.no_grad()
def test_softplus_floors_match_oracle(sub, t):
    for i, (name, floor) in enumerate([("nu", 0.1), ("alpha", 2.0), ("beta", 0.01)], start=1):
        sp = F.softplus(sub[f"nig_{t}/02_raw_params"][:, i:i + 1])
        ol.assert_close(sp, sub[f"nig_{t}/04_softplus_{name}"], "forward", f"{t} softplus {name}")
        ol.assert_close(sp + floor, sub[f"nig_{t}/05_final_{name}"], "forward", f"{t} final {name}")


@pytest.mark.parametrize("t", TARGETS)
@torch.no_grad()
def test_final_outputs_match_layer_outputs(w, lo, meta, t):
    assert meta[f"nig_{t}"]["inputs"] == ["trunk_drop"]
    outs = nig_head(w, t)(lo["017_trunk_drop"])
    for got, k in zip(outs, out_keys(meta, f"nig_{t}"), strict=True):
        ol.assert_close(got, lo[k], "forward", k)


@torch.no_grad()
def test_ordered_means_matches_oracle(lo, meta):
    assert meta["ordered_means"]["inputs"] == ["nig_close[0]", "nig_high[0]", "nig_low[0]"]
    mu_c, raw_h, raw_l = (lo[out_keys(meta, f"nig_{t}")[0]] for t in ("close", "high", "low"))
    for got, k in zip(OrderedMeans()(mu_c, raw_h, raw_l), out_keys(meta, "ordered_means"), strict=True):
        ol.assert_close(got, lo[k], "forward", k)


@pytest.mark.parametrize("t", TARGETS)
@torch.no_grad()
def test_nig_uncertainty_matches_oracle(lo, meta, t):
    nu, alpha, beta = (lo[k] for k in out_keys(meta, f"nig_{t}")[1:])
    for got, k in zip(NIGUncertainty(unc_max=20.0)(nu, alpha, beta), out_keys(meta, f"unc_{t}"), strict=True):
        ol.assert_close(got, lo[k], "forward", k)


def test_nig_uncertainty_floor_and_cap():
    one = torch.ones(3, 1)
    alpha = torch.tensor([[1.0], [1.5], [3.0]])
    epi, ale = NIGUncertainty(unc_max=None)(one, alpha, one)
    torch.testing.assert_close(ale[0], torch.tensor([10.0]))  # sqrt(1 / NIG_ALPHA_DEN_MIN)
    epi, ale = NIGUncertainty(unc_max=2.0)(one, alpha, torch.tensor([[float("inf")], [1.0], [1.0]]))
    assert ale[0].item() == 2.0 and epi.max().item() <= 2.0


@pytest.mark.parametrize("t", TARGETS)
@torch.no_grad()
def test_confidence_head_matches_oracle(w, lo, meta, t):
    head = ConfidenceHead(D, hidden=32)
    copy_linear(head.d1, w, f"conf_{t}.d1")
    copy_linear(head.d2, w, f"conf_{t}.d2")
    h, nu, alpha, beta = (lo[out_keys(meta, ref.split("[")[0])[int(ref[-2]) if "[" in ref else 0]]
                          for ref in meta[f"conf_{t}"]["inputs"])
    ol.assert_close(head(h, nu, alpha, beta), lo[out_keys(meta, f"conf_{t}")[0]], "forward", f"conf_{t}")


def test_confidence_head_stops_gradient():
    head = ConfidenceHead(4)
    h = torch.randn(2, 4, requires_grad=True)
    head(h, torch.ones(2, 1), torch.full((2, 1), 3.0), torch.ones(2, 1)).sum().backward()
    assert h.grad is None and head.d1.weight.grad is not None


@pytest.mark.parametrize("t", TARGETS)
@torch.no_grad()
def test_classification_head_matches_oracle(w, lo, meta, t):
    fc = copy_linear(nn.Linear(D, 64), w, f"class_fc_{t}")
    out = copy_linear(nn.Linear(64, 1), w, f"y_{t}_class_logits")
    ol.assert_close(torch.relu(fc(lo["017_trunk_drop"])), lo[out_keys(meta, f"class_fc_{t}")[0]], "forward", t)
    prob = classification_head_forward(lo["017_trunk_drop"], fc, out, nn.Dropout(0.1).eval())
    ol.assert_close(prob, lo[out_keys(meta, f"y_{t}_class_logits")[0]], "forward", f"y_{t}_class_logits")
