import dataclasses

import pytest
import torch

from nig_timenet_torch.model.builder import build_nig_timenet_v2
from nig_timenet_torch.model.contract import ContractMismatchError, check_contract, contract_of
from tests.oracle import oracle_loader as ol


def _model_and_input():
    kw = ol.load_config()["constructor_kwargs"]
    torch.manual_seed(0)
    model = build_nig_timenet_v2(**kw)
    return model, torch.randn(3, kw["seq_len"], kw["n_features"])


def test_contract_of_matches_nig_timenet_v2():
    model, x = _model_and_input()
    c = contract_of(model, x)
    assert c.inputs[0].shape == (None, x.shape[1], x.shape[2]) and c.inputs[0].dtype == "float32"
    names = {o.name: o for o in c.outputs}
    assert len(names) == 24
    expected = set(ol.load_outputs())
    assert set(names) == expected
    for t in ("high", "low", "close"):
        assert names[f"y_{t}_class_logits"].head_kind == "binary_classification"
        for suffix in ("", "_nu", "_alpha", "_beta", "_epistemic", "_aleatoric", "_confidence"):
            assert names[f"y_{t}{suffix}"].head_kind == "nig_regression"
    oracle = ol.load_outputs()
    for name, spec in names.items():
        assert spec.shape == (None, *oracle[name].shape[1:]), name
    check_contract(model, c, x)


def test_check_contract_raises_on_mismatch():
    model, x = _model_and_input()
    c = contract_of(model, x)
    bad = tuple(dataclasses.replace(o, shape=(None, 2)) if o.name == "y_close" else o for o in c.outputs)
    msg = r"outputs\['y_close'\]\.shape: expected \(None, 2\), got \(None, 1\)"
    with pytest.raises(ContractMismatchError, match=msg):
        check_contract(model, dataclasses.replace(c, outputs=bad), x)
