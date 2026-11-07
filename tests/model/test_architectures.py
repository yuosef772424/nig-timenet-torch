import torch

from nig_timenet_torch.model.architectures import (
    ARCHITECTURES,
    NIG_TIMENET_V2_DEFAULT_CONFIG,
    ModelSettings,
    build_model,
)
from nig_timenet_torch.model.builder import NigTimeNetV2
from nig_timenet_torch.model.contract import check_contract, contract_of
from tests.oracle import oracle_loader as ol


def test_default_config_is_the_oracle_constructor_kwargs():
    kw = dict(ol.load_config()["constructor_kwargs"])
    kw.pop("seq_len"), kw.pop("n_features")
    assert NIG_TIMENET_V2_DEFAULT_CONFIG == kw


def test_default_architecture_builds_and_matches_contract():
    kw = ol.load_config()["constructor_kwargs"]
    meta = ARCHITECTURES.entry("nig_timenet_v2").meta
    torch.manual_seed(0)
    model = ARCHITECTURES.get("nig_timenet_v2")(kw["seq_len"], kw["n_features"], **meta["default_config"])
    x = ol.load_inputs()
    oracle_model = meta["build_fn"](**kw)
    check_contract(model, contract_of(oracle_model, x), x)


def test_architecture_default_is_nig_timenet_v2():
    assert ModelSettings().architecture == "nig_timenet_v2"
    assert isinstance(build_model(ModelSettings(), 16, 18), NigTimeNetV2)
