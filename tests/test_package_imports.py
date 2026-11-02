def test_import_package():
    import nig_timenet_torch

    assert isinstance(nig_timenet_torch.__version__, str)


def test_seeds_stub_is_deterministic():
    import torch

    from nig_timenet_torch.core.seeds import set_determinism

    set_determinism(0)
    a = torch.randn(4)
    set_determinism(0)
    assert torch.equal(a, torch.randn(4))
    assert torch.are_deterministic_algorithms_enabled()
