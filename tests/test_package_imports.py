def test_import_package():
    import nig_timenet_torch

    assert isinstance(nig_timenet_torch.__version__, str)


def test_seeds_stub_is_deterministic():
    import torch

    from nig_timenet_torch.core.seeds import seed_everything

    seed_everything(0)
    a = torch.randn(4)
    seed_everything(0)
    assert torch.equal(a, torch.randn(4))
    assert torch.are_deterministic_algorithms_enabled()
