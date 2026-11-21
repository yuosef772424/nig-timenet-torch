

def test_stride_groups_keeps_whole_groups_evenly_over_the_range():
    """C-02: the health report's subsample of a cross-section run must not cut a timestamp."""
    import numpy as np

    from nig_timenet_torch.workflow.run_common import stride_groups

    gid = np.repeat(np.arange(50), 7)                  # 50 timestamps x 7 coins
    pos = stride_groups(gid, 70)
    kept = np.unique(gid[pos])
    assert len(kept) == 10 and all((gid[pos] == g).sum() == 7 for g in kept)
    assert kept[0] == 0 and kept[-1] == 49            # spread over the whole range
    assert len(stride_groups(gid, None)) == len(gid) and len(stride_groups(gid, 10_000)) == len(gid)
