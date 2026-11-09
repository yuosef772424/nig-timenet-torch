"""PURPOSE: The representative batch set of MODEL_DEV_PROTOCOL section 1: R batches of the training split drawn with a
    fixed seed, each holding rows of >= 32 coins (all if fewer) spread over the whole training time range.
TAGS: representative batch set, gradient gate, trainer trace, stratified batches, coins, time range, fixed seed
PITFALLS: never "the first rows" (the dataset is coin-major, so the first rows are one coin's earliest windows, H0
    audit); indices are positions in the TRAINING arrays the caller passes, not artifact rows; a coin with fewer rows
    than its share gives what it has and the batch is topped up from the other coins, so a batch never repeats a row.

``representative_batches(coins, times, batch_size)`` returns ``n_batches`` index arrays. Per batch: ``min_coins``
coins are drawn (all of them when there are fewer), each coin gets an equal share of the batch, and a coin's share is
taken one row per equal-count time stratum of that coin (a random row inside each stratum), so every batch covers the
coin's whole time range. Used by ``trainer.grad_report`` (gate A) and by ``Trainer.fit(trace_batches=...)``.
"""

import numpy as np

__all__ = ["representative_batches", "N_BATCHES", "MIN_COINS"]

N_BATCHES = 16                      # R of MODEL_DEV_PROTOCOL section 1
MIN_COINS = 32                      # coins per batch (all coins if fewer)


def _coin_rows(coin_rows: np.ndarray, times: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` rows of one coin, one per equal-count time stratum (all rows when the coin has <= k)."""
    if coin_rows.shape[0] <= k:
        return coin_rows
    order = coin_rows[np.argsort(times[coin_rows], kind="stable")]
    edges = np.linspace(0, order.shape[0], k + 1).astype(np.int64)
    return np.array([order[rng.integers(edges[i], edges[i + 1])] for i in range(k)], dtype=np.int64)


def representative_batches(coins, times, batch_size: int, *, n_batches: int = N_BATCHES, min_coins: int = MIN_COINS,
                           seed: int = 0) -> list[np.ndarray]:
    """``n_batches`` arrays of row positions (each of length ``min(batch_size, rows)``, no repeated row inside a
    batch) into arrays aligned with ``coins`` (one coin id per training row) and ``times`` (one int time per row)."""
    coins, times = np.asarray(coins), np.asarray(times).astype(np.int64)
    if coins.shape != times.shape or coins.ndim != 1 or coins.shape[0] == 0:
        raise ValueError("coins and times must be non-empty 1-D arrays of the same length")
    if batch_size < 1 or n_batches < 1 or min_coins < 1:
        raise ValueError("batch_size, n_batches and min_coins must be >= 1")
    rng = np.random.default_rng(seed)
    names, inv = np.unique(coins, return_inverse=True)
    by_coin = [np.flatnonzero(inv == i) for i in range(names.shape[0])]
    size = min(batch_size, coins.shape[0])
    out = []
    for _ in range(n_batches):
        chosen = rng.permutation(names.shape[0])[:min(min_coins, names.shape[0])]
        share = np.full(chosen.shape[0], size // chosen.shape[0])
        share[:size - share.sum()] += 1
        rows = np.concatenate([_coin_rows(by_coin[c], times, int(k), rng) for c, k in zip(chosen, share, strict=True)])
        if rows.shape[0] < size:                     # small coins: top up from the rows not taken yet
            rest = np.setdiff1d(np.arange(coins.shape[0]), rows)
            rows = np.concatenate([rows, rng.choice(rest, size - rows.shape[0], replace=False)])
        out.append(np.sort(rows))
    return out
