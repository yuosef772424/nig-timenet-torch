"""Coin names

PURPOSE: One normaliser of coin / symbol names: the comparison key every module uses when it asks "is this the same
    coin" (build reference coin, sealed list, artifact loader, split exclusion).
TAGS: coin name, symbol, normalise, usdt suffix, sealed coin, reference coin
PITFALLS: the key drops a trailing USDT (``BTC`` and ``BTCUSDT`` are one coin) but a name that IS ``USDT`` stays;
    it is a comparison key, not the name to write into an artifact.
"""

__all__ = ["norm_coin"]


def norm_coin(symbol: str) -> str:
    """Name as the TF guard compares it: stripped, upper case, with or without the trailing USDT."""
    n = str(symbol).strip().upper()
    return n[:-4] if n.endswith("USDT") and len(n) > 4 else n
