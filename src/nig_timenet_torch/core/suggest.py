"""PURPOSE: The ONE "did you mean" helper: the closest known name to a mistyped one, and the text "'a' (did you mean
    'b'?), 'c'" every strict config / registry / coin-list error uses.
TAGS: did you mean, suggestion, difflib, unknown key, strict config, error message, typo
PITFALLS: matching is difflib's default cutoff (0.6) on ``str(name)``; a message that must stay byte-identical to an
    older one is built by the CALLER around ``did_you_mean`` (the wording around the hint differs per error).
"""

import difflib
from collections.abc import Callable, Iterable

__all__ = ["did_you_mean", "unknown_names_text"]


def did_you_mean(name, known: Iterable) -> str | None:
    """The known name closest to ``name`` (difflib, cutoff 0.6), or None when none is close or ``known`` is empty."""
    close = difflib.get_close_matches(str(name), [str(k) for k in known], n=1)
    return close[0] if close else None


def unknown_names_text(names: Iterable, known: Iterable, *, match: Callable[[str], str] = str) -> str:
    """``'x' (did you mean 'y'?), 'z'`` for the unknown ``names``; ``match`` maps a name to what is compared with
    ``known`` (e.g. a coin-name normaliser), the name itself is printed as given."""
    known = list(known)
    parts = []
    for name in names:
        close = did_you_mean(match(str(name)), known)
        parts.append(f"{name!r}" + (f" (did you mean {close!r}?)" if close else ""))
    return ", ".join(parts)
