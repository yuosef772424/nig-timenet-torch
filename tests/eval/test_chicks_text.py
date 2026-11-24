"""eval/chicks_text.py: the two languages hold the same messages, every key the code uses exists, nothing is blank."""

import re
from pathlib import Path

import pytest

from nig_timenet_torch.eval import chicks_text
from nig_timenet_torch.eval.chicks_text import AR, EN, Say, t

SRC = Path(chicks_text.__file__).parent.parent


def test_both_languages_have_exactly_the_same_keys():
    assert set(AR) == set(EN), (set(AR) ^ set(EN))
    assert all(AR[k].strip() and EN[k].strip() for k in AR)


def test_every_key_the_chicks_code_uses_is_in_the_catalog():
    used = set()
    files = [*(SRC / "eval").glob("chicks_*.py"), SRC / "workflow" / "chicks_run.py"]
    for p in files:
        if p.name == "chicks_text.py":
            continue
        s = p.read_text(encoding="utf-8")
        used |= set(re.findall(r'\b(?:say|say\.text)\(\s*"([a-z_0-9]+)"', s))
        used |= {k for k in AR if f'"{k}"' in s}               # a key chosen by a conditional or a table of keys
    called = set()
    for p in files:
        if p.name != "chicks_text.py":
            called |= set(re.findall(r'\b(?:say|say\.text)\(\s*"([a-z_0-9]+)"', p.read_text(encoding="utf-8")))
    assert called and called <= set(AR), sorted(called - set(AR))
    unused = set(AR) - used
    assert not unused, f"messages nothing uses: {sorted(unused)}"


def test_say_formats_prints_and_refuses_unknowns(capsys):
    Say("en")("trade_empty")
    assert capsys.readouterr().out == "(no trades pass the filter)\n"
    assert Say("ar").text("ver_all_ok") == "✅ جميع الأهداف والأصول اجتازت التحقق"
    with pytest.raises(ValueError, match="lang"):
        Say("fr")
    with pytest.raises(KeyError, match="no_such_key"):
        t("ar", "no_such_key")
    with pytest.raises(KeyError):
        Say("ar").text("trade_none", k=1.0)          # a missing format field is an error, never a blank
