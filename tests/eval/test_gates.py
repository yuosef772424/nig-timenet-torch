"""Gates B / C / canary / rule D (eval/gates.py) on known answers: the oracle of E1 passes every blocking check (its
high / low coverage judged relative to itself, D-019), a constant confidence fails B, a zero std fails C coverage, a
label leak fails the canary on E2; summarize ANDs over seeds (ROADMAP G2)."""

import numpy as np
import pytest

from nig_timenet_torch.data_io.environments import permute_labels
from nig_timenet_torch.data_io.synthetic import oracle_keys
from nig_timenet_torch.eval import gates
from nig_timenet_torch.eval.target_report import target_report
from nig_timenet_torch.eval.vol_baseline import window_ranges
from nig_timenet_torch.model.heads import head_spec
from tests.data_io import envs

TARGETS = ("high", "low", "close")


@pytest.fixture(scope="module")
def e1():
    return envs.small(n_coins=32, n_bars=800)


def _stage(ds, table, labels, anchors, last_close, state, env="E1", seed=0, oracle=True, losses=None):
    rep = target_report(table, labels, anchors=anchors, last_close=last_close, codec_state=state)
    orc = {t: tuple(np.asarray(ds[k]) for k in oracle_keys(t)) for t in TARGETS} if oracle else None
    x, fo = np.asarray(ds[f"X_{ds['base_tf']}"]), list(ds["feature_order"])
    return gates.GateStage(env=env, seed=seed, table=table, labels=labels, report=rep, anchors=anchors,
                           last_close=last_close, codec_state=state, oracle=orc, losses=losses,
                           ranges=window_ranges(x, fo), last_move=x[:, -1, fo.index("close")])


def _blocking(rows, gate=None):
    return [r for r in rows if r["blocking"] and (gate is None or r["gate"] == gate)]


def _set(table, t, key, values):
    table[head_spec("nig_regression", t).output_keys[key]] = values


@pytest.fixture(scope="module")
def oracle_rows(e1):
    return gates.stage_rows(_stage(e1, *envs.oracle_table(e1)))


def test_oracle_passes_every_blocking_e1_check(oracle_rows):
    rows = oracle_rows
    bad = [(r["gate"], r["check"], r["head"], r["value"]) for r in _blocking(rows) if r["verdict"] == "fail"]
    assert not bad
    cov = {r["head"]: r for r in rows if r["check"] == "coverage_80"}
    for t in ("high", "low"):
        assert cov[t]["mode"] == "relative to the oracle" and cov[t]["value"] > gates.COVERAGE_BAND[1], \
            "B-056: the TRUE mean / sd over-covers with a symmetric interval; judged relative to the oracle"
    assert cov["close"]["mode"] == "band"
    assert {r["gate"] for r in rows} == {"B", "C", "S", "D"}
    assert not any(r["blocking"] for r in rows if r["gate"] in ("S", "D"))


def test_constant_confidence_fails_gate_b(e1):
    table, *rest = envs.oracle_table(e1)
    for t in TARGETS:
        _set(table, t, "confidence", np.zeros(len(table)))
    rows = gates.stage_rows(_stage(e1, table, *rest))
    failed = {(r["check"], r["head"]) for r in _blocking(rows, "B") if r["verdict"] == "fail"}
    assert {("decile_error_spearman", t) for t in TARGETS} <= failed
    assert not [r for r in _blocking(rows, "C") if r["verdict"] == "fail"], "gate C is not affected"


def test_zero_std_fails_gate_c_coverage(e1):
    table, *rest = envs.oracle_table(e1)
    for t in TARGETS:
        _set(table, t, "beta", np.full(len(table), 1e-14))
    rows = gates.stage_rows(_stage(e1, table, *rest))
    cov = {r["head"]: r["verdict"] for r in rows if r["check"] == "coverage_80"}
    assert cov == {t: "fail" for t in TARGETS}


def test_label_leak_fails_the_canary_on_e2(e1):
    e2 = permute_labels(e1, seed=1)
    table, labels, anchors, last_close, state = envs.oracle_table(e1)      # predictions of the source rows
    labels = {k: np.asarray(e2[k]) for k in labels}
    honest = gates.stage_rows(_stage(e2, table.copy(), labels, anchors, last_close, state, env="E2", oracle=False,
                                     losses={"model": -1.0, "constant": -1.0}))
    canary = [r for r in honest if r["gate"] == "canary"]
    assert canary and all(r["verdict"] == "pass" for r in canary), [(r["head"], r["value"]) for r in canary]
    for t in TARGETS:                                                       # the leak: mu IS the (permuted) label
        _set(table, t, "mu", labels[head_spec("nig_regression", t).label_key])
    leak = gates.stage_rows(_stage(e2, table, labels, anchors, last_close, state, env="E2", oracle=False,
                                   losses={"model": -2.0, "constant": -1.0}))
    failed = {(r["check"], r["head"]) for r in leak if r["gate"] == "canary" and r["verdict"] == "fail"}
    assert {("rank_ic_vs_shuffled", t) for t in TARGETS} | {("val_loss_vs_constant", "all")} <= failed


def test_relative_rule_only_when_the_oracle_fails_the_band():
    assert gates._band(0.80, 0.75, 0.85, oracle=0.80) == ("pass", "band")
    assert gates._band(0.88, 0.75, 0.85, oracle=0.80) == ("fail", "band")
    assert gates._band(0.88, 0.75, 0.85, oracle=0.884) == ("pass", "relative to the oracle")
    assert gates._band(0.80, 0.75, 0.85, oracle=0.884)[0] == "fail", "half width 0.05 around the oracle"
    assert gates._at_most(-0.75, -0.8, oracle=-0.7) == ("pass", "relative to the oracle")
    assert gates._at_least(np.nan, 0.7)[0] == "fail"


def _row(env, seed, check, verdict, gate="B"):
    return {"env": env, "seed": seed, "gate": gate, "check": check, "head": "close", "verdict": verdict,
            "blocking": True}


def test_summarize_ands_over_seeds_and_needs_every_env():
    rows = [_row(e, s, "x", "pass") for e in gates.ENVS for s in range(3)]
    assert gates.summarize(rows)["passed"]
    assert not gates.summarize(rows + [_row("E2", 1, "x", "fail")])["passed"], "one failing seed fails the gate"
    assert not gates.summarize([r for r in rows if r["env"] != "E3"])["passed"], "an env not run cannot pass"
    a6 = [_row("E1", s, "A6", v, gate="A") for s, v in enumerate(("fail", "fail", "pass"))]
    s = gates.summarize(rows + a6)
    assert s["passed"] and s["per_check"]["E1|A|A6|close"] == "warn", "A6 blocks only when it fails on every seed"
    a6[2]["verdict"] = "fail"
    assert not gates.summarize(rows + a6)["passed"]
    info = {**_row("E3", 0, "coverage_80", "fail", gate="C"), "blocking": False}
    assert gates.summarize(rows + [info])["passed"], "information rows never block"


def test_report_markdown_lists_every_row(oracle_rows):
    rows = oracle_rows
    rep = {"key": "k" * 64, "summary": gates.summarize(rows), "rows": rows, "baseline_arm": True}
    md = gates.report_markdown(rep)
    assert md.count("\n| E1 |") == len(rows) + 1 and "Failed checks" in md     # + the verdict table line
