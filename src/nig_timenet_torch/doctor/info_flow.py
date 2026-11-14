"""PURPOSE: Card A — where is information lost or gained inside the model? Walks the model's streams in forward order
    and, at every stage, measures how much of each signal (the labels, the E1 oracle means, and tracers that are
    known functions of the input) a probe can still read, next to the strong simple baseline on the same rows.
TAGS: doctor, card a, information flow, stage probe, tracer, path ablation, readout gap, effective rank, dead units
PITFALLS: a probe score is what a ridge readout (features + squares) finds, a LOWER bound of the information; the
    judged comparisons are therefore differences between stages measured the same way, and a stage is judged against
    its PARENT streams (the model is a graph: stats / level / linear_path are side paths). Rows are VALIDATION rows
    (never used to fit the weights). A model without stream taps (``model._tap``) is walked by module hooks, in
    registration order, each stage against the one before it.

``info_flow(ctx) -> (checks, table)``: ``table`` has one row per stage (scores of every signal, delta vs the parents,
effective rank, constant share); the Checks are the readable verdicts:

* ``trunk vs simple baseline`` per head: is what predicts the label still there when it reaches the heads?
* ``readout gap`` per head: does the head's own mu use what the trunk holds?
* ``largest loss`` per signal: the stage with the biggest drop that is not recovered at the trunk;
* ``path use``: how much each path changes the outputs when its rows are shuffled (an unused path is dead weight);
* ``training gain``: the same trunk probe on an untrained twin (does training create information?);
* ``instrument self-check``: a deliberately destroyed stage must be the one the instrument names.
"""

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.doctor.common import (
    Check,
    DoctorContext,
    fresh_trainer,
    judge,
    probe_scores,
    reg_key,
    row_times,
    simple_features,
    time_folds,
)
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.taps import record_taps, stream_taps, tap_points
from nig_timenet_torch.stats.metrics import spearman

__all__ = ["info_flow", "collect_streams", "tracers", "stage_table", "PARENTS", "ABLATION_PATHS", "destroy_rows"]

#: stream -> the streams it is computed from (NigTimeNetV2.branch); a stage is judged against the best of them
PARENTS = {"scaled": ("input",), "inst_norm": ("scaled", "input"), "stats": ("scaled", "input"),
           "level": ("scaled", "input"), "tokens": ("inst_norm", "level"), "enc_in": ("tokens",),
           "enc_out": ("enc_in",), "pooled": ("enc_out", "stats"), "readout": ("pooled",),
           "linear_path": ("inst_norm", "level"), "branch_out": ("readout", "linear_path"), "trunk": ("branch_out",)}
#: the paths whose rows are shuffled one at a time to see how much the outputs use them
ABLATION_PATHS = ("stats", "level", "inst_norm", "enc_out", "readout", "linear_path")
_FLATTEN_MAX = 512        # a (T, D) stream up to this many values is probed flat, a wider one as [last, mean, sd]
_TOL = 0.03               # probe noise: differences below this are not read


def _pool(t: torch.Tensor) -> np.ndarray:
    t = t.detach().float().cpu()
    if t.dim() == 3:
        if t.shape[1] * t.shape[2] <= _FLATTEN_MAX:
            return t.flatten(1).numpy()
        return torch.cat([t[:, -1], t.mean(dim=1), t.std(dim=1)], dim=1).numpy()
    return t.reshape(t.shape[0], -1).numpy()


def destroy_rows(v: torch.Tensor) -> torch.Tensor:
    """The row <-> value link of a stream broken with a fixed row permutation (its distribution is kept)."""
    g = torch.Generator().manual_seed(0)
    return v[torch.randperm(v.shape[0], generator=g).to(v.device)]


@torch.no_grad()
def collect_streams(model, x: torch.Tensor, *, edit: dict | None = None, chunk: int = 2048) -> tuple[dict, dict]:
    """({stage: pooled activations (N, d)} in forward order, {output key: (N,)}) of ``model(x)`` in eval mode.
    ``edit`` ({stream: fn}) replaces streams (ablation). Stages: the model's stream taps, else every leaf module;
    plus ``head_<t>`` = the hidden layer of each NIG head (silu(fc(trunk)))."""
    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    acts: dict[str, list] = {}
    outs: dict[str, list] = {}
    heads = model.head_modules() if hasattr(model, "head_modules") else {}
    names = {m: n for n, m in model.named_modules()}
    try:
        for i in range(0, len(x), chunk):
            xb = x[i:i + chunk].to(dev)
            if hasattr(model, "_tap"):
                with stream_taps(model, edit) as taps, record_taps(model, [names[h.fc] for h in heads.values()]) as hk:
                    out = model(xb)
                taps = dict(taps)
                for t, h in heads.items():
                    taps[f"head_{t}"] = torch.nn.functional.silu(hk[names[h.fc]])
            else:
                if edit:
                    raise TypeError("stream edits need a model with stream taps")
                with record_taps(model, tap_points(model, leaves_only=True)) as hk:
                    out = model(xb)
                taps = {"input": xb, **hk}
            for k, v in taps.items():
                acts.setdefault(k, []).append(_pool(v))
            for k, v in out.items():
                outs.setdefault(k, []).append(v.float().cpu().numpy().reshape(len(xb), -1)[:, 0])
    finally:
        model.train(was)
    return {k: np.concatenate(v) for k, v in acts.items()}, {k: np.concatenate(v) for k, v in outs.items()}


def tracers(x: np.ndarray, feature_order) -> dict:
    """Signals that are KNOWN functions of the raw window, so their probe score at the input is ~1 and any later
    drop is a loss inside the model: ``vol_level`` (log of the mean bar range, or of the mean absolute value without
    high / low channels), ``last_move`` (the last bar of the close channel, or of channel 0), ``drift`` (its mean)."""
    x = np.asarray(x, dtype=np.float64)
    fo = list(feature_order)
    c = fo.index("close") if "close" in fo else 0
    if "high" in fo and "low" in fo:
        level = (x[:, :, fo.index("high")] - x[:, :, fo.index("low")]).mean(axis=1)
    else:
        level = np.abs(x[:, :, c]).mean(axis=1)
    return {"vol_level": np.log(np.maximum(level, 1e-12)), "last_move": x[:, -1, c], "drift": x[:, :, c].mean(axis=1)}


def _eff_rank(a: np.ndarray) -> float:
    """Participation ratio (sum l)^2 / sum l^2 of the covariance eigenvalues: how many directions carry variance."""
    a = torch.from_numpy(np.asarray(a, dtype=np.float64))
    c = torch.cov(a.T).reshape(a.shape[1], a.shape[1])      # (sum l)^2 / sum l^2 = trace(C)^2 / ||C||_F^2
    return float(torch.trace(c) ** 2 / torch.clamp_min((c * c).sum(), 1e-300))


def _parents(stage: str, order: list[str]) -> list[str]:
    if stage.startswith("enc_block_"):
        i = int(stage.rsplit("_", 1)[1])
        return ["enc_in" if i == 1 else f"enc_block_{i - 1}"]
    if stage == "enc_out":
        blocks = [s for s in order if s.startswith("enc_block_")]
        return [blocks[-1]] if blocks else ["enc_in"]
    if stage.startswith("head_"):
        return ["trunk"]
    if stage in PARENTS:
        return [p for p in PARENTS[stage] if p in order][:1] if stage in ("inst_norm", "stats", "level") else [
            p for p in PARENTS[stage] if p in order]
    i = order.index(stage)
    return order[i - 1:i]


def stage_table(acts: dict, signals: dict, folds, metric: dict) -> pd.DataFrame:
    """One row per stage: ``<signal>`` = the probe score (``metric[signal]``: "ic" or "r2"), ``d_<signal>`` = the
    score minus the best parent's, ``dim`` / ``eff_rank`` / ``const_share`` of the pooled activations."""
    order = list(acts)
    scores = {s: probe_scores(acts[s], signals, folds) for s in order}
    rows = []
    for s in order:
        a = acts[s]
        sd = a.std(0)
        row = {"stage": s, "dim": int(a.shape[1]), "eff_rank": _eff_rank(a),
               "const_share": float((sd <= 1e-6 * max(float(sd.max()), 1e-300)).mean())}
        par = [p for p in _parents(s, order) if p in scores]
        for k in signals:
            v = scores[s][k][metric[k]]
            row[k] = v
            row[f"d_{k}"] = v - max(scores[p][k][metric[k]] for p in par) if par else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _largest_loss(table: pd.DataFrame, k: str, end: str = "trunk") -> tuple[str, float, float]:
    """(stage, its drop vs its parents, the score at ``end``) of the most negative ``d_<k>`` on the way to ``end``
    (head stages are judged by the readout check)."""
    t = table[~table["stage"].str.startswith("head_")]
    i = t[f"d_{k}"].idxmin()
    at_end = float(table.loc[table["stage"] == end, k].iloc[0]) if (table["stage"] == end).any() else float("nan")
    return str(t.loc[i, "stage"]), float(t.loc[i, f"d_{k}"]), at_end


def info_flow(ctx: DoctorContext, *, max_rows: int | None = None) -> tuple[list[Check], pd.DataFrame]:
    max_rows = max_rows or (3000 if ctx.quick else 12000)
    ds, model = ctx.dataset, ctx.model
    idx, x, y = ctx.xy("val", max_rows)
    xn = x.numpy()
    fo = list(ds["feature_order"])
    bar = int(pd.Timedelta(ds["base_tf"]).value)
    folds = time_folds(row_times(ds, idx), 4, embargo=bar * (x.shape[1] + int(ds["forecast_horizon"])))
    signals = {f"y_{t}": y[reg_key(t)] for t in ctx.targets}
    metric = {k: "ic" for k in signals}
    if ctx.is_e1:
        for t in ctx.targets:
            signals[f"oracle_{t}"] = np.asarray(ds[f"oracle_{reg_key(t)}_mean"])[idx]
            metric[f"oracle_{t}"] = "r2"
    tr = tracers(xn, fo)
    signals.update(tr)
    metric.update({k: "r2" for k in tr})

    acts, outs = collect_streams(model, x)
    table = stage_table(acts, signals, folds, metric)
    end = "trunk" if "trunk" in acts else list(acts)[-1]
    feats, _ = simple_features(xn, fo)
    base = probe_scores(feats, signals, folds)
    checks: list[Check] = []

    def row(stage):
        return table[table["stage"] == stage].iloc[0]

    # 1. what predicts each label: still there at the trunk? does the head's mu use it?
    for t in ctx.targets:
        k = f"y_{t}"
        b, tk = base[k]["ic"], float(row(end)[k])
        mu = outs[head_spec("nig_regression", t).output_keys["mu"]]
        mu_ic = float(spearman(signals[k], mu))
        if b <= _TOL and tk <= _TOL:
            verdict, action = "info", ("neither the simple baseline nor the trunk reads this label from the inputs: "
                                       "a data limit, not a model fault (card C, input information)")
        else:
            verdict = judge(tk - b, fail_below=-max(2 * _TOL, 0.2 * abs(b)), warn_below=-_TOL)
            action = "" if verdict == "pass" else ("the trunk holds less of the label than the simple features: "
                                                   "read the stage table for the stage that loses it")
        checks.append(Check("A", f"model/{end}", f"{t}: probe IC at the {end} vs simple baseline", tk, b, verdict,
                            action, {"baseline_r2": base[k]["r2"]}))
        gap = mu_ic - tk
        checks.append(Check("A", f"model/head_{t}", f"{t}: IC of the model's own mu vs the {end} probe", mu_ic, tk,
                            "info" if max(mu_ic, tk) <= _TOL else judge(gap, fail_below=-max(2 * _TOL, 0.2 * abs(tk)),
                                                                        warn_below=-_TOL),
                            "" if gap >= -_TOL else "the head does not use what the trunk holds: head parametrisation "
                            "or optimisation (card B: one-batch overfit, mode gap, loss balance)"))
    # 2. the stage that loses each signal most
    for k in signals:
        stage, drop, at_end = _largest_loss(table, k, end)
        at_in = float(row("input")[k]) if "input" in acts else float("nan")
        lost = drop < -3 * _TOL and at_end < max(at_in, base[k][metric[k]]) - 3 * _TOL
        if k in tr or k.startswith("oracle_"):
            checks.append(Check("A", f"model/{stage}", f"{k}: largest loss on the way to the {end} ({metric[k]})",
                                drop, f"input {at_in:.3f} -> {end} {at_end:.3f}", "warn" if lost else "pass",
                                f"stage `{stage}` erases `{k}` and it is not restored: give that stage a bypass or "
                                "remove the normalisation that erases it" if lost else ""))
    # 3. constant features and collapsed stages
    for _, r in table.iterrows():
        if r["const_share"] > 0.5 or r["eff_rank"] < 1.5:
            checks.append(Check("A", f"model/{r['stage']}", "constant share / effective rank",
                                f"{r['const_share']:.2f} / {r['eff_rank']:.1f} of {r['dim']}", "", "warn",
                                "most features of this stage are constant over rows (dead units or a saturated "
                                "normalisation)"))
    # 4. path use: shuffle one path's rows, how much do the outputs move?
    if hasattr(model, "_tap"):
        for path in [p for p in ABLATION_PATHS if p in acts]:
            _, ab = collect_streams(model, x, edit={path: destroy_rows})
            change = {t: 1.0 - abs(float(np.corrcoef(outs[head_spec("nig_regression", t).output_keys["mu"]],
                                                     ab[head_spec("nig_regression", t).output_keys["mu"]])[0, 1]))
                      for t in ctx.targets}
            m = max(change.values())
            checks.append(Check("A", f"model/{path}", "path use: 1 - |corr(mu, mu with this path's rows shuffled)|",
                                m, "0 = the outputs ignore the path", "warn" if m < 0.01 else "info",
                                "the outputs do not depend on this path (dead weight, or its gradient is blocked)"
                                if m < 0.01 else "", change))
    # 5. does training create information? the same probe on an untrained twin
    twin = fresh_trainer(ctx, seed=ctx.seed + 1000).model
    acts0, _ = collect_streams(twin, x)
    if end in acts0:
        s0 = probe_scores(acts0[end], signals, folds)
        for t in ctx.targets:
            k = f"y_{t}"
            gain = float(row(end)[k]) - s0[k]["ic"]
            checks.append(Check("A", f"model/{end}", f"{t}: training gain of the {end} probe IC (trained - untrained)",
                                gain, f"untrained {s0[k]['ic']:.3f}",
                                "info" if abs(base[k]["ic"]) <= _TOL else judge(gain, warn_below=-_TOL),
                                "training did not add readable information to the trunk (a random encoder holds as "
                                "much): the loss or the optimiser is not shaping the representation (card B)"
                                if gain < -_TOL else ""))
    # 6. instrument self-check: destroy one stage on purpose, the table must name it
    if hasattr(model, "_tap") and "branch_out" in acts:
        acts_d, _ = collect_streams(model, x, edit={"branch_out": destroy_rows})
        td = stage_table({k: acts_d[k] for k in acts_d if not k.startswith("head_")}, {"vol_level": tr["vol_level"]},
                         folds, {"vol_level": "r2"})
        r = td[td["stage"] == "branch_out"].iloc[0]
        before, after = float(r["vol_level"] - r["d_vol_level"]), float(td[td["stage"] == end]["vol_level"].iloc[0])
        if before < 0.4:
            checks.append(Check("A", "instrument", "self-check: a destroyed stage (branch_out) is seen", "n/a",
                                f"tracer R2 before the stage {before:.2f}", "info",
                                "the tracer is too weak before the planted stage to self-check on this model"))
        else:
            ok = r["d_vol_level"] < -0.5 * before and after < 0.5 * before
            checks.append(Check("A", "instrument", "self-check: tracer R2 across a destroyed stage (branch_out)",
                                f"{before:.2f} -> {after:.2f}", "must fall to ~0", "pass" if ok else "fail",
                                "" if ok else "the information-flow instrument did not see a planted loss: do not "
                                "trust its table until this passes"))
    ctx.notes["info_flow_baseline"] = {k: base[k][metric[k]] for k in signals}
    return checks, table
