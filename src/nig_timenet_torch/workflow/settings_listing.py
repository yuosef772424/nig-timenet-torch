"""PURPOSE: The generated list of EVERY tunable setting (RunSettings, DataSettings, RollingSettings, the model options):
    path, type, default, valid values from the registries and a one-line meaning taken from the field comments.
TAGS: settings listing, all settings, tunable settings, valid values, registries, notebook settings, settings cli
PITFALLS: walks the dataclasses, so a new field appears by itself; the meaning is the field's own comment (inline, else
    the comment block right above, else the section docstring) and the model options' meanings are the table
    MODEL_OPTION_DOCS (tests/workflow/test_settings_listing.py pins its keys to BuilderOptions); ``choices`` come from
    the registries at call time and the test checks each one against the real validation; ``EXCLUDED`` holds the
    leaves that are not a setting (none today).

Why: the owner asked that ALL the settings that can be tuned are shown (2026-11-23). ``notebooks/main.ipynb`` shows
``CHANGES`` (every RunSettings field with its default and an Arabic comment) and then this listing, so the valid values
and the settings that ``CHANGES`` cannot hold (the data pipeline's DataSettings, the model options beyond the
anti-memorization preset, the sub-sections ``train.staged`` / ``train.day_weights``, the rolling study) are on screen
too and cannot go stale: nothing here is typed by hand except the model options' meanings.

    python -m nig_timenet_torch.workflow.settings_listing [--tree RunSettings|DataSettings|RollingSettings] [--markdown]
"""

import argparse
import ast
import dataclasses
import inspect
import io
import sys
import textwrap
import tokenize
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.data.coins import COINS_BY_CATEGORY
from nig_timenet_torch.data.settings import (
    FETCH_SOURCES,
    HIGHER_TF_MODES,
    PRICE_NORM_MODES,
    SCALER_TYPES,
    SOURCE_KINDS,
    DataSettings,
)
from nig_timenet_torch.data.targets import ENTRY_CLOSE_REGS, TARGET_RAW
from nig_timenet_torch.data_io.loading import FORMATS
from nig_timenet_torch.eval.battery_scorecard import TRIM_KEYS
from nig_timenet_torch.model.architectures import ARCHITECTURES, DEFAULT_ARCHITECTURE, NIG_TIMENET_V2_DEFAULT_CONFIG
from nig_timenet_torch.model.attn_heads import NIG_HEADS
from nig_timenet_torch.model.builder import LEVEL_NORMS, BuilderOptions
from nig_timenet_torch.model.cross_coin import CROSS_COIN_KINDS, CROSS_INPUTS
from nig_timenet_torch.model.encoders import ENCODERS
from nig_timenet_torch.model.label_scale import LABEL_SCALES
from nig_timenet_torch.model.layers import STATS_TRANSFORMS, TransformerBlock
from nig_timenet_torch.model.parts import INPUT_SKIP_VARIANTS
from nig_timenet_torch.model.scalers import INPUT_SCALERS
from nig_timenet_torch.model.target_norm import TARGET_NORMS
from nig_timenet_torch.trainer.cross_section_loss import CROSS_SECTION_LOSSES
from nig_timenet_torch.trainer.groups import BATCH_ORDERS
from nig_timenet_torch.trainer.loss_balance import LOSS_BALANCERS
from nig_timenet_torch.trainer.schedules import LR_KINDS
from nig_timenet_torch.trainer.selection_metric import SELECTION_METRICS
from nig_timenet_torch.trainer.staged import GROWTH
from nig_timenet_torch.trainer.switches import INITS, WEIGHT_DECAY_GROUPS
from nig_timenet_torch.trainer.tracker import SMOOTHERS
from nig_timenet_torch.trainer.trainer import BATCHINGS
from nig_timenet_torch.workflow.settings import ROLLING_MODES, RollingSettings, RunSettings, default_model_settings
from nig_timenet_torch.workflow.settings_roles import TRAIN_FIELD_ROLES

__all__ = ["SettingRow", "TREES", "ALL_TREES", "EXCLUDED", "CHOICES", "MODEL_OPTION_DOCS", "SHOW_CHOICES", "rows",
           "all_rows", "changed_rows", "render", "show", "main"]

REQUIRED = "(required)"
DOC_WIDTH = 150
VALUE_WIDTH = 80

#: the settings trees the listing walks: name -> the dataclass and how to get a default instance (None: no instance
#: can be built, the field defaults are read; RunSettings needs a run folder, any text does here)
TREES: dict[str, tuple[type, Callable | None]] = {
    "RunSettings": (RunSettings, lambda: RunSettings(run_dir="run")),
    "DataSettings": (DataSettings, DataSettings),
    "RollingSettings": (RollingSettings, None),
}

#: leaves of a tree that are NOT a setting and so are not listed: (tree, path) -> why. None today: every field of
#: RunSettings, DataSettings and RollingSettings is something the owner can set.
EXCLUDED: dict[tuple[str, str], str] = {}

_META = "per_row"


def _names(registry) -> tuple:
    return tuple(registry.names())


#: (tree, path) -> (valid values from the registry or the vocabulary the setting's validation uses, any subset?)
#: Each callable runs at listing time, so a name registered later (a new target mode, an architecture) shows up.
CHOICES: dict[tuple[str, str], tuple[Callable[[], tuple], bool]] = {
    ("RunSettings", "data.format"): (lambda: tuple(FORMATS), False),
    ("RunSettings", "model.architecture"): (lambda: _names(ARCHITECTURES), False),
    ("RunSettings", "train.selection_metric"): (
        lambda: tuple(n for n in SELECTION_METRICS.names() if SELECTION_METRICS.entry(n).meta["selectable"]), False),
    ("RunSettings", "train.early_stop.smoothing"): (lambda: tuple(SMOOTHERS), False),
    ("RunSettings", "train.init"): (lambda: _names(INITS), False),
    ("RunSettings", "train.weight_decay_groups"): (lambda: _names(WEIGHT_DECAY_GROUPS), False),
    ("RunSettings", "train.loss_balance"): (lambda: _names(LOSS_BALANCERS), False),
    ("RunSettings", "train.cross_section_loss"): (lambda: tuple(CROSS_SECTION_LOSSES), False),
    ("RunSettings", "train.batch_by"): (lambda: tuple(BATCHINGS), False),
    ("RunSettings", "train.batch_order"): (lambda: _names(BATCH_ORDERS), False),
    ("RunSettings", "train.lr_schedule"): (lambda: tuple(LR_KINDS), False),
    ("RunSettings", "train.staged.growth"): (lambda: _names(GROWTH), False),
    ("RunSettings", "evaluation.battery.verdict_trim"): (lambda: tuple(TRIM_KEYS), False),
    ("RunSettings", "model.overrides.encoder"): (lambda: _names(ENCODERS), False),
    ("RunSettings", "model.overrides.input_skip"): (lambda: _names(INPUT_SKIP_VARIANTS), False),
    ("RunSettings", "model.overrides.stats_mode"): (lambda: tuple(STATS_TRANSFORMS), False),
    ("RunSettings", "model.overrides.level_norm"): (lambda: tuple(LEVEL_NORMS), False),
    ("RunSettings", "model.overrides.nig_scale_fit"): (lambda: _names(LABEL_SCALES), False),
    ("RunSettings", "model.overrides.input_scaler"): (lambda: _names(INPUT_SCALERS), False),
    ("RunSettings", "model.overrides.nig_head"): (lambda: tuple(NIG_HEADS), False),
    ("RunSettings", "model.overrides.ffn"): (lambda: tuple(TransformerBlock.FFN_KINDS), False),
    ("RunSettings", "model.overrides.cross_input"): (lambda: tuple(CROSS_INPUTS), False),
    ("RunSettings", "model.overrides.cross_coin_kind"): (lambda: tuple(CROSS_COIN_KINDS), False),
    ("RunSettings", "model.overrides.target_norm"): (lambda: _names(TARGET_NORMS), False),
    ("RunSettings", "model.overrides.price_targets"): (lambda: tuple(TARGET_COLUMNS), True),
    ("DataSettings", "source.kind"): (lambda: tuple(SOURCE_KINDS), False),
    ("DataSettings", "source.fetch"): (lambda: tuple(FETCH_SOURCES), False),
    ("DataSettings", "source.categories"): (lambda: (*COINS_BY_CATEGORY, "all", "all_global"), True),
    ("DataSettings", "windows.higher_tf_mode"): (lambda: tuple(HIGHER_TF_MODES), False),
    ("DataSettings", "targets.targets"): (lambda: tuple(TARGET_COLUMNS), True),
    ("DataSettings", "targets.reg_target_mode"): (
        lambda: tuple(n for n in TARGET_RAW.names() if TARGET_RAW.entry(n).meta[_META]), False),
    ("DataSettings", "targets.entry_close_reg"): (lambda: tuple(ENTRY_CLOSE_REGS), False),
    ("DataSettings", "scaling.scaler_type"): (lambda: tuple(SCALER_TYPES), False),
    ("DataSettings", "scaling.price_norm_mode"): (lambda: tuple(PRICE_NORM_MODES), False),
    ("RollingSettings", "mode"): (lambda: tuple(ROLLING_MODES), False),
}

#: the meaning of each option of the model builder (``model.builder.BuilderOptions``), which carries no comments;
#: tests/workflow/test_settings_listing.py fails when this table and BuilderOptions name different options
MODEL_OPTION_DOCS: dict[str, str] = {
    "d_model": "width of the internal representation (channels); bigger = more capacity and easier memorisation",
    "num_layers": "number of encoder blocks",
    "num_heads": "attention heads of the transformer encoders",
    "num_kv_heads": "key/value heads (grouped-query attention); must divide num_heads",
    "patch_len": "bars per patch of the transformer's patch embedding",
    "stride": "bars between two patches of the patch embedding (the model's stride, not the data windows' stride)",
    "kernel_sizes": "kernel sizes of the causal multi-scale decomposition in front of the encoder",
    "max_rel_pos": "clip of the relative-position bias of the attention (bars)",
    "dropout": "dropout of the encoder blocks and (unless readout_dropout is set) of the readout; training only",
    "attn_dropout": "dropout on the attention weights; training only",
    "causal": "True = a position attends only to earlier positions",
    "window": "attention window in tokens (None = the whole sequence)",
    "price_targets": "the price targets the model has heads for (a subset of the schema's targets)",
    "head_types": "head kinds per target (None = NIG regression + binary classification for each); class-only = "
                  "only the classification kind",
    "enforce_order": "True = the means are ordered low <= close <= high by construction",
    "head_hidden": "hidden width of the NIG regression heads",
    "nu_min": "floor of the NIG parameter nu",
    "alpha_min": "floor of the NIG parameter alpha (> 1 keeps the predictive variance finite)",
    "beta_min": "floor of the NIG parameter beta",
    "unc_max": "ceiling of the confidence (uncertainty) head output",
    "class_head_hidden": "hidden width of the classification heads",
    "encoder": "the registered encoder: transformer, conv_transformer, gru, tcn, itransformer",
    "input_skip": "how the raw input is added back into the transformer (transformer encoder only)",
    "n_classes": "classes of the multiclass head (an int, or {target: int})",
    "norm_eps": "epsilon of the per-window normalisation",
    "linear_path_l2": "L2 penalty on the linear path (stops it carrying the signal alone)",
    "input_clip": "clip of the normalised input to +-this value (None = no clip)",
    "stats_mode": "which per-window statistics the normaliser hands to the model: full, symlog or none",
    "input_noise_std": "gaussian noise added to the input; training only",
    "feature_dropout": "share of whole feature channels dropped; training only",
    "level_passthrough": "True = the last price level goes to the heads directly",
    "level_norm": "normalisation of that level: 'batch' or None",
    "order_detach_close": "True = the high / low offsets no longer pull the close mean (D-023)",
    "nig_scale": "fixed scale of the NIG heads (a number), 'train_std' = fitted on the train labels, None = unset",
    "bn_warmup": "True = BatchNorm moving statistics follow the data from the first step (D-023)",
    "readout_dropout": "dropout of the trunk and the NIG heads (None = the model's dropout)",
    "nig_trunk_grad": "share of the NIG heads' gradient that reaches the trunk (0 = a read-out, 1 = all)",
    "nig_scale_fit": "how the NIG label scale is set: fixed (the nig_scale value) or train_std (fitted on train)",
    "input_scaler": "input standardisation: none or train_std (train statistics, a fitted component)",
    "nig_head": "NIG head kind: dense, linear, scalar_joint, scalar_factorized",
    "nig_head_dropout": "attention dropout of the scalar NIG heads",
    "ffn": "feed-forward sublayer of the transformer blocks: swiglu or none (attention only)",
    "ffn_dim": "hidden width of that sublayer (None = 8/3 x d_model)",
    "cross_coin_layers": "blocks that let coins of the same timestamp see each other (0 = none; panel model)",
    "cross_coin_heads": "attention heads of those blocks",
    "cross_coin_peer_dropout": "probability that a peer coin is hidden from the others; training only",
    "cross_coin_kind": "cross-coin block: attention or mean (a mean pool)",
    "cross_input": "'relative' = every coin's window also enters relative to its timestamp's other coins (panel)",
    "cross_center": "True = the close mean leaves the model minus its mean over the rows of the timestamp (D-032)",
    "target_norm": "none or logret_std (NIG heads work on per-head train-standardised labels, D-033)",
    "har_range": "(high, low) channel indices that switch the HAR range terms of the raw window on (None = off)",
    "use_instance_norm": "False = skip the per-window normalisation (the scaled input goes in as it is)",
    "order_in_label_units": "True = the order low <= close <= high is applied on prices, after the inverse label map",
    "aux_heads": "auxiliary close-head kinds (side_payoff, upper_quantile; () = none; train.aux_* read them)",
    "aux_trunk_grad": "share of the auxiliary heads' gradient that reaches the trunk (0 = a read-out)",
}


@dataclass(frozen=True)
class SettingRow:
    tree: str                     # RunSettings | DataSettings | RollingSettings
    path: str                     # dotted path inside the tree, as ``updated({...})`` and CHANGES write it
    type: str                     # the annotation, as text
    default: str                  # repr of the default; REQUIRED when the field has none
    choices: tuple | None         # valid values from the registry / vocabulary, None = free
    subset: bool                  # choices: any subset (of a tuple field) rather than one of
    doc: str                      # one-line meaning: the field's own comment, else its section's summary
    note: str                     # "" | when the setting is read at all
    inherited: bool               # doc is the section's summary (the field has no comment of its own)


def _section_type(hint) -> type | None:
    """The dataclass a field annotation names: ``DC``, ``DC | None`` or ``Optional[DC]``; else None."""
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return hint
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        found = [a for a in typing.get_args(hint) if isinstance(a, type) and dataclasses.is_dataclass(a)]
        return found[0] if len(found) == 1 else None
    return None


def _type_text(hint) -> str:
    if isinstance(hint, type):
        return hint.__name__
    return str(hint).replace("typing.", "").replace("NoneType", "None")


def _field_docs(cls: type) -> dict[str, str]:
    """field name -> its comment: the inline comment (and the comment lines that continue it further right), else the
    comment block directly above the field. Read from the class source, so it is the text the author wrote."""
    source = textwrap.dedent(inspect.getsource(cls))
    lines = source.splitlines()
    comments = {tok.start[0]: tok for tok in tokenize.generate_tokens(io.StringIO(source).readline)
                if tok.type == tokenize.COMMENT}

    def text(row):
        return comments[row].string.lstrip("#").strip()

    def only_comment(row):
        return row in comments and lines[row - 1].strip().startswith("#")

    out = {}
    for node in ast.parse(source).body[0].body:
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        indent = node.col_offset
        parts = [text(r) for r in range(node.lineno, node.end_lineno + 1) if r in comments]
        row = node.end_lineno + 1
        while parts and row <= len(lines) and only_comment(row) and comments[row].start[1] > indent:
            parts.append(text(row))
            row += 1
        row = node.lineno - 1
        above = []
        while not parts and row >= 1 and only_comment(row) and comments[row].start[1] == indent:
            above.insert(0, text(row))
            row -= 1
        out[node.target.id] = " ".join(p for p in (parts or above) if p)
    return out


def _summary(cls: type) -> str:
    """The first sentence of the class docstring ('' for a dataclass without one: its generated doc is a signature)."""
    doc = inspect.getdoc(cls) or ""
    if doc.startswith(f"{cls.__name__}("):
        return ""
    return doc.split("\n\n")[0].replace("\n", " ").split(". ")[0].rstrip(".").strip()


def _note_of(tree: str, path: str, inside: str) -> str:
    if inside:
        return f"read only when {inside} is set"
    if tree == "RunSettings" and path.startswith("train.") and path.count(".") == 1:
        for tag in TRAIN_FIELD_ROLES.get(path.removeprefix("train."), ()):
            if tag.startswith("dormant:"):
                return f"read only while {tag.removeprefix('dormant:')} is on"
    return ""


def _make_row(tree: str, path: str, hint, value, required: bool, cls: type, docs: dict, inside: str) -> SettingRow:
    choice = CHOICES.get((tree, path))
    own = docs.get(path.rsplit(".", 1)[-1], "")
    return SettingRow(tree=tree, path=path, type=_type_text(hint), default=REQUIRED if required else repr(value),
                      choices=choice[0]() if choice else None, subset=bool(choice and choice[1]),
                      doc=own or _summary(cls), note=_note_of(tree, path, inside), inherited=not own)


def _walk(tree: str, cls: type, inst, prefix: str, inside: str):
    hints, docs = typing.get_type_hints(cls), _field_docs(cls)
    for f in dataclasses.fields(cls):
        path = prefix + f.name
        required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        if inst is not None:
            value = getattr(inst, f.name)
        elif not required:
            value = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        else:
            value = None
        sub = _section_type(hints[f.name])
        if sub is not None and isinstance(hints[f.name], type):
            yield from _walk(tree, sub, value, path + ".", inside)
            continue
        if (tree, path) not in EXCLUDED:
            yield _make_row(tree, path, hints[f.name], value, required, cls, docs, inside)
        if sub is not None:                       # an optional section (None = off): its fields, read only when on
            yield from _walk(tree, sub, sub(), path + ".", path)


def _builder_config(overrides: dict) -> dict:
    """The options ``build_model`` gives the default architecture's builder: its own defaults, then the architecture's
    default config, then ``overrides`` (the same merge as ``model.architectures.build_model``)."""
    return {**{f.name: f.default for f in dataclasses.fields(BuilderOptions)}, **NIG_TIMENET_V2_DEFAULT_CONFIG,
            **overrides}


def _model_option_rows() -> list[SettingRow]:
    """``model.overrides.<option>`` for every option of the default architecture's builder, with the value the default
    run builds it with."""
    on_by_default = default_model_settings().overrides
    effective = _builder_config(on_by_default)
    out = []
    for f in dataclasses.fields(BuilderOptions):
        path = f"model.overrides.{f.name}"
        value = effective[f.name]
        choice = CHOICES.get(("RunSettings", path))
        out.append(SettingRow(
            tree="RunSettings", path=path, type=_type_text(typing.get_type_hints(BuilderOptions)[f.name]),
            default=repr(value), choices=choice[0]() if choice else None, subset=bool(choice and choice[1]),
            doc=MODEL_OPTION_DOCS[f.name],
            note="set by the default run's overrides" if f.name in on_by_default else "", inherited=False))
    return out


def rows(tree: str) -> list[SettingRow]:
    """Every setting of one tree, in declaration order; the model options follow ``model.overrides``."""
    if tree not in TREES:
        raise ValueError(f"unknown settings tree {tree!r}; known: {sorted(TREES)}")
    cls, make = TREES[tree]
    found = list(_walk(tree, cls, make() if make else None, "", ""))
    if tree != "RunSettings":
        return found
    at = next(i for i, r in enumerate(found) if r.path == "model.overrides") + 1
    return found[:at] + _model_option_rows() + found[at:]


ALL_TREES = tuple(TREES)


def all_rows() -> list[SettingRow]:
    return [r for tree in ALL_TREES for r in rows(tree)]


def _current(obj, path: str) -> str | None:
    """repr of the value of ``path`` in a settings object; None when there is nothing to show: an optional section on
    the way is off. A model option is read as the builder gets it (the architecture's defaults under the overrides)."""
    if path.startswith("model.overrides.") and hasattr(obj, "model"):
        overrides = obj.model.overrides
        effective = _builder_config(overrides) if obj.model.architecture == DEFAULT_ARCHITECTURE else overrides
        option = path.rsplit(".", 1)[1]
        return repr(effective[option]) if option in effective else None
    here = obj
    for part in path.split("."):
        if here is None:
            return None
        here = getattr(here, part)
    return repr(here)


def _shown(r: SettingRow, current) -> str:
    now = _current(current, r.path) if current is not None and r.default != REQUIRED else None
    return "" if now is None or now == r.default else f"   <- NOW {_cut(now, VALUE_WIDTH)}"


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 3].rstrip() + "..."


def _line(r: SettingRow, current) -> str:
    values = ""
    if r.choices is not None:
        values = f"  [{'any subset of ' if r.subset else ''}{' | '.join(repr(c) for c in r.choices)}]"
    doc = "" if r.inherited else f"  # {_cut(r.doc, DOC_WIDTH)}"
    note = f" ({r.note})" if r.note else ""
    return f"  {r.path} = {_cut(r.default, VALUE_WIDTH)}{_shown(r, current)}{values}{doc}{note}"


def _group(r: SettingRow) -> tuple[str, str]:
    return r.tree, r.path.rsplit(".", 1)[0] if "." in r.path else ""


def render(listing: list[SettingRow], current) -> str:
    """The rows as text, one block per section (the section's summary on its header). ``current`` (a RunSettings, a
    DataSettings or None) adds ``<- NOW value`` to every setting that differs from its default."""
    blocks: dict[tuple[str, str], list[SettingRow]] = {}
    for r in listing:
        blocks.setdefault(_group(r), []).append(r)         # a section's fields stay together, first appearance first
    out = []
    for group, members in blocks.items():
        about = next((r.doc for r in members if r.inherited), "")
        out += ["", f"-- {group[0]}: {group[1] or '(top level)'}{f'  ({about})' if about else ''} --"]
        out += [_line(r, current if type(current).__name__ == r.tree else None) for r in members]
    return "\n".join(out).lstrip("\n")


def changed_rows(listing: list[SettingRow], current) -> list[SettingRow]:
    """The rows of ``listing`` whose value in ``current`` differs from the default."""
    out = []
    for r in listing:
        now = _current(current, r.path) if type(current).__name__ == r.tree and r.default != REQUIRED else None
        if now is not None and now != r.default:
            out.append(r)
    return out


def _all(listing: list[SettingRow], current) -> list[SettingRow]:
    return listing


def _none(listing: list[SettingRow], current) -> list[SettingRow]:
    return []


#: what the notebooks' ``SHOW_SETTINGS`` can be: how much of the listing is printed (a dict, so no mode is compared)
SHOW_CHOICES: dict[str, Callable[[list[SettingRow], object], list[SettingRow]]] = {
    "all": _all, "changed": changed_rows, "none": _none}


def show(mode: str, trees: tuple, current) -> str:
    """The text a notebook prints: the rows of ``trees`` that ``mode`` ("all" | "changed" | "none") keeps, with the
    values of ``current`` (a RunSettings or a DataSettings) marked ``<- NOW`` where they differ from the default."""
    if mode not in SHOW_CHOICES:
        raise ValueError(f"SHOW_SETTINGS must be one of {tuple(SHOW_CHOICES)}, got {mode!r}")
    listing = [r for tree in trees for r in rows(tree)]
    return render(SHOW_CHOICES[mode](listing, current), current)


def _markdown(listing: list[SettingRow]) -> str:
    head = ["| path | type | default | valid values | meaning |", "|---|---|---|---|---|"]
    body = []
    for r in listing:
        values = " / ".join(f"`{c}`" for c in r.choices) if r.choices is not None else ""
        doc = (r.doc + (f" ({r.note})" if r.note else "")).replace("|", "/")
        body.append(f"| `{r.tree}:{r.path}` | `{r.type}` | `{r.default}` | {values} | {doc} |")
    return "\n".join(head + body)


def main(argv: list[str] | None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.workflow.settings_listing", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", choices=sorted(TREES), help="only this settings tree (default: all)")
    ap.add_argument("--markdown", action="store_true", help="a markdown table instead of the notebook text")
    a = ap.parse_args(argv)
    listing = rows(a.tree) if a.tree else all_rows()
    print(_markdown(listing) if a.markdown else render(listing, None))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
