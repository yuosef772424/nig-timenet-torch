"""The generated list of every tunable setting (workflow/settings_listing.py) cannot go stale (owner 2026-11-23: show
ALL the settings that can be tuned).

  * every leaf field of RunSettings, DataSettings and RollingSettings is in the listing or in ``EXCLUDED`` (reason);
  * the listing holds nothing that is not a field (but the model options, which come from the model builder);
  * every listed set of valid values equals its registry, and the setting's own validation accepts exactly those;
  * the notebooks print the listing, and the defaults the notebook's CHANGES holds equal the listed ones.
"""

import ast
import dataclasses
import typing

import pytest

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data.coins import COINS_BY_CATEGORY
from nig_timenet_torch.data.settings import DataSettings
from nig_timenet_torch.data.targets import ENTRY_CLOSE_REGS, TARGET_RAW
from nig_timenet_torch.eval.battery_scorecard import TRIM_KEYS
from nig_timenet_torch.model.architectures import ARCHITECTURES, ModelSettings, build_model
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
from nig_timenet_torch.trainer.staged import GROWTH, StageSettings
from nig_timenet_torch.trainer.switches import INITS, WEIGHT_DECAY_GROUPS
from nig_timenet_torch.trainer.tracker import SMOOTHERS
from nig_timenet_torch.trainer.trainer import BATCHINGS
from nig_timenet_torch.workflow import settings_listing as sl
from nig_timenet_torch.workflow.settings import ROLLING_MODES, RollingSettings, RunSettings
from tests.notebooks.helpers import code_cells, is_settings, load
from tests.scan import ROOT

RUN = RunSettings(run_dir="x")
DEFAULTS = {"RunSettings": RUN, "DataSettings": DataSettings()}


def _section_type(hint):
    """The dataclass of an ``Optional[DC]`` annotation (None when the annotation is anything else)."""
    args = [a for a in typing.get_args(hint) if isinstance(a, type) and dataclasses.is_dataclass(a)]
    return args[0] if args and not isinstance(hint, type) else None


def _leaves(obj, prefix=""):
    """Dotted path of every settings field of an INSTANCE: a section is entered, a dict or a None is a leaf; the fields
    of an optional section that is off are yielded too (read only when it is on)."""
    hints = typing.get_type_hints(type(obj))
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            yield from _leaves(v, f"{prefix}{f.name}.")
            continue
        yield f"{prefix}{f.name}"
        sub = _section_type(hints[f.name])
        if sub is not None:
            yield from _leaves(sub(), f"{prefix}{f.name}.")


def _expected(tree):
    if tree == "RollingSettings":
        return {f.name for f in dataclasses.fields(RollingSettings)}
    return set(_leaves(DEFAULTS[tree]))


MODEL_PATHS = {f"model.overrides.{f.name}" for f in dataclasses.fields(BuilderOptions)}


# ---------------------------------------------------------------------------------------------------- coverage

@pytest.mark.parametrize("tree", sl.ALL_TREES)
def test_every_leaf_field_is_listed_or_excluded_with_a_reason(tree):
    listed = {r.path for r in sl.rows(tree)}
    excluded = {p for t, p in sl.EXCLUDED if t == tree}
    missing = _expected(tree) - listed - excluded
    assert not missing, (f"{tree} fields neither in the listing nor in settings_listing.EXCLUDED: {sorted(missing)}. "
                         "The listing walks the dataclasses, so this means the walker skips them")
    extra = listed - _expected(tree) - (MODEL_PATHS if tree == "RunSettings" else set())
    assert not extra, f"listed paths that are not fields of {tree}: {sorted(extra)}"
    stale = sorted(excluded - _expected(tree))
    assert not stale, f"stale EXCLUDED entries (not a field any more): {stale}"


def test_every_exclusion_is_justified():
    for key, reason in sl.EXCLUDED.items():
        assert isinstance(reason, str) and len(reason) > 20, f"{key}: say why it is not a setting"


def test_the_listing_covers_the_three_trees_and_the_model_options():
    assert set(sl.ALL_TREES) == {"RunSettings", "DataSettings", "RollingSettings"}
    paths = {r.path for r in sl.rows("RunSettings")}
    assert MODEL_PATHS <= paths and {"train.staged.gamma", "train.day_weights.every", "evaluation.battery.drop_flat",
                                     "run_dir", "split.coin_channel"} <= paths
    assert {"source.kind", "features.custom.range_windows", "windows.stride", "output.latest"} <= {
        r.path for r in sl.rows("DataSettings")}
    assert len(sl.all_rows()) > 250


def test_a_new_field_shows_up_without_touching_the_listing():
    """The walker reads the dataclasses: a field added to TrainSettings is listed by the very same code."""
    @dataclasses.dataclass(frozen=True)
    class Probe:
        flag: bool = False   # a made-up switch
        other: int = 3

    @dataclasses.dataclass(frozen=True)
    class Outer:
        inner: Probe = dataclasses.field(default_factory=Probe)
        maybe: Probe | None = None   # off unless given

    found = list(sl._walk("Outer", Outer, Outer(), "", ""))
    assert [r.path for r in found] == ["inner.flag", "inner.other", "maybe", "maybe.flag", "maybe.other"]
    assert found[0].doc == "a made-up switch" and found[1].inherited and found[3].note == "read only when maybe is set"


def test_every_row_has_a_meaning_a_type_and_a_default_that_is_the_real_one():
    seen = set()
    for r in sl.all_rows():
        assert (r.tree, r.path) not in seen, f"{r.tree}:{r.path} listed twice"
        seen.add((r.tree, r.path))
        assert r.doc.strip() and r.type, (r.tree, r.path)
        if r.default != sl.REQUIRED:
            ast.literal_eval(r.default)   # a default the notebook can write down
    for r in sl.rows("DataSettings"):
        here = DEFAULTS["DataSettings"]
        for part in r.path.split("."):
            here = getattr(here, part)
        assert repr(here) == r.default, r.path
    for r in sl.rows("RunSettings"):
        optional = ("train.staged.", "train.day_weights.")
        if r.path in MODEL_PATHS or r.default == sl.REQUIRED or r.path.startswith(optional):
            continue
        here = RUN
        for part in r.path.split("."):
            here = getattr(here, part)
        assert repr(here) == r.default, r.path


def test_the_model_options_are_the_builders_and_every_one_has_a_meaning():
    names = {f.name for f in dataclasses.fields(BuilderOptions)}
    assert set(sl.MODEL_OPTION_DOCS) == names, "BuilderOptions changed: add / remove its line in MODEL_OPTION_DOCS"
    assert all(len(d) > 15 for d in sl.MODEL_OPTION_DOCS.values())
    base = {**ARCHITECTURES.entry("nig_timenet_v2").meta["default_config"], **RUN.model.overrides}   # as build_model
    for f in dataclasses.fields(BuilderOptions):   # the default run builds exactly what the listing says
        row = next(r for r in sl.rows("RunSettings") if r.path == f"model.overrides.{f.name}")
        want = base[f.name] if f.name in base else f.default
        assert _norm(ast.literal_eval(row.default)) == _norm(want), f.name


# ---------------------------------------------------------------------------------------------------- choices

def _reg(registry):
    return set(registry.names())


#: the registries / vocabularies, spelled again here (not through settings_listing) so a hard-coded list there fails
EXPECTED_CHOICES = {
    ("RunSettings", "model.architecture"): lambda: _reg(ARCHITECTURES),
    ("RunSettings", "train.selection_metric"): lambda: {n for n in SELECTION_METRICS.names()
                                                        if SELECTION_METRICS.entry(n).meta["selectable"]},
    ("RunSettings", "train.early_stop.smoothing"): lambda: set(SMOOTHERS),
    ("RunSettings", "train.init"): lambda: _reg(INITS),
    ("RunSettings", "train.weight_decay_groups"): lambda: _reg(WEIGHT_DECAY_GROUPS),
    ("RunSettings", "train.loss_balance"): lambda: _reg(LOSS_BALANCERS),
    ("RunSettings", "train.cross_section_loss"): lambda: set(CROSS_SECTION_LOSSES),
    ("RunSettings", "train.batch_by"): lambda: set(BATCHINGS),
    ("RunSettings", "train.batch_order"): lambda: _reg(BATCH_ORDERS),
    ("RunSettings", "train.lr_schedule"): lambda: set(LR_KINDS),
    ("RunSettings", "train.staged.growth"): lambda: _reg(GROWTH),
    ("RunSettings", "evaluation.battery.verdict_trim"): lambda: set(TRIM_KEYS),
    ("RunSettings", "data.format"): lambda: {"auto", "pkl.gz", "npy_dir"},
    ("RunSettings", "model.overrides.encoder"): lambda: _reg(ENCODERS),
    ("RunSettings", "model.overrides.input_skip"): lambda: _reg(INPUT_SKIP_VARIANTS),
    ("RunSettings", "model.overrides.stats_mode"): lambda: set(STATS_TRANSFORMS),
    ("RunSettings", "model.overrides.level_norm"): lambda: set(LEVEL_NORMS),
    ("RunSettings", "model.overrides.nig_scale_fit"): lambda: _reg(LABEL_SCALES),
    ("RunSettings", "model.overrides.input_scaler"): lambda: _reg(INPUT_SCALERS),
    ("RunSettings", "model.overrides.nig_head"): lambda: set(NIG_HEADS),
    ("RunSettings", "model.overrides.ffn"): lambda: set(TransformerBlock.FFN_KINDS),
    ("RunSettings", "model.overrides.cross_input"): lambda: set(CROSS_INPUTS),
    ("RunSettings", "model.overrides.cross_coin_kind"): lambda: set(CROSS_COIN_KINDS),
    ("RunSettings", "model.overrides.target_norm"): lambda: _reg(TARGET_NORMS),
    ("RunSettings", "model.overrides.price_targets"): lambda: {"high", "low", "close"},
    ("DataSettings", "source.kind"): lambda: {"files", "synthetic"},
    ("DataSettings", "source.fetch"): lambda: {"", "futures", "vision"},
    ("DataSettings", "source.categories"): lambda: {*COINS_BY_CATEGORY, "all", "all_global"},
    ("DataSettings", "windows.higher_tf_mode"): lambda: {"legacy", "closed"},
    ("DataSettings", "targets.targets"): lambda: {"high", "low", "close"},
    ("DataSettings", "targets.reg_target_mode"): lambda: {n for n in TARGET_RAW.names()
                                                          if TARGET_RAW.entry(n).meta["per_row"]},
    ("DataSettings", "targets.entry_close_reg"): lambda: set(ENTRY_CLOSE_REGS),
    ("DataSettings", "scaling.scaler_type"): lambda: {"robust", "minmax"},
    ("DataSettings", "scaling.price_norm_mode"): lambda: {"window_scale", "pct_change"},
    ("RollingSettings", "mode"): lambda: set(ROLLING_MODES),
}


def test_every_listed_choice_set_equals_its_registry_and_none_is_stale():
    assert set(sl.CHOICES) == set(EXPECTED_CHOICES), (
        "a CHOICES entry without a registry here (or the other way round): "
        f"{sorted(set(sl.CHOICES) ^ set(EXPECTED_CHOICES))}")
    listed = {(r.tree, r.path): r for r in sl.all_rows()}
    for key, expected in EXPECTED_CHOICES.items():
        assert key in listed, f"CHOICES names {key}, which is not a listed setting"
        assert set(listed[key].choices) == expected(), key
        assert len(listed[key].choices) == len(set(listed[key].choices)), f"{key}: repeated choice"


def test_a_name_registered_later_shows_up_in_the_valid_values():
    ARCHITECTURES.register("zz_listing_probe", lambda seq_len, n_features, **kw: None, version=1, default_config={})
    try:
        row = next(r for r in sl.rows("RunSettings") if r.path == "model.architecture")
        assert "zz_listing_probe" in row.choices
    finally:
        ARCHITECTURES.unregister("zz_listing_probe")


#: settings the setting's own validation does not check (a trainer / model / loader reads them later): a wrong value
#: is refused at build time, not by ``updated``. Everything else in CHOICES must be refused by the settings themselves.
LAZY = {("RunSettings", "model.architecture"), ("DataSettings", "source.categories")}
COMPANION = {("RunSettings", "train.cross_section_loss"): {"split.cross_section": True}}


def _apply(tree, path, value):
    if tree == "RollingSettings":
        base = dict(train_span="30D", test_span="7D", step="7D", embargo="1D")
        return RollingSettings(**{**base, path: value})
    return updated(DEFAULTS[tree], {path: value, **COMPANION.get((tree, path), {})})


def _refused(tree, path, value) -> bool:
    try:
        _apply(tree, path, value)
    except (ValueError, TypeError, KeyError):
        return True
    return False


def test_the_settings_accept_exactly_the_listed_values():
    checked = 0
    for r in sl.all_rows():
        if r.choices is None or r.path.startswith("model.overrides."):
            continue
        wrap = (lambda v: (v,)) if r.subset else (lambda v: v)
        for choice in r.choices:
            assert not _refused(r.tree, r.path, wrap(choice)), f"{r.tree}:{r.path} refuses its valid value {choice!r}"
        refused = _refused(r.tree, r.path, wrap("zz-not-a-value"))
        assert refused == ((r.tree, r.path) not in LAZY), (
            f"{r.tree}:{r.path}: the settings {'refuse' if refused else 'accept'} a made-up value; keep LAZY in step")
        checked += 1
    assert checked >= 20


def test_every_listed_model_option_value_builds():
    """The model options are checked by the builder, not by the settings: each valid value builds a model."""
    needs = {"cross_coin_kind": {"cross_coin_layers": 1}}
    built = 0
    for r in sl.rows("RunSettings"):
        if r.choices is None or not r.path.startswith("model.overrides.") or r.path.endswith("price_targets"):
            continue
        option = r.path.removeprefix("model.overrides.")
        for choice in r.choices:
            overrides = {"d_model": 8, "num_layers": 1, option: choice, **needs.get(option, {})}
            if option == "input_skip" and choice != "none":
                overrides["encoder"] = "transformer"
            build_model(ModelSettings("nig_timenet_v2", overrides), 8, 4)
            built += 1
    assert built >= 25


# ---------------------------------------------------------------------------------------------------- rendering

def test_render_marks_what_differs_from_the_default_and_shows_the_valid_values():
    default = RUN
    assert sl.show("changed", sl.ALL_TREES, default) == ""
    assert sl.show("none", sl.ALL_TREES, default) == ""
    changed = updated(default, {"train.epochs": 1, "train.staged": {"gamma": 0.3}, "model.overrides": {"d_model": 32}})
    text = sl.show("changed", sl.ALL_TREES, changed)
    assert "train.epochs = 5   <- NOW 1" in text and "train.staged.gamma = 0.5   <- NOW 0.3" in text
    assert "model.overrides.d_model = 64   <- NOW 32" in text
    assert "model.overrides.dropout = 0.25   <- NOW 0.1" in text   # the replaced dict falls back to the architecture's
    assert "train.lr = " not in text and "train.seed" not in text
    full = sl.show("all", sl.ALL_TREES, default)
    assert "'constant' | 'cosine' | 'cosine_restarts'" in full and "NOW" not in full
    assert "-- DataSettings: windows" in full and "-- RollingSettings: (top level)" in full
    assert "run_dir = (required)" in full
    assert "run_dir = (required)   <- NOW" not in sl.show("all", sl.ALL_TREES, changed)
    with pytest.raises(ValueError, match="SHOW_SETTINGS must be one of"):
        sl.show("everything", sl.ALL_TREES, default)
    with pytest.raises(ValueError, match="unknown settings tree"):
        sl.rows("TrainSettings")


def test_dormant_settings_say_when_they_are_read():
    notes = {r.path: r.note for r in sl.rows("RunSettings")}
    assert notes["train.beta_nll_normalize"] == "read only while beta_nll is on"
    assert notes["train.staged.k"] == "read only when train.staged is set"
    assert notes["train.epochs"] == ""


def test_the_cli_prints_the_listing(capsys):
    assert sl.main(["--tree", "DataSettings"]) == 0
    out = capsys.readouterr().out
    assert "source.kind = 'files'" in out and "train.epochs" not in out
    assert sl.main(["--markdown"]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].startswith("| path |") and any(line.startswith("| `RunSettings:train.epochs`") for line in table)
    with pytest.raises(SystemExit):
        sl.main(["--tree", "Nope"])


def test_stage_settings_listing_follows_the_dataclass():
    sub = {r.path for r in sl.rows("RunSettings") if r.path.startswith("train.staged.")}
    assert sub == {f"train.staged.{f.name}" for f in dataclasses.fields(StageSettings)}


# ---------------------------------------------------------------------------------------------------- notebooks

def _norm(v):
    return tuple(_norm(x) for x in v) if isinstance(v, (list, tuple)) else v


def _settings_literal(nb, name):
    cell = next(src for _, c, src in code_cells(nb) if is_settings(c))
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not in the settings cell")


def _prints_the_listing(nb, trees):
    for _, _, src in code_cells(nb):
        if "settings_listing import" in src and "show(SHOW_SETTINGS" in src and trees in src:
            return True
    return False


@pytest.mark.parametrize("name,trees", [("main", "ALL_TREES"), ("data_pipeline", '("DataSettings",)')])
def test_the_notebooks_print_the_generated_listing(name, trees):
    nb = load(name)
    assert _settings_literal(nb, "SHOW_SETTINGS") in sl.SHOW_CHOICES
    assert _prints_the_listing(nb, trees), f"{name}: no cell prints show(SHOW_SETTINGS, {trees}, settings)"
    markdown = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "SHOW_SETTINGS" in markdown or "<- NOW" in markdown
    assert "settings_listing" in markdown or name == "data_pipeline"


def test_the_main_notebooks_defaults_are_the_listed_defaults():
    """CHANGES is typed by hand; the listing is generated: where both name a setting they must agree."""
    changes = _settings_literal(load("main"), "CHANGES")
    listed = {r.path: r for r in sl.rows("RunSettings")}
    compared = 0
    for path, value in changes.items():
        assert path in listed, f"CHANGES sets {path}, which the listing does not know"
        if listed[path].default != sl.REQUIRED:
            assert _norm(ast.literal_eval(listed[path].default)) == _norm(value), path
            compared += 1
    assert compared == len(changes) > 60


def test_the_notebook_readme_points_at_the_listing():
    text = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    assert "settings_listing" in text and "SHOW_SETTINGS" in text
