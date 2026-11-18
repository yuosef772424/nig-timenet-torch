"""PURPOSE: The snapshot panel model (owner 2026-11-18, "100 x 3"): the WHOLE moment (every coin of a timestamp, its
    channels over the window) enters as one input with per-coin input and output weights; NIG + class outputs per coin.
TAGS: snapshot panel, joint moment encoder, coin slots, per coin weights, multi coin, cross section, architecture,
    nig heads, hybrid, coin embedding, accuracy
PITFALLS: the rows of one timestamp must arrive TOGETHER (``split.cross_section``: batches and evaluation chunks are
    cut at timestamp borders) and every row carries its coin id (``split.coin_channel``: the last feature channel,
    before the timestamp channel). A coin's slot is FITTED on the train rows (``CoinSlots``, the ``n_slots`` coins
    with most train rows; rule B4: stored as a buffer, never refitted); a coin without a slot (new in val / test)
    gets the "unknown" slot: it does not enter the joint input and reads the moment through a shared embedding.
    A row's outputs depend on the other coins of its timestamp, by design. Rows of a partial timestamp (a probe
    subsample) see a partial moment. Parameters grow with ``n_slots``: with ~100 x fewer independent rows than
    rows, memorisation is the first thing to watch (train - val gap), hence the dropout on whole slots.

Design (docs/audit/2026-11-18-snapshot-panel/DESIGN.md):
* joint path (the owner's design): per bar, the vector [z_1 .. z_S, present_1 .. present_S] of all S coin slots
  (z = standardised channels, 0 for an absent coin) -> Linear (per-coin input weights) -> LayerNorm -> GRU over the
  window -> the moment state m; per-coin OUTPUT weights map m to raw NIG parameters of each coin (``slot_out``, zero
  at start), so each coin can have its own function of the whole market (lead / lag, beta to the market).
* shared path (switch ``shared_path``): today's per-coin reading (relative inputs within the timestamp -> GRU), the
  same weights for every coin: what the shared model already reads is not lost to the joint path's few samples.
* per coin: [shared state, m, coin embedding] -> readout -> NIG heads (floors, label scale, ordered means) + raw
  terms from ``slot_out``, confidence heads, class heads: the output dict of every architecture (HEAD_REGISTRY names).
"""

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.model import label_scale
from nig_timenet_torch.model.cross_coin import RelativeInputs, group_grid, split_group
from nig_timenet_torch.model.heads import (
    HEAD_REGISTRY,
    class_head_kind,
    class_outputs,
    head_spec,
    head_width,
    kinds_with_task,
    nig_outputs,
)
from nig_timenet_torch.model.layers import RMSNorm
from nig_timenet_torch.model.nig_layers import ConfidenceHead, NIGHead, NIGUncertainty, OrderedMeans
from nig_timenet_torch.model.scalers import InputScaler

__all__ = ["CoinSlots", "FeatureScaler", "SnapshotPanelModel", "build_snapshot_panel", "SNAPSHOT_DEFAULT_CONFIG"]

_RAW = 4  # raw NIG parameters per head: mu, nu, alpha, beta


class FeatureScaler(InputScaler):
    """InputScaler over the feature channels only: ``fit`` also accepts the input WITH the trailing coin-id channel
    (what ``fit_scalers`` hands over) and ignores that channel."""

    def fit(self, x: torch.Tensor, chunk: int = 200_000) -> dict:
        n = self.mean.shape[0]
        return super().fit(x[..., :n] if x.dim() == 3 and x.shape[-1] == n + 1 else x, chunk)


class CoinSlots(nn.Module):
    """Coin id (the dataset's asset index) -> slot 0 .. n_slots - 1, or n_slots ("unknown"). Fitted on the TRAIN
    rows: the ``n_slots`` coins with the most train rows, in decreasing count (ties: lower id). Buffer
    ``slot_of_asset`` (max_assets,), -1 = no slot; all -1 until fitted or loaded (then every coin is unknown and the
    first forward raises)."""

    def __init__(self, n_slots: int, max_assets: int = 4096):
        super().__init__()
        self.n_slots, self.max_assets = int(n_slots), int(max_assets)
        self.register_buffer("slot_of_asset", torch.full((self.max_assets,), -1, dtype=torch.long))

    @torch.no_grad()
    def fit_from_inputs(self, x: torch.Tensor) -> dict:
        """Fit from the train inputs (N, T, F) whose LAST channel is the coin id (``fit_scalers`` calls this)."""
        ids = x[:, -1, -1].round().to(torch.long).cpu()
        if ids.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= self.max_assets:
            raise ValueError(f"coin ids must lie in [0, {self.max_assets}) (is split.coin_channel on?)")
        counts = torch.bincount(ids, minlength=self.max_assets)
        order = sorted((i for i in range(self.max_assets) if counts[i] > 0), key=lambda i: (-int(counts[i]), i))
        keep = order[:self.n_slots]
        table = torch.full((self.max_assets,), -1, dtype=torch.long)   # built on the CPU, copied to the buffer's device
        table[torch.as_tensor(keep, dtype=torch.long)] = torch.arange(len(keep))
        self.slot_of_asset.copy_(table)
        return {"coins_with_slot": len(keep), "coins_in_train": len(order)}

    def forward(self, asset: torch.Tensor) -> torch.Tensor:
        if not bool((self.slot_of_asset >= 0).any()):
            raise RuntimeError("CoinSlots is not fitted: run.build fits it on the train rows (model.scalers."
                               "fit_scalers) or load a state_dict that holds it")
        slot = self.slot_of_asset.to(asset.device)[asset.clamp(0, self.max_assets - 1)]
        return torch.where((slot < 0) | (asset >= self.max_assets) | (asset < 0), self.n_slots, slot)


SNAPSHOT_DEFAULT_CONFIG: dict = {
    "price_targets": list(TARGET_COLUMNS), "n_slots": 128, "max_assets": 4096, "joint_path": True,
    "shared_path": True, "d_joint": 256, "joint_layers": 2, "d_coin": 64, "coin_layers": 2, "d_emb": 32,
    "d_read": 128, "head_hidden": 64, "class_head_hidden": 32, "dropout": 0.25, "slot_dropout": 0.1,
    "input_noise_std": 0.1, "input_clip": 5.0, "nu_min": 0.1, "alpha_min": 2.0, "beta_min": 0.01, "unc_max": 20.0,
    "enforce_order": True, "order_detach_close": True, "nig_scale_fit": "train_std",
}


class SnapshotPanelModel(nn.Module):
    """See the module docstring. ``n_features`` counts the feature channels PLUS the coin-id channel (the input of a
    cross-section run also carries the timestamp channel after them)."""

    default_probe_points = ("readout_norm",)
    balance_point = "readout_norm"

    def __init__(self, seq_len: int, n_features: int, *, price_targets=TARGET_COLUMNS, head_types=None,
                 n_slots=128, max_assets=4096, joint_path=True, shared_path=True, d_joint=256, joint_layers=2,
                 d_coin=64, coin_layers=2, d_emb=32, d_read=128, head_hidden=64, class_head_hidden=32,
                 dropout=0.25, slot_dropout=0.1, input_noise_std=0.1, input_clip=5.0, nu_min=0.1, alpha_min=2.0,
                 beta_min=0.01, unc_max=20.0, enforce_order=True, order_detach_close=True, nig_scale_fit="train_std"):
        super().__init__()
        if not (joint_path or shared_path):
            raise ValueError("snapshot_panel needs joint_path and / or shared_path")
        if int(n_features) < 2:
            raise ValueError("snapshot_panel needs >= 1 feature channel plus the coin-id channel (split.coin_channel)")
        self.seq_len, self.n_in, self.n_feat = int(seq_len), int(n_features), int(n_features) - 1
        self.joint_path, self.shared_path = bool(joint_path), bool(shared_path)
        self.n_slots, self.dropout_p, self.slot_dropout = int(n_slots), float(dropout), float(slot_dropout)
        self.input_noise_std = float(input_noise_std)
        self.label_scale_name, fit_s = nig_scale_fit, label_scale.fitted(nig_scale_fit)
        self.price_targets = tuple(price_targets)
        head_types = head_types or {t: ["nig_regression", "binary_classification"] for t in self.price_targets}
        self.nig_targets = [t for t in self.price_targets if kinds_with_task(head_types.get(t, []), "evidential")]
        self.head_specs = {t: head_spec("nig_regression", t) for t in self.nig_targets}
        self.class_kind = {t: k for t in self.price_targets if (k := class_head_kind(head_types.get(t, []), t))}
        self.class_targets = list(self.class_kind)

        self.input_scaler = FeatureScaler(self.n_feat, clip=input_clip)
        self.coin_slots = CoinSlots(self.n_slots, max_assets)
        self.coin_emb = nn.Embedding(self.n_slots + 1, d_emb)
        read_in = d_emb
        if self.joint_path:
            self.joint_in = nn.Linear(self.n_slots * (self.n_feat + 1), d_joint)
            self.joint_in_norm = nn.LayerNorm(d_joint)
            self.joint_gru = nn.GRU(d_joint, d_joint, num_layers=joint_layers, batch_first=True,
                                    dropout=dropout if joint_layers > 1 else 0.0)
            self.joint_norm = RMSNorm(d_joint)
            self.slot_out = nn.Parameter(torch.zeros(self.n_slots + 1, d_joint, len(self.nig_targets) * _RAW))
            self.slot_out_bias = nn.Parameter(torch.zeros(self.n_slots + 1, len(self.nig_targets) * _RAW))
            read_in += d_joint
        if self.shared_path:
            self.relative = RelativeInputs()
            self.coin_in = nn.Linear(self.n_feat * RelativeInputs.width_mult, d_coin)
            self.coin_gru = nn.GRU(d_coin, d_coin, num_layers=coin_layers, batch_first=True,
                                   dropout=dropout if coin_layers > 1 else 0.0)
            self.coin_norm = RMSNorm(d_coin)
            read_in += d_coin
        self.readout_fc = nn.Linear(read_in, d_read)
        self.readout_drop = nn.Dropout(dropout)
        self.readout_norm = RMSNorm(d_read)
        for t in self.nig_targets:
            setattr(self, f"nig_{t}", NIGHead(d_read, head_hidden, dropout, nu_min, alpha_min, beta_min,
                                              fitted_scale=fit_s))
            setattr(self, f"unc_{t}", NIGUncertainty(unc_max))
            setattr(self, f"conf_{t}", ConfidenceHead(d_read))
        self.apply_order = enforce_order and set(TARGET_COLUMNS) <= set(self.nig_targets)
        if self.apply_order:
            self.ordered_means = OrderedMeans(None, order_detach_close, fitted_scale=fit_s)
        self.class_specs = {t: head_spec(k, t) for t, k in self.class_kind.items()}
        for t, kind in self.class_kind.items():
            setattr(self, f"class_fc_{t}", nn.Linear(d_read, class_head_hidden))
            setattr(self, f"class_drop_{t}", nn.Dropout(dropout))
            setattr(self, self.class_specs[t].output_keys["logits"], nn.Linear(class_head_hidden, head_width(kind, 3)))

    # ------------------------------------------------------------------------------------------------ parts
    def _inputs(self, x: torch.Tensor):
        """(scaled features (B, T, n_feat), coin slot (B,), timestamp ids (B,) or None)."""
        x, gid = split_group(x, self.n_in)
        z = self.input_scaler(x[..., :self.n_feat])
        if self.training and self.input_noise_std > 0:
            z = z + torch.randn_like(z) * self.input_noise_std
        return z, self.coin_slots(x[:, -1, -1].round().to(torch.long)), gid

    def _moment(self, z, slot, inv, n_groups):
        """The moment state m (G, d_joint): every known coin's channels and a presence flag, slot by slot, per bar."""
        known = slot < self.n_slots
        grid = z.new_zeros(n_groups, self.n_slots, z.shape[1], self.n_feat + 1)
        grid[inv[known], slot[known], :, :self.n_feat] = z[known]
        grid[inv[known], slot[known], :, self.n_feat] = 1.0
        if self.training and self.slot_dropout > 0:   # whole coins hidden: no reliance on one coin or a fixed universe
            keep = (torch.rand(n_groups, self.n_slots, 1, 1, device=z.device) >= self.slot_dropout).to(z.dtype)
            grid = grid * keep
        seq = grid.permute(0, 2, 1, 3).reshape(n_groups, z.shape[1], -1)
        h = F.gelu(self.joint_in_norm(self.joint_in(seq)))
        return self.joint_norm(self.joint_gru(h)[0][:, -1])

    def _features(self, x: torch.Tensor):
        z, slot, gid = self._inputs(x)
        inv, _, n_groups, _ = group_grid(gid if gid is not None else torch.arange(len(z), device=z.device))
        parts, extra = [self.coin_emb(slot)], None
        if self.joint_path:
            m = self._moment(z, slot, inv, n_groups)
            parts.append(m[inv])
            per_slot = torch.einsum("gd,sdk->gsk", m, self.slot_out) + self.slot_out_bias   # per-coin output weights
            extra = per_slot[inv, slot]
        if self.shared_path:
            r = self.relative(z, gid)
            parts.append(self.coin_norm(self.coin_gru(F.gelu(self.coin_in(r)))[0][:, -1]))
        h = self.readout_norm(self.readout_drop(F.gelu(self.readout_fc(torch.cat(parts, dim=-1)))))
        return h, extra

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, extra = self._features(x)
        raw = {}
        for i, t in enumerate(self.nig_targets):
            head = getattr(self, f"nig_{t}")
            p = head.raw_params(h)
            if extra is not None:
                p = p + extra[:, i * _RAW:(i + 1) * _RAW]
            raw[t] = head.from_raw(p)
        mus = {t: raw[t][0] for t in self.nig_targets}
        if self.apply_order:
            mus["high"], mus["close"], mus["low"] = self.ordered_means(raw["close"][0], raw["high"][0], raw["low"][0])
        out: dict[str, torch.Tensor] = {}
        for t in self.nig_targets:
            _, nu, alpha, beta = raw[t]
            epi, ale = getattr(self, f"unc_{t}")(nu, alpha, beta)
            out.update(nig_outputs(self.head_specs[t], mus[t], nu, alpha, beta, epi, ale,
                                   getattr(self, f"conf_{t}")(h, nu, alpha, beta)))
        for t, kind in self.class_kind.items():
            probs = HEAD_REGISTRY.get(kind)(h, getattr(self, f"class_fc_{t}"),
                                            getattr(self, self.class_specs[t].output_keys["logits"]),
                                            getattr(self, f"class_drop_{t}"))
            out.update(class_outputs(self.class_specs[t], probs))
        return out

    # ------------------------------------------------------------------------------------------- hooks
    def head_modules(self) -> dict[str, NIGHead]:
        return {t: getattr(self, f"nig_{t}") for t in self.nig_targets}

    @property
    def activation_points(self) -> tuple[str, ...]:
        return ("readout_fc", *(f"nig_{t}.fc" for t in self.nig_targets), *(f"conf_{t}.d1" for t in self.nig_targets),
                *(f"class_fc_{t}" for t in self.class_targets))

    def regularization_loss(self) -> torch.Tensor:
        return sum((m.regularization_loss() for m in self.head_modules().values()), torch.zeros(()))


def build_snapshot_panel(seq_len: int, n_features: int, **config) -> SnapshotPanelModel:
    return SnapshotPanelModel(int(seq_len), int(n_features), **config)
