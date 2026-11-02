# STATUS: PyTorch port build plan

> Read this first, then ONLY your step in `BUILD_PLAN.md`. Update your row in the same PR. Append real obstacles
> to `friction_log.csv`. Keep this file under 60 lines (the table only; move any discussion to `BUILD_PLAN.md`'s
> own change log).

Not started yet: no code in `src/` yet. This file and `BUILD_PLAN.md` were written together in one planning
session; no step below has a PR yet.

| Step | Title | State | PR / job | Note |
|---|---|---|---|---|
| P00 | Repo skeleton | todo | | |
| P01 | Data-artifact contract + second (real-default) oracle | todo | | needs a `crypto-signal-prediction` checkout at the pinned commit; closes BUILD_PLAN.md V1 |
| P02 | core/registry.py | todo | | port, not a redesign |
| P03 | core/schema.py + constants.py | todo | | |
| P04 | core/config.py | todo | | |
| P05 | Review 1 + CI finalize | todo | | M1 close |
| P06 | data_io/npy_dir.py | todo | | needs P01 |
| P07 | Oracle loader | todo | | handles ORACLE_REVIEW.md MAJOR-2 in one place |
| P08 | InstanceNorm + SymLog | todo | | needs P07 |
| P09 | CausalMultiScaleDecomp | todo | | needs P08 |
| P10 | PatchEmbedding | todo | | needs P09 |
| P11 | RMSNorm + RelativeGQAttention | todo | | needs P10 |
| P12 | SwiGLU + TransformerBlock | todo | | needs P11 |
| P13 | Readout layers | todo | | needs P12 |
| P14 | NIG head stack | todo | | needs P13 |
| P15 | Full model assembly + e2e parity | todo | | needs P08-P14; **M2 exit gate** |
| P16 | Review 2 | todo | | M2 close |
| P17 | tasks.py + heads.py registries | todo | | needs P02, P15 |
| P18 | Kendall + compute_losses | todo | | needs P17; **M3 exit gate** (with P19) |
| P19 | One optimizer step | todo | | needs P18; gradients only, see tolerance table |
| P20 | Review 3 | todo | | M3 close |
| P21 | stats/metrics.py | todo | | |
| P22 | selection_metric + Trainer loop | todo | | needs P18, P21 |
| P23 | seeds.py + identity.py | todo | | |
| P24 | Review 4 | todo | | M4 close |
| P25 | roles.py + split_plan.py | todo | | needs P01 |
| P26 | target_codec.py | todo | | needs P02 |
| P27 | prediction_schema.py + prediction_table.py | todo | | needs P15, P25 |
| P28 | Evaluation parity on fixture | todo | | wiring test, not a quality claim |
| P29 | Calibrator | todo | | needs P25 |
| P30 | Review 5 | todo | | M5 close |
| P31 | model_contract.py | todo | | needs P17 |
| P32 | ARCHITECTURES registry | todo | | needs P15, P31 |
| P33 | tiny_mlp plugin | todo | | needs P32 |
| P33b | GRU/anti-memorization parity (closes V1) | todo | | needs P01 (second oracle) |
| P34 | input_skip seam | todo | | needs P15; default stays "none" forever |
| P35 | Per-layer probe report | todo | | needs P15; tool only, no findings |
| P36 | Review 6 | todo | | M6 close |
| P37 | RollingSettings + rolling.py seam | todo | | seam only, no experiment run |
| P38 | staleness.py seam | todo | | seam only, no experiment run |
| P39 | Final review | todo | | plan close |

States: `todo` / `in progress` / `PR open` / `merged` / `closed (rejected)` / `skipped (reason)`.
