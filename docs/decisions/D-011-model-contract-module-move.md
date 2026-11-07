# D-011: model contract moves from core/ to model/ (package H3)

Date: 2026-11-07
Owner decision: hardening package H3 of `docs/hardening/PLAN.md` (finding HF-27: tier-0 `core` imported torch and the
head registry; rule G7). No value changes: the moved function keeps its default.
Why: `core` must import nothing from the package; the contract reads `HEAD_REGISTRY`, so it belongs to `model/`.

- `nig_timenet_torch.core.model_contract.contract_of(input_name)`: `'x'` -> `<removed>`
- `nig_timenet_torch.model.contract.contract_of(input_name)`: `<new>` -> `'x'`
