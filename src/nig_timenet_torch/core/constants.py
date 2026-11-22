"""PURPOSE: Numeric constants shared by the model layer and the evaluation decoding (today: the floor of the NIG
    ``alpha - 1`` denominator).
TAGS: constants, nig alpha floor, uncertainty, shared numerics, model and evaluation, schema pinned
PITFALLS: pinned by tests/test_schema_pinned.py and copied verbatim from the TF project; the model layer
    (``model.nig_layers.alpha_minus_one``) and ``stats.nig`` must read the same value. Other floors (the logged
    evidential stats, the side-payoff weights) are named in ``trainer/tasks.py`` and differ on purpose.

Numeric constants shared by model and evaluation (BUILD_PLAN.md P03).

SOURCE: copied verbatim from yuosef772424/crypto-signal-prediction core/constants.py at
aeafc564f25c78f252936affbd374f68aa49b8d9. Pinned by tests/test_schema_pinned.py.
"""

#: Floor of the (alpha - 1) denominator of the NIG uncertainty. No effect at all when alpha >= 1.01 (the NIG head
#: guarantees alpha >= alpha_min = 2 by default); prevents beta / (alpha - 1) -> inf if alpha_min is lowered towards 1.
#: The model layer and the evaluation decoding MUST use the same value.
NIG_ALPHA_DEN_MIN = 1e-2
