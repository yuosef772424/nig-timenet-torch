# E4 known-answer look (Colab L4, 2026-11-18, commit a10bdbc; 50 coins, 180,000 rows, 8 epochs, seeds 0 / 1)

Raw rows: `e4_look.jsonl` (script `scripts/snapshot_look.py`, the same loss / selection / rows for every arm).
Oracle on E4 (the conditional means, `summary` of the artifact): close rank IC by timestamp 0.130, high 0.277,
low 0.276; the "own minus timestamp mean" reader alone reaches 0.122 on close.

| arm | params | TEST rank IC close (s0 / s1) | high | low | best epoch | 80 % coverage close |
|---|---|---|---|---|---|---|
| base (today's panel model, relative inputs + rank term) | 142 k | **0.129 / 0.125** | 0.268 / 0.267 | 0.256 / 0.254 | 6 / 7 | 0.82 / 0.77 |
| joint (the moment as one vector, per-coin weights) | 1.41 M | 0.010 / 0.012 | 0.159 / 0.184 | 0.209 / 0.187 | 7 / 8 | 0.77 / 0.78 |
| hybrid (joint + shared per-coin path) | 1.47 M | 0.083 / 0.102 | 0.177 / 0.202 | 0.226 / 0.226 | 7 / 8 | 0.76 / 0.78 |

Reading (pre-registered rule: "if the joint arm does NOT beat base on E4, the 100 x 3 view has no advantage even where
one is built into the data"):
- base reaches the oracle on close (0.127 against 0.130): on E4 there is almost no room left for the per-coin beta
  that only the joint view can learn (the readers' ladder is 0.122 -> 0.129).
- joint does NOT read the planted coin-specific signal at all (close 0.01 on both seeds) and loses 0.08-0.11 on high
  and 0.05-0.07 on low: with 2,880 training moments for 1.4 M parameters, each coin's input / output weights see one
  example per moment; the shared model sees 50 per moment for the same function.
- hybrid sits between: adding the joint path to the shared path makes the shared path WORSE (close 0.09 against
  0.127), i.e. the joint features are noise the readout has to learn to ignore.
- Both snapshot arms still improve at epoch 8 (best epoch = last or one before); train and val NLL move together
  (no memorisation yet): the failure is sample efficiency, not overfitting.
Caveat: E4 has 50 coins and ~2,900 training moments; the real folds have 100 coins and 3,700-13,000 moments; the
walk-forward jobs decide on the real data.
