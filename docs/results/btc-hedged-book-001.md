# btc-hedged-book-001: long the top coins against BTC, versus the market-neutral book (owner 2026-11-19)

Pre-registered: `docs/audit/2026-11-18-panel-std/PREREGISTERED.md` addendum 4 (written before any number). No
retraining (a common reference does not change the ranking): books built from the saved TEST frames of the finished
walk-forwards `panel-rolling-ref-002`, `-std-001`, `-ls-001`, 7 folds pooled per seed (about 7300 timestamps, 1225
days). Script `docs/audit/2026-11-19-rolling-memorisation/btc_book.py`, output on Drive
`nig_timenet_runs/memorisation-check-001/btc_book_*.json`. Exploratory (rule C5).

Failure reading: not applicable (a use of existing predictions, no trained arm; the arms' failure readings are in
`docs/results/panel-rolling-std-001.md`).

Units: one unit long against one unit short per 4h timestamp, P&L = r_long - r_short, the top / bottom 20 % by the
close mean; cost = 2 x cost per side on every leg that trades, FULL turnover of the alt legs every 4 h for both
books (an upper bound; this is not the panel report's spread convention, so the net levels differ from it).

| per seed (0 / 1) | ref-002 | std-001 | ls-001 |
|---|---|---|---|
| gross per timestamp, neutral (top - bottom) | 0.093 % / 0.085 % | 0.089 % / 0.093 % | 0.089 % / 0.093 % |
| gross per timestamp, BTC-hedged (top - BTC) | 0.013 % / 0.007 % | 0.010 % / 0.007 % | 0.009 % / 0.005 % |
| gross per timestamp, pair sign vs BTC (information) | 0.054 % / 0.054 % | 0.058 % / 0.054 % | 0.057 % / 0.058 % |
| net day, neutral | -0.229 % / -0.237 % | -0.234 % / -0.229 % | -0.234 % / -0.230 % |
| hedged - neutral, net day, cost (a) flat 0.0008 every leg | -0.079 [-0.131, -0.034] / -0.077 [-0.122, -0.023] | -0.077 / -0.084 (CI < 0) | -0.078 / -0.086 (CI < 0) |
| (b) BTC 0.0004 per side | +0.001 [-0.051, +0.046] / +0.003 [-0.042, +0.057] | +0.003 / -0.004 (CI holds 0) | +0.002 / -0.006 (CI holds 0) |
| (c) BTC 0.0004 and the BTC leg held | **+0.081 [+0.029, +0.126] / +0.083 [+0.038, +0.137]** | +0.083 / +0.076 (CI > 0) | +0.082 / +0.074 (CI > 0) |
| net day, BTC-hedged, (c) | -0.148 % / -0.154 % | -0.151 % / -0.153 % | -0.152 % / -0.155 % |

## Reading
1. **Formally accepted under cost model (c), for the wrong reason to celebrate.** The hedged book beats the neutral
   one by about +0.08 % a day on every seed of every arm, but ONLY because a held BTC hedge costs nothing to keep:
   its gross is far smaller (0.01 % against 0.09 % per timestamp), and with the flat cost it loses (-0.08 % a day).
   Both books lose money after cost; neither is profitable.
2. **Where the model's edge sits:** the top-20 % coins barely beat BTC (0.005-0.013 % per 4 h), so most of the
   neutral spread comes from the BOTTOM 20 % falling behind (about 0.08 % per 4 h against BTC). The model is better
   at naming the coins to avoid or short than the coins to buy.
3. **The cost wall again (B-074):** with full turnover every 4 h, every book pays 0.32 % per round of two legs
   against a gross of 0.01-0.09 %. The real turnover (how many coins leave the top / bottom 20 % between two
   timestamps) was not measured: it is the number that decides whether any of these books can pay.

## Next hypotheses
1. Turnover-aware cost: charge only the coins that enter or leave the book each timestamp (and a lower rebalance
   frequency: daily). Refuted if the net day mean of the neutral book stays below 0 with its CI under 0 on both
   seeds at the measured turnover.
2. "Short the bottom 20 %, long BTC held" (seen on TEST, NOT pre-registered: a hypothesis for the next
   walk-forward, never a result): about 0.08 % gross per 4 h against 0.16 % of alt cost at full turnover.
