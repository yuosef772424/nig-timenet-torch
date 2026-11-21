# Pre-registration BR1: BTC-relative inputs AND a BTC-relative close label (2026-11-19)

Written before any number of the run. Exploratory (rule C5). Hypothesis BR1 of `docs/results/btc-relative-001.md`
(owner: "continue"). The BR arm improved the close rank IC in all 7 folds but its loss was still spent on the coin's
own return, most of which is BTC's move; BR1 aims the close loss at the part the inputs now show.

## Data
`/content/art_btcrel_lbl` = `/content/art_btcrel` (11 BTC-relative channels) with `relative_close_labels`:
y_close_reg = clip(log(coin future / last close) - log(BTC future / BTC last close), +-1) over the label's own span,
y_close_class = (it > 0), last_candles future_close rebuilt so the codec decodes it. High / low labels unchanged.

## Arm
BR1 = BR with only the close label changed (GRU 64 x 2, lr 3e-4, 8 epochs, patience 3, 400k rows, top 100 coins,
7 expanding folds, seeds 0 and 1). Reference: BR (and R) on the same folds.

## Reading and accept rule
The battery's y_ret is now the coin's return relative to BTC, so `read_arms.py`'s `long` = (relative return - one
round trip). A hedged trade pays a second round trip (the BTC leg): the primary number is `long` - 0.16 %.
Accept (both seeds): (1) `long_lo` - 0.0016 > 0 (day-block CI at 1 - 0.05 / 2); (2) `long_wo_best5` - 0.0016 > 0;
(3) the primary mean above BR's `long_btc_real` mean on the same seed. Reported next to it: close rank IC by timestamp
per fold against BR's (the ranks of a relative label inside a timestamp equal the ranks of the own label, so the two
ICs are comparable), the short book (information only). If BR1 fails, the failure reading of rule C7 comes first.
