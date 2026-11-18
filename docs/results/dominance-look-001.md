# dominance-look-001: USDT.D / BTC.D, and BTC + ETH instead of the cross-section as the panel's reference

Owner question 2026-11-19 (Arabic): evaluate adding the USDT.D and BTC.D indicators, and computing the shared window
from BTC and ETH only instead of all coins. Model-free look (`docs/PHILOSOPHY.md` section 4.5) on the TRAIN era of
the 4h artifact (13185 timestamps 2020-02 .. 2026-02, median 120 coins), script
`docs/audit/2026-11-19-rolling-memorisation/dominance.py`, output on Drive
`nig_timenet_runs/memorisation-check-001/dominance_look.json`. Correlations with a 5-day-block 95 % CI. Exploratory.

Failure reading: not applicable (a model-free look, no trained arm).

## Data
The real USDT.D / BTC.D (TradingView) are shares of the TOTAL crypto market cap, i.e. they need the caps of every
coin and the stablecoin supply: not in the artifact, and their history needs an outside source with a key (CoinGecko
Pro, CoinMarketCap). Measured here are their price parts: the USDT.D change is about minus the cap-weighted market
move (proxy: -(0.75 BTC + 0.25 ETH) over 24 h); the BTC.D change is about BTC's move minus the market's (proxy: BTC
24 h minus the mean of the coins).

## Results
| question | number [95 % CI] |
|---|---|
| BTC + ETH mean move vs the cross-section mean move (24 h; last bar; next 8 h) | 0.863 [0.75, 1.00]; 0.863; 0.863 |
| market 24 h -> market next 8 h | -0.043 [-0.071, -0.013] |
| BTC + ETH 24 h -> market next 8 h | -0.048 [-0.076, -0.019] |
| USDT.D proxy -> market next 8 h | +0.047 [+0.019, +0.072] (the mirror of the line above) |
| BTC.D proxy -> market next 8 h | +0.022 [-0.011, +0.051] |
| BTC.D proxy 24 h -> BTC.D proxy next 8 h (does a rotation persist?) | -0.009 [-0.042, +0.024] |
| BTC.D proxy -> cross-section dispersion next 8 h | +0.005 [-0.060, +0.092] |
| relative 24 h move -> next relative move, rank IC: reference = mean of all coins / BTC + ETH | -0.0463 / -0.0463 (identical) |
| coin beta to BTC + ETH -> next relative move, rank IC, after BTC.D proxy up / down | -0.017 / -0.030 |
| the same after market 24 h up / down | -0.037 / -0.008 |

## Reading
1. **USDT.D / BTC.D as price proxies add nothing new.** The USDT.D proxy is the market's own short reversal with
   the sign flipped (same information, |r| 0.047, r^2 0.2 %); the BTC.D proxy predicts neither the market, nor the
   next rotation, nor the dispersion (every CI holds 0). The price part of both is already in the relative inputs'
   "mean" channel. What the real indices add beyond prices is the stablecoin supply (money waiting to enter) and the
   caps of coins outside the panel: testable only with an outside history (a key), first model-free with this
   same table.
2. **A BTC + ETH reference does not change the ranking.** Subtracting the same number from every coin of a timestamp
   leaves their order unchanged: the rank IC of the relative move is the same to every digit. What would change: the
   market-state channel would see 74 % of the market's variance (r 0.86: alt-specific market moves are lost) and the
   dispersion channel (the sd across coins, which needs a cross-section) would go. What would be GAINED is practical:
   a model that needs only BTC and ETH as context does not need whole timestamps of 100 coins, so it is not tied to
   the 100 longest-listed coins (rolling-memorisation-001) and can score any coin alone.
3. Side finding: high-beta coins tend to lag in the next 8 h relative to the others (rank IC -0.02 .. -0.04), more
   after a market rise; small and without a CI here.

## Next hypotheses
1. A single-coin model with BTC and ETH context channels (their last 32 bars beside the coin's) against the panel
   relative arm: refuted if the close relative rank IC is more than 0.01 below it on the walk-forward folds.
2. The real USDT.D / BTC.D histories (outside source, key needed): refuted if their 24 h change does not predict the
   market's next 8 h beyond the market's own 24 h move (partial correlation CI holding 0).

## Addendum: BTC alone as the reference, dispersion over all coins except BTC and ETH (owner 2026-11-19)
Same rows and method, script `btcref.py`, output `btc_ref_look.json`.

| question | number |
|---|---|
| BTC vs the cross-section mean move (24 h; last bar; next 8 h) | r 0.79; 0.80; 0.79 (BTC + ETH: 0.86) |
| dispersion over the alts only vs over all coins (24 h; last bar) | r 0.9998; 0.9998 |
| BTC 24 h -> market next 8 h / market 24 h -> market next 8 h | -0.044 [-0.070, -0.017] / -0.043 [-0.071, -0.013] |
| dispersion 24 h -> dispersion next 8 h (alts / all) | 0.70 / 0.70 |
| dispersion of the last bar -> mean absolute relative move next 8 h | 0.61 |
| absolute BTC 24 h move -> dispersion next 8 h | 0.18 [0.13, 0.24] |
| rank IC with the next relative move: own - market / own - BTC / (own - BTC) / alt dispersion | -0.04633 / -0.04633 / -0.04633 |

Reading: (1) the ranking does not change with any of these references (identical to five digits: a reference and a
dispersion common to the timestamp keep the order). (2) BTC alone sees 62 % of the market's variance (r 0.79; BTC +
ETH 74 %), yet as a timing signal for the market's next 8 h it is as good as the mean (-0.044 vs -0.043). (3) Leaving
BTC and ETH out of the dispersion changes nothing (r 0.9998: 2 of about 120 coins). (4) The dispersion is the strong
channel: it persists (0.70) and predicts the size of the next relative moves (0.61), i.e. the volatility regime the
high / low heads and the uncertainty need. (5) Practical cost: the dispersion still needs the cross-section of the
timestamp, so a model with this channel still needs whole timestamps at training and at prediction time (the panel's
coin-set limit stays); a fixed basket of liquid alts would lift that limit (not measured). (6) Practical gain of a BTC
reference: (own - BTC) is the P&L of "long the coin, short BTC", a hedge with one liquid instrument instead of a short
basket of alts (relevant to the cost wall B-074; the battery's flat cost per side does not model it yet).
Hypothesis: a BTC-relative target and book (long top coins, short BTC) against the market-neutral book, with a cost
model per instrument; refuted if its net day mean is not above the neutral book's on the pooled walk-forward days.
