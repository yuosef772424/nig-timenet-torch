# Chart readings by three Haiku agents (2026-11-17)

Raw output of the readers, condensed only in layout. The lead checked two claims against the 1h candles (HIGHUSDT, BTRUSDT: start hours and levels match; Haiku put HIGH's peak at 10:00-12:00 UTC, the candles put it at 07:00, so the hours are +-3 h). The readers did not judge the model.

## Group A (worst shorts, VAL)

### AIOTUSDT 2026-04-02 (short, -172 %)
- before: flat to slightly down (0.0125-0.013 03-22..03-28, dip to 0.0095 03-29, quiet base 0.0105-0.0125 into 04-01); no earlier spike.
- volume before: near zero until 03-31, then a burst (~2.5-3e7 USDT per 4h bar) on 03-31 / 04-01.
- event day: slow staircase from ~0.0125 at 00:00, steeper from ~18:00-20:00, vertical 0.02 -> 0.038 around 22:00-00:00, peak ~0.058 just after 00:00 on 04-03.
- warning: the volume burst and a small break above the base 1-2 days before; little on the day itself.
- after: partial reversal (0.03-0.045, well above the base).
- type: pump / squeeze on thin volume after a month-long decline.
- numbers: max_up_2bar 231.0; volume x30d 20.7; ret_7d +14.9; window_vol_4h 5.33.

### HIGHUSDT 2026-04-18 (short, -172 %)
- before: flat (0.12-0.13 04-07..04-14, dip 0.11, tight 0.115-0.125 range to 04-17).
- volume before: essentially none until 04-17 late.
- event day: staircase from 04-17 ~20:00-22:00: 0.12 -> 0.2 by 02:00, 0.4 by 06:00, peak ~0.5 (wick 0.59); about 4x the base.
- warning: volume and a break above the range ~4-6 h before the main leg.
- after: gave back 35-40 %, second spike on 04-19, ends ~2x the base.
- type: news- or listing-style pump on a thin, flat base.
- numbers: max_up_2bar 139.4; volume x30d 1207; ret_7d +10.8; window_vol_4h 3.22.

### DAMUSDT 2026-04-27 (short, -158 %)
- before: down (0.045 04-17 -> 0.021 04-25), small bounce, back to 0.021-0.022.
- volume before: near zero; one bar ~1 day before (with a dump).
- event day: flat until ~08:00-10:00, one vertical candle 0.034 -> 0.052 (wick 0.072), plateau 0.05-0.06.
- warning: none clear.
- after: ~80 % given back within a day.
- type: short-lived squeeze inside a downtrend, single vertical candle on a quiet base.
- numbers: max_up_2bar 292.7; volume x30d 179; ret_7d -41.6; window_vol_4h 5.02.

### PORTALUSDT 2026-05-31 (short, -147 %)
- before: flat with a slow drift down (0.0085-0.009, slide to 0.0078 05-29, bounce).
- volume before: near zero until 05-29, small bars 05-29 / 05-30.
- event day: big green candle at 00:00 (0.0125 -> 0.0175), half given back by 12:00, steady climb to 0.02 by 23:00; main vertical leg at 00:00 06-01 (0.025 -> 0.042).
- warning: volume and a break above the base 12-24 h before.
- after: partial reversal, ~2x the base.
- type: breakout after a long flat base with a volume ramp the day before.
- numbers: max_up_2bar 201.2; volume x30d 270; ret_7d +47.2; window_vol_4h 3.52.

Common pattern (reader A): all four start from a quiet, flat, very-low-volume base; event-day volume 20x-1200x the 30-day median; the move is a staircase or a vertical candle; a visible warning in HIGH and PORTAL (volume + breakout 4-24 h before), a volume burst 1-2 days before in AIOT, none in DAM; give-back differs (DAM ~80 %, HIGH / PORTAL 35-50 %, AIOT holds).

## Group B (worst shorts, TEST and worst VAL days)

### LSKUSDT 2026-09-12 (short, -172 %)
- before: flat ~0.10 for ~8 days, slow climb from 09-09 to 0.13; quiet base then a grind up.
- volume before: near zero until 09-09, small bars rising from 09-11.
- event day: staircase from 00:00 (0.15) to 0.26 by 13:00 and 0.36 by 23:00; the vertical spike (wick ~2.5, body ~1.2) comes at 02:00-05:00 on 09-13.
- warning: slow grind and rising volume 12-24 h before.
- after: continuation (0.9-1.0 through 09-14).
- type: breakout from a long flat base, staircase pump on thin volume, larger spike the next day.
- numbers: max_up_2bar 833.1; volume x30d 559; ret_7d +37.6; window_vol_4h 2.55.

### TAIKOUSDT 2026-07-01 (short, -172 %)
- before: down 0.10 -> 0.06, then a base 0.06-0.065; an earlier reversed spike ~06-22.
- volume before: low, small spikes 06-21/22 and 06-30.
- event day: staircase 0.08 -> 0.13 by 14:00, vertical leg 16:00-20:00 to 0.42-0.46.
- warning: breakout above 0.08 and rising volume 12-24 h before; little in the final hours.
- after: full reversal of the vertical leg (0.13 by the end of 07-02).
- type: pump on thin volume with a vertical final leg, then a fast retrace.
- numbers: max_up_2bar 322.3; volume x30d 885; ret_7d +14.5; window_vol_4h 5.31.

### BTRUSDT 2026-08-26 (short, -172 %)
- before: flat ~0.030 for ~10 days with tiny ranges.
- volume before: essentially zero, a small bump 08-24/25.
- event day: flat until ~01:00, steep staircase 02:00-06:00 (to ~0.09), ~0.15 by 15:00-18:00; ~5x in 18 h.
- warning: weak (a small wick breakout and a first volume bump 12-24 h before).
- after: continuation (0.14-0.17 through 08-28).
- type: breakout from a dead, near-zero-volume base, steep staircase pump.
- numbers: max_up_2bar 208.0; volume x30d 628; ret_7d +11.5; window_vol_4h 4.25.

### SIRENUSDT 2026-03-22 (short, worst VAL day coin, -217)
- before: strong uptrend 0.5 -> 0.95-1.0 (03-11..03-21) with a crash and recovery 03-16/17.
- volume before: low-to-moderate, spikes 03-16/17, 03-19, a bump 03-21.
- event day: climb to 1.5 by 08:00-10:00, then wick spikes up to ~4.4 around 16:00-18:00, close ~2.3-2.8.
- warning: rising pace and a volume bump 12-24 h before.
- after: partial reversal (20-25 % from the high), then flat.
- type: trend continuation with a violent, wick-heavy squeeze.
- numbers: max_up_2bar 175.7; volume x30d 20.0; ret_7d +60.7; window_vol_4h 6.94.

### STOUSDT 2026-04-01 (short, worst VAL day coin, -234)
- before: steep rally 0.07 (03-25) -> 0.2 (03-31 close).
- volume before: low until 03-29, rising 03-29..03-31.
- event day: staircase 0.21 -> 0.47 by 18:00-22:00, closes near the high; after the day a vertical spike to 1.1-1.8 (04-02 01:00-04:00) then a dump to ~0.2 by 04-03.
- warning: multi-day rally and rising volume 12-72 h before.
- after: full reversal of the spike.
- type: squeeze after a downtrend-to-rally, late vertical blow-off and dump.
- numbers: max_up_2bar 122.3; volume x30d 183; ret_7d +165.6; window_vol_4h 7.20.

Common pattern (reader B): all five rose before the event (ret_7d +11 .. +166 %, volume 20x-885x); up 2-5x inside the day; LSK and BTR are breakouts from silent near-zero-volume bases and held their highs; TAIKO and STO ended with a vertical leg and retraced most of it within 1-2 days; a volume ramp was visible 12-72 h before in four of five (not BTR).

## Group C (BEST shorts, the contrast)

### POWERUSDT 2026-03-03 (short, best VAL day coin, +184)
- before: strong uptrend ~0.4 (02-20) -> ~1.8-2.0 (03-02); volatile, 2.2 spike with wicks near the top.
- volume before: rising in spikes (~5.8e8 on 02-26, ~3e8 on 02-27), ~2.9e8 on the bar before the event.
- event day (1h): flat near 1.8 until ~11:00-12:00 UTC, then a staircase of large red candles to ~0.2 by ~20:00; flat after.
- warning: stall at the top on 4h (2.2 spike with long upper wick, lower close), ~4-8h before.
- type: collapse after a parabolic run-up.
- numbers: max_down_2bar -83.96; vol x30d 10.26; ret_7d +273; ret_30d +880; vol_4h 10.0.

### VELVETUSDT 2026-06-12 (short, best VAL day coin, +156)
- before: accelerating rise ~0.10 (06-01) -> ~1.8 (06-11), steepest in the last 2-3 days; volatile.
- volume before: near zero until ~06-05, rising from 06-07, spikes ~1e8 (06-09) and ~3e8 (06-10/11).
- event day: flat 1.5-1.8 until ~13:00 UTC, then a steep cascade to ~0.4 by ~22:00; bounce to ~0.5 late 06-13.
- warning: stall at the top on 06-11 with upper wicks, peak volume 06-11, ~4-8h before.
- type: collapse after a parabolic run-up.
- numbers: max_down_2bar -75.0; vol x30d 518; ret_7d +1174; ret_30d +1485; vol_4h 11.2.

### TAIKOUSDT 2026-07-02 (short, best TEST day coin, +168)
- before: flat/slightly declining 0.08 -> 0.06 (06-21..06-30), then pump 06-30 -> peak ~0.45-0.5 on 07-01 (one huge green 4h candle).
- volume before: near zero until 06-30, spike ~2.5e8 on 07-01, declining through 07-02.
- event day: opens ~0.43-0.45, red staircase with lower highs to ~0.22 by ~08:00, ~0.12 by ~18:00. Fall starts at the day open.
- warning: 07-01 top shows a long upper wick and volume fade, 4-8h before.
- type: fade of yesterday's pump.
- numbers: max_down_2bar -66.9; vol x30d 964; ret_7d +624; ret_30d +343; vol_4h 17.2.

Common pattern (reader C):
- All three came after a parabolic run-up (ret_7d 273..1174 %, volume 10x..964x the 30d median, 4h vol 10..17 %).
- Falls fast and large: -67..-84 % within two 4h bars; most of the drop in the first 8-12 h.
- The visible warning was short and weak (upper wicks, stalling, volume peaking then fading) only ~4-8 h before.
