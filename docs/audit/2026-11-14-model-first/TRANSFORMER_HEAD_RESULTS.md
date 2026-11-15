# Transformer encoder, SwiGLU and the output head (2026-11-15, third session)

Owner questions: (1) does information reach the transformer blocks; (2) SwiGLU is > 70 % of a block's weights: does
the model need it; (3) which useful parts of the other models belong in a transformer; (4) the attention output head
against the Dense head gave contradicting results (the owner's own run at lr 1e-5: the attention head trained for more
epochs and scored better).

Data of sections 1-3: the public-kline real slice (`data_io/vision_slice.py`: 60 unsealed coins, 4h, 706,104 windows of
32 bars x 3 channels). NOT the Drive artifact. CPU, batch 128, lr 3e-4, default model (preset + `MODEL_FIRST_FIXES`,
d_model 64, 2 layers) with the encoder / head / feed-forward changed, early stopping (patience 8 on the 3-epoch mean of
`val_raw_loss`; the head arms of 3a: patience 15). Every arm: seeds 0 and 1 on one commit. `best val` = the lowest
RAW `val_raw_loss` of the run (lower is better; the sum over the three NIG heads and the three class heads); IC = rank
IC by timestamp on the TEST rows with the best-epoch weights. One epoch of validation loss moves by about +-0.015
from epoch to epoch and the two seeds of one arm differ by 0.005-0.013: differences below ~0.015 are not read (rule
C4: 2 seeds = a look).

Commands (one arm; `--model` = JSON of model overrides, `--card-a` = card A on the trained weights):

```bash
python docs/audit/2026-11-14-model-first/quick_arms.py ARTIFACT OUT --rows 20000 --epochs 30 --patience 8 --seed 0 \
    --arms transformer:3e-4 --model '{"ffn": "none"}' --tag _noffn --card-a --device cpu
```

## 1. Does information reach the transformer blocks? (card A, new rows `attn use / ffn use / block use`)

Yes, it reaches them, and the outputs barely use them. Trained transformer (default size), validation rows:

| measurement (seed 0 / seed 1) | value | reading |
|---|---|---|
| probe IC of `y_high` at `enc_in -> enc_block_1 -> enc_block_2 -> enc_out` (seed 0) | 0.268 -> 0.263 -> 0.259 -> 0.250 | the label information is present at the entry of the blocks and at their exit: nothing is lost, nothing is added |
| update / stream of a block's attention; of its SwiGLU | 0.31-0.38; 0.44-0.53 | both sublayers write into the stream (SwiGLU more) |
| attention entropy, share of uniform | 0.91-0.95 | the attention is close to an average of the tokens: it does not select |
| ALL blocks switched off: change of the label rank IC (high / low / close) | seed 1: -0.019 / 0.000 / -0.001 | the whole attention stack is worth <= 0.02 IC of 0.33-0.39 |
| encoder output rows shuffled (`enc_out` path use): IC change high / low | -0.052 / -0.007 | the encoder as a whole: <= 0.05 |
| `linear_path` rows shuffled: IC change high / low | -0.110 / -0.079 | the flat linear path carries the model |
| `stats` rows shuffled (window mean / log sd): IC change high / low | -0.030 / -0.047 | then the volatility level |

The same holds for the GRU encoder (enc_out -0.029 / -0.006, linear_path -0.112 / -0.107, stats -0.071 / -0.099) and
for every variant below. Reading: on these inputs the high / low labels are a volatility forecast that a linear map of
the window plus its level already gives (card C: simple-model ceiling 0.34 / 0.41, the model sits on it), so any
sequence encoder is close to dead weight and the encoder comparison below is a comparison of small differences.
Nothing is broken on the way to the blocks; there is nothing more for them to find in three log-move channels (N-08).

## 2. SwiGLU: remove it? (`ffn`: `swiglu` default | `none`; `ffn_dim`)

Parameters of the two blocks: 89,608 with SwiGLU, 24,968 without (whole model 144,558 -> 79,918). 20,000 strided rows
per split:

| arm | params | best val s0 / s1 | mean | IC high | IC low | s / epoch |
|---|---|---|---|---|---|---|
| transformer (SwiGLU, 2 blocks) | 144,558 | -5.899 / -5.912 | -5.906 | 0.333 | 0.381 | 12 |
| transformer, `ffn: none` | 79,918 | -5.924 / -5.914 | -5.919 | 0.333 | 0.389 | 8.5 |
| transformer, `ffn: none`, 4 blocks, `num_kv_heads` 4 (the weights moved into attention) | 121,270 | -5.917 / -5.910 | -5.914 | 0.331 | 0.395 | 13 |

Reading: taking SwiGLU out costs nothing on either seed (validation loss equal or better, IC equal within the seed
spread), with 45 % fewer weights and 30 % less time. Spending the saved weights on more attention (4 blocks, full
key / value heads) buys nothing either, which agrees with section 1 (the blocks are hardly used at all). In the
trained SwiGLU model, switching SwiGLU off in one block moves the label IC by <= 0.009.
Not run: the owner's size (d_model 384, 6 blocks, 8 heads, SwiGLU width 1024: 1.18 M of 1.62 M weights per block) and
the Drive artifact; see section 4.

## 3. A transformer with the useful parts of the other encoders (`encoder: conv_transformer`)

`conv_transformer` (model/encoders.py): one token per bar (the GRU / TCN `step_proj`, no patching), a causal
convolution stem (2 TCN blocks, dilations 1 and 2), then the transformer blocks (relative-position attention; `ffn`
applies). 20,000 rows:

| arm | params | best val s0 / s1 | mean | IC high | IC low | s / epoch |
|---|---|---|---|---|---|---|
| transformer (patches, SwiGLU) | 144,558 | -5.899 / -5.912 | -5.906 | 0.333 | 0.381 | 12 |
| transformer, `ffn: none` | 79,918 | -5.924 / -5.914 | -5.919 | 0.333 | 0.389 | 8.5 |
| conv_transformer | 166,702 | -5.923 / -5.919 | -5.921 | 0.333 | 0.388 | 21 |
| conv_transformer, `ffn: none` | 102,062 | -5.927 / -5.923 | -5.925 | 0.331 | 0.393 | 16 |
| gru (the default encoder) | 102,310 | -5.926 / -5.933 | -5.929 | 0.334 | 0.390 | 11 |

Reading: the patch transformer with SwiGLU is the weakest arm on both seeds; per-bar tokens + the conv stem, or just
dropping SwiGLU, bring the transformer level with the GRU (the top four arms are inside the seed spread). No arm beats
the GRU, and no arm moves the IC: the encoder is not what limits this model (section 1).

## 3a. Output head: Dense vs attention vs one Linear layer (`nig_head`)

`linear` (new, model/attn_heads.py: one Linear(trunk -> 4), no hidden layer) is the control: the least free head.
Transformer encoder (SwiGLU), until early stop:

| rows per split | head | best val s0 / s1 | mean | best epoch | epochs run | IC high | IC low | IC close |
|---|---|---|---|---|---|---|---|---|
| 20,000 | dense | -5.899 / -5.912 | -5.906 | 11 / 7 | 26 / 22 | 0.333 | 0.381 | 0.010 |
| 20,000 | scalar_joint | -5.902 / -5.902 | -5.902 | 9 / 9 | 25 / 24 | 0.332 | 0.392 | 0.018 |
| 20,000 | linear | -5.901 / -5.899 | -5.900 | 9 / 11 | 19 / 21 (patience 8) | 0.330 | 0.379 | 0.015 |
| 60,000 | dense | -5.930 / -5.935 | -5.932 | 5 / 4 | 14 / 14 | 0.354 | 0.396 | 0.006 |
| 60,000 | scalar_joint | -5.944 / -5.946 | -5.945 | 6 / 5 | 16 / 14 | 0.352 | 0.403 | 0.027 |

Head x encoder at 60,000 rows (patience 8, 2 seeds; `encoder` = the trunk the head sits on):

| encoder | head | best val s0 / s1 | mean | best epoch | IC high | IC low | IC close |
|---|---|---|---|---|---|---|---|
| transformer (SwiGLU) | dense | -5.930 / -5.935 | -5.932 | 5 / 4 | 0.354 | 0.396 | 0.006 |
| transformer (SwiGLU) | scalar_joint | -5.944 / -5.946 | -5.945 | 6 / 5 | 0.352 | 0.403 | 0.027 |
| transformer, `ffn: none` | dense | -5.946 / -5.952 | -5.949 | 12 / 11 | 0.354 | 0.391 | -0.002 |
| transformer, `ffn: none` | scalar_joint | -5.929 / -5.933 | -5.931 | 6 / 5 | 0.354 | 0.407 | 0.047 |
| gru | dense | -5.952 / -5.961 | -5.956 | 10 / 14 | 0.358 | 0.389 | 0.004 |
| gru | scalar_joint | -5.928 / -5.919 | -5.923 | 14 / 14 | 0.353 | 0.407 | 0.021 |

lr 1e-5 (20,000 rows, seed 0, stopped by hand: 150 epochs on this CPU reach only the lr x steps of ~5 epochs at
3e-4): selection loss at epoch 20 dense -5.673, scalar_joint -5.311; at epoch 37 -5.766 and -5.541; dense at epoch
55 -5.819. Both still falling.

Reading:
- 20,000 rows: the three heads tie (means within 0.006), the same best epochs, the same run length, the same decay
  after the best epoch. A head with one Linear layer does as well as the Dense head: on this model the freedom of the
  head is not what decides anything.
- 60,000 rows: the sign of the head effect depends on the trunk, the same way on both seeds of each pair. On the
  SwiGLU transformer the attention head is better by 0.012-0.014; on the transformer without SwiGLU it is worse by
  0.017-0.019; on the GRU it is worse by 0.024-0.042. This is why earlier results contradicted each other: the
  attention head wins only on the weakest trunk (the SwiGLU patch transformer, which itself overfits earliest: best
  epoch 4-5), where it stands in for a regulariser; on the trunks that are better to begin with, the Dense head is
  better. The best arms are Dense on the GRU (-5.956) and Dense on the transformer without SwiGLU (-5.949).
- The attention head has the higher `low` rank IC in all three pairs (+0.007 .. +0.018) and an equal or lower `high`
  IC; its validation NLL is nevertheless worse on two trunks, i.e. it ranks the low move slightly better and
  calibrates worse. Inside the seed spread of the IC (0.01-0.02): a look, not a result.
- At 60,000 rows `ffn: none` again beats SwiGLU with the Dense head on both seeds (-5.949 vs -5.932) and its best
  epoch moves from 4-5 to 11-12: without SwiGLU the transformer overfits later.
- Why it trains for more epochs (measured at init, `ScalarOutputAttention` on RMS-normalised tokens): its output is
  an average over all T x d cells, so across rows it starts almost constant (sd 0.025 against 0.16 for the Dense head)
  and its range is bounded (about +-5 x gain). It has to grow its queries and gain before it says anything, which at
  lr 1e-5 is slow: after the same 37 epochs it is 0.22 behind. More epochs before the stop is therefore expected from
  the parametrisation alone and is not by itself a sign of less memorisation; whether it ENDS better is the question,
  and 60,000 rows say "slightly, yes" on the transformer.
- Against the earlier Colab result (GRU encoder, Drive artifact, 60,000 rows, 10 epochs: Dense better by 0.04-0.08):
  those runs ended at their 10-epoch limit, not at an early stop (best epochs 7-10 of 10), so the slower head was
  compared before it had finished; and the encoder was the GRU, not the transformer. Section 4 repeats it to the
  stop.

## 4. Drive artifact, Colab T4 (2026-11-15): partial, the runtime went offline

Artifact `preprocessing_output_1` (535 coins, 2,506,192 windows), code 43b317d and later, seed 0 unless said, batch
128. `sel` = `val_raw_loss` per epoch (lower is better). The tunnel answered HTTP 530 before the last arms finished,
so there are NO test numbers (IC) from these runs and the lr 3e-4 pair of 4b has two epochs only.

4a. lr 1e-5, 200,000 rows per split, default-size transformer (SwiGLU), first two epochs (then stopped: a validation
pass over 200,000 rows per epoch made an epoch ~7 minutes, 60 epochs ~7 hours):

| head | sel epoch 1 | epoch 2 |
|---|---|---|
| dense | -6.089 | -6.226 |
| scalar_joint | -5.513 | -5.693 |

4b. The owner's fast setting: small transformer without SwiGLU (`ffn: none`, 79,918 weights), 60,000 rows, 10 epochs,
`steps_per_execution` 32 (45 s per epoch):

| lr | head | sel by epoch (1 .. 10) |
|---|---|---|
| 1e-5 | dense | -5.558 -5.854 -6.015 -6.096 -6.147 -6.184 -6.213 -6.241 -6.264 -6.280 |
| 1e-5 | scalar_joint | -5.404 -5.436 -5.476 -5.522 -5.573 -5.624 -5.674 -5.724 -5.770 -5.812 |
| 3e-4 | dense | -6.458 -6.478 (runtime lost) |
| 3e-4 | scalar_joint | -6.323 -6.393 (runtime lost) |

4c. lr 3e-4, 60,000 rows, default-size transformer (SwiGLU), Dense head, stopped by hand at epoch 14 to free the
runtime: best sel -6.537 (seed 0, epoch 9) and -6.534 (seed 1, epoch 9), flat from epoch 4.

Reading: on the Drive artifact at lr 1e-5 the attention head is far behind the Dense head at every epoch of a 10-epoch
run (-5.81 against -6.28) and both are still falling at epoch 10, the attention head almost linearly: at this lr it
has not left its start (N-24). With an early stop the Dense run plateaus and stops first, and the attention run keeps
improving for many more epochs: the owner's "more epochs". Whether it then ENDS below the Dense head's plateau
(about -6.53 at lr 3e-4) was not reached here. What settles it: both heads at lr 1e-5 to the early stop on the same
rows and seed, 2 seeds (at 45 s per epoch and ~0.045 per epoch the attention head needs roughly 20 more epochs to
reach -6.5; `--epochs 60 --patience 8`, about 45 minutes per arm on the T4), or the lr 3e-4 pair of 4b to the stop
(10 minutes per arm):

```bash
python -m nig_timenet_torch.colab_ops setup ccr-adb40d84-31jfyt
python -m nig_timenet_torch.colab_ops start ". /content/nig_env.sh && cd /content/nig-timenet-torch && python -u \
  docs/audit/2026-11-14-model-first/quick_arms.py /content/art1 /content/out_small --rows 60000 --epochs 60 \
  --patience 8 --seed 0 --spe 32 --model '{\"ffn\": \"none\"}' --tag _noffn \
  --arms transformer:1e-5,transformer:1e-5:scalar_joint,transformer:3e-4,transformer:3e-4:scalar_joint \
  --device cuda" small
```

## 5. A direct line into the transformer (`input_skip`), fourth session 2026-11-15

Owner: "if there is doubt about information loss, a line can go to the transformer directly". Card A found no loss on
the way in (section 1), and the three existing direct paths of the transformer encoder (model/parts.py) agree:
real slice, 20,000 rows, 2 seeds, to the early stop.

| arm | best val s0 / s1 | mean | IC high | IC low |
|---|---|---|---|---|
| transformer (no direct path) | -5.899 / -5.912 | -5.906 | 0.333 | 0.381 |
| `input_skip: concat_each_block` (the raw patches re-enter before every block) | -5.917 / -5.903 | -5.910 | 0.330 | 0.383 |
| `input_skip: dense_skip` (every block sees the raw patches and all earlier outputs) | -5.905 / -5.907 | -5.906 | 0.333 | 0.386 |
| `input_skip: late_raw` (the last 4 bars join the readout) | -5.897 / -5.902 | -5.900 | 0.330 | 0.375 |

Reading: no arm leaves the seed spread. The blocks are not starved of the input; they have nothing more to add.

## 6. The multi-coin model (`cross_coin_layers`, `SplitSettings.cross_section`; D-025)

What existed: `model/panel.py`, the TF port (its own two heads, a (day, pos) grid built by the caller, no training
path in this project: parked). What was built instead, inside the default architecture so that the trainer, the
gates, the battery and the reports apply unchanged:

- `model/cross_coin.py`: between the trunk and the heads, the rows of a batch that share a timestamp exchange
  information; the result enters as `h + gate * (block(h) - h)` with `gate` = 0 at the start, so the model begins
  as the per-coin model exactly. Two kinds: `attention` (masked self-attention over the coins, optional
  `cross_coin_peer_dropout`) and `mean` (each coin gets the timestamp's mean, its own minus the mean, and the sd of
  the trunk, through one hidden layer).
- The timestamp of a row travels as one extra input channel; `SplitSettings.cross_section` puts every split in
  time-major order, makes `max_rows` keep WHOLE timestamps, and the trainer then cuts batches and evaluation chunks
  at timestamp borders (`trainer/groups.py`).

6a. Is the target of a multi-coin model prepared well? (owner question)

Measured on the real slice (150,000 train rows, whole timestamps, ~56-59 coins each; ridge on the simple features;
rank IC by timestamp on val / test):

| label | share of its variance that is the timestamp's common move | absolute target, own features | + features relative to the timestamp mean | relative target, own features | relative target + relative features |
|---|---|---|---|---|---|
| close | 0.60 | -0.008 / -0.010 | +0.007 / +0.013 | -0.003 / +0.002 | +0.007 / +0.012 |
| high | 0.38 | 0.336 / 0.365 | 0.337 / 0.364 | 0.291 / 0.325 | 0.337 / 0.364 |
| low | 0.81 | 0.393 / 0.410 | 0.434 / 0.445 | 0.375 / 0.414 | 0.434 / 0.445 |

Reading:
- The labels are per-coin log returns; 60 % of the close label (and 81 % of the low label) is the move every coin
  shares at that timestamp. A per-coin model spends its loss on that common part and cannot see it coming; a
  cross-sectional rank (what a market-neutral trade uses) never needs it.
- Making the TARGET relative (label minus the timestamp mean) does not by itself raise the ceiling (it lowers it
  with own features only): the rank metric already ignores the common part.
- Giving the model the coin RELATIVE TO ITS TIMESTAMP on the input side is what pays: `low` +0.04 rank IC on both
  val and test, `close` from ~-0.01 to ~+0.01 (inside the noise), `high` nothing.
- So the present target preparation is sound for the NIG heads (they must stay in price units for the limit-order
  rows of the battery), and the improvement is on the input / trunk side: cross-sectional context. The TF panel
  port's own targets ("beats the day median", "scaled relative return") are a different product (a ranking model);
  they are not needed for that gain and they drop the price units the battery reads. Worth adding later as an EXTRA
  head (a rank head next to the NIG heads), not as a replacement.

6b. Does the cross-coin block find that gain? (GRU, lr 3e-4, real slice, whole timestamps)

60,000 rows (~1,000 timestamps), 2 seeds:

| arm | best val s0 / s1 | mean | IC high | IC low | IC close |
|---|---|---|---|---|---|
| A same rows, ordinary shuffled batches, no block | -5.812 / -5.830 | -5.821 | 0.341 | 0.366 | 0.009 |
| B whole-timestamp batches, no block | -5.814 / -5.822 | -5.818 | 0.338 | 0.372 | 0.003 |
| C + attention block | -5.810 / -5.815 | -5.813 | 0.336 | 0.375 | 0.002 |
| D + attention block, peer dropout 0.2 | -5.811 / -5.811 | -5.811 | 0.339 | 0.367 | -0.002 |

200,000 rows (~3,600 timestamps), seed 0 (seed 1: see the note under the table):

| arm | best val | best epoch | IC high | IC low | IC close |
|---|---|---|---|---|---|
| B whole-timestamp batches, no block | -5.912 | 8 | 0.356 | 0.397 | 0.018 |
| C + attention block | -5.910 | 8 | 0.350 | 0.418 | 0.039 |
| E + mean block | -5.910 | 6 | 0.359 | 0.427 | 0.058 |

Reading:
- Whole-timestamp batches cost nothing against shuffled batches on the same rows (A vs B). But whole timestamps at
  a fixed ROW budget is a poorer sample: 60,000 rows as 1,000 timestamps reach -5.82, the same number of rows
  strided over all dates reach -5.956 (section 3a), and arm A's best epoch is the first. Cross-section runs need
  more rows than per-coin runs to cover the same dates.
- 60,000 rows: the block changes nothing. 200,000 rows, one seed: the validation loss is the same, the TEST rank IC
  of `low` rises 0.397 -> 0.418 (attention) -> 0.427 (mean) and of `close` 0.018 -> 0.039 -> 0.058. That is the size
  and the place the ridge check of 6a predicts (+0.04 on `low`), and the simpler `mean` kind does at least as well
  as attention. One seed: a look (rule C4), to be confirmed by the second seed and by the Drive artifact (535 coins
  per timestamp), section 7.

