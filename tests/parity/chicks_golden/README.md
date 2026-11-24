# chicks_golden: the OLD project's output on a small known world

These files were written by the OLD project's own code (`yuosef772424/crypto-signal-prediction`,
`evaluation/full_analysis.py::run_full_analysis`, read-only checkout), not by this project:

    PYTHONPATH=src:. python -m tests.parity.make_chicks_golden [--old /path/to/crypto-signal-prediction]

`tests/parity/chicks_world.py` builds the synthetic world (a dataset with `last_candles`, a "skilled" model's
PredictionTable with a varying NIG scale) and writes it in the old layout; `tests/parity/run_old_chicks.py` (its own
interpreter, TensorFlow replaced by an empty stub module, the one function that calls the model replaced by a fake that
replays the table's columns) runs the unchanged old code on it. One folder per case (`CASES` in `chicks_world.py`):

- `printed.txt`: the old stdout (the output folder's absolute path replaced by `<OUT>`);
- `analysis_outputs/`: every file the old code saved, except the PNGs (their names are the keys of `figures.json`);
- `figures.json`: a signature of each figure saved (titles, labels, legend, the data of lines, bars and scatters,
  `tests/parity/figsig.py`), not pixels;
- `tables.json`: the structures `run_full_analysis` returned (trust report, calibration table, movement / confidence /
  batch analyses, trade selection, patterns, integrity, tearsheet and significance, coverage, verification);
- `args.json`: the keyword arguments both projects were given.

`tests/parity/test_chicks_parity.py` runs this project's report on the same world and compares the printed text line by
line, the saved files, the figure signatures and the returned tables. Regenerate only when the world or the cases
change (never to make a failing test pass): a golden file is the old project's answer.
