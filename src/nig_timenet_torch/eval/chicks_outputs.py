"""PURPOSE: The output manager of the TF chicks report: a big table or text is saved to a file with a short preview
    (save_or_print), figures go to <out>/figures, plus the number formats the tables use.
TAGS: chicks, save or print, save figure, analysis outputs, number format, price format, output folder, tf parity
PITFALLS: a table of at most ``max_rows`` rows (40) or a text of at most ``max_chars`` (4000) is printed, a bigger one
    is saved as ``<name>.csv`` (``utf-8-sig``) / ``<name>.txt``; ``timestamp=True`` appends the clock time to the
    file name, so a file written that way is never the same twice; the TF code wrote some of these files to the
    fixed folder ``analysis_outputs`` whatever ``out_dir`` was, here every file goes to the ``out_dir`` the caller
    passes (the same folder with the default); figures are shown only on a display backend, never on Agg.
SOURCE: crypto-signal-prediction ``evaluation/outputs.py`` (save_or_print, save_figure, ensure_output_dir) and the
    format helpers of ``evaluation/latest_table.py`` (``_fmt_price``, ``_fmt_num``).
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.chicks_text import Say

__all__ = ["DEFAULT_OUTPUT_DIR", "DEFAULT_MAX_ROWS", "DEFAULT_MAX_CHARS", "ensure_output_dir", "save_or_print",
           "save_figure", "show_figure", "fmt_price", "fmt_num"]

DEFAULT_OUTPUT_DIR = "analysis_outputs"
DEFAULT_MAX_ROWS = 40
DEFAULT_MAX_CHARS = 4000


def ensure_output_dir(out_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_or_print(data, name: str, out_dir: str = DEFAULT_OUTPUT_DIR, max_rows: int = DEFAULT_MAX_ROWS,
                  max_chars: int = DEFAULT_MAX_CHARS, verbose: bool = True, timestamp: bool = False,
                  say: Say | None = None) -> str | None:
    """Print ``data`` (a DataFrame or a text) when it is small, else save it whole in ``out_dir`` and print a preview
    and the path. Returns the saved path, or None when it was printed."""
    say = say or Say()
    ensure_output_dir(out_dir)
    suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if timestamp else ""
    if isinstance(data, pd.DataFrame):
        if len(data) <= max_rows:
            if verbose:
                print(data.to_string(index=False))
            return None
        path = os.path.join(out_dir, f"{name}{suffix}.csv")
        data.to_csv(path, index=False, encoding="utf-8-sig")
        if verbose:
            say("out_df_big", n=len(data), path=path)
            say("out_df_preview", k=min(max_rows, len(data)))
            print(data.head(max_rows).to_string(index=False))
            say("out_df_more", n=len(data) - max_rows)
        return path
    if isinstance(data, str):
        if len(data) <= max_chars:
            if verbose:
                print(data)
            return None
        path = os.path.join(out_dir, f"{name}{suffix}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
        if verbose:
            say("out_txt_big", n=len(data), path=path)
            print(data[:max_chars])
            say("out_txt_cut")
        return path
    raise TypeError("save_or_print takes a pd.DataFrame or a str only")


def save_figure(fig, name: str, out_dir: str = DEFAULT_OUTPUT_DIR, dpi: int = 130) -> str:
    """Save ``fig`` as ``<out_dir>/figures/<name>.png`` and return the path."""
    fig_dir = ensure_output_dir(os.path.join(out_dir, "figures"))
    path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def show_figure(plt, fig) -> None:
    """What the TF code did after saving (``plt.show()`` then close), without the Agg warning: show only when the
    backend can display; always close the figure."""
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def fmt_price(x) -> str:
    """A price with its natural number of decimals ("—" for a missing value)."""
    if x is None or not np.isfinite(x):
        return "—"
    a = abs(x)
    if a >= 1000:
        return f"{x:,.2f}"
    if a >= 1:
        return f"{x:.4f}"
    if a >= 0.01:
        return f"{x:.5f}"
    return f"{x:.7f}"


def fmt_num(x, fmt: str) -> str:
    return "—" if (x is None or not np.isfinite(x)) else format(x, fmt)
