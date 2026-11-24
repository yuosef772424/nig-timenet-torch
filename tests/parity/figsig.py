"""A comparable signature of a matplotlib figure: titles, labels, legend, the data of its lines, bars and scatters (not
pixels). Stand-alone (numpy only, imports nothing of this project), so the OLD-project runner and the test use the
same function."""

import numpy as np


def _num(v):
    v = float(v)
    return None if not np.isfinite(v) else float(f"{v:.9g}")


def _arr(a, limit: int = 200):
    a = np.asarray(a, dtype=object if np.asarray(a).dtype.kind in "OUS" else float).ravel()
    return [str(v) if isinstance(v, str) else _num(v) for v in a[:limit]]


def _color(c):
    try:
        return [round(float(v), 4) for v in np.asarray(c, dtype=float).ravel()[:4]]
    except (TypeError, ValueError):
        return str(c)


def axis_signature(ax) -> dict:
    legend = ax.get_legend()
    out = {"title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
           "legend": [t.get_text() for t in legend.get_texts()] if legend else [],
           "texts": [t.get_text() for t in ax.texts], "yticklabels": [t.get_text() for t in ax.get_yticklabels()][:30]}
    out["lines"] = [{"label": ln.get_label(), "n": len(ln.get_xdata()), "x": _arr(ln.get_xdata()),
                     "y": _arr(ln.get_ydata()), "color": str(ln.get_color()), "ls": ln.get_linestyle()}
                    for ln in ax.lines]
    bars = [p for p in ax.patches if hasattr(p, "get_height") and hasattr(p, "get_x")]
    out["bars"] = {"n": len(bars), "x": _arr([p.get_x() for p in bars]), "w": _arr([p.get_width() for p in bars]),
                   "h": _arr([p.get_height() for p in bars]), "color": [_color(p.get_facecolor()) for p in bars[:12]]}
    out["collections"] = []
    for c in ax.collections:
        off = np.asarray(c.get_offsets(), dtype=float)
        item = {"n": int(len(off)), "x_sum": _num(np.nansum(off[:, 0])) if len(off) else None,
                "y_sum": _num(np.nansum(off[:, 1])) if len(off) else None,
                "color": [_color(v) for v in np.asarray(c.get_facecolor())[:3]]}
        out["collections"].append(item)
    return out


def fig_signature(fig) -> dict:
    w, h = fig.get_size_inches()
    return {"size": [round(float(w), 3), round(float(h), 3)], "axes": [axis_signature(ax) for ax in fig.axes]}
