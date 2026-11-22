"""DY1: one markdown row per arm (both seeds) from read_dy1.txt lines.  python summary_table.py READ_DY1_TXT"""
import re
import sys

PAT = re.compile(r"^(\S+)/(own/)?s(\d) folds \d+ IC ([+-][\d.]+) \(t~([-\d.]+)\) \| CAGR ([+-][\d.]+)% .*?Sharpe "
                 r"([-\d.]+) DD (-?\d+)% expo [\d.]+ \| always long ([+-][\d.]+)% placebo p95 ([+-][\d.]+)% .*?accept "
                 r"(True|False) \| period CI \[([+-][\d.]+), ([+-][\d.]+)\] bp \| long-short CAGR ([+-][\d.]+)%")


def main(path):
    arms = {}
    for line in open(path):
        m = PAT.match(line)
        if m:
            arm = m.group(1) + (" (own)" if m.group(2) else (" (residual)" if "mres" in m.group(1) else ""))
            arms.setdefault(arm, []).append(m.groups()[2:])
    print("| arm | IC s0 / s1 | CAGR s0 / s1 | Sharpe | maxDD | placebo p95 | accept s0 / s1 | long-short CAGR |")
    print("|---|---|---|---|---|---|---|---|")
    for arm, rows in sorted(arms.items()):
        rows = sorted(rows)
        j = lambda i, f=str: " / ".join(f(r[i]) for r in rows)  # noqa: E731
        print(f"| {arm} | {j(1)} | {j(3, lambda v: v + ' %')} | {j(4)} | {j(5, lambda v: v + ' %')} | "
              f"{j(7, lambda v: v + ' %')} | {j(8, lambda v: 'yes' if v == 'True' else 'no')} | "
              f"{j(11, lambda v: v + ' %')} |")


if __name__ == "__main__":
    main(sys.argv[1])
