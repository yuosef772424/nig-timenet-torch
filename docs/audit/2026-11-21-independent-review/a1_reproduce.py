from load import load, COST
for s in (0, 1):
    d = load(s)
    # overlap check: rows per coin step
    g = d.sort_values(["coin", "end_time"]).groupby("coin")["end_time"].diff().dt.total_seconds().div(3600)
    lg = d[d.long]
    day = (lg.y_ret - COST).groupby(lg.day).mean()
    mkt = d.groupby("day").y_ret.mean() - COST
    print(f"seed {s}: rows {len(d):,} coins {d.coin.nunique()} step-hours mode {g.mode()[0]} | long trades {len(lg):,} days {len(day)}")
    print(f"  long book mean day {day.mean()*100:+.3f}%  w/o best5 {day[day<day.quantile(.95)].mean()*100:+.3f}%  always-long {mkt.mean()*100:+.3f}%")
    print(f"  dup (coin,end_time): {d.duplicated(['coin','end_time']).sum()}  folds overlap in time: ",
          any(d[d.fold==k].end_time.max() >= d[d.fold==k+1].end_time.min() for k in range(6)))
