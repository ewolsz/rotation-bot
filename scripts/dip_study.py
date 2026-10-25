#!/usr/bin/env python3
"""Does buying a dip in an uptrend make money after costs?

    python scripts/dip_study.py --csv data/daily.csv --explore
    python scripts/dip_study.py --csv data/daily.csv --final

The rule and the pass mark live in rules/dip_buy.json, written before any data
was looked at. This script only reads them. It prints PASS or FAIL against that
file so the bar cannot quietly move after you see a number.

--explore   runs on the older dates (up to explore_until). Look as much as you
            like. The verdict is printed for information only.
--final     runs once on the newer dates (from final_test_from). It writes
            rules/dip_buy.final.txt with the result, and refuses to run a second
            time. Those dates are then spent.

What it measures, per symbol:

  trades / win rate      counted after costs
  avg win / avg loss     the size of each, so the win rate means something
  expectancy             average trade after costs, with a 95% range
  vs random              the same exit rule on random uptrend days. If the dip
                         does not beat that, the rule is just 'be long in a bull
                         market' with extra steps.
  CAGR / max drawdown    against buying and holding the same symbol over the
                         same dates. Cash earns nothing here, which flatters
                         buy-and-hold slightly.

Entries and exits are at the close. A live bot has to act a minute or two
before the close, so the real fill will differ a little from these numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOT = 5000


@dataclass(slots=True)
class Day:
    date: str
    open: float
    high: float
    low: float
    close: float


def load(path: Path) -> dict[str, list[Day]]:
    out: dict[str, list[Day]] = defaultdict(list)
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            c, a = float(r["close"]), float(r["adjclose"])
            f = a / c if c else 1.0
            out[r["symbol"]].append(Day(r["date"], float(r["open"]) * f,
                                        float(r["high"]) * f, float(r["low"]) * f, a))
    for s in out:
        out[s].sort(key=lambda d: d.date)
    return out


def sma(values: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    run = 0.0
    for i, v in enumerate(values):
        run += v
        if i >= n:
            run -= values[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

def in_uptrend(i, closes, avg) -> bool:
    return avg[i] is not None and closes[i] > avg[i]


def is_dip(i, closes, k) -> bool:
    if i < k:
        return False
    return all(closes[i - j] < closes[i - j - 1] for j in range(k))


def exit_index(days: list[Day], i: int, last: int, max_hold: int) -> int:
    """First close above the prior day's high, or max_hold days, or the period end."""
    for j in range(i + 1, last + 1):
        if days[j].close > days[j - 1].high or j - i >= max_hold:
            return j
    return last


def run_rule(days: list[Day], first: int, last: int, rule: dict) -> list[tuple[int, int, float]]:
    closes = [d.close for d in days]
    avg = sma(closes, rule["trend_days"])
    cost = rule["cost_pct_round_trip"] / 100.0
    trades = []
    i = first
    while i < last:
        if in_uptrend(i, closes, avg) and is_dip(i, closes, rule["down_days"]):
            j = exit_index(days, i, last, rule["max_hold_days"])
            trades.append((i, j, closes[j] / closes[i] - 1.0 - cost))
            i = j + 1            # one position at a time
            continue
        i += 1
    return trades


def random_control(days, first, last, rule, n, rng) -> list[float]:
    """Same exit rule, same trend filter, random entry day."""
    closes = [d.close for d in days]
    avg = sma(closes, rule["trend_days"])
    cost = rule["cost_pct_round_trip"] / 100.0
    pool = [i for i in range(first, last) if in_uptrend(i, closes, avg)]
    if not pool:
        return []
    out = []
    for _ in range(n):
        i = rng.choice(pool)
        j = exit_index(days, i, last, rule["max_hold_days"])
        out.append(closes[j] / closes[i] - 1.0 - cost)
    return out


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def equity_curve(days, first, last, trades, cost) -> list[float]:
    """Daily marked-to-market equity. Flat between trades, cost at each exit."""
    held = {}
    for i, j, _ in trades:
        for k in range(i + 1, j + 1):
            held[k] = j
    eq, cur = [1.0], 1.0
    for k in range(first + 1, last + 1):
        if k in held:
            cur *= days[k].close / days[k - 1].close
            if held[k] == k:
                cur *= (1.0 - cost)
        eq.append(cur)
    return eq


def max_drawdown(eq: list[float]) -> float:
    peak, worst = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        worst = max(worst, 1.0 - v / peak)
    return worst


def cagr(eq: list[float]) -> float:
    years = (len(eq) - 1) / 252.0
    return eq[-1] ** (1.0 / years) - 1.0 if years > 0 and eq[-1] > 0 else 0.0


def boot_mean(x, rng):
    if len(x) < 2:
        return (float("nan"),) * 2
    means = sorted(statistics.fmean(rng.choices(x, k=len(x))) for _ in range(BOOT))
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT)]


def boot_diff(a, b, rng):
    if len(a) < 2 or len(b) < 2:
        return (float("nan"),) * 2
    d = sorted(statistics.fmean(rng.choices(a, k=len(a))) -
               statistics.fmean(rng.choices(b, k=len(b))) for _ in range(BOOT))
    return d[int(0.025 * BOOT)], d[int(0.975 * BOOT)]


def ratio(c, dd):
    return c / dd if dd > 0 else float("inf")


def study(days: list[Day], since: str | None, until: str | None, rule: dict, rng) -> dict | None:
    idx = [i for i, d in enumerate(days)
           if (not since or d.date >= since) and (not until or d.date <= until)]
    if len(idx) < 2:
        return None
    first, last = idx[0], idx[-1]
    trades = run_rule(days, first, last, rule)
    rets = [r for _, _, r in trades]
    ctrl = random_control(days, first, last, rule, max(300, 3 * len(trades)), rng)
    cost = rule["cost_pct_round_trip"] / 100.0

    eq = equity_curve(days, first, last, trades, cost)
    bh = [days[k].close / days[first].close for k in range(first, last + 1)]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    held_days = sum(j - i for i, j, _ in trades)

    res = {
        "start": days[first].date, "end": days[last].date,
        "years": (last - first) / 252.0,
        "trades": len(rets),
        "win_rate": len(wins) / len(rets) if rets else 0.0,
        "avg_win": statistics.fmean(wins) if wins else 0.0,
        "avg_loss": statistics.fmean(losses) if losses else 0.0,
        "worst": min(rets) if rets else 0.0,
        "expectancy": statistics.fmean(rets) if rets else 0.0,
        "exp_ci": boot_mean(rets, rng),
        "ctrl_mean": statistics.fmean(ctrl) if ctrl else 0.0,
        "excess": (statistics.fmean(rets) - statistics.fmean(ctrl)) if rets and ctrl else 0.0,
        "excess_ci": boot_diff(rets, ctrl, rng),
        "exposure": held_days / max(last - first, 1),
        "cagr": cagr(eq), "maxdd": max_drawdown(eq),
        "bh_cagr": cagr(bh), "bh_maxdd": max_drawdown(bh),
    }
    res["rtd"] = ratio(res["cagr"], res["maxdd"])
    res["bh_rtd"] = ratio(res["bh_cagr"], res["bh_maxdd"])
    return res


def judge(r: dict, pm: dict) -> list[tuple[str, bool]]:
    checks = [(f"at least {pm['min_trades']} trades", r["trades"] >= pm["min_trades"])]
    if pm.get("expectancy_low_end_above_zero"):
        checks.append(("average trade > 0, low end above zero", r["exp_ci"][0] > 0))
    if pm.get("beats_random_low_end_above_zero"):
        checks.append(("beats random entries, low end above zero", r["excess_ci"][0] > 0))
    if pm.get("return_to_drawdown_at_least_buy_and_hold"):
        checks.append(("return/drawdown at least buy-and-hold's", r["rtd"] >= r["bh_rtd"]))
    return checks


def pct(x, d=2):
    return f"{x * 100:+.{d}f}%"


def report(sym: str, r: dict, checks) -> bool:
    print(f"\n{'=' * 66}\n{sym} · {r['start']} to {r['end']} · {r['years']:.1f} years\n{'=' * 66}")
    print(f"  trades        {r['trades']}  ({r['trades'] / max(r['years'], 1e-9):.1f} a year, "
          f"in the market {r['exposure']:.0%} of days)")
    print(f"  win rate      {r['win_rate']:.1%}")
    print(f"  avg win       {pct(r['avg_win'])}     avg loss  {pct(r['avg_loss'])}     "
          f"worst  {pct(r['worst'])}")
    lo, hi = r["exp_ci"]
    print(f"  expectancy    {pct(r['expectancy'], 3)} per trade after costs   "
          f"95% [{pct(lo, 3)}, {pct(hi, 3)}]")
    lo, hi = r["excess_ci"]
    print(f"  vs random     {pct(r['excess'], 3)} per trade   95% [{pct(lo, 3)}, {pct(hi, 3)}]   "
          f"(random entry: {pct(r['ctrl_mean'], 3)})")
    print(f"  {'':14}{'yearly':>10}{'worst drop':>13}{'ratio':>9}")
    print(f"  {'this rule':<14}{pct(r['cagr'], 1):>10}{pct(-r['maxdd'], 1):>13}{r['rtd']:>9.2f}")
    print(f"  {'buy and hold':<14}{pct(r['bh_cagr'], 1):>10}{pct(-r['bh_maxdd'], 1):>13}"
          f"{r['bh_rtd']:>9.2f}")
    print("  pass mark")
    ok = True
    for name, passed in checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    return ok


def main():
    p = argparse.ArgumentParser(description="dip buy in an uptrend: does it work after costs?")
    p.add_argument("--csv", required=True)
    p.add_argument("--rules", default=str(ROOT / "rules" / "dip_buy.json"))
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--explore", action="store_true")
    mode.add_argument("--final", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--i-know-its-spent", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    rules_path = Path(args.rules)
    cfg = json.loads(rules_path.read_text())
    rule, pm = cfg["rule"], cfg["pass_mark"]
    stamp = rules_path.with_suffix(".final.txt")

    if args.final and stamp.exists() and not args.i_know_its_spent:
        print(stamp.read_text())
        sys.exit("\nThe final test for this rule has already been run (above). Those dates are\n"
                 "spent. Running it again would make it a second look, not a test.")

    data = load(Path(args.csv))
    rng = random.Random(args.seed)
    since, until = (cfg["final_test_from"], None) if args.final else (None, cfg["explore_until"])

    print(f"\nrule: {cfg['name']}   (written {cfg['written']})")
    for line in cfg.get("rule_in_words", []):
        print(f"  - {line}")
    if args.final:
        print(f"\nFINAL TEST: {since} onward. Judged against the pass mark in {rules_path.name}.")
    else:
        print(f"\nEXPLORE: up to {until}. Verdict is for information; the final test decides.")

    results, passes = {}, 0
    for sym in pm["symbols"]:
        if sym not in data:
            print(f"\n{sym}: not in {args.csv}")
            continue
        r = study(data[sym], since, until, rule, rng)
        if r is None:
            print(f"\n{sym}: no data in range")
            continue
        results[sym] = r
        passes += report(sym, r, judge(r, pm))

    verdict = "PASS" if passes >= pm["must_pass_on"] else "FAIL"
    print(f"\n{'=' * 66}\nOVERALL: {verdict}  ({passes} of {len(pm['symbols'])} passed, "
          f"{pm['must_pass_on']} needed)\n{'=' * 66}")
    print("Not financial advice. Past results, especially of a well-known rule, can fade.\n")

    if args.final:
        lines = [f"final test run {datetime.now().isoformat(timespec='seconds')}",
                 f"rule written {cfg['written']}, dates {since} onward",
                 f"OVERALL: {verdict} ({passes} of {len(pm['symbols'])})"]
        for sym, r in results.items():
            lines.append(f"  {sym}: trades {r['trades']}, win {r['win_rate']:.1%}, "
                         f"expectancy {pct(r['expectancy'], 3)}, "
                         f"CAGR {pct(r['cagr'], 1)} vs {pct(r['bh_cagr'], 1)}, "
                         f"max DD {pct(-r['maxdd'], 1)} vs {pct(-r['bh_maxdd'], 1)}")
        stamp.write_text("\n".join(lines) + "\n")
        print(f"recorded -> {stamp}")


if __name__ == "__main__":
    main()
