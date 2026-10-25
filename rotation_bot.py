#!/usr/bin/env python3
"""Dip rotation bot: runs the tested rule on an Alpaca PAPER account.

    python bot/rotation_bot.py status          # account, positions, what it is tracking
    python bot/rotation_bot.py run --dry-run   # what it would do right now, no orders
    python bot/rotation_bot.py run             # do it (paper account only)

Meant to run unattended, either from GitHub Actions (.github/workflows/rotation.yml,
free, no computer needed) or from a Mac schedule (bot/install_schedule.sh). It
checks Alpaca's clock and only acts in the last 15 minutes before the close, so
it can be started as often as you like; it trades at most once per day.

Memory lives in the Alpaca account, not on disk. Every order the bot places is
tagged "rot-<date>-<symbol>-<side>". From those tags it knows which positions it
opened, when, and whether it already traded today. Nothing to sync, nothing to
lose when a cloud runner is thrown away.

The rule is read from rules/dip_rotation.json, the same file the backtest used,
and signals use the same functions, so the bot cannot drift from what was tested.

Safety:
  - Paper only. The trading URL is Alpaca's paper endpoint, hard-coded.
  - Kill switch: a file named STOP in the project root (or state/STOP), or the
    environment variable ROTATION_STOP=1, and every run does nothing. On GitHub,
    simply disable the workflow.
  - At most once a day, at most max_positions open, never more cash than it has.
  - Positions it did not open are never touched.

Alerts (optional, free): set NTFY_TOPIC to a private topic name and install the
ntfy app on your phone. It pushes when it trades, when something fails, and a
summary on Fridays. Silence means nothing happened.

Keys: ALPACA_KEY / ALPACA_SECRET environment variables, or ~/.scalp_alpaca.

Differences from the backtest, small and on purpose:
  - "Today's close" is the last trade a few minutes before the bell.
  - History is split-adjusted only, to match today's live price.
  - Idle cash earns nothing in the paper account (the backtest assumed T-bills).
  - An ETF sold today is not bought back the same day.

Not financial advice. Paper first; the bot has not earned real money.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dip_study import in_uptrend, is_dip, sma  # noqa: E402

ET = ZoneInfo("America/New_York")
TRADING_URL = "https://paper-api.alpaca.markets"     # paper. Do not change casually.
DATA_URL = "https://data.alpaca.markets"
STATE = ROOT / "state"
RULES = ROOT / "rules" / "dip_rotation.json"
TAG = "rot"
WINDOW_MIN = 15          # act only in the last 15 minutes before the close
HISTORY_DAYS = 420       # calendar days of daily bars; covers 200 trading days + margin


# ---------------------------------------------------------------------------
# the decision: pure, no network, fully testable
# ---------------------------------------------------------------------------

def decide(cfg: dict, history: dict, prices: dict, tracked: dict,
           broker_positions: dict, equity: float, cash: float, today: str) -> dict:
    """What to do today.

    history:  symbol -> completed daily bars (dicts with date, high, close), oldest first
    prices:   symbol -> latest trade price (stands in for today's close)
    tracked:  symbol -> {"entry_date": ...} for positions this bot opened
    broker_positions: symbol -> market value, as the broker reports it
    Returns {"sell": [(sym, why)], "buy": [(sym, notional, drop)], "notes": [...]}.
    """
    rule = cfg["rule"]
    k, slots = rule["down_days"], rule["max_positions"]
    notes, sell = [], []

    for sym in broker_positions:
        if sym not in tracked:
            notes.append(f"{sym}: held in the account but not opened by this bot; ignoring it")

    open_now = {}
    for sym, info in tracked.items():
        if sym not in broker_positions:
            continue
        bars, px = history.get(sym) or [], prices.get(sym)
        if not bars or px is None:
            notes.append(f"{sym}: no price today; holding")
            open_now[sym] = info
            continue
        held = sum(1 for b in bars if b["date"] > info["entry_date"]) + 1   # +1 for today
        if px > bars[-1]["high"]:
            sell.append((sym, f"closed above yesterday's high ({px:.2f} > {bars[-1]['high']:.2f})"))
        elif held >= rule["max_hold_days"]:
            sell.append((sym, f"held {held} trading days"))
        else:
            open_now[sym] = info

    freed = sum(broker_positions.get(s, 0.0) for s, _ in sell)
    cash_after = cash + freed
    free_slots = slots - len(open_now)
    sold = {s for s, _ in sell}
    sigs = []
    for sym in cfg["universe"]:
        if sym in open_now or sym in sold or sym in broker_positions and sym not in tracked:
            continue
        bars, px = history.get(sym) or [], prices.get(sym)
        if len(bars) < rule["trend_days"] or px is None:
            continue
        closes = [b["close"] for b in bars] + [px]
        i = len(closes) - 1
        avg = sma(closes, rule["trend_days"])
        if in_uptrend(i, closes, avg) and is_dip(i, closes, k):
            sigs.append((closes[i] / closes[i - k] - 1.0, sym))
    sigs.sort()

    buy = []
    for drop, sym in sigs:
        if free_slots <= 0:
            notes.append(f"{sym}: signalled ({drop:+.2%}) but no free slot")
            continue
        notional = round(min(equity / slots, cash_after) * 0.995, 2)   # small cushion
        if notional < 1.0:
            notes.append(f"{sym}: signalled but no cash left")
            continue
        buy.append((sym, notional, drop))
        cash_after -= notional
        free_slots -= 1

    return {"sell": sell, "buy": buy, "notes": notes}


def order_tag(day: str, sym: str, side: str) -> str:
    return f"{TAG}-{day}-{sym}-{side}"


def memory_from_orders(orders: list[dict], held: dict, today: str) -> tuple[dict, bool]:
    """Rebuild what the bot knows from its own tagged orders.

    Returns (tracked, traded_today). tracked maps each currently held symbol the
    bot bought to the date of its most recent filled buy.
    """
    tracked, traded_today = {}, False
    latest_buy: dict[str, str] = {}
    for o in orders:
        cid = o.get("client_order_id") or ""
        parts = cid.split("-")
        if len(parts) < 6 or parts[0] != TAG:
            continue
        day, sym, side = "-".join(parts[1:4]), parts[4], parts[5]
        if day == today and o.get("status") not in ("canceled", "rejected", "expired"):
            traded_today = True
        if side == "buy" and o.get("status") in ("filled", "partially_filled"):
            if day > latest_buy.get(sym, ""):
                latest_buy[sym] = day
    for sym in held:
        if sym in latest_buy:
            tracked[sym] = {"entry_date": latest_buy[sym]}
    return tracked, traded_today


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

def load_keys() -> tuple[str, str]:
    key, secret = os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")
    f = Path.home() / ".scalp_alpaca"
    if (not key or not secret) and f.exists():
        for line in f.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip().upper(), v.strip().strip('"').strip("'")
                if k in ("KEY", "ALPACA_KEY"):
                    key = v
                elif k in ("SECRET", "ALPACA_SECRET"):
                    secret = v
    if not key or not secret:
        sys.exit("no Alpaca keys: set ALPACA_KEY / ALPACA_SECRET or create ~/.scalp_alpaca")
    return key, secret


class Alpaca:
    def __init__(self, key: str, secret: str):
        import httpx
        h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.trade = httpx.Client(base_url=TRADING_URL, headers=h, timeout=30.0)
        self.data = httpx.Client(base_url=DATA_URL, headers=h, timeout=30.0)

    def _ok(self, r):
        if r.status_code >= 400:
            raise RuntimeError(f"alpaca {r.request.method} {r.request.url.path}: "
                               f"{r.status_code} {r.text[:300]}")
        return r.json() if r.text else {}

    def clock(self):
        return self._ok(self.trade.get("/v2/clock"))

    def account(self):
        return self._ok(self.trade.get("/v2/account"))

    def positions(self) -> dict:
        """symbol -> {"mv": market value, "qty": shares as Alpaca reports them}"""
        return {p["symbol"]: {"mv": float(p["market_value"]), "qty": p["qty"]}
                for p in self._ok(self.trade.get("/v2/positions"))}

    def orders(self, since: date) -> list[dict]:
        out, until = [], None
        while True:
            params = {"status": "all", "limit": 500, "direction": "desc",
                      "after": datetime(since.year, since.month, since.day, tzinfo=ET).isoformat()}
            if until:
                params["until"] = until
            page = self._ok(self.trade.get("/v2/orders", params=params))
            out += page
            if len(page) < 500:
                return out
            until = page[-1]["submitted_at"]

    def sell_all(self, sym, qty, day):
        return self._ok(self.trade.post("/v2/orders", json={
            "symbol": sym, "qty": str(qty), "side": "sell", "type": "market",
            "time_in_force": "day", "client_order_id": order_tag(day, sym, "sell")}))

    def buy_notional(self, sym, notional, day):
        return self._ok(self.trade.post("/v2/orders", json={
            "symbol": sym, "notional": f"{notional:.2f}", "side": "buy",
            "type": "market", "time_in_force": "day",
            "client_order_id": order_tag(day, sym, "buy")}))

    def daily_history(self, symbols, today: date) -> dict:
        """Completed daily bars up to yesterday."""
        # Midnight of the trading day, but never later than 20 minutes ago: on a
        # weekend the next close is days away, and asking for data up to then is
        # 'recent' as far as the free plan is concerned.
        end = min(datetime(today.year, today.month, today.day, tzinfo=ET),
                  datetime.now(timezone.utc) - timedelta(minutes=20))
        params = {"symbols": ",".join(symbols), "timeframe": "1Day",
                  "start": (today - timedelta(days=HISTORY_DAYS)).isoformat(),
                  "end": end.isoformat(),
                  "adjustment": "split", "feed": "sip", "limit": 10000}
        out: dict = {s: [] for s in symbols}
        while True:
            r = self.data.get("/v2/stocks/bars", params=params)
            if r.status_code == 403 and params["feed"] == "sip":
                # free plan refused full-market data: fall back to IEX. Daily
                # closes and highs differ slightly; the signal rarely does.
                print("  note: full-market data refused; using IEX daily bars")
                params["feed"] = "iex"
                params.pop("page_token", None)
                out = {s: [] for s in symbols}
                continue
            d = self._ok(r)
            for sym, bars in (d.get("bars") or {}).items():
                for b in bars:
                    out[sym].append({"date": b["t"][:10], "high": float(b["h"]),
                                     "close": float(b["c"])})
            if not d.get("next_page_token"):
                break
            params["page_token"] = d["next_page_token"]
        for sym in out:
            out[sym] = [b for b in sorted(out[sym], key=lambda b: b["date"])
                        if b["date"] < today.isoformat()]
        return out

    def latest_prices(self, symbols) -> dict:
        d = self._ok(self.data.get("/v2/stocks/trades/latest",
                                   params={"symbols": ",".join(symbols), "feed": "iex"}))
        return {s: float(t["p"]) for s, t in (d.get("trades") or {}).items()}


# ---------------------------------------------------------------------------
# alerts, journal, kill switch
# ---------------------------------------------------------------------------

def notify(title: str, body: str, urgent: bool = False) -> None:
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    try:
        import httpx
        httpx.post(f"https://ntfy.sh/{topic}", content=body.encode(),
                   headers={"Title": title, "Priority": "high" if urgent else "default"},
                   timeout=15.0)
    except Exception as exc:          # an alert failing must never stop trading
        print(f"  (alert not sent: {exc})")


def journal(rows: list[dict]) -> None:
    """Local CSV when there is a disk to keep it; Alpaca's order history is the record."""
    try:
        STATE.mkdir(exist_ok=True)
        p = STATE / "rotation_journal.csv"
        new = not p.exists()
        with p.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["ts", "action", "symbol", "amount", "detail"])
            if new:
                w.writeheader()
            w.writerows(rows)
    except OSError:
        pass
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and rows:
        with open(summary, "a") as fh:
            fh.write("| action | symbol | amount | detail |\n|---|---|---|---|\n")
            for r in rows:
                fh.write(f"| {r['action']} | {r['symbol']} | {r['amount']} | {r['detail']} |\n")


def stopped() -> bool:
    return (os.environ.get("ROTATION_STOP") == "1" or (ROOT / "STOP").exists()
            or (STATE / "STOP").exists())


# ---------------------------------------------------------------------------

def cmd_status(api) -> None:
    a, c = api.account(), api.clock()
    today = datetime.fromisoformat(c["next_close"].replace("Z", "+00:00")).astimezone(ET).date()
    pos = api.positions()
    tracked, traded = memory_from_orders(api.orders(today - timedelta(days=30)), pos,
                                         today.isoformat())
    print(f"paper account  equity ${float(a['equity']):,.2f}   cash ${float(a['cash']):,.2f}")
    print(f"market         {'open' if c['is_open'] else 'closed'} · next close {c['next_close']}")
    print(f"traded today   {'yes' if traded else 'no'}")
    if not pos:
        print("positions      none")
    for sym, p in pos.items():
        t = tracked.get(sym)
        tag = f"bought by bot {t['entry_date']}" if t else "not opened by this bot"
        print(f"  {sym:<6} ${p['mv']:>10,.2f}   {tag}")
    if stopped():
        print("KILL SWITCH ON: runs will do nothing")


def cmd_run(api, cfg: dict, dry: bool, force_window: bool) -> int:
    now = datetime.now(timezone.utc)
    if stopped():
        print("kill switch on; doing nothing")
        return 0

    c = api.clock()
    close = datetime.fromisoformat(c["next_close"].replace("Z", "+00:00"))
    mins_left = (close - now).total_seconds() / 60.0
    if not c["is_open"] or mins_left > WINDOW_MIN:
        if not force_window:
            print(f"not in the last {WINDOW_MIN} minutes before the close "
                  f"({'market closed' if not c['is_open'] else f'{mins_left:.0f} min left'}); "
                  "doing nothing")
            return 0
        print(f"outside the trading window ({mins_left:.0f} min to close); dry run only")
        dry = True

    today = close.astimezone(ET).date()
    day = today.isoformat()
    pos = api.positions()
    tracked, traded_today = memory_from_orders(api.orders(today - timedelta(days=30)), pos, day)
    if traded_today and not dry:
        print(f"already traded today ({day}); doing nothing")
        return 0

    syms = cfg["universe"]
    history = api.daily_history(syms, today)
    prices = api.latest_prices(syms)
    acct = api.account()
    equity, cash = float(acct["equity"]), float(acct["cash"])
    plan = decide(cfg, history, prices, tracked, {s: p["mv"] for s, p in pos.items()},
                  equity, cash, day)

    ts = now.isoformat(timespec="seconds")
    rows = [{"ts": ts, "action": "note", "symbol": "", "amount": "", "detail": n}
            for n in plan["notes"]]
    print(f"{day} · equity ${equity:,.2f} · cash ${cash:,.2f}{'  [DRY RUN]' if dry else ''}")
    for n in plan["notes"]:
        print(f"  note  {n}")
    if not plan["sell"] and not plan["buy"]:
        print("  nothing to do today")

    done, failed = [], []
    for sym, why in plan["sell"]:
        print(f"  SELL  {sym}  ({why})")
        if dry:
            continue
        try:
            api.sell_all(sym, pos[sym]["qty"], day)
            done.append(f"SELL {sym} ${pos[sym]['mv']:,.0f} ({why})")
            rows.append({"ts": ts, "action": "sell", "symbol": sym,
                         "amount": f"{pos[sym]['mv']:.2f}", "detail": why})
        except Exception as exc:
            failed.append(f"SELL {sym}: {exc}")
            rows.append({"ts": ts, "action": "sell_failed", "symbol": sym,
                         "amount": "", "detail": str(exc)})

    for sym, notional, drop in plan["buy"]:
        print(f"  BUY   {sym}  ${notional:,.2f}  (down {drop:+.2%} over "
              f"{cfg['rule']['down_days']} days)")
        if dry:
            continue
        try:
            api.buy_notional(sym, notional, day)
            done.append(f"BUY {sym} ${notional:,.0f} (down {drop:+.1%} in 3 days)")
            rows.append({"ts": ts, "action": "buy", "symbol": sym,
                         "amount": f"{notional:.2f}", "detail": f"3-day drop {drop:+.2%}"})
        except Exception as exc:
            failed.append(f"BUY {sym}: {exc}")
            rows.append({"ts": ts, "action": "buy_failed", "symbol": sym,
                         "amount": f"{notional:.2f}", "detail": str(exc)})

    if dry:
        return 0

    rows.append({"ts": ts, "action": "equity", "symbol": "", "amount": f"{equity:.2f}",
                 "detail": f"cash {cash:.2f}"})
    journal(rows)

    if done:
        notify(f"Rotation bot traded · equity ${equity:,.0f}", "\n".join(done))
    if failed:
        notify("Rotation bot: order failed", "\n".join(failed), urgent=True)
    if today.weekday() == 4:   # Friday summary, traded or not
        held = ", ".join(f"{s} ${p['mv']:,.0f}" for s, p in pos.items()) or "none"
        notify("Rotation bot · weekly check-in",
               f"Equity ${equity:,.2f}\nCash ${cash:,.2f}\nPositions before today: {held}\n"
               f"Today: {'; '.join(done) if done else 'no trades'}")
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description="dip rotation bot (Alpaca paper)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true", help="show the plan, place nothing")
    r.add_argument("--any-time", action="store_true",
                   help="allow a dry run outside the closing window (never places orders)")
    t = sub.add_parser("test-alert", help="send a test notification to your phone")
    args = p.parse_args()

    if args.cmd == "test-alert":
        if not os.environ.get("NTFY_TOPIC"):
            sys.exit("set NTFY_TOPIC first")
        notify("Rotation bot", "Test alert: if you can read this, alerts work.")
        print("sent")
        return 0

    cfg = json.loads(RULES.read_text())
    api = Alpaca(*load_keys())
    if args.cmd == "status":
        cmd_status(api)
        return 0
    try:
        return cmd_run(api, cfg, args.dry_run, args.any_time)
    except Exception:
        err = traceback.format_exc()
        print(err)
        notify("Rotation bot: run failed", err[-900:], urgent=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
