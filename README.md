# Rotation bot

Runs the dip rotation rule on an Alpaca **paper** account once a day, a few minutes
before the close, from GitHub Actions. No computer needs to be on.

- Rule and universe: `rules/dip_rotation.json` (the exact file the backtest used)
- Bot: `bot/rotation_bot.py`
- Schedule: `.github/workflows/rotation.yml`
- Secrets needed: `ALPACA_KEY`, `ALPACA_SECRET`, `NTFY_TOPIC`
- Pause: Actions tab → Rotation bot → "..." → Disable workflow

Paper only. Not financial advice.
