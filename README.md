# Screener

A config-driven market screener: it pulls daily OHLCV data for a universe of
tickers, runs each symbol through a pipeline of pass/fail filters, scores and
ranks whoever survives, and reports the result — as a CSV, a console table,
and a self-contained HTML dashboard. It can run once on demand, on a schedule
(always-on loop or free via GitHub Actions), and can ping you on Discord,
Telegram, or email when something worth noticing happens.

Almost nothing about *what* it screens for lives in the code. It lives in
[`config/settings.yaml`](config/settings.yaml), which is the only file most
people ever need to touch.

> **This screener finds candidates by fixed rules. It does not know news,
> context, or whether a rule stopped making sense.** A filter that was sound
> when it was written can quietly stop being sound as market conditions
> change, and nothing here will tell you that's happened — that judgment is
> yours. **Candidates feed a research process; they are not signals.** Treat
> every name that comes out of a run as a starting point for your own
> homework, never as an instruction.

## Quick start

```bash
pip install -r requirements.txt
python main.py all
```

That screens the three example profiles shipped in `config/settings.yaml`
against a small starter universe, writes CSVs and `output/dashboard.html`,
and prints a summary. Open the dashboard in any browser — it's one static
file, no server needed.

## How it fits together

```
config/settings.yaml   <- almost everything: universe, filters, ranking,
                           profiles, schedule, alerts, data-quality — all here

data/universe.py       -> resolves the ticker list
data/fetcher.py        -> pulls OHLCV bars (yfinance), point-in-time truncated,
                           with on-disk caching and clear failure handling
data/quality.py        -> flags stale/gappy/suspicious data before screening

filters/base.py        -> the Filter contract every filter implements
filters/library.py     -> the 8 built-in filters (see below)

engine.py              -> runs the universe through data-quality + filters
ranking.py             -> scores and ranks survivors from configurable weights
profiles.py            -> resolves a named profile as a config override

output/report.py       -> CSV + console output, per-profile or combined
output/dashboard.py    -> the self-contained HTML dashboard

scheduler.py           -> "when is a run due" — shared by both deployment modes
alerts.py              -> rule evaluation + Discord/Telegram/email delivery
audit.py               -> the look-back (point-in-time) audit, runnable standalone

main.py                -> ties everything together; the one thing you run
```

**The filter library** (`filters/library.py`): `liquidity`, `price_range`,
`trend`, `momentum`, `proximity`, `volatility`, `volume_surge`, `gap_event`.
Each is independently enabled/configured in settings.yaml, and four of them
(`momentum`, `trend`, `volume_surge`, `proximity`) also contribute a
normalized "how good, not just pass/fail" score that ranking combines by
weight.

**Point-in-time discipline** is enforced in one place — `engine.py` truncates
every symbol's data (and any benchmark data a filter requested) to the
evaluation date before a filter ever sees it — and independently verified by
`audit.py` (below), which re-runs every registered filter against poisoned
future data and asserts the result never changes.

## Config surface

Everything in `config/settings.yaml` is heavily commented in place; this is
just the map. Top-level sections:

| Section | Controls |
|---|---|
| `universe` | The ticker list (or future: an index source) |
| `data` | Provider, history window, evaluation date (incl. backtesting a past date), caching |
| `data_quality` | Staleness/gap/suspicious-price thresholds (see below) |
| `filters` | Default parameters for all 8 filters — a shared baseline every profile inherits |
| `ranking` | Default composite-score weights, `top_n`, `min_score` |
| `output` | Report formats, paths, history archive, dashboard settings |
| `profiles` | Named screens, each a **partial override** of everything above |
| `schedule` | When automated runs fire (see Deployment) |
| `logging` | Where diagnostics go |
| `alerts` | Rules + delivery channels (see Alerts) |

**Profiles are the main customization surface.** A profile only lists what's
*different* — filters it turns on and tunes, its own ranking weights, maybe
its own output path — and inherits everything else (the universe, the
liquidity floor, any filter field it doesn't mention) unchanged. Three ship
as examples: `momentum_screen`, `pullback_screen`, `breakout_watch`. There's
a step-by-step "clone and tune" comment directly above the `profiles:`
section in settings.yaml.

## Guardrails

- **Data quality** (`data_quality` in settings.yaml, `data/quality.py`): before
  any filter runs, each symbol's point-in-time data is checked for staleness,
  large gaps between bars, and suspicious prices (non-positive values,
  High < Low, an implausible single-day move). A flagged symbol is excluded
  outright — no filter runs on it — with the specific reason surfaced in the
  console output and the dashboard's "Data quality exclusions" section. A
  filter has no way to tell "real news" from "bad data"; this is what keeps
  a data problem from quietly becoming a confidently-wrong result.
- **Look-back audit** (`audit.py`): re-uses the point-in-time regression-test
  technique (feed the real engine a "poisoned" history with extreme values
  planted after the evaluation date, assert the result is unaffected)
  against every filter actually registered in the library, using each
  filter's real default config. Run it any time, especially after adding a
  new filter:
  ```bash
  python audit.py
  ```
  It's also wired into the test suite (`tests/test_lookback_audit.py`), so
  `pytest` catches a regression here automatically.
- **Small list is fine**: nothing in the reporting or ranking path ever pads
  a short result. `min_score`/`top_n` only ever shrink a list. Zero survivors
  renders as an explicit "No qualifying candidates today" — in the console,
  the CSV, and the dashboard — never a blank table that looks like something
  broke.

## The dashboard

`output/dashboard.html` is regenerated as the last step of every run — one
self-contained file, no build step, no JavaScript framework (expand/collapse
per symbol uses plain HTML5 `<details>`). Per profile it shows: today's
ranked results with full filter-by-filter detail, what changed since the
last run (new entrants, dropped names, big rank moves), a small recent-runs
history view, and any data-quality exclusions. The disclaimer above appears
in its footer on every regeneration.

## Alerts

Off by default (`alerts.enabled: false`). Three rule types — a symbol newly
entering a profile's top N, a composite score crossing a threshold, or a
symbol from a hand-picked `watchlist` appearing in any screen — batched into
one digest per run per channel (Discord webhook, Telegram bot, and/or email).
Secrets are **never** written to settings.yaml, only the *name* of an
environment variable that holds them. `alerts.dry_run: true` (the default)
prints what would be sent instead of sending it — see `DEPLOYMENT.md` for
wiring up real secrets per deployment mode.

## Deployment

Three ways to run it, same `main.py`:

- `python main.py <profile|all>` — one-off manual run.
- `python main.py schedule` — always-on loop for a machine you keep running.
- `python main.py scheduled-run` — one-shot check-and-maybe-run, fired by the
  included GitHub Actions workflow (`.github/workflows/screener.yml`) for
  free, with no server, committing the report archive back to the repo.

**Full setup instructions, the tradeoffs between the two automated modes, and
a recommendation for getting started live in [`DEPLOYMENT.md`](DEPLOYMENT.md).**
Short version: GitHub Actions if you're starting out — nothing to provision
or keep alive, and the archive lands as browsable commits.

## Making it your own

1. **Point it at your own list**: edit `universe.tickers` in settings.yaml.
2. **Tune an existing profile, or clone one**: copy a block under `profiles:`,
   rename it, change what it screens for. See the comment above `profiles:`
   for the exact steps — this needs zero code changes.
3. **Re-weight ranking**: every profile has its own `ranking.weights`; only
   `momentum`, `trend`, `volume_surge`, and `proximity` have a score to
   weight (the rest are pure gates).
4. **Add a new filter**: subclass `Filter` in `filters/library.py`
   (`pass_fail` required, `score_component` optional), register it in
   `FILTER_REGISTRY`, give it a config block in settings.yaml. Run
   `python audit.py` afterward — it picks up a new registered filter
   automatically.
5. **Loosen or tighten the guardrails**: `data_quality` thresholds are
   per-profile-overridable like everything else.
6. **Wire up alerts**: flip `alerts.enabled: true`, pick rules and a channel,
   set the channel's secret as an environment variable, leave `dry_run: true`
   until the digest looks right.

## Testing

```bash
pip install -r requirements.txt
pytest
python audit.py
```

The test suite covers every module — filters (including the four score
components), the engine (including point-in-time truncation and data-quality
gating), ranking, profile resolution, the scheduler (including the DST
cron-bracketing math the GitHub Actions workflow relies on), report/dashboard
rendering, and alerts (rule evaluation and each delivery channel, with no
real network calls). `python audit.py` is the same look-back audit `pytest`
runs, runnable standalone with human-readable PASS/FAIL output.
