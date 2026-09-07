# Running the screener on a schedule

The screener runs itself in three ways, all driven by the `schedule:` section
of `config/settings.yaml` — retiming a run or adding a new one is a config
change, not a code change:

- `python main.py <profile>` / `python main.py all` — a one-off manual run
  (what you've been doing so far).
- `python main.py schedule` — **deployment (a)**: an always-on loop for a
  server or always-on machine you already have.
- `python main.py scheduled-run` — **deployment (b)**: a single
  check-and-maybe-run, fired by GitHub Actions' cron on a schedule, for
  free, with no server of your own.

Every run — manual, looped, or scheduled — writes a timestamped snapshot of
its report to `output/history/` (e.g. `output/history/20260906T133000Z_all.csv`),
so the archive builds up the same way regardless of which mode produced it.
Turn it off with `output.save_history: false` if you don't want it.

## Deployment (a): always-on loop

For a machine that's already running 24/7 (your own server, a Raspberry Pi,
a always-on cloud VM, `tmux`/`screen` on a machine you leave on).

```bash
pip install -r requirements.txt
python main.py schedule
```

This blocks forever. It wakes up every `schedule.poll_interval_seconds`
(default 30s), checks whether any `schedule.run_times` entry is due, and
runs it — skipping weekends (and any dates listed in `schedule.holidays`)
automatically. Logs go to stdout and, if `logging.file` is set, to that file
too.

To keep it running across reboots/crashes, use whatever process supervisor
you already have — `systemd`, `supervisord`, a `pm2` process, or even a
`screen`/`tmux` session with `nohup`. A minimal systemd unit:

```ini
[Unit]
Description=Screener scheduler
After=network.target

[Service]
WorkingDirectory=/path/to/Screener
ExecStart=/path/to/Screener/.venv/bin/python main.py schedule
Restart=always

[Install]
WantedBy=multi-user.target
```

## Deployment (b): GitHub Actions (free, no server)

The workflow file is already at `.github/workflows/screener.yml`. To turn it
on:

1. Make sure this project is a git repo with a GitHub remote (`git init`,
   create a repo on GitHub, `git remote add origin ...`, push).
2. In the repo's Settings → Actions → General → Workflow permissions,
   select **"Read and write permissions"** (needed for the workflow to
   commit the report back).
3. Push `.github/workflows/screener.yml` (already included) — that's it.
   GitHub will start firing it on the schedule below automatically. You can
   also trigger it manually from the Actions tab (`workflow_dispatch`) to
   test it right away without waiting for a cron trigger.

GitHub Actions' cron is UTC-only and has no idea what "30 minutes before the
NYSE open" or "DST" means, so the workflow fires **two** cron triggers per
configured run time — one tuned for EDT, one for EST, an hour apart — and
`python main.py scheduled-run` does the real, timezone-aware check itself
(via `scheduler.find_due_run`), no-opping cleanly on whichever trigger
doesn't actually correspond to a due run right now. Exchange holidays
(`schedule.holidays` in settings.yaml) are respected the same way, so a
cron firing on July 4th just no-ops instead of screening a closed market.

Each run either no-ops (nothing committed) or commits and pushes whatever
changed under `output/history/` and the profile CSVs — so your repo's commit
history *is* the archive. If the screener run itself fails (e.g. a total
network outage), the workflow still commits whatever was produced (including
a `RUN FAILED: ...` report row) and then marks the job as failed, so you get
a GitHub notification rather than a silent gap in the archive.

**Caveat:** GitHub's own docs note that scheduled workflows can be delayed
by several minutes during periods of high load on their infrastructure, and
a repo with no activity for 60+ days has its scheduled workflows disabled
until someone visits the Actions tab. Neither matters much for a
twice-a-day screener, but worth knowing.

## Which one should you use?

| | (a) Always-on loop | (b) GitHub Actions |
|---|---|---|
| **Cost** | Whatever the machine costs (or free if you already leave one on) | Free for a public repo; a private repo gets 2,000 free minutes/month, and this job takes well under a minute per run |
| **Setup effort** | Install deps, run it, keep it alive (systemd/supervisor) yourself | Push one YAML file, flip a permissions toggle |
| **Reliability** | As reliable as your machine + supervisor config | GitHub's infrastructure; the docs warn scheduled runs can occasionally lag by minutes |
| **Timing precision** | Exact — checks the real clock every 30s | Approximate — bounded by `trigger_tolerance_minutes` (default ±20 min) around the DST-bracketing cron pairs |
| **Where output lives** | Wherever the machine's disk is (you decide backup/access) | Committed straight into the repo — free versioned archive, browsable on GitHub from anywhere |
| **You maintain** | The machine staying up, OS updates, the process not dying | Nothing beyond the repo itself |

**Recommendation for someone starting out: GitHub Actions (b).** There's no
server to provision, patch, or forget about, no process supervisor to
configure, and the archive lands as reviewable, versioned commits you can
browse from your phone — for a twice-a-day screen the ~20-minute timing
tolerance is irrelevant. Reach for the always-on loop (a) only once you
already have a machine running anyway, want tighter timing precision, need
to run far more often than twice a day, or want output that never touches a
third-party service.

## Alerts

`config/settings.yaml`'s `alerts:` section (off by default) can ping you via
Discord, Telegram, and/or email when a rule fires — a symbol newly enters a
profile's top N, a composite score crosses a threshold, or a symbol from your
`watchlist` survives any screen. See `alerts.py` and the comments in
settings.yaml for the rules/channels themselves; this section is just about
wiring up the secrets each channel needs, since **they never go in
settings.yaml or any file in the repo** — only the *name* of an environment
variable does (e.g. `webhook_url_env: DISCORD_WEBHOOK_URL`), read from the
environment at send time.

**Test before you wire up real delivery.** `alerts.dry_run: true` (the
default once you flip `alerts.enabled: true`) prints the exact digest that
would be sent instead of sending it — safe to leave on indefinitely, or
force it either way without editing the file:

```bash
SCREENER_ALERTS_DRY_RUN=1 python main.py all   # force dry-run
SCREENER_ALERTS_DRY_RUN=0 python main.py all   # force real delivery
```

**Setting the secrets, per deployment mode:**

- **Manual / always-on loop (a):** export them in the shell (or a systemd
  unit's `Environment=`/`EnvironmentFile=`) before running `main.py`:
  ```bash
  export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
  python main.py schedule
  ```
- **GitHub Actions (b):** add each one under this repo's Settings → Secrets
  and variables → Actions → New repository secret, using the exact name
  referenced by `*_env` in settings.yaml. `.github/workflows/screener.yml`
  already forwards `DISCORD_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN`,
  `TELEGRAM_CHAT_ID`, `SMTP_USERNAME`, and `SMTP_PASSWORD` into the run step's
  environment — you only need to actually set the ones for the channel(s)
  you enabled; an unset one is skipped with a clear log line, not a failure.

A missing/misconfigured channel never breaks a run — it's logged clearly
(same philosophy as a failed data fetch) and the rest of the run, including
every other enabled channel, proceeds normally.
