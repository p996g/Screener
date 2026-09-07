"""
Evaluates alert rules against a run's results and delivers a concise
digest to any enabled channel (Discord, Telegram, email).

Design:
  - Rule evaluation (`evaluate_rules`) is pure: it takes the same
    `run_results` / `previous_by_profile` data main.py already computed
    for the dashboard's "changes since last run" section (see
    output/dashboard.py) — there's exactly one notion of "what changed",
    reused for both. No extra history scanning happens here.
  - Delivery is deliberately dumb: everything this run's rules produced is
    batched into ONE digest message per channel (not one notification per
    alert), both to stay within each channel's rate limits and because a
    short digest is what "concise" means for a dozen alerts firing on a
    broad market move.
  - Secrets are never read from settings.yaml. Every channel config names
    an *environment variable* (`..._env`) holding the actual secret; it's
    read from `os.environ` only at send time. A missing/unset variable is
    logged clearly and that channel is skipped — it never crashes the run
    (same philosophy as a failed data fetch — see data/fetcher.py).
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from dataclasses import dataclass
from email.message import EmailMessage
from typing import TYPE_CHECKING

from output.dashboard import RUN_FAILED_SENTINEL

if TYPE_CHECKING:
    import pandas as pd

    from main import ProfileRunResult

logger = logging.getLogger(__name__)

DISCORD_MAX_CHARS = 1900  # Discord's hard cap is 2000; leave headroom
TELEGRAM_MAX_CHARS = 4000  # Telegram's hard cap is 4096


@dataclass(frozen=True)
class Alert:
    profile: str
    symbol: str
    score: float
    rule: str
    reason: str


# -----------------------------------------------------------------------------
# Rule evaluation (pure — no I/O)
# -----------------------------------------------------------------------------


def evaluate_rules(
    run_results: list["ProfileRunResult"],
    previous_by_profile: dict[str, "pd.DataFrame | None"],
    rules_config: dict,
) -> list[Alert]:
    """Every alert triggered by this run, across every profile. A failed
    profile run contributes nothing — there's nothing meaningful to alert
    on when a run didn't produce results."""
    alerts: list[Alert] = []
    for run_result in run_results:
        if run_result.error:
            continue
        previous_df = previous_by_profile.get(run_result.profile_name)

        new_top_n_cfg = rules_config.get("new_top_n") or {}
        if new_top_n_cfg.get("enabled"):
            alerts.extend(_rule_new_top_n(run_result, previous_df, new_top_n_cfg["n"]))

        threshold_cfg = rules_config.get("score_threshold") or {}
        if threshold_cfg.get("enabled"):
            alerts.extend(_rule_score_threshold(run_result, previous_df, threshold_cfg["threshold"]))

        watchlist_cfg = rules_config.get("watchlist") or {}
        if watchlist_cfg.get("enabled"):
            watched = set(watchlist_cfg.get("symbols") or [])
            alerts.extend(_rule_watchlist(run_result, watched))

    return alerts


def _rule_new_top_n(run_result: "ProfileRunResult", previous_df, n: int) -> list[Alert]:
    previously_in_top_n = _previous_top_n_symbols(previous_df, n)
    alerts = []
    for position, ranked_symbol in enumerate(run_result.ranked[:n], start=1):
        if ranked_symbol.symbol in previously_in_top_n:
            continue
        alerts.append(
            Alert(
                profile=run_result.profile_name,
                symbol=ranked_symbol.symbol,
                score=ranked_symbol.score,
                rule="new_top_n",
                reason=f"entered the top {n} (now #{position})",
            )
        )
    return alerts


def _rule_score_threshold(run_result: "ProfileRunResult", previous_df, threshold: float) -> list[Alert]:
    previous_scores = _previous_scores_by_symbol(previous_df)
    alerts = []
    for ranked_symbol in run_result.ranked:
        if ranked_symbol.score < threshold:
            continue
        previous_score = previous_scores.get(ranked_symbol.symbol)
        if previous_score is not None and previous_score >= threshold:
            continue  # already above threshold last run -- not a new crossing
        alerts.append(
            Alert(
                profile=run_result.profile_name,
                symbol=ranked_symbol.symbol,
                score=ranked_symbol.score,
                rule="score_threshold",
                reason=f"composite score crossed {threshold:.2f} (now {ranked_symbol.score:.3f})",
            )
        )
    return alerts


def _rule_watchlist(run_result: "ProfileRunResult", watched_symbols: set[str]) -> list[Alert]:
    if not watched_symbols:
        return []
    alerts = []
    for position, ranked_symbol in enumerate(run_result.ranked, start=1):
        if ranked_symbol.symbol not in watched_symbols:
            continue
        alerts.append(
            Alert(
                profile=run_result.profile_name,
                symbol=ranked_symbol.symbol,
                score=ranked_symbol.score,
                rule="watchlist",
                reason=f"on your watchlist, survived {run_result.profile_name} (#{position})",
            )
        )
    return alerts


def _previous_top_n_symbols(previous_df, n: int) -> set[str]:
    usable = _usable_previous_rows(previous_df)
    if usable is None or "rank" not in usable.columns:
        return set()
    return set(usable[usable["rank"] <= n]["symbol"])


def _previous_scores_by_symbol(previous_df) -> dict[str, float]:
    usable = _usable_previous_rows(previous_df)
    if usable is None or "score" not in usable.columns:
        return {}
    return dict(zip(usable["symbol"], usable["score"]))


def _usable_previous_rows(previous_df):
    if previous_df is None or "symbol" not in previous_df.columns:
        return None
    return previous_df[previous_df["symbol"] != RUN_FAILED_SENTINEL]


# -----------------------------------------------------------------------------
# Digest formatting
# -----------------------------------------------------------------------------


def resolve_dashboard_link(alerts_config: dict, output_config: dict) -> str:
    """A public URL if one's configured (alerts.dashboard_url — set this
    if the dashboard is published somewhere reachable, e.g. GitHub Pages);
    otherwise the local file path, which is still useful context even if
    it isn't clickable from a phone notification."""
    return alerts_config.get("dashboard_url") or output_config.get("dashboard_path", "output/dashboard.html")


def _format_alert_line(alert: Alert) -> str:
    return f"• [{alert.profile}] {alert.symbol} — score {alert.score:.3f} — {alert.reason}"


def build_digest(alerts: list[Alert], dashboard_link: str, suppressed_count: int = 0) -> str:
    lines = [_format_alert_line(a) for a in alerts]
    if suppressed_count:
        lines.append(f"...and {suppressed_count} more alert(s) suppressed by max_alerts_per_run.")
    lines.append("")
    lines.append(f"Full dashboard: {dashboard_link}")
    return "\n".join(lines)


def _cap_alerts(alerts: list[Alert], max_alerts: int | None) -> tuple[list[Alert], int]:
    if max_alerts is None or len(alerts) <= max_alerts:
        return alerts, 0
    return alerts[:max_alerts], len(alerts) - max_alerts


# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------


def resolve_dry_run(alerts_config: dict) -> bool:
    """SCREENER_ALERTS_DRY_RUN, if set, overrides settings.yaml's
    alerts.dry_run — the quickest way to test rules/channels without
    editing config: `SCREENER_ALERTS_DRY_RUN=1 python main.py all`."""
    override = os.environ.get("SCREENER_ALERTS_DRY_RUN")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    return bool(alerts_config.get("dry_run", False))


def dispatch_alerts(alerts: list[Alert], alerts_config: dict, dashboard_link: str) -> None:
    """Cap, format, and deliver (or, in dry-run/quiet mode, log/print
    instead of deliver) this run's alerts."""
    if not alerts:
        return

    capped, suppressed = _cap_alerts(alerts, alerts_config.get("max_alerts_per_run"))
    message = build_digest(capped, dashboard_link, suppressed)

    if resolve_dry_run(alerts_config):
        logger.info("[DRY RUN] %d alert(s) would be sent:\n%s", len(capped), message)
        print(f"[DRY RUN] {len(capped)} alert(s) would be sent:\n{message}")
        return

    if alerts_config.get("quiet"):
        logger.info("Quiet mode enabled -- suppressing %d alert(s).", len(capped))
        return

    channels = [c for c in (alerts_config.get("channels") or []) if c.get("enabled")]
    if not channels:
        logger.warning("%d alert(s) triggered but no delivery channel is enabled.", len(capped))
        return

    for channel_config in channels:
        _send_via_channel(channel_config, message)


def _send_via_channel(channel_config: dict, message: str) -> None:
    channel_type = channel_config.get("type")
    sender = _CHANNEL_SENDERS.get(channel_type)
    if sender is None:
        logger.error("Unknown alert channel type '%s' -- skipping.", channel_type)
        return
    try:
        sent = sender(message, channel_config)
        if sent:
            logger.info("Sent alert digest via %s.", channel_type)
        # else: the sender already logged exactly why it skipped (e.g. a
        # missing secret) via _read_secret -- nothing more to say here.
    except Exception:
        logger.error("Failed to send alert digest via %s.", channel_type, exc_info=True)


def _read_secret(channel_config: dict, env_key_field: str) -> str | None:
    env_var_name = channel_config.get(env_key_field)
    if not env_var_name:
        logger.error("Alert channel config is missing '%s'.", env_key_field)
        return None
    value = os.environ.get(env_var_name)
    if not value:
        logger.error(
            "Environment variable '%s' is not set -- cannot send this alert channel.", env_var_name
        )
        return None
    return value


def send_discord(message: str, channel_config: dict) -> bool:
    """Returns True if a send was actually attempted (and didn't raise),
    False if it was skipped (e.g. a missing secret) — so callers can tell
    "skipped" apart from "sent" rather than logging both as success."""
    webhook_url = _read_secret(channel_config, "webhook_url_env")
    if not webhook_url:
        return False
    body = json.dumps({"content": _truncate(message, DISCORD_MAX_CHARS)}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=10):
        pass
    return True


def send_telegram(message: str, channel_config: dict) -> bool:
    bot_token = _read_secret(channel_config, "bot_token_env")
    chat_id = _read_secret(channel_config, "chat_id_env")
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    body = json.dumps({"chat_id": chat_id, "text": _truncate(message, TELEGRAM_MAX_CHARS)}).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10):
        pass
    return True


def send_email(message: str, channel_config: dict) -> bool:
    username = _read_secret(channel_config, "username_env")
    password = _read_secret(channel_config, "password_env")
    if not username or not password:
        return False

    email_message = EmailMessage()
    email_message["Subject"] = "Screener Alerts"
    email_message["From"] = channel_config["from_address"]
    email_message["To"] = ", ".join(channel_config["to_addresses"])
    email_message.set_content(message)

    smtp_host = channel_config["smtp_host"]
    smtp_port = channel_config.get("smtp_port", 587)
    with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as smtp:
        smtp.starttls()
        smtp.login(username, password)
        smtp.send_message(email_message)
    return True


def _truncate(message: str, max_chars: int) -> str:
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 15] + "\n...(truncated)"


_CHANNEL_SENDERS = {
    "discord": send_discord,
    "telegram": send_telegram,
    "email": send_email,
}
