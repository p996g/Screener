"""
Tests for alerts.py: rule evaluation (pure), digest formatting, capping,
dry-run/quiet dispatch behavior, and each delivery channel — all without
ever touching a real network or SMTP server.
"""

from dataclasses import dataclass, field

import pandas as pd
import pytest

from alerts import (
    Alert,
    _cap_alerts,
    build_digest,
    dispatch_alerts,
    evaluate_rules,
    resolve_dashboard_link,
    resolve_dry_run,
    send_discord,
    send_email,
    send_telegram,
)
from ranking import RankedSymbol


@dataclass
class _FakeRunResult:
    profile_name: str
    ranked: list = field(default_factory=list)
    error: str | None = None


def _ranked(symbol: str, score: float) -> RankedSymbol:
    return RankedSymbol(symbol=symbol, score=score, close=100.0, component_scores={})


def _previous_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# new_top_n rule
# -----------------------------------------------------------------------------


def test_new_top_n_alerts_when_no_prior_run():
    run_result = _FakeRunResult("demo", ranked=[_ranked("AAA", 0.9)])
    alerts = evaluate_rules([run_result], {"demo": None}, {"new_top_n": {"enabled": True, "n": 5}})
    assert len(alerts) == 1
    assert alerts[0] == Alert("demo", "AAA", 0.9, "new_top_n", "entered the top 5 (now #1)")


def test_new_top_n_silent_when_symbol_already_in_top_n():
    run_result = _FakeRunResult("demo", ranked=[_ranked("AAA", 0.9)])
    previous = _previous_df([{"symbol": "AAA", "rank": 3}])
    alerts = evaluate_rules([run_result], {"demo": previous}, {"new_top_n": {"enabled": True, "n": 5}})
    assert alerts == []


def test_new_top_n_alerts_when_symbol_moves_into_top_n_from_outside():
    run_result = _FakeRunResult("demo", ranked=[_ranked("AAA", 0.9)])
    previous = _previous_df([{"symbol": "AAA", "rank": 8}])
    alerts = evaluate_rules([run_result], {"demo": previous}, {"new_top_n": {"enabled": True, "n": 5}})
    assert len(alerts) == 1


def test_new_top_n_ignores_symbols_outside_top_n_today():
    ranked = [_ranked(f"S{i}", 1.0 - i * 0.1) for i in range(6)]  # 6 symbols, ranks 1..6
    run_result = _FakeRunResult("demo", ranked=ranked)
    alerts = evaluate_rules([run_result], {"demo": None}, {"new_top_n": {"enabled": True, "n": 5}})
    assert {a.symbol for a in alerts} == {"S0", "S1", "S2", "S3", "S4"}  # not S5


def test_new_top_n_disabled_produces_no_alerts():
    run_result = _FakeRunResult("demo", ranked=[_ranked("AAA", 0.9)])
    alerts = evaluate_rules([run_result], {"demo": None}, {"new_top_n": {"enabled": False, "n": 5}})
    assert alerts == []


# -----------------------------------------------------------------------------
# score_threshold rule
# -----------------------------------------------------------------------------


def test_score_threshold_alerts_on_first_crossing():
    run_result = _FakeRunResult("demo", ranked=[_ranked("AAA", 0.9)])
    alerts = evaluate_rules(
        [run_result], {"demo": None}, {"score_threshold": {"enabled": True, "threshold": 0.85}}
    )
    assert len(alerts) == 1
    assert alerts[0].rule == "score_threshold"


def test_score_threshold_silent_when_already_above_last_run():
    run_result = _FakeRunResult("demo", ranked=[_ranked("AAA", 0.9)])
    previous = _previous_df([{"symbol": "AAA", "score": 0.87}])
    alerts = evaluate_rules(
        [run_result], {"demo": previous}, {"score_threshold": {"enabled": True, "threshold": 0.85}}
    )
    assert alerts == []


def test_score_threshold_silent_when_below_threshold():
    run_result = _FakeRunResult("demo", ranked=[_ranked("AAA", 0.5)])
    alerts = evaluate_rules(
        [run_result], {"demo": None}, {"score_threshold": {"enabled": True, "threshold": 0.85}}
    )
    assert alerts == []


def test_score_threshold_fires_again_after_dropping_and_recrossing():
    run_result = _FakeRunResult("demo", ranked=[_ranked("AAA", 0.9)])
    previous = _previous_df([{"symbol": "AAA", "score": 0.5}])  # was below threshold last run
    alerts = evaluate_rules(
        [run_result], {"demo": previous}, {"score_threshold": {"enabled": True, "threshold": 0.85}}
    )
    assert len(alerts) == 1


# -----------------------------------------------------------------------------
# watchlist rule
# -----------------------------------------------------------------------------


def test_watchlist_fires_every_run_not_just_when_new():
    run_result = _FakeRunResult("demo", ranked=[_ranked("NVDA", 0.5)])
    previous = _previous_df([{"symbol": "NVDA", "rank": 1, "score": 0.5}])  # identical to last run
    alerts = evaluate_rules(
        [run_result], {"demo": previous}, {"watchlist": {"enabled": True, "symbols": ["NVDA"]}}
    )
    assert len(alerts) == 1
    assert alerts[0].rule == "watchlist"


def test_watchlist_ignores_symbols_not_on_the_list():
    run_result = _FakeRunResult("demo", ranked=[_ranked("AAPL", 0.5)])
    alerts = evaluate_rules(
        [run_result], {"demo": None}, {"watchlist": {"enabled": True, "symbols": ["NVDA"]}}
    )
    assert alerts == []


def test_watchlist_empty_list_produces_no_alerts():
    run_result = _FakeRunResult("demo", ranked=[_ranked("NVDA", 0.5)])
    alerts = evaluate_rules([run_result], {"demo": None}, {"watchlist": {"enabled": True, "symbols": []}})
    assert alerts == []


# -----------------------------------------------------------------------------
# Cross-cutting
# -----------------------------------------------------------------------------


def test_failed_profile_run_contributes_no_alerts():
    run_result = _FakeRunResult("demo", ranked=[], error="network down")
    rules = {
        "new_top_n": {"enabled": True, "n": 5},
        "score_threshold": {"enabled": True, "threshold": 0.0},
        "watchlist": {"enabled": True, "symbols": ["AAA"]},
    }
    alerts = evaluate_rules([run_result], {"demo": None}, rules)
    assert alerts == []


def test_one_symbol_can_trigger_multiple_rules():
    run_result = _FakeRunResult("demo", ranked=[_ranked("NVDA", 0.95)])
    rules = {
        "new_top_n": {"enabled": True, "n": 5},
        "score_threshold": {"enabled": True, "threshold": 0.85},
        "watchlist": {"enabled": True, "symbols": ["NVDA"]},
    }
    alerts = evaluate_rules([run_result], {"demo": None}, rules)
    assert {a.rule for a in alerts} == {"new_top_n", "score_threshold", "watchlist"}


def test_evaluate_rules_covers_every_profile_independently():
    run_a = _FakeRunResult("profile_a", ranked=[_ranked("AAA", 0.9)])
    run_b = _FakeRunResult("profile_b", ranked=[_ranked("BBB", 0.9)])
    alerts = evaluate_rules(
        [run_a, run_b], {"profile_a": None, "profile_b": None}, {"new_top_n": {"enabled": True, "n": 5}}
    )
    assert {a.profile for a in alerts} == {"profile_a", "profile_b"}


# -----------------------------------------------------------------------------
# Digest formatting / capping
# -----------------------------------------------------------------------------


def test_cap_alerts_under_limit_is_unchanged():
    alerts = [Alert("p", "A", 0.9, "r", "reason")]
    capped, suppressed = _cap_alerts(alerts, max_alerts=10)
    assert capped == alerts
    assert suppressed == 0


def test_cap_alerts_truncates_and_counts_suppressed():
    alerts = [Alert("p", f"S{i}", 0.9, "r", "reason") for i in range(5)]
    capped, suppressed = _cap_alerts(alerts, max_alerts=2)
    assert len(capped) == 2
    assert suppressed == 3


def test_cap_alerts_none_means_unlimited():
    alerts = [Alert("p", f"S{i}", 0.9, "r", "reason") for i in range(5)]
    capped, suppressed = _cap_alerts(alerts, max_alerts=None)
    assert len(capped) == 5
    assert suppressed == 0


def test_build_digest_includes_profile_symbol_score_reason_and_link():
    alert = Alert("momentum_screen", "NVDA", 0.913, "new_top_n", "entered the top 5 (now #1)")
    digest = build_digest([alert], "output/dashboard.html")
    assert "momentum_screen" in digest
    assert "NVDA" in digest
    assert "0.913" in digest
    assert "entered the top 5" in digest
    assert "output/dashboard.html" in digest


def test_build_digest_notes_suppressed_count():
    alert = Alert("p", "A", 0.9, "r", "reason")
    digest = build_digest([alert], "link", suppressed_count=4)
    assert "4 more alert(s) suppressed" in digest


def test_resolve_dashboard_link_prefers_configured_url():
    link = resolve_dashboard_link({"dashboard_url": "https://example.com/dash"}, {"dashboard_path": "output/dashboard.html"})
    assert link == "https://example.com/dash"


def test_resolve_dashboard_link_falls_back_to_local_path():
    link = resolve_dashboard_link({"dashboard_url": None}, {"dashboard_path": "output/dashboard.html"})
    assert link == "output/dashboard.html"


# -----------------------------------------------------------------------------
# resolve_dry_run
# -----------------------------------------------------------------------------


def test_resolve_dry_run_defaults_to_config_value(monkeypatch):
    monkeypatch.delenv("SCREENER_ALERTS_DRY_RUN", raising=False)
    assert resolve_dry_run({"dry_run": True}) is True
    assert resolve_dry_run({"dry_run": False}) is False


def test_resolve_dry_run_env_override_forces_true(monkeypatch):
    monkeypatch.setenv("SCREENER_ALERTS_DRY_RUN", "1")
    assert resolve_dry_run({"dry_run": False}) is True


def test_resolve_dry_run_env_override_forces_false(monkeypatch):
    monkeypatch.setenv("SCREENER_ALERTS_DRY_RUN", "0")
    assert resolve_dry_run({"dry_run": True}) is False


# -----------------------------------------------------------------------------
# dispatch_alerts
# -----------------------------------------------------------------------------


def test_dispatch_alerts_dry_run_prints_and_never_calls_channels(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("alerts._CHANNEL_SENDERS", {"discord": lambda m, c: called.append(m)})
    monkeypatch.setenv("SCREENER_ALERTS_DRY_RUN", "1")

    alerts = [Alert("p", "A", 0.9, "r", "reason")]
    config = {"channels": [{"type": "discord", "enabled": True}], "max_alerts_per_run": 10}
    dispatch_alerts(alerts, config, "link")

    assert called == []
    assert "[DRY RUN]" in capsys.readouterr().out


def test_dispatch_alerts_quiet_mode_suppresses_delivery(monkeypatch):
    called = []
    monkeypatch.setattr("alerts._CHANNEL_SENDERS", {"discord": lambda m, c: called.append(m)})
    monkeypatch.setenv("SCREENER_ALERTS_DRY_RUN", "0")

    alerts = [Alert("p", "A", 0.9, "r", "reason")]
    config = {"quiet": True, "channels": [{"type": "discord", "enabled": True}], "max_alerts_per_run": 10}
    dispatch_alerts(alerts, config, "link")

    assert called == []


def test_dispatch_alerts_sends_to_every_enabled_channel(monkeypatch):
    sent_to = []
    monkeypatch.setattr(
        "alerts._CHANNEL_SENDERS",
        {
            "discord": lambda m, c: sent_to.append("discord"),
            "telegram": lambda m, c: sent_to.append("telegram"),
        },
    )
    monkeypatch.setenv("SCREENER_ALERTS_DRY_RUN", "0")

    alerts = [Alert("p", "A", 0.9, "r", "reason")]
    config = {
        "channels": [
            {"type": "discord", "enabled": True},
            {"type": "telegram", "enabled": True},
            {"type": "email", "enabled": False},
        ],
        "max_alerts_per_run": 10,
    }
    dispatch_alerts(alerts, config, "link")

    assert set(sent_to) == {"discord", "telegram"}


def test_dispatch_alerts_no_alerts_is_a_no_op(monkeypatch):
    called = []
    monkeypatch.setattr("alerts._CHANNEL_SENDERS", {"discord": lambda m, c: called.append(m)})
    dispatch_alerts([], {"channels": [{"type": "discord", "enabled": True}]}, "link")
    assert called == []


def test_dispatch_alerts_channel_exception_does_not_propagate(monkeypatch):
    def boom(message, config):
        raise RuntimeError("webhook rejected")

    monkeypatch.setattr("alerts._CHANNEL_SENDERS", {"discord": boom})
    monkeypatch.setenv("SCREENER_ALERTS_DRY_RUN", "0")

    alerts = [Alert("p", "A", 0.9, "r", "reason")]
    config = {"channels": [{"type": "discord", "enabled": True}]}
    dispatch_alerts(alerts, config, "link")  # must not raise


def test_dispatch_alerts_unknown_channel_type_is_skipped_gracefully(monkeypatch):
    monkeypatch.setenv("SCREENER_ALERTS_DRY_RUN", "0")
    alerts = [Alert("p", "A", 0.9, "r", "reason")]
    config = {"channels": [{"type": "carrier_pigeon", "enabled": True}]}
    dispatch_alerts(alerts, config, "link")  # must not raise


def test_dispatch_alerts_does_not_log_sent_when_channel_actually_skipped(monkeypatch, caplog):
    """Regression test: a sender returning False (e.g. a missing secret)
    must not be logged as a successful send."""
    monkeypatch.setattr("alerts._CHANNEL_SENDERS", {"discord": lambda m, c: False})
    monkeypatch.setenv("SCREENER_ALERTS_DRY_RUN", "0")

    alerts = [Alert("p", "A", 0.9, "r", "reason")]
    config = {"channels": [{"type": "discord", "enabled": True}]}

    with caplog.at_level("INFO"):
        dispatch_alerts(alerts, config, "link")

    assert "Sent alert digest via discord" not in caplog.text


# -----------------------------------------------------------------------------
# Channel senders — secrets and payloads, no real network
# -----------------------------------------------------------------------------


def test_send_discord_skips_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    calls = []
    monkeypatch.setattr("alerts.urllib.request.urlopen", lambda *a, **k: calls.append(1))
    send_discord("hello", {"webhook_url_env": "DISCORD_WEBHOOK_URL"})
    assert calls == []


def test_send_discord_posts_json_payload_to_the_webhook_url(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook/123")
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = request.data
        captured["headers"] = request.headers
        return _FakeResponse()

    monkeypatch.setattr("alerts.urllib.request.urlopen", fake_urlopen)

    send_discord("hello world", {"webhook_url_env": "DISCORD_WEBHOOK_URL"})

    assert captured["url"] == "https://discord.example/webhook/123"
    assert b"hello world" in captured["body"]
    assert captured["headers"]["Content-type"] == "application/json"


def test_send_telegram_skips_when_secrets_missing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    calls = []
    monkeypatch.setattr("alerts.urllib.request.urlopen", lambda *a, **k: calls.append(1))
    send_telegram("hello", {"bot_token_env": "TELEGRAM_BOT_TOKEN", "chat_id_env": "TELEGRAM_CHAT_ID"})
    assert calls == []


def test_send_telegram_posts_to_the_bot_api(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = request.data
        return _FakeResponse()

    monkeypatch.setattr("alerts.urllib.request.urlopen", fake_urlopen)

    send_telegram("hi", {"bot_token_env": "TELEGRAM_BOT_TOKEN", "chat_id_env": "TELEGRAM_CHAT_ID"})

    assert captured["url"] == "https://api.telegram.org/botabc123/sendMessage"
    assert b'"chat_id": "999"' in captured["body"]


def test_send_email_skips_when_credentials_missing(monkeypatch):
    monkeypatch.delenv("SMTP_USERNAME", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    calls = []
    monkeypatch.setattr("alerts.smtplib.SMTP", lambda *a, **k: calls.append(1))
    send_email("hello", {"username_env": "SMTP_USERNAME", "password_env": "SMTP_PASSWORD"})
    assert calls == []


def test_send_email_logs_in_and_sends_via_smtp(monkeypatch):
    monkeypatch.setenv("SMTP_USERNAME", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")

    calls = {"login": None, "sent": False, "starttls": False}

    class _FakeSMTP:
        def __init__(self, host, port, timeout=None):
            calls["host"] = host
            calls["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            calls["starttls"] = True

        def login(self, username, password):
            calls["login"] = (username, password)

        def send_message(self, message):
            calls["sent"] = True
            calls["subject"] = message["Subject"]

    monkeypatch.setattr("alerts.smtplib.SMTP", _FakeSMTP)

    send_email(
        "hello",
        {
            "username_env": "SMTP_USERNAME",
            "password_env": "SMTP_PASSWORD",
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "from_address": "bot@example.com",
            "to_addresses": ["you@example.com"],
        },
    )

    assert calls["starttls"] is True
    assert calls["login"] == ("user@example.com", "secret")
    assert calls["sent"] is True
    assert calls["subject"] == "Screener Alerts"
