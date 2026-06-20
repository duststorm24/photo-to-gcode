from __future__ import annotations

import pytest

from photo_to_gcode import machine_control as mc


def _ok_response(text: str = "ok") -> mc.BridgeResponse:
    return mc.BridgeResponse(ok=True, response_text=text, status_code=200)


def _transient_response(message: str = "Read timed out") -> mc.BridgeResponse:
    return mc.BridgeResponse(ok=False, response_text="", error=message)


def _snapshot(
    entries: list[tuple[int, str]] | None = None,
    *,
    state: str = "Idle",
) -> mc.BridgeStatusSnapshot:
    recent_entries = [] if entries is None else entries
    return mc.BridgeStatusSnapshot(
        raw_payload="",
        recent_log=[line for _, line in recent_entries],
        recent_log_entries=recent_entries,
        grbl_status=mc.GrblStatus(raw=f"<{state}|MPos:0.000,0.000,0.000>", state=state),
    )


def test_stream_replays_overlap_after_transient_status_disconnect(monkeypatch):
    sent_commands: list[str] = []
    progress_messages: list[str] = []
    status_disconnect_seen = False

    monkeypatch.setattr(mc.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(mc, "send_grbl_command", lambda _settings, command, **_kwargs: sent_commands.append(command) or _ok_response())

    def fake_status(settings: mc.BridgeSettings, **_kwargs):
        nonlocal status_disconnect_seen
        if not sent_commands:
            return settings, _ok_response(), _snapshot()

        if len(sent_commands) == 1 and not status_disconnect_seen:
            status_disconnect_seen = True
            return settings, _transient_response(), None

        if len(sent_commands) == 1:
            return settings, _ok_response(), _snapshot()

        ack_count = min(len(sent_commands) - 1, 3)
        entries = [(sequence, "ok") for sequence in range(1, ack_count + 1)]
        return settings, _ok_response(), _snapshot(entries)

    monkeypatch.setattr(mc, "_fetch_bridge_status_with_recovery", fake_status)

    result = mc.stream_gcode_to_bridge(
        mc.BridgeSettings(base_url="http://esp32", timeout_seconds=0.01),
        "G1 X1\nG1 X2\nG1 X3\n",
        batch_line_limit=3,
        batch_timeout_seconds=0.1,
        ready_timeout_seconds=0.1,
        max_in_flight_commands=1,
        inter_command_delay_seconds=0.0,
        bridge_recovery_timeout_seconds=0.1,
        poll_interval_seconds=0.001,
        progress_callback=lambda _completed, _total, message: progress_messages.append(message),
    )

    assert result.ok
    assert result.completed_commands == 3
    assert sent_commands == ["G1 X1", "G1 X1", "G1 X2", "G1 X3"]
    assert any("replaying" in message.lower() for message in progress_messages)


def test_stream_retries_low_risk_line_after_transient_command_disconnect(monkeypatch):
    sent_commands: list[str] = []
    accepted_count = 0

    monkeypatch.setattr(mc.time, "sleep", lambda _seconds: None)

    def fake_send(_settings: mc.BridgeSettings, command: str, **_kwargs):
        nonlocal accepted_count
        sent_commands.append(command)
        if len(sent_commands) == 1:
            return _transient_response()
        accepted_count += 1
        return _ok_response()

    def fake_status(settings: mc.BridgeSettings, **_kwargs):
        entries = [(sequence, "ok") for sequence in range(1, accepted_count + 1)]
        return settings, _ok_response(), _snapshot(entries)

    monkeypatch.setattr(mc, "send_grbl_command", fake_send)
    monkeypatch.setattr(mc, "_fetch_bridge_status_with_recovery", fake_status)

    result = mc.stream_gcode_to_bridge(
        mc.BridgeSettings(base_url="http://esp32", timeout_seconds=0.01),
        "G1 X1\nG1 X2\n",
        batch_line_limit=2,
        batch_timeout_seconds=0.1,
        ready_timeout_seconds=0.1,
        max_in_flight_commands=1,
        inter_command_delay_seconds=0.0,
        bridge_recovery_timeout_seconds=0.1,
        poll_interval_seconds=0.001,
    )

    assert result.ok
    assert result.completed_commands == 2
    assert sent_commands == ["G1 X1", "G1 X1", "G1 X2"]


def test_stream_stops_instead_of_crawling_through_repeated_command_timeouts(monkeypatch):
    sent_commands: list[str] = []
    accepted_count = 0

    monkeypatch.setattr(mc.time, "sleep", lambda _seconds: None)

    def fake_send(_settings: mc.BridgeSettings, command: str, **_kwargs):
        nonlocal accepted_count
        sent_commands.append(command)
        if len(sent_commands) % 2 == 1:
            return _transient_response("Read timed out")
        accepted_count += 1
        return _ok_response()

    def fake_status(settings: mc.BridgeSettings, **_kwargs):
        entries = [(sequence, "ok") for sequence in range(1, accepted_count + 1)]
        return settings, _ok_response(), _snapshot(entries)

    monkeypatch.setattr(mc, "send_grbl_command", fake_send)
    monkeypatch.setattr(mc, "_fetch_bridge_status_with_recovery", fake_status)

    result = mc.stream_gcode_to_bridge(
        mc.BridgeSettings(base_url="http://esp32", timeout_seconds=10.0),
        "G1 X1\nG1 X2\nG1 X3\n",
        batch_line_limit=3,
        batch_timeout_seconds=0.1,
        ready_timeout_seconds=0.1,
        max_in_flight_commands=1,
        inter_command_delay_seconds=0.0,
        bridge_recovery_timeout_seconds=0.1,
        poll_interval_seconds=0.001,
    )

    assert not result.ok
    assert "timeout-paced slow mode" in result.message
    assert result.failed_command == "G1 X2"
    assert sent_commands == ["G1 X1", "G1 X1", "G1 X2", "G1 X2"]


def test_stream_stops_when_grbl_reset_marker_appears(monkeypatch):
    sent_commands: list[str] = []

    monkeypatch.setattr(mc.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(mc, "send_grbl_command", lambda _settings, command, **_kwargs: sent_commands.append(command) or _ok_response())

    def fake_status(settings: mc.BridgeSettings, **_kwargs):
        if not sent_commands:
            return settings, _ok_response(), _snapshot()
        return settings, _ok_response(), _snapshot([(1, "Grbl 1.1h ['$' for help]")])

    monkeypatch.setattr(mc, "_fetch_bridge_status_with_recovery", fake_status)

    result = mc.stream_gcode_to_bridge(
        mc.BridgeSettings(base_url="http://esp32", timeout_seconds=0.01),
        "G1 X1\nG1 X2\n",
        batch_line_limit=2,
        batch_timeout_seconds=0.1,
        ready_timeout_seconds=0.1,
        max_in_flight_commands=1,
        inter_command_delay_seconds=0.0,
        bridge_recovery_timeout_seconds=0.1,
        poll_interval_seconds=0.001,
    )

    assert not result.ok
    assert "reset" in result.message.lower()
    assert result.failed_command == "G1 X1"


def test_stream_cancel_check_stops_before_next_command(monkeypatch):
    sent_commands: list[str] = []
    canceled = False

    monkeypatch.setattr(mc.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(mc, "send_grbl_command", lambda _settings, command, **_kwargs: sent_commands.append(command) or _ok_response())
    monkeypatch.setattr(mc, "_fetch_bridge_status_with_recovery", lambda settings, **_kwargs: (settings, _ok_response(), _snapshot()))

    def cancel_check() -> None:
        if canceled:
            raise RuntimeError("Drawing canceled by user.")

    def progress_callback(_completed: int, _total: int, message: str) -> None:
        nonlocal canceled
        if message.startswith("Sent 1/"):
            canceled = True

    with pytest.raises(RuntimeError, match="Drawing canceled"):
        mc.stream_gcode_to_bridge(
            mc.BridgeSettings(base_url="http://esp32", timeout_seconds=0.01),
            "G1 X1\nG1 X2\n",
            batch_line_limit=2,
            batch_timeout_seconds=0.1,
            ready_timeout_seconds=0.1,
            max_in_flight_commands=1,
            inter_command_delay_seconds=0.0,
            bridge_recovery_timeout_seconds=0.1,
            poll_interval_seconds=0.001,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )

    assert sent_commands == ["G1 X1"]


def test_bridge_status_recovery_switches_to_candidate_base_url(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(mc.time, "sleep", lambda _seconds: None)

    def fake_fetch(settings: mc.BridgeSettings, **_kwargs):
        calls.append(settings.normalized_base_url)
        if settings.normalized_base_url == "http://esp32-backup":
            return _ok_response(), _snapshot()
        return _transient_response(), None

    monkeypatch.setattr(mc, "_fetch_bridge_status_snapshot_with_retries", fake_fetch)

    recovered_settings, response, snapshot = mc._fetch_bridge_status_with_recovery(
        mc.BridgeSettings(base_url="http://esp32-primary", timeout_seconds=0.01),
        request_fresh_status=False,
        bridge_base_url_candidates=["http://esp32-backup"],
        recovery_timeout_seconds=0.1,
        retry_sleep_seconds=0.0,
    )

    assert response.ok
    assert snapshot is not None
    assert recovered_settings.normalized_base_url == "http://esp32-backup"
    assert calls[:3] == ["http://esp32-primary", "http://esp32-primary", "http://esp32-backup"]
