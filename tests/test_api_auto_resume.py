from __future__ import annotations

import api_app
from photo_to_gcode.machine_control import BridgeResponse, BridgeSettings, BridgeStatusSnapshot, GcodeStreamResult, GrblStatus


def _job(job_id: str, total: int) -> api_app.DrawJob:
    return api_app.DrawJob(
        id=job_id,
        status="running",
        message="Drawing started.",
        total_commands=total,
        completed_commands=0,
        sent_commands=0,
        progress=0.0,
        estimated_seconds=total * 0.25,
        remaining_seconds=total * 0.25,
        seconds_per_command=0.25,
        started_at=0.0,
        updated_at=0.0,
    )


def test_auto_resume_rewinds_and_continues_same_job(monkeypatch):
    job_id = "auto-resume-test"
    commands = [
        "G21",
        "G90",
        "G1 Z20.000 F7200.0",
        "G0 X0.000 Y0.000",
        "G1 Z28.000 F7200.0",
        "G1 X1.000 Y0.000 F1200",
        "G1 X2.000 Y0.000",
        "G1 Z20.000 F7200.0",
    ]
    streamed_texts: list[str] = []
    uncertain_replay_flags: list[bool] = []
    api_app._jobs[job_id] = _job(job_id, len(commands))
    api_app._job_cancel_flags.pop(job_id, None)

    machine = api_app.MachineSettingsPayload()
    machine.auto_resume_enabled = True
    machine.auto_resume_rewind_commands = 1
    machine.auto_resume_max_attempts = 3
    machine.auto_resume_retry_delay_seconds = 0.0

    def fake_stream(_settings: BridgeSettings, gcode_text: str, **kwargs):
        streamed_texts.append(gcode_text)
        uncertain_replay_flags.append(bool(kwargs.get("allow_uncertain_batch_replay", True)))
        if len(streamed_texts) == 1:
            return GcodeStreamResult(
                ok=False,
                total_commands=len(commands),
                completed_commands=6,
                sent_commands=6,
                message="Read timed out while waiting for bridge status.",
                failed_command="G1 X2.000 Y0.000",
                failed_command_index=6,
                active_base_url="http://esp32",
            )
        completed_commands = len(api_app.prepare_gcode_for_streaming(gcode_text))
        return GcodeStreamResult(
            ok=True,
            total_commands=completed_commands,
            completed_commands=completed_commands,
            sent_commands=completed_commands,
            message="Drawing stream completed successfully.",
            active_base_url="http://esp32",
        )

    monkeypatch.setattr(api_app, "stream_gcode_to_bridge", fake_stream)
    monkeypatch.setattr(api_app, "_store_resume_payload", lambda **_kwargs: "hash")
    monkeypatch.setattr(
        api_app,
        "_bridge_recovery_handshake",
        lambda **kwargs: (
            kwargs["bridge_settings"],
            api_app.BridgeHealthReport(
                ok=True,
                score=96,
                label="healthy",
                message="Bridge health is good.",
                latency_ms=25.0,
                active_base_url=kwargs["bridge_settings"].normalized_base_url,
            ),
        ),
    )
    monkeypatch.setattr(api_app.time, "sleep", lambda _seconds: None)

    result = api_app._stream_gcode_with_auto_resume(
        job_id=job_id,
        bridge_settings=BridgeSettings(base_url="http://esp32"),
        command_context=commands,
        machine=machine,
        progress_callback=lambda *_args: None,
    )

    assert result.ok
    assert result.completed_commands == len(commands)
    assert uncertain_replay_flags == [False, False]
    assert streamed_texts[1] == (
        "G90\n"
        "G1 Z20.000 F7200.0\n"
        "G0 X0.000 Y0.000\n"
        "G1 Z28.000 F7200.0\n"
        "G1 X1.000 Y0.000 F1200\n"
        "G1 X2.000 Y0.000\n"
        "G1 Z20.000 F7200.0\n"
    )
    assert api_app._jobs[job_id].auto_resume_attempts == 1
    assert api_app._jobs[job_id].connection_loss_count == 1
    assert api_app._jobs[job_id].connection_recovery_count == 1

    api_app._jobs.pop(job_id, None)


def test_bridge_recovery_handshake_restarts_bridge_after_degraded_health(monkeypatch):
    job_id = "handshake-restart-test"
    api_app._jobs[job_id] = _job(job_id, 10)
    health_reports = [
        api_app.BridgeHealthReport(
            ok=False,
            score=35,
            label="slow",
            message="Bridge is slow.",
            latency_ms=2500.0,
            active_base_url="http://esp32",
        ),
        api_app.BridgeHealthReport(
            ok=True,
            score=100,
            label="healthy",
            message="Bridge recovered.",
            latency_ms=35.0,
            active_base_url="http://esp32",
        ),
    ]
    restart_calls: list[str] = []

    def fake_measure(settings: BridgeSettings, _machine: api_app.MachineSettingsPayload, **_kwargs):
        return settings, health_reports.pop(0)

    def fake_restart(settings: BridgeSettings):
        restart_calls.append(settings.normalized_base_url)
        return True, "restart accepted"

    monkeypatch.setattr(api_app, "_measure_bridge_health", fake_measure)
    monkeypatch.setattr(api_app, "_request_bridge_restart", fake_restart)
    monkeypatch.setattr(api_app, "_sleep_with_cancel", lambda *_args, **_kwargs: None)

    try:
        settings, report = api_app._bridge_recovery_handshake(
            job_id=job_id,
            bridge_settings=BridgeSettings(base_url="http://esp32"),
            machine=api_app.MachineSettingsPayload(bridgeUrl="http://esp32"),
            attempt_index=1,
            max_attempts=3,
        )

        assert settings.normalized_base_url == "http://esp32"
        assert report.ok
        assert report.label == "healthy"
        assert restart_calls == ["http://esp32"]
        assert api_app._jobs[job_id].bridge_reboot_count == 1
        assert api_app._jobs[job_id].connection_quality_score == 100
    finally:
        api_app._jobs.pop(job_id, None)


def test_home_request_during_active_draw_stops_job_without_homing(monkeypatch):
    job_id = "home-during-draw-test"
    cancel_flag = api_app.threading.Event()
    api_app._jobs[job_id] = _job(job_id, 10)
    api_app._job_cancel_flags[job_id] = cancel_flag
    realtime_commands: list[str] = []
    grbl_commands: list[str] = []

    def fake_realtime(_settings: BridgeSettings, command: str, **_kwargs):
        realtime_commands.append(command)
        return BridgeResponse(ok=True, response_text="hold")

    def fake_grbl(_settings: BridgeSettings, command: str):
        grbl_commands.append(command)
        return BridgeSettings(base_url="http://esp32"), BridgeResponse(ok=True, response_text="ok")

    monkeypatch.setattr(api_app, "send_bridge_realtime_command", fake_realtime)
    monkeypatch.setattr(api_app, "_send_grbl_command_with_bridge_fallback", fake_grbl)

    try:
        response = api_app.machine_action(
            api_app.MachineActionPayload(
                action="home_all",
                machine=api_app.MachineSettingsPayload(bridgeUrl="http://esp32"),
            )
        )

        assert response["ok"] is True
        assert response["completed"] is False
        assert response["stoppedJobs"] == 1
        assert realtime_commands == ["!"]
        assert grbl_commands == []
        assert cancel_flag.is_set()
        assert api_app._jobs[job_id].status == "canceled"
    finally:
        api_app._jobs.pop(job_id, None)
        api_app._job_cancel_flags.pop(job_id, None)


def test_home_request_when_controller_busy_does_not_send_homing(monkeypatch):
    grbl_commands: list[str] = []

    def fake_status(settings: BridgeSettings, **_kwargs):
        return (
            settings,
            BridgeResponse(ok=True, response_text="ok"),
            BridgeStatusSnapshot(
                raw_payload="",
                grbl_status=GrblStatus(raw="<Hold:0|MPos:0.000,0.000,0.000>", state="Hold:0"),
            ),
        )

    def fake_grbl(_settings: BridgeSettings, command: str):
        grbl_commands.append(command)
        return BridgeSettings(base_url="http://esp32"), BridgeResponse(ok=True, response_text="ok")

    monkeypatch.setattr(api_app, "_fetch_status_with_bridge_fallback", fake_status)
    monkeypatch.setattr(api_app, "_send_grbl_command_with_bridge_fallback", fake_grbl)

    response = api_app.machine_action(
        api_app.MachineActionPayload(
            action="home_all",
            machine=api_app.MachineSettingsPayload(bridgeUrl="http://esp32"),
        )
    )

    assert response["ok"] is True
    assert response["completed"] is False
    assert response["state"] == "Hold"
    assert "was not sent" in str(response["message"])
    assert grbl_commands == []
