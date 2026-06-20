from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import requests
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from photo_to_gcode.cura_backend import CuraSettings, build_cura_page_mask, build_cura_page_tone, slice_image_with_cura
from photo_to_gcode.gcode import generate_gcode
from photo_to_gcode.machine_control import (
    BridgeSettings,
    GcodeStreamResult,
    PenMotionSettings,
    _is_transient_bridge_error,
    build_jog_command,
    clear_bridge_log,
    fetch_bridge_status_snapshot,
    prepare_gcode_for_streaming,
    replace_pen_control_commands_with_axis_moves,
    send_bridge_realtime_command,
    send_grbl_command,
    stream_gcode_to_bridge,
)
from photo_to_gcode.models import ProcessingSettings
from photo_to_gcode.openai_images import DEFAULT_PLOTTER_AI_PROMPT, convert_image_to_plotter_friendly_ai
from photo_to_gcode.planner import plan_page_mask
from photo_to_gcode.preview import render_toolpath_preview
from photo_to_gcode.toolpaths import calculate_path_metrics
from photo_to_gcode.triangle_mesh import TriangleMeshSettings, plan_triangle_mesh_from_tone_map

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
DRAW_RESUME_STATE_DIR = PROJECT_ROOT / ".draw_resume_state"
APP_VERSION = "V3.0"
DEFAULT_PAGE_WIDTH_MM = 215.9
DEFAULT_PAGE_HEIGHT_MM = 279.4
DEFAULT_GRBL_HOMING_PULLOFF_MM = 3.0
DEFAULT_PEN_UP_GAP_MM = 8.0
DEFAULT_PEN_AXIS = "Z"
DEFAULT_PEN_FEED_RATE_MM_MIN = 7200.0
DEFAULT_PROCESSING_RESOLUTION_PX_MM = 18.0
DEFAULT_MIN_INFILL_SPACING_MM = 0.40
BRIDGE_DISCOVERY_CANDIDATES = (
    "http://10.0.0.90",
    "http://10.0.0.89",
    "http://esp32-grbl-bridge.local",
)
BRIDGE_HEALTH_PROBE_COUNT = 3
BRIDGE_HEALTH_POLL_SECONDS = 1.0
BRIDGE_RESTART_PATH = "/restart"


class PlacementPayload(BaseModel):
    x_mm: float = Field(default=28.0, alias="xMm")
    y_mm: float = Field(default=38.0, alias="yMm")
    width_mm: float = Field(default=160.0, alias="widthMm")
    height_mm: float = Field(default=120.0, alias="heightMm")
    rotation_degrees: float = Field(default=0.0, alias="rotationDeg")


class WorkflowSettingsPayload(BaseModel):
    mode: Literal["vector_trace", "cura_slice", "triangle_mesh"] = "vector_trace"
    page_width_mm: float = Field(default=DEFAULT_PAGE_WIDTH_MM, alias="pageWidthMm")
    page_height_mm: float = Field(default=DEFAULT_PAGE_HEIGHT_MM, alias="pageHeightMm")
    threshold: int = 165
    invert_input: bool = Field(default=False, alias="invertInput")
    margin_mm: float = Field(default=0.0, alias="marginMm")
    mask_resolution_px_mm: float = Field(default=DEFAULT_PROCESSING_RESOLUTION_PX_MM, alias="maskResolutionPxMm")
    line_width_mm: float = Field(default=0.050, alias="lineWidthMm")
    wall_lines: int = Field(default=1, alias="wallLines")
    infill_density_percent: int = Field(default=100, alias="infillDensityPercent")
    draw_speed_mm_sec: float = Field(default=120.0, alias="drawSpeedMmSec")
    travel_speed_mm_sec: float = Field(default=120.0, alias="travelSpeedMmSec")
    fill_strategy: Literal["continuous_zigzag", "separate_paths"] = Field(
        default="continuous_zigzag",
        alias="fillStrategy",
    )
    fill_turn_split_angle_deg: float = Field(default=20.0, alias="fillTurnSplitAngleDeg")
    continuous_fill_chunk_segments: int = Field(default=0, alias="continuousFillChunkSegments")
    path_simplify_tolerance_mm: float = Field(default=0.08, alias="pathSimplifyToleranceMm")
    min_segment_length_mm: float = Field(default=0.10, alias="minSegmentLengthMm")
    min_toolpath_length_mm: float = Field(default=0.10, alias="minToolpathLengthMm")
    coordinate_decimals: int = Field(default=3, alias="coordinateDecimals")
    placement: PlacementPayload = Field(default_factory=PlacementPayload)


class MachineSettingsPayload(BaseModel):
    bridge_url: str = Field(default="http://10.0.0.90", alias="bridgeUrl")
    timeout_seconds: float = Field(default=8.0, alias="timeoutSeconds")
    pen_up_gap_mm: float = Field(default=DEFAULT_PEN_UP_GAP_MM, alias="penUpGapMm")
    queue_window_size: int = Field(default=24, alias="queueWindowSize")
    batch_ack_timeout_seconds: float = Field(default=90.0, alias="batchAckTimeoutSeconds")
    max_in_flight: int = Field(default=1, alias="maxInFlight")
    send_spacing_ms: int = Field(default=6, alias="sendSpacingMs")
    recovery_timeout_seconds: float = Field(default=180.0, alias="recoveryTimeoutSeconds")
    bridge_recovery_cooldown_seconds: float = Field(default=5.0, alias="bridgeRecoveryCooldownSeconds")
    bridge_health_max_latency_ms: int = Field(default=1200, alias="bridgeHealthMaxLatencyMs")
    bridge_restart_enabled: bool = Field(default=True, alias="bridgeRestartEnabled")
    bridge_restart_wait_seconds: float = Field(default=12.0, alias="bridgeRestartWaitSeconds")
    auto_resume_enabled: bool = Field(default=True, alias="autoResumeEnabled")
    auto_resume_rewind_commands: int = Field(default=6, alias="autoResumeRewindCommands")
    auto_resume_max_attempts: int = Field(default=20, alias="autoResumeMaxAttempts")
    auto_resume_retry_delay_seconds: float = Field(default=5.0, alias="autoResumeRetryDelaySeconds")
    use_pen_axis: bool = Field(default=True, alias="usePenAxis")
    pen_axis: str = Field(default=DEFAULT_PEN_AXIS, alias="penAxis")
    pen_up_position_mm: float = Field(default=20.0, alias="penUpPositionMm")
    pen_down_position_mm: float = Field(default=28.0, alias="penDownPositionMm")
    pen_feed_rate_mm_min: float = Field(default=DEFAULT_PEN_FEED_RATE_MM_MIN, alias="penFeedRateMmMin")
    pen_up_dwell_seconds: float = Field(default=0.0, alias="penUpDwellSeconds")
    pen_down_dwell_seconds: float = Field(default=0.0, alias="penDownDwellSeconds")


class DrawJobPayload(BaseModel):
    gcode: str
    machine: MachineSettingsPayload = Field(default_factory=MachineSettingsPayload)


class ResumeJobPayload(BaseModel):
    machine: MachineSettingsPayload = Field(default_factory=MachineSettingsPayload)
    rewind_commands: int = Field(default=6, alias="rewindCommands")


class MachineActionPayload(BaseModel):
    machine: MachineSettingsPayload = Field(default_factory=MachineSettingsPayload)
    action: Literal["home_all", "home_pen", "pen_up", "pen_down", "jog_pen", "status", "clear_log"]
    distance_mm: float = Field(default=0.0, alias="distanceMm")
    page_width_mm: float = Field(default=DEFAULT_PAGE_WIDTH_MM, alias="pageWidthMm")
    page_height_mm: float = Field(default=DEFAULT_PAGE_HEIGHT_MM, alias="pageHeightMm")


class SaveAiImagePayload(BaseModel):
    image_data: str = Field(alias="imageData")
    image_name: str = Field(default="plotter_ai.png", alias="imageName")


@dataclass
class DrawJob:
    id: str
    status: str
    message: str
    total_commands: int
    completed_commands: int
    sent_commands: int
    progress: float
    estimated_seconds: float
    remaining_seconds: float
    seconds_per_command: float
    started_at: float
    updated_at: float
    finished_at: float | None = None
    error: str | None = None
    failed_command: str | None = None
    failed_command_index: int | None = None
    command_context: list[str] | None = None
    recent_log: list[str] | None = None
    resume_available: bool = False
    resume_index: int | None = None
    resume_command_hash: str = ""
    active_bridge_url: str = ""
    parent_job_id: str | None = None
    auto_resume_attempts: int = 0
    connection_loss_count: int = 0
    connection_recovery_count: int = 0
    connection_quality_score: int = 0
    connection_quality_label: str = "unknown"
    connection_latency_ms: float | None = None
    connection_health_message: str = "Bridge health has not been checked yet."
    bridge_reboot_count: int = 0


@dataclass
class BridgeHealthReport:
    ok: bool
    score: int
    label: str
    message: str
    latency_ms: float | None = None
    active_base_url: str = ""


app = FastAPI(title="Photo to G-code API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=("http://localhost:5173", "http://127.0.0.1:5173"),
    allow_credentials=True,
    allow_methods=("*",),
    allow_headers=("*",),
)

_jobs: dict[str, DrawJob] = {}
_jobs_lock = threading.Lock()
_job_cancel_flags: dict[str, threading.Event] = {}
_ACTIVE_DRAW_STATUSES = {"queued", "running"}


def _active_draw_jobs_and_flags() -> tuple[list[str], list[threading.Event]]:
    with _jobs_lock:
        active_job_ids = [job_id for job_id, job in _jobs.items() if job.status in _ACTIVE_DRAW_STATUSES]
        active_cancel_flags = [flag for job_id, flag in _job_cancel_flags.items() if job_id in active_job_ids]
    return active_job_ids, active_cancel_flags


def _request_active_draw_stop(bridge_settings: BridgeSettings, *, message: str):
    active_job_ids, active_cancel_flags = _active_draw_jobs_and_flags()
    if not active_job_ids:
        return [], None

    for cancel_flag in active_cancel_flags:
        cancel_flag.set()
    response = send_bridge_realtime_command(bridge_settings, "!")
    for job_id in active_job_ids:
        _update_job(
            job_id,
            status="canceled",
            message=message,
            error=None if response.ok else response.error,
            finished_at=time.time(),
        )
    return active_job_ids, response


def _snapshot_grbl_state(snapshot) -> str | None:
    state = None if snapshot is None or snapshot.grbl_status is None else snapshot.grbl_status.state
    if not state:
        return None
    return state.split(":", 1)[0]


def _blocked_home_response(
    *,
    label: str,
    state: str,
    bridge_settings: BridgeSettings,
) -> dict[str, object]:
    return {
        "ok": True,
        "completed": False,
        "label": label,
        "message": (
            f"{label} was not sent because GRBL is {state}. "
            "Reset or power-cycle the controller to clear any queued motion, then home again."
        ),
        "state": state,
        "syncedZMm": None,
        "activeBridgeUrl": bridge_settings.normalized_base_url,
        "stoppedJobs": 0,
    }


def _block_home_if_controller_busy(label: str, bridge_settings: BridgeSettings) -> tuple[BridgeSettings, dict[str, object] | None]:
    bridge_settings, status_response, snapshot = _fetch_status_with_bridge_fallback(
        bridge_settings,
        request_fresh_status=True,
    )
    if not status_response.ok:
        return bridge_settings, None
    state = _snapshot_grbl_state(snapshot)
    if state is not None and state not in {"Idle", "Alarm"}:
        return bridge_settings, _blocked_home_response(label=label, state=state, bridge_settings=bridge_settings)
    return bridge_settings, None


def _command_context_hash(commands: list[str]) -> str:
    digest = hashlib.sha256()
    for command in commands:
        digest.update(command.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resume_payload_path(job_id: str) -> Path:
    safe_job_id = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id)
    return DRAW_RESUME_STATE_DIR / f"api_{safe_job_id}.json"


def _latest_resume_payload_path() -> Path:
    return DRAW_RESUME_STATE_DIR / "api_latest.json"


def _store_resume_payload(
    *,
    job_id: str,
    commands: list[str],
    resume_index: int,
    failed_command: str | None,
    failed_command_index: int | None,
) -> str:
    DRAW_RESUME_STATE_DIR.mkdir(parents=True, exist_ok=True)
    command_hash = _command_context_hash(commands)
    payload = {
        "version": 1,
        "job_id": job_id,
        "commands": commands,
        "resume_index": min(max(resume_index, 0), len(commands)),
        "failed_command": failed_command,
        "failed_command_index": failed_command_index,
        "command_hash": command_hash,
        "created_at": time.time(),
    }
    serialized = json.dumps(payload)
    _resume_payload_path(job_id).write_text(serialized, encoding="utf-8")
    _latest_resume_payload_path().write_text(serialized, encoding="utf-8")
    return command_hash


def _load_resume_payload(job_id: str) -> dict[str, object] | None:
    payload_path = _resume_payload_path(job_id)
    if not payload_path.exists():
        return None
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    commands = payload.get("commands")
    if not isinstance(commands, list) or not all(isinstance(command, str) for command in commands):
        return None
    return payload


def _clear_resume_payload(job_id: str) -> None:
    for payload_path in (_resume_payload_path(job_id), _latest_resume_payload_path()):
        try:
            if payload_path.exists():
                payload_path.unlink()
        except OSError:
            pass


def _resume_payload_to_job(payload: dict[str, object]) -> DrawJob:
    commands = [str(command) for command in payload.get("commands", [])]
    try:
        resume_index = int(payload.get("resume_index", 0))
    except (TypeError, ValueError):
        resume_index = 0
    resume_index = min(max(resume_index, 0), len(commands))
    try:
        now = float(payload.get("created_at", time.time()))
    except (TypeError, ValueError):
        now = time.time()
    failed_command_index = None
    try:
        if payload.get("failed_command_index") is not None:
            failed_command_index = int(payload["failed_command_index"])
    except (TypeError, ValueError):
        failed_command_index = None
    return DrawJob(
        id=str(payload.get("job_id", "")),
        status="error",
        message="Stored resume point from a failed draw.",
        total_commands=len(commands),
        completed_commands=resume_index,
        sent_commands=resume_index,
        progress=0.0 if not commands else min(max(resume_index / len(commands), 0.0), 1.0),
        estimated_seconds=len(commands) * 0.25,
        remaining_seconds=max(0, len(commands) - resume_index) * 0.25,
        seconds_per_command=0.25,
        started_at=now,
        updated_at=now,
        finished_at=now,
        error="Stored resume point from a failed draw.",
        failed_command=payload.get("failed_command") if isinstance(payload.get("failed_command"), str) else None,
        failed_command_index=failed_command_index,
        command_context=commands,
        resume_available=True,
        resume_index=resume_index,
        resume_command_hash=str(payload.get("command_hash", "")),
    )


def _build_resume_stream_text(
    commands: list[str],
    resume_index: int,
    rewind_commands: int,
    machine: MachineSettingsPayload,
) -> tuple[str, int]:
    if not commands:
        raise HTTPException(status_code=400, detail="No stored commands are available to resume.")
    bounded_resume_index = min(max(int(resume_index), 0), len(commands))
    bounded_rewind = min(max(int(rewind_commands), 0), 250)
    start_index = max(0, bounded_resume_index - bounded_rewind)
    stream_text, _prefix_command_count = _safe_resume_stream_text(commands, start_index, machine)
    return stream_text, start_index


def _gcode_word(command: str, word: str) -> float | None:
    match = re.search(rf"(^|\s){re.escape(word.upper())}([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?=\s|$)", command.upper())
    if match is None:
        return None
    return float(match.group(2))


def _command_has_motion_code(command: str, motion_codes: tuple[str, ...]) -> bool:
    normalized = command.upper()
    for motion_code in motion_codes:
        if re.search(rf"(^|\s){re.escape(motion_code)}(?=\s|$)", normalized):
            return True
    return False


def _command_contains_xy_motion(command: str) -> bool:
    normalized = command.upper()
    if not _command_has_motion_code(normalized, ("G0", "G00", "G1", "G01")):
        return False
    return _gcode_word(normalized, "X") is not None or _gcode_word(normalized, "Y") is not None


def _command_is_draw_motion(command: str) -> bool:
    normalized = command.upper()
    return _command_has_motion_code(normalized, ("G1", "G01")) and _command_contains_xy_motion(normalized)


def _command_sets_pen_axis(command: str, pen_settings: PenMotionSettings) -> Literal["up", "down"] | None:
    axis_value = _gcode_word(command, pen_settings.axis.upper())
    if axis_value is None:
        return None
    if abs(axis_value - pen_settings.pen_up_position_mm) <= 0.001:
        return "up"
    if abs(axis_value - pen_settings.pen_down_position_mm) <= 0.001:
        return "down"
    return None


def _command_sets_servo_pen(command: str) -> Literal["up", "down"] | None:
    normalized = command.strip().upper()
    if normalized == "M5" or normalized.startswith("M5 "):
        return "up"
    if normalized == "M3" or normalized.startswith("M3 "):
        return "down"
    return None


def _resume_state_before_command(
    commands: list[str],
    start_index: int,
    machine: MachineSettingsPayload,
) -> tuple[float | None, float | None, Literal["up", "down"]]:
    pen_settings = _pen_motion_settings(machine)
    x_pos: float | None = None
    y_pos: float | None = None
    pen_state: Literal["up", "down"] = "up"

    for command in commands[:max(0, start_index)]:
        if machine.use_pen_axis:
            next_pen_state = _command_sets_pen_axis(command, pen_settings)
        else:
            next_pen_state = _command_sets_servo_pen(command)
        if next_pen_state is not None:
            pen_state = next_pen_state

        next_x = _gcode_word(command, "X")
        next_y = _gcode_word(command, "Y")
        if next_x is not None:
            x_pos = next_x
        if next_y is not None:
            y_pos = next_y

    return x_pos, y_pos, pen_state


def _pen_resume_command(
    machine: MachineSettingsPayload,
    position: Literal["up", "down"],
) -> str:
    if machine.use_pen_axis:
        return _pen_move_command(_pen_motion_settings(machine), position)
    return "M5" if position == "up" else "M3 S30"


def _safe_resume_stream_text(
    commands: list[str],
    start_index: int,
    machine: MachineSettingsPayload,
) -> tuple[str, int]:
    resume_commands = commands[start_index:]
    if not resume_commands:
        raise HTTPException(status_code=400, detail="The stored resume point is already at the end of the draw.")

    x_pos, y_pos, pen_state = _resume_state_before_command(commands, start_index, machine)
    first_command = resume_commands[0] if resume_commands else ""
    first_command_changes_pen = (
        _command_sets_pen_axis(first_command, _pen_motion_settings(machine))
        if machine.use_pen_axis
        else _command_sets_servo_pen(first_command)
    ) is not None

    prelude = ["G90", _pen_resume_command(machine, "up")]
    if x_pos is not None or y_pos is not None:
        xy_words = []
        if x_pos is not None:
            xy_words.append(f"X{x_pos:.3f}")
        if y_pos is not None:
            xy_words.append(f"Y{y_pos:.3f}")
        prelude.append("G0 " + " ".join(xy_words))
    if pen_state == "down" and _command_is_draw_motion(first_command) and not first_command_changes_pen:
        prelude.append(_pen_resume_command(machine, "down"))

    return "\n".join([*prelude, *resume_commands]) + "\n", len(prelude)


def _is_auto_resumable_stream_result(result: GcodeStreamResult) -> bool:
    message = result.message.lower()
    non_resumable_hints = (
        "drawing canceled",
        "grbl appears to have reset",
        "grbl entered alarm",
        "grbl reported an error",
        "no grbl commands",
    )
    if any(hint in message for hint in non_resumable_hints):
        return False
    if _is_transient_bridge_error(result.message):
        return True
    return any(
        hint in message
        for hint in (
            "acknowledge",
            "bridge",
            "connection",
            "host",
            "timed out",
            "timeout",
        )
    )


def _connection_event_flags(message: str) -> tuple[bool, bool]:
    normalized = message.lower()
    loss_hints = (
        "briefly disconnected",
        "bridge lost",
        "bridge status timed out",
        "connection",
        "connect timeout",
        "lost track",
        "timed out",
        "timeout",
    )
    recovery_hints = (
        "bridge recovered",
        "continuing",
        "recovered bridge contact",
        "replaying from the last confirmed",
        "replaying the last unconfirmed",
    )
    return (
        any(hint in normalized for hint in loss_hints),
        any(hint in normalized for hint in recovery_hints),
    )


def _increment_connection_counters(
    job_id: str,
    *,
    loss_delta: int = 0,
    recovery_delta: int = 0,
) -> None:
    if loss_delta <= 0 and recovery_delta <= 0:
        return
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.connection_loss_count += max(0, int(loss_delta))
        job.connection_recovery_count += max(0, int(recovery_delta))
        job.updated_at = time.time()


def _connection_counters(job_id: str) -> tuple[int, int]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return 0, 0
        return job.connection_loss_count, job.connection_recovery_count


def _update_job_bridge_health(
    job_id: str,
    report: BridgeHealthReport,
    *,
    message: str | None = None,
) -> None:
    updates = {
        "connection_quality_score": report.score,
        "connection_quality_label": report.label,
        "connection_latency_ms": report.latency_ms,
        "connection_health_message": report.message,
    }
    if report.active_base_url:
        updates["active_bridge_url"] = report.active_base_url
    if message is not None:
        updates["message"] = message
    _update_job(job_id, **updates)


def _bridge_health_report_from_samples(
    *,
    samples: list[float],
    failures: int,
    max_latency_ms: float,
    active_base_url: str,
    last_error: str,
) -> BridgeHealthReport:
    if not samples:
        return BridgeHealthReport(
            ok=False,
            score=0,
            label="offline",
            message=last_error or "Bridge did not answer health probes.",
            active_base_url=active_base_url,
        )

    average_latency = sum(samples) / len(samples)
    worst_latency = max(samples)
    success_ratio = len(samples) / max(len(samples) + failures, 1)
    latency_ratio = min(max(average_latency / max(max_latency_ms, 1.0), 0.0), 3.0)
    score = int(max(0.0, min(100.0, (success_ratio * 100.0) - max(0.0, latency_ratio - 1.0) * 35.0)))
    healthy = failures == 0 and worst_latency <= max_latency_ms
    if healthy:
        label = "healthy"
        message = f"Bridge answered {len(samples)} health probes quickly; average {average_latency:.0f} ms."
    elif failures == 0:
        label = "slow"
        message = f"Bridge answered, but average response was {average_latency:.0f} ms."
    else:
        label = "degraded"
        message = f"Bridge health is inconsistent: {len(samples)} ok, {failures} failed."
    return BridgeHealthReport(
        ok=healthy,
        score=score,
        label=label,
        message=message,
        latency_ms=average_latency,
        active_base_url=active_base_url,
    )


def _measure_bridge_health(
    bridge_settings: BridgeSettings,
    machine: MachineSettingsPayload,
    *,
    probe_count: int = BRIDGE_HEALTH_PROBE_COUNT,
) -> tuple[BridgeSettings, BridgeHealthReport]:
    active_settings = bridge_settings
    samples: list[float] = []
    failures = 0
    last_error = ""
    max_latency_ms = max(100.0, float(machine.bridge_health_max_latency_ms))

    for _probe_index in range(max(1, probe_count)):
        started_at = time.monotonic()
        active_settings, response, _snapshot = _fetch_status_with_bridge_fallback(
            active_settings,
            request_fresh_status=True,
        )
        elapsed_ms = (time.monotonic() - started_at) * 1000.0
        if response.ok:
            samples.append(elapsed_ms)
        else:
            failures += 1
            last_error = response.error or response.response_text or "Bridge health probe failed."

    report = _bridge_health_report_from_samples(
        samples=samples,
        failures=failures,
        max_latency_ms=max_latency_ms,
        active_base_url=active_settings.normalized_base_url,
        last_error=last_error,
    )
    return active_settings, report


def _bridge_restart_url(settings: BridgeSettings) -> str:
    return f"{settings.normalized_base_url}{BRIDGE_RESTART_PATH}"


def _request_bridge_restart(settings: BridgeSettings) -> tuple[bool, str]:
    if not settings.normalized_base_url:
        return False, "Bridge URL is empty."
    try:
        response = requests.post(
            _bridge_restart_url(settings),
            headers={"Connection": "close"},
            timeout=max(0.5, min(settings.timeout_seconds, 2.0)),
        )
    except requests.RequestException as exc:
        if _is_transient_bridge_error(str(exc)):
            return True, "ESP32 restart request was sent; the bridge closed the connection while rebooting."
        return False, str(exc)
    if response.ok:
        return True, response.text or "ESP32 restart request accepted."
    return False, response.text or f"ESP32 restart endpoint returned HTTP {response.status_code}."


def _sleep_with_cancel(seconds: float, cancel_check=None) -> None:
    deadline = time.time() + max(0.0, seconds)
    while time.time() < deadline:
        if cancel_check is not None:
            cancel_check()
        time.sleep(min(0.25, max(deadline - time.time(), 0.0)))


def _bridge_recovery_handshake(
    *,
    job_id: str,
    bridge_settings: BridgeSettings,
    machine: MachineSettingsPayload,
    attempt_index: int,
    max_attempts: int,
    cancel_check=None,
) -> tuple[BridgeSettings, BridgeHealthReport]:
    cooldown_seconds = max(0.0, float(machine.bridge_recovery_cooldown_seconds))
    if cooldown_seconds > 0:
        cooldown_report = BridgeHealthReport(
            ok=False,
            score=20,
            label="cooldown",
            message=f"Cooling down the bridge for {cooldown_seconds:.1f} seconds before health checks.",
            active_base_url=bridge_settings.normalized_base_url,
        )
        _update_job_bridge_health(
            job_id,
            cooldown_report,
            message=f"Recovery cooldown before attempt {attempt_index}/{max_attempts}.",
        )
        _sleep_with_cancel(cooldown_seconds, cancel_check=cancel_check)

    _update_job_bridge_health(
        job_id,
        BridgeHealthReport(
            ok=False,
            score=35,
            label="checking",
            message="Checking bridge response quality before resuming.",
            active_base_url=bridge_settings.normalized_base_url,
        ),
        message=f"Checking bridge health before recovery attempt {attempt_index}/{max_attempts}.",
    )
    active_settings, report = _measure_bridge_health(bridge_settings, machine)
    _update_job_bridge_health(job_id, report)
    if report.ok:
        return active_settings, report

    if not machine.bridge_restart_enabled:
        return active_settings, report

    reboot_report = BridgeHealthReport(
        ok=False,
        score=10,
        label="rebooting",
        message="Bridge health is degraded; requesting ESP32 bridge restart.",
        active_base_url=active_settings.normalized_base_url,
    )
    _update_job_bridge_health(
        job_id,
        reboot_report,
        message="Bridge health is degraded; restarting ESP32 bridge before resuming.",
    )
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.bridge_reboot_count += 1
            job.updated_at = time.time()

    restart_ok, restart_message = _request_bridge_restart(active_settings)
    if not restart_ok:
        return active_settings, BridgeHealthReport(
            ok=False,
            score=report.score,
            label="degraded",
            message=f"{report.message} ESP32 restart was not available: {restart_message}",
            latency_ms=report.latency_ms,
            active_base_url=active_settings.normalized_base_url,
        )

    deadline = time.time() + max(1.0, float(machine.bridge_restart_wait_seconds))
    latest_report = BridgeHealthReport(
        ok=False,
        score=10,
        label="rebooting",
        message=restart_message,
        active_base_url=active_settings.normalized_base_url,
    )
    while time.time() < deadline:
        _sleep_with_cancel(BRIDGE_HEALTH_POLL_SECONDS, cancel_check=cancel_check)
        active_settings, latest_report = _measure_bridge_health(active_settings, machine)
        _update_job_bridge_health(job_id, latest_report)
        if latest_report.ok:
            return active_settings, latest_report

    return active_settings, latest_report


def _absolute_stream_result(
    result: GcodeStreamResult,
    *,
    command_context: list[str],
    start_index: int,
    prefix_command_count: int = 0,
) -> GcodeStreamResult:
    total_commands = len(command_context)
    relative_completed = max(result.completed_commands - max(prefix_command_count, 0), 0)
    relative_sent = max(result.sent_commands - max(prefix_command_count, 0), 0)
    completed_commands = min(max(start_index + relative_completed, 0), total_commands)
    sent_commands = min(max(start_index + relative_sent, 0), total_commands)
    failed_command_index = None
    failed_command = result.failed_command
    if result.failed_command_index is not None:
        relative_failed_index = max(result.failed_command_index - max(prefix_command_count, 0), 0)
        failed_command_index = min(max(start_index + relative_failed_index, 0), total_commands)
        if failed_command_index < total_commands:
            failed_command = command_context[failed_command_index]
    return GcodeStreamResult(
        ok=result.ok,
        total_commands=total_commands,
        completed_commands=completed_commands,
        sent_commands=sent_commands,
        message=result.message,
        failed_command=failed_command,
        failed_command_index=failed_command_index,
        last_snapshot=result.last_snapshot,
        active_base_url=result.active_base_url,
    )


def _stream_gcode_with_auto_resume(
    *,
    job_id: str,
    bridge_settings: BridgeSettings,
    command_context: list[str],
    machine: MachineSettingsPayload,
    cancel_check=None,
    progress_callback=None,
) -> GcodeStreamResult:
    current_start_index = 0
    current_stream_text = "\n".join(command_context) + "\n"
    current_prefix_command_count = 0
    auto_attempts = 0
    max_auto_attempts = max(0, int(machine.auto_resume_max_attempts)) if machine.auto_resume_enabled else 0
    rewind_commands = min(max(int(machine.auto_resume_rewind_commands), 0), 250)
    active_bridge_settings = bridge_settings

    while True:
        segment_start_index = current_start_index
        loss_count_before_segment, recovery_count_before_segment = _connection_counters(job_id)

        def segment_progress(completed: int, _total: int, message: str) -> None:
            relative_completed = max(completed - current_prefix_command_count, 0)
            remapped_message = message
            sent_match = re.search(r"Sent\s+(\d+)/\d+", message)
            if sent_match is not None:
                relative_sent = max(int(sent_match.group(1)) - current_prefix_command_count, 0)
                absolute_sent = min(segment_start_index + relative_sent, len(command_context))
                remapped_message = re.sub(
                    r"Sent\s+\d+/\d+",
                    f"Sent {absolute_sent}/{len(command_context)}",
                    message,
                    count=1,
                )
            progress_callback(
                min(segment_start_index + relative_completed, len(command_context)),
                len(command_context),
                remapped_message,
            )

        result = stream_gcode_to_bridge(
            active_bridge_settings,
            current_stream_text,
            batch_line_limit=max(1, int(machine.queue_window_size)),
            batch_timeout_seconds=max(2.0, float(machine.batch_ack_timeout_seconds)),
            max_in_flight_commands=max(1, int(machine.max_in_flight)),
            inter_command_delay_seconds=max(0.0, float(machine.send_spacing_ms) / 1000.0),
            bridge_base_url_candidates=_bridge_discovery_candidates(active_bridge_settings.normalized_base_url),
            bridge_recovery_timeout_seconds=max(0.0, float(machine.recovery_timeout_seconds)),
            allow_uncertain_batch_replay=False,
            cancel_check=cancel_check,
            progress_callback=segment_progress,
        )
        absolute_result = _absolute_stream_result(
            result,
            command_context=command_context,
            start_index=segment_start_index,
            prefix_command_count=current_prefix_command_count,
        )
        if result.active_base_url:
            active_bridge_settings = replace(active_bridge_settings, base_url=result.active_base_url)

        if absolute_result.ok:
            _loss_count_after_segment, recovery_count_after_segment = _connection_counters(job_id)
            if auto_attempts > 0 and recovery_count_after_segment == recovery_count_before_segment:
                _increment_connection_counters(job_id, recovery_delta=1)
            return absolute_result

        failure_anchor = min(
            max(
                absolute_result.failed_command_index
                if absolute_result.failed_command_index is not None
                else absolute_result.completed_commands,
                0,
            ),
            len(command_context),
        )
        command_hash = _store_resume_payload(
            job_id=job_id,
            commands=command_context,
            resume_index=failure_anchor,
            failed_command=absolute_result.failed_command,
            failed_command_index=absolute_result.failed_command_index,
        )
        _update_job(
            job_id,
            command_context=command_context,
            resume_available=True,
            resume_index=failure_anchor,
            resume_command_hash=command_hash,
            active_bridge_url=active_bridge_settings.normalized_base_url,
        )

        if auto_attempts >= max_auto_attempts or not _is_auto_resumable_stream_result(absolute_result):
            return absolute_result

        loss_count_after_segment, _recovery_count_after_segment = _connection_counters(job_id)
        if loss_count_after_segment == loss_count_before_segment:
            _increment_connection_counters(job_id, loss_delta=1)
        auto_attempts += 1
        active_bridge_settings, health_report = _bridge_recovery_handshake(
            job_id=job_id,
            bridge_settings=active_bridge_settings,
            machine=machine,
            attempt_index=auto_attempts,
            max_attempts=max_auto_attempts,
            cancel_check=cancel_check,
        )
        if not health_report.ok:
            return GcodeStreamResult(
                ok=False,
                total_commands=len(command_context),
                completed_commands=failure_anchor,
                sent_commands=absolute_result.sent_commands,
                message=f"Bridge recovery handshake failed: {health_report.message}",
                failed_command=absolute_result.failed_command,
                failed_command_index=absolute_result.failed_command_index,
                last_snapshot=absolute_result.last_snapshot,
                active_base_url=active_bridge_settings.normalized_base_url,
            )
        current_start_index = max(0, failure_anchor - rewind_commands)
        current_stream_text, current_prefix_command_count = _safe_resume_stream_text(
            command_context,
            current_start_index,
            machine,
        )
        _update_job(
            job_id,
            status="running",
            message=(
                f"Connection recovery attempt {auto_attempts}/{max_auto_attempts}; bridge health {health_report.score}/100; "
                f"replaying from command {current_start_index + 1}."
            ),
            error=None,
            completed_commands=current_start_index,
            sent_commands=current_start_index,
            progress=current_start_index / max(len(command_context), 1),
            remaining_seconds=max(0, len(command_context) - current_start_index) * 0.25,
            auto_resume_attempts=auto_attempts,
            active_bridge_url=active_bridge_settings.normalized_base_url,
        )


def _queue_draw_job(
    *,
    gcode: str,
    machine: MachineSettingsPayload,
    message: str = "Draw queued.",
    parent_job_id: str | None = None,
) -> DrawJob:
    commands = prepare_gcode_for_streaming(gcode)
    if not commands:
        raise HTTPException(status_code=400, detail="No GRBL commands were found in the G-code.")

    job_id = uuid.uuid4().hex
    now = time.time()
    job = DrawJob(
        id=job_id,
        status="queued",
        message=message,
        total_commands=len(commands),
        completed_commands=0,
        sent_commands=0,
        progress=0.0,
        estimated_seconds=len(commands) * 0.25,
        remaining_seconds=len(commands) * 0.25,
        seconds_per_command=0.25,
        started_at=now,
        updated_at=now,
        command_context=commands,
        active_bridge_url=_bridge_settings(machine).normalized_base_url,
        parent_job_id=parent_job_id,
    )
    cancel_flag = threading.Event()
    with _jobs_lock:
        _jobs[job_id] = job
        _job_cancel_flags[job_id] = cancel_flag

    thread = threading.Thread(
        target=_run_draw_job,
        args=(job_id, gcode, machine),
        daemon=True,
    )
    thread.start()
    return job


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"ok": "true", "version": APP_VERSION}


@app.post("/api/generate")
async def generate_from_image(
    image: UploadFile = File(...),
    settings_json: str = Form(..., alias="settings"),
) -> dict[str, object]:
    payload = _parse_workflow_settings(settings_json)
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data was uploaded.")

    try:
        source_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}") from exc

    started_at = time.time()
    plan_result = _build_plan_and_gcode(source_image, payload)
    elapsed_seconds = time.time() - started_at
    total_commands = len(prepare_gcode_for_streaming(str(plan_result["gcode"])))
    seconds_per_command = 0.25

    return {
        **plan_result,
        "version": APP_VERSION,
        "sourceName": image.filename,
        "totalCommands": total_commands,
        "estimatedSeconds": total_commands * seconds_per_command,
        "secondsPerCommand": seconds_per_command,
        "commandsPerSecond": 1.0 / seconds_per_command,
        "planningSeconds": elapsed_seconds,
        "page": {
            "widthMm": payload.page_width_mm,
            "heightMm": payload.page_height_mm,
        },
    }


@app.post("/api/ai/convert")
async def convert_source_image_to_ai(
    image: UploadFile = File(...),
    additional_comments: str = Form("", alias="additionalComments"),
) -> dict[str, object]:
    api_key = _configured_openai_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data was uploaded.")

    prompt = DEFAULT_PLOTTER_AI_PROMPT
    cleaned_comments = " ".join(str(additional_comments).split())
    if cleaned_comments:
        prompt = f"{prompt}\n\nAdditional user comments: {cleaned_comments}"

    result = convert_image_to_plotter_friendly_ai(image_bytes, api_key=api_key, prompt=prompt)
    if not result.ok or result.image_bytes is None:
        raise HTTPException(status_code=400, detail=result.error or "AI conversion failed.")

    stem = Path(image.filename or "plotter_image").stem
    return {
        "imageData": bytes_to_data_url(result.image_bytes),
        "imageName": f"{stem}_ai.png",
        "revisedPrompt": result.revised_prompt,
    }


@app.post("/api/ai/save-to-desktop")
def save_ai_image_to_desktop(payload: SaveAiImagePayload) -> dict[str, object]:
    image_bytes = _decode_data_url(payload.image_data)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No AI image data was provided.")

    desktop_path = Path.home() / "Desktop"
    if not desktop_path.exists():
        raise HTTPException(status_code=400, detail=f"Desktop folder was not found at {desktop_path}.")

    filename = _safe_image_filename(payload.image_name)
    target_path = _unique_path(desktop_path / filename)
    try:
        target_path.write_bytes(image_bytes)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save AI image: {exc}") from exc

    return {"ok": True, "fileName": target_path.name, "path": str(target_path)}


@app.post("/api/machine/action")
def machine_action(payload: MachineActionPayload) -> dict[str, object]:
    machine = payload.machine
    bridge_settings = _bridge_settings(machine)
    pen_settings = _pen_motion_settings(machine)

    if payload.action == "status":
        bridge_settings, response, snapshot = _fetch_status_with_bridge_fallback(
            bridge_settings,
            request_fresh_status=True,
        )
        return _machine_response("Status", response, snapshot=snapshot, bridge_settings=bridge_settings)

    if payload.action == "clear_log":
        bridge_settings, response = _clear_bridge_log_with_fallback(bridge_settings)
        return {
            "ok": response.ok,
            "label": "Clear log",
            "message": response.error or response.response_text,
            "activeBridgeUrl": bridge_settings.normalized_base_url,
        }

    if payload.action == "home_all":
        active_job_ids, stop_response = _request_active_draw_stop(
            bridge_settings,
            message="Home All was requested while drawing. Feed-hold was sent and the active draw was canceled.",
        )
        if active_job_ids:
            return {
                "ok": stop_response.ok if stop_response is not None else False,
                "completed": False,
                "label": "Home All",
                "message": (
                    (None if stop_response is None else stop_response.error)
                    or "Drawing was stopped before homing. Press Home All again after the machine is idle."
                ),
                "state": "Hold",
                "syncedZMm": None,
                "activeBridgeUrl": bridge_settings.normalized_base_url,
                "stoppedJobs": len(active_job_ids),
            }
        bridge_settings, blocked_response = _block_home_if_controller_busy("Home All", bridge_settings)
        if blocked_response is not None:
            return blocked_response
        bridge_settings, response = _send_grbl_command_with_bridge_fallback(bridge_settings, "$H")
        if response.ok:
            _wait_for_idle(bridge_settings, timeout_seconds=45.0)
            _sync_page_coordinates(bridge_settings, payload.page_width_mm, payload.page_height_mm, pen_settings)
        return _machine_response(
            "Home All",
            response,
            synced_z_mm=DEFAULT_GRBL_HOMING_PULLOFF_MM,
            bridge_settings=bridge_settings,
        )

    if payload.action == "home_pen":
        active_job_ids, stop_response = _request_active_draw_stop(
            bridge_settings,
            message="Home Pen was requested while drawing. Feed-hold was sent and the active draw was canceled.",
        )
        if active_job_ids:
            return {
                "ok": stop_response.ok if stop_response is not None else False,
                "completed": False,
                "label": "Home Pen",
                "message": (
                    (None if stop_response is None else stop_response.error)
                    or "Drawing was stopped before homing. Press Home Pen again after the machine is idle."
                ),
                "state": "Hold",
                "syncedZMm": None,
                "activeBridgeUrl": bridge_settings.normalized_base_url,
                "stoppedJobs": len(active_job_ids),
            }
        bridge_settings, blocked_response = _block_home_if_controller_busy("Home Pen", bridge_settings)
        if blocked_response is not None:
            return blocked_response
        bridge_settings, response = _send_grbl_command_with_bridge_fallback(
            bridge_settings,
            f"$H{pen_settings.axis.upper()}",
        )
        if response.ok:
            _wait_for_idle(bridge_settings, timeout_seconds=30.0)
            _sync_page_coordinates(bridge_settings, payload.page_width_mm, payload.page_height_mm, pen_settings)
        return _machine_response(
            "Home Pen",
            response,
            synced_z_mm=DEFAULT_GRBL_HOMING_PULLOFF_MM,
            bridge_settings=bridge_settings,
        )

    if payload.action == "pen_up":
        command = _pen_move_command(pen_settings, "up")
        bridge_settings, response = _send_grbl_command_with_bridge_fallback(bridge_settings, command)
        return _machine_response(
            "Pen Up",
            response,
            synced_z_mm=pen_settings.pen_up_position_mm,
            bridge_settings=bridge_settings,
        )

    if payload.action == "pen_down":
        command = _pen_move_command(pen_settings, "down")
        bridge_settings, response = _send_grbl_command_with_bridge_fallback(bridge_settings, command)
        return _machine_response(
            "Pen Down",
            response,
            synced_z_mm=pen_settings.pen_down_position_mm,
            bridge_settings=bridge_settings,
        )

    if payload.action == "jog_pen":
        jog_command = build_jog_command(
            pen_settings.axis,
            max(-50.0, min(50.0, payload.distance_mm)),
            pen_settings.feed_rate_mm_min,
        )
        bridge_settings, response = _send_grbl_command_with_bridge_fallback(bridge_settings, jog_command)
        return _machine_response("Jog Pen", response, bridge_settings=bridge_settings)

    raise HTTPException(status_code=400, detail=f"Unsupported machine action: {payload.action}")


@app.post("/api/machine/log")
def machine_log(machine: MachineSettingsPayload = Body(default_factory=MachineSettingsPayload)) -> dict[str, object]:
    bridge_settings = _bridge_settings(machine)
    response, snapshot = fetch_bridge_status_snapshot(bridge_settings, request_fresh_status=True)
    grbl_status = None if snapshot is None else snapshot.grbl_status
    return {
        "ok": response.ok,
        "message": response.error or response.response_text,
        "state": None if grbl_status is None else grbl_status.state,
        "lastCommand": "" if snapshot is None else snapshot.last_command,
        "recentLog": [] if snapshot is None or snapshot.recent_log is None else snapshot.recent_log[-120:],
        "raw": None if snapshot is None else snapshot.raw_payload,
    }


@app.post("/api/machine/log/clear")
def machine_log_clear(machine: MachineSettingsPayload = Body(default_factory=MachineSettingsPayload)) -> dict[str, object]:
    response = clear_bridge_log(_bridge_settings(machine))
    return {"ok": response.ok, "message": response.error or response.response_text}


@app.post("/api/machine/draw")
def start_draw(payload: DrawJobPayload) -> dict[str, object]:
    job = _queue_draw_job(gcode=payload.gcode, machine=payload.machine)
    return {"jobId": job.id, "job": asdict(job)}


@app.get("/api/jobs/{job_id}")
def get_draw_job(job_id: str) -> dict[str, object]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is not None:
        return asdict(job)

    resume_payload = _load_resume_payload(job_id)
    if resume_payload is not None:
        return asdict(_resume_payload_to_job(resume_payload))

    raise HTTPException(status_code=404, detail="Draw job was not found.")


@app.post("/api/jobs/{job_id}/resume")
def resume_draw_job(job_id: str, payload: ResumeJobPayload) -> dict[str, object]:
    with _jobs_lock:
        source_job = _jobs.get(job_id)

    resume_payload = _load_resume_payload(job_id)
    if resume_payload is not None:
        commands = [str(command) for command in resume_payload.get("commands", [])]
        resume_index = int(resume_payload.get("resume_index", 0))
    elif source_job is not None and source_job.resume_available and source_job.command_context:
        commands = list(source_job.command_context)
        resume_index = source_job.resume_index if source_job.resume_index is not None else source_job.completed_commands
    else:
        raise HTTPException(status_code=400, detail="No resume point is available for this job.")

    resume_text, start_index = _build_resume_stream_text(
        commands,
        resume_index,
        payload.rewind_commands,
        payload.machine,
    )
    resumed_job = _queue_draw_job(
        gcode=resume_text,
        machine=payload.machine,
        message=f"Resume queued from command {start_index + 1}.",
        parent_job_id=job_id,
    )
    return {"jobId": resumed_job.id, "job": asdict(resumed_job), "startIndex": start_index}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_draw_job(
    job_id: str,
    machine: MachineSettingsPayload = Body(default_factory=MachineSettingsPayload),
) -> dict[str, object]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        cancel_flag = _job_cancel_flags.get(job_id)
    if job is None or cancel_flag is None:
        raise HTTPException(status_code=404, detail="Draw job was not found.")

    cancel_flag.set()
    response = send_bridge_realtime_command(_bridge_settings(machine), "!")
    _update_job(
        job_id,
        status="canceled",
        message="Feed-hold was sent. Drawing cancellation requested.",
        error=None if response.ok else response.error,
        finished_at=time.time(),
    )
    with _jobs_lock:
        updated_job = _jobs[job_id]
    return {"ok": response.ok, "message": response.error or response.response_text, "job": asdict(updated_job)}


@app.post("/api/machine/estop")
def emergency_stop(machine: MachineSettingsPayload = Body(default_factory=MachineSettingsPayload)) -> dict[str, object]:
    with _jobs_lock:
        active_job_ids = [job_id for job_id, job in _jobs.items() if job.status in {"queued", "running"}]
        active_cancel_flags = [flag for job_id, flag in _job_cancel_flags.items() if job_id in active_job_ids]
    for cancel_flag in active_cancel_flags:
        cancel_flag.set()
    response = send_bridge_realtime_command(_bridge_settings(machine), "!")
    for job_id in active_job_ids:
        _update_job(
            job_id,
            status="canceled",
            message="E-STOP feed-hold was sent. Active drawing jobs were canceled.",
            error=None if response.ok else response.error,
            finished_at=time.time(),
        )
    with _jobs_lock:
        stopped_jobs = [asdict(_jobs[job_id]) for job_id in active_job_ids if job_id in _jobs]
    return {
        "ok": response.ok,
        "message": response.error or "E-STOP feed-hold was sent. Active drawing jobs were canceled.",
        "stoppedJobs": len(active_job_ids),
        "jobs": stopped_jobs,
    }


def _parse_workflow_settings(settings_json: str) -> WorkflowSettingsPayload:
    try:
        raw_settings = json.loads(settings_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Settings JSON is invalid: {exc}") from exc
    return WorkflowSettingsPayload(**raw_settings)


def _build_plan_and_gcode(source_image: Image.Image, payload: WorkflowSettingsPayload) -> dict[str, object]:
    cura_settings = _build_cura_settings(source_image, payload)
    if payload.mode == "cura_slice":
        result = slice_image_with_cura(source_image, cura_settings)
        gcode = result.plotter_gcode
        toolpaths = result.toolpaths
        mask_image = Image.fromarray(result.threshold_mask_preview).convert("RGB")
        engine = result.engine_path
        triangle_count = result.triangle_count
    elif payload.mode == "triangle_mesh":
        tone_map = build_cura_page_tone(source_image, cura_settings)
        plan = plan_triangle_mesh_from_tone_map(
            tone_map,
            TriangleMeshSettings(
                page_width_mm=payload.page_width_mm,
                page_height_mm=payload.page_height_mm,
                processing_resolution_ppmm=payload.mask_resolution_px_mm,
                threshold=payload.threshold,
                invert_input=payload.invert_input,
            ),
        )
        processing_settings = _build_processing_settings(payload)
        gcode = generate_gcode(plan.toolpaths, processing_settings)
        toolpaths = plan.toolpaths
        mask_image = Image.fromarray(plan.threshold_mask).convert("RGB")
        engine = "triangle_mesh"
        triangle_count = None
    else:
        page_mask = build_cura_page_mask(source_image, cura_settings)
        processing_settings = _build_processing_settings(payload, page_mask=page_mask)
        plan = plan_page_mask(page_mask, processing_settings)
        gcode = generate_gcode(plan.toolpaths, processing_settings)
        toolpaths = plan.toolpaths
        mask_image = Image.fromarray(plan.threshold_mask).convert("RGB")
        engine = "vector_trace"
        triangle_count = None

    metrics = calculate_path_metrics(toolpaths)
    preview_image = render_toolpath_preview(
        toolpaths,
        page_width_mm=payload.page_width_mm,
        page_height_mm=payload.page_height_mm,
        line_width_mm=max(0.03, payload.line_width_mm),
        preview_width_px=980,
        preview_height_px=760,
    )
    return {
        "mode": payload.mode,
        "gcode": gcode,
        "previewImage": image_to_data_url(preview_image),
        "maskImage": image_to_data_url(mask_image),
        "metrics": metrics,
        "diagnostics": {},
        "engine": engine,
        "triangleCount": triangle_count,
    }


def _build_processing_settings(
    payload: WorkflowSettingsPayload,
    *,
    page_mask=None,
) -> ProcessingSettings:
    pixels_per_mm = payload.mask_resolution_px_mm
    if page_mask is not None:
        mask_height_px, mask_width_px = page_mask.shape
        pixels_per_mm = max(
            0.1,
            min(mask_width_px / payload.page_width_mm, mask_height_px / payload.page_height_mm),
        )
    infill_density = max(0, int(payload.infill_density_percent))
    fill_mode = "zigzag" if infill_density > 0 else "none"
    fill_spacing = max(
        payload.line_width_mm,
        payload.line_width_mm * (100.0 / max(float(infill_density), 1.0)),
        DEFAULT_MIN_INFILL_SPACING_MM,
    )
    return ProcessingSettings(
        page_width_mm=payload.page_width_mm,
        page_height_mm=payload.page_height_mm,
        margin_mm=0.0,
        threshold=payload.threshold,
        invert_input=payload.invert_input,
        pen_width_mm=max(0.03, payload.line_width_mm),
        min_feature_width_mm=0.05,
        min_region_area_mm2=0.05,
        wall_count=max(1, int(payload.wall_lines)),
        thin_feature_mode=True,
        thin_feature_max_width_mm=max(0.75, payload.line_width_mm * 1.5),
        centerline_min_length_mm=0.35,
        simplify_tolerance_px=0.50,
        curve_smoothing_passes=2,
        curve_sample_step_mm=0.25,
        fill_mode=fill_mode,
        fill_spacing_mm=fill_spacing,
        processing_resolution_ppmm=pixels_per_mm,
        potrace_turdsize=2,
        potrace_alphamax=1.0,
        potrace_opttolerance=0.2,
        feed_rate=max(60, int(round(payload.draw_speed_mm_sec * 60.0))),
        pen_up_command="M5",
        pen_down_command="M3 S30",
        pen_pause_seconds=0.0,
    )


def _build_cura_settings(source_image: Image.Image, payload: WorkflowSettingsPayload) -> CuraSettings:
    placement = payload.placement
    scale_multiplier = _placement_scale_multiplier(source_image, payload)
    center_x_mm = placement.x_mm + (placement.width_mm / 2.0)
    center_y_mm_from_top = placement.y_mm + (placement.height_mm / 2.0)
    offset_x_mm = center_x_mm - (payload.page_width_mm / 2.0)
    offset_y_mm = (payload.page_height_mm / 2.0) - center_y_mm_from_top
    return CuraSettings(
        page_width_mm=payload.page_width_mm,
        page_height_mm=payload.page_height_mm,
        margin_mm=payload.margin_mm,
        threshold=payload.threshold,
        invert_input=payload.invert_input,
        placement_scale=scale_multiplier,
        placement_rotation_degrees=placement.rotation_degrees,
        placement_offset_x_mm=offset_x_mm,
        placement_offset_y_mm=offset_y_mm,
        processing_resolution_ppmm=payload.mask_resolution_px_mm,
        line_width_mm=max(0.03, payload.line_width_mm),
        wall_line_count=max(0, int(payload.wall_lines)),
        infill_density_percent=max(0, min(100, int(payload.infill_density_percent))),
        draw_speed_mm_per_s=payload.draw_speed_mm_sec,
        travel_speed_mm_per_s=payload.travel_speed_mm_sec,
        plotter_fill_mode=payload.fill_strategy,
        fill_turn_split_angle_degrees=payload.fill_turn_split_angle_deg,
        continuous_fill_chunk_segments=payload.continuous_fill_chunk_segments,
        path_simplify_tolerance_mm=payload.path_simplify_tolerance_mm,
        min_segment_length_mm=payload.min_segment_length_mm,
        min_toolpath_length_mm=payload.min_toolpath_length_mm,
        coordinate_decimals=payload.coordinate_decimals,
    )


def _placement_scale_multiplier(source_image: Image.Image, payload: WorkflowSettingsPayload) -> float:
    source_width_px, source_height_px = _rotated_source_size(source_image, payload.placement.rotation_degrees)
    usable_width_mm = max(1.0, payload.page_width_mm - (payload.margin_mm * 2.0))
    usable_height_mm = max(1.0, payload.page_height_mm - (payload.margin_mm * 2.0))
    fit_scale_mm_per_px = min(usable_width_mm / source_width_px, usable_height_mm / source_height_px)
    fit_width_mm = source_width_px * fit_scale_mm_per_px
    fit_height_mm = source_height_px * fit_scale_mm_per_px
    width_multiplier = payload.placement.width_mm / max(fit_width_mm, 0.01)
    height_multiplier = payload.placement.height_mm / max(fit_height_mm, 0.01)
    return max(0.01, min(width_multiplier, height_multiplier))


def _rotated_source_size(source_image: Image.Image, rotation_degrees: float) -> tuple[int, int]:
    if abs(rotation_degrees) <= 1e-6:
        return source_image.size
    rotated = source_image.rotate(-rotation_degrees, expand=True)
    return rotated.size


def _run_draw_job(job_id: str, gcode: str, machine: MachineSettingsPayload) -> None:
    bridge_settings = _bridge_settings(machine)
    pen_settings = _pen_motion_settings(machine)
    if machine.use_pen_axis:
        gcode = replace_pen_control_commands_with_axis_moves(gcode, pen_settings)
    command_context = prepare_gcode_for_streaming(gcode)
    with _jobs_lock:
        cancel_flag = _job_cancel_flags.get(job_id)

    def raise_if_canceled() -> None:
        if cancel_flag is not None and cancel_flag.is_set():
            raise RuntimeError("Drawing canceled by user.")

    def progress_callback(completed: int, total: int, message: str) -> None:
        raise_if_canceled()
        loss_seen, recovery_seen = _connection_event_flags(message)
        _update_job_progress(
            job_id,
            completed,
            total,
            message,
            status="running",
            sent_commands=_sent_count_from_progress_message(message),
        )
        _increment_connection_counters(
            job_id,
            loss_delta=1 if loss_seen else 0,
            recovery_delta=1 if recovery_seen else 0,
        )

    started_at = time.time()
    _update_job(job_id, status="running", message="Drawing started.", started_at=started_at, updated_at=started_at)
    try:
        result = _stream_gcode_with_auto_resume(
            job_id=job_id,
            bridge_settings=bridge_settings,
            command_context=command_context,
            machine=machine,
            cancel_check=raise_if_canceled,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        if str(exc) == "Drawing canceled by user.":
            _update_job(job_id, status="canceled", message="Drawing canceled by user.", finished_at=time.time())
        else:
            _update_job(job_id, status="error", message=f"Drawing failed: {exc}", error=str(exc), finished_at=time.time())
        with _jobs_lock:
            _job_cancel_flags.pop(job_id, None)
        return

    recent_log = None if result.last_snapshot is None or result.last_snapshot.recent_log is None else result.last_snapshot.recent_log[-80:]
    if result.ok:
        _clear_resume_payload(job_id)
        _update_job(
            job_id,
            status="complete",
            message=result.message,
            completed_commands=result.completed_commands,
            sent_commands=result.sent_commands,
            progress=1.0,
            remaining_seconds=0.0,
            finished_at=time.time(),
            recent_log=recent_log,
            active_bridge_url=result.active_base_url,
            resume_available=False,
            resume_index=None,
        )
    else:
        failure_anchor = min(
            max(
                result.failed_command_index if result.failed_command_index is not None else result.completed_commands,
                0,
            ),
            len(command_context),
        )
        command_hash = _store_resume_payload(
            job_id=job_id,
            commands=command_context,
            resume_index=failure_anchor,
            failed_command=result.failed_command,
            failed_command_index=result.failed_command_index,
        )
        _update_job(
            job_id,
            status="error",
            message=result.message,
            error=result.message,
            completed_commands=result.completed_commands,
            sent_commands=result.sent_commands,
            progress=result.completed_commands / max(result.total_commands, 1),
            failed_command=result.failed_command,
            failed_command_index=result.failed_command_index,
            command_context=command_context,
            recent_log=recent_log,
            resume_available=True,
            resume_index=failure_anchor,
            resume_command_hash=command_hash,
            active_bridge_url=result.active_base_url,
            finished_at=time.time(),
        )
    with _jobs_lock:
        _job_cancel_flags.pop(job_id, None)


def _update_job_progress(
    job_id: str,
    completed: int,
    total: int,
    message: str,
    *,
    status: str,
    sent_commands: int | None = None,
) -> None:
    progress = 0.0 if total <= 0 else min(max(completed / total, 0.0), 1.0)
    with _jobs_lock:
        job = _jobs[job_id]
        seconds_per_command = job.seconds_per_command
        job.status = status
        job.message = message
        job.total_commands = total
        job.completed_commands = completed
        if sent_commands is not None:
            job.sent_commands = sent_commands
        job.progress = progress
        job.remaining_seconds = max(0.0, (total - completed) * seconds_per_command)
        job.updated_at = time.time()


def _update_job(job_id: str, **updates) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = updates.get("updated_at", time.time())


def _sent_count_from_progress_message(message: str) -> int | None:
    match = re.search(r"Sent\s+(\d+)/", message)
    if match:
        return int(match.group(1))
    return None


def _bridge_settings(machine: MachineSettingsPayload) -> BridgeSettings:
    return BridgeSettings(base_url=machine.bridge_url, timeout_seconds=max(0.5, machine.timeout_seconds))


def _candidate_bridge_settings(settings: BridgeSettings):
    seen = {settings.normalized_base_url}
    for candidate in _bridge_discovery_candidates(settings.normalized_base_url):
        candidate_settings = BridgeSettings(
            base_url=candidate,
            timeout_seconds=settings.timeout_seconds,
        )
        normalized = candidate_settings.normalized_base_url
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        yield candidate_settings


def _fetch_status_with_bridge_fallback(
    settings: BridgeSettings,
    *,
    request_fresh_status: bool,
) -> tuple[BridgeSettings, object, object | None]:
    response, snapshot = fetch_bridge_status_snapshot(settings, request_fresh_status=request_fresh_status)
    if response.ok:
        return settings, response, snapshot

    for candidate_settings in _candidate_bridge_settings(settings):
        candidate_response, candidate_snapshot = fetch_bridge_status_snapshot(
            candidate_settings,
            request_fresh_status=request_fresh_status,
        )
        if candidate_response.ok:
            return candidate_settings, candidate_response, candidate_snapshot

    return settings, response, snapshot


def _send_grbl_command_with_bridge_fallback(
    settings: BridgeSettings,
    command: str,
) -> tuple[BridgeSettings, object]:
    response = send_grbl_command(settings, command)
    if response.ok:
        return settings, response

    for candidate_settings in _candidate_bridge_settings(settings):
        probe_response, _ = fetch_bridge_status_snapshot(candidate_settings, request_fresh_status=False)
        if not probe_response.ok:
            continue
        retry_response = send_grbl_command(candidate_settings, command)
        if retry_response.ok:
            return candidate_settings, retry_response

    return settings, response


def _clear_bridge_log_with_fallback(settings: BridgeSettings) -> tuple[BridgeSettings, object]:
    response = clear_bridge_log(settings)
    if response.ok:
        return settings, response

    for candidate_settings in _candidate_bridge_settings(settings):
        retry_response = clear_bridge_log(candidate_settings)
        if retry_response.ok:
            return candidate_settings, retry_response

    return settings, response


def _pen_motion_settings(machine: MachineSettingsPayload) -> PenMotionSettings:
    return PenMotionSettings(
        axis=(machine.pen_axis or DEFAULT_PEN_AXIS).upper(),
        pen_up_position_mm=machine.pen_up_position_mm,
        pen_down_position_mm=machine.pen_down_position_mm,
        feed_rate_mm_min=max(1.0, machine.pen_feed_rate_mm_min),
        pen_up_dwell_seconds=max(0.0, machine.pen_up_dwell_seconds),
        pen_down_dwell_seconds=max(0.0, machine.pen_down_dwell_seconds),
    )


def _pen_move_command(pen_settings: PenMotionSettings, position: Literal["up", "down"]) -> str:
    target = pen_settings.pen_up_position_mm if position == "up" else pen_settings.pen_down_position_mm
    return f"G1 {pen_settings.axis.upper()}{target:.3f} F{pen_settings.feed_rate_mm_min:.1f}"


def _sync_page_coordinates(
    bridge_settings: BridgeSettings,
    page_width_mm: float,
    page_height_mm: float,
    pen_settings: PenMotionSettings,
) -> None:
    send_grbl_command(bridge_settings, "G90")
    send_grbl_command(
        bridge_settings,
        (
            f"G92 X{page_width_mm:.3f} "
            f"Y{page_height_mm:.3f} "
            f"{pen_settings.axis.upper()}{DEFAULT_GRBL_HOMING_PULLOFF_MM:.3f}"
        ),
    )


def _wait_for_idle(bridge_settings: BridgeSettings, *, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response, snapshot = fetch_bridge_status_snapshot(bridge_settings, request_fresh_status=True)
        state = None if snapshot is None or snapshot.grbl_status is None else snapshot.grbl_status.state
        if response.ok and state == "Idle":
            return
        time.sleep(0.2)


def _machine_response(
    label: str,
    response,
    *,
    snapshot=None,
    synced_z_mm: float | None = None,
    bridge_settings: BridgeSettings | None = None,
) -> dict[str, object]:
    if snapshot is None:
        _, snapshot = fetch_bridge_status_snapshot(
            BridgeSettings(base_url="", timeout_seconds=0.5),
            request_fresh_status=False,
        ) if False else (None, None)
    grbl_status = None if snapshot is None else snapshot.grbl_status
    return {
        "ok": response.ok,
        "label": label,
        "message": response.error or response.response_text or f"{label} sent.",
        "state": None if grbl_status is None else grbl_status.state,
        "syncedZMm": synced_z_mm,
        "activeBridgeUrl": None if bridge_settings is None else bridge_settings.normalized_base_url,
    }


def _bridge_discovery_candidates(current_base_url: str | None = None) -> list[str]:
    candidates: list[str] = []
    for candidate in (*BRIDGE_DISCOVERY_CANDIDATES, current_base_url):
        if candidate is None:
            continue
        normalized = BridgeSettings(base_url=str(candidate)).normalized_base_url
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def bytes_to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _decode_data_url(data_url: str) -> bytes:
    if not data_url:
        return b""
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"AI image data is not valid base64: {exc}") from exc


def _safe_image_filename(filename: str) -> str:
    stem = Path(filename or "plotter_ai.png").stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "plotter_ai"
    return f"{safe_stem}.png"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise HTTPException(status_code=500, detail=f"Could not find an available filename for {path.name}.")


def _configured_openai_api_key() -> str:
    import os

    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key:
        return env_key
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return ""
    try:
        import tomllib

        payload = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(payload.get("OPENAI_API_KEY", "")).strip()


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str):
    requested_path = FRONTEND_DIST / full_path
    if FRONTEND_DIST.exists() and requested_path.is_file():
        return FileResponse(requested_path)
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend has not been built. Run `npm run build` in frontend/.")
