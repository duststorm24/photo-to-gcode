from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from urllib.parse import urljoin

import requests


@dataclass(slots=True)
class BridgeSettings:
    base_url: str
    command_path: str = "/command"
    realtime_path: str = "/realtime"
    status_path: str = "/status"
    clear_log_path: str = "/clear-log"
    servo_move_path: str = "/servo/move"
    servo_status_path: str = "/servo/status"
    command_field: str = "cmd"
    realtime_field: str = "rt"
    timeout_seconds: float = 2.0

    @property
    def normalized_base_url(self) -> str:
        value = self.base_url.strip()
        if not value:
            return ""
        if not re.match(r"^https?://", value, flags=re.IGNORECASE):
            value = f"http://{value}"
        return value.rstrip("/")

    @property
    def command_url(self) -> str:
        return _resolve_url(self.normalized_base_url, self.command_path)

    @property
    def realtime_url(self) -> str:
        return _resolve_url(self.normalized_base_url, self.realtime_path)

    @property
    def status_url(self) -> str:
        return _resolve_url(self.normalized_base_url, self.status_path)

    @property
    def clear_log_url(self) -> str:
        return _resolve_url(self.normalized_base_url, self.clear_log_path)

    @property
    def servo_move_url(self) -> str:
        return _resolve_url(self.normalized_base_url, self.servo_move_path)

    @property
    def servo_status_url(self) -> str:
        return _resolve_url(self.normalized_base_url, self.servo_status_path)


@dataclass(slots=True)
class BridgeResponse:
    ok: bool
    response_text: str
    status_code: int | None = None
    error: str | None = None


@dataclass(slots=True)
class GrblStatus:
    raw: str
    state: str = "Unknown"
    machine_position: tuple[float, float, float] | None = None
    work_position: tuple[float, float, float] | None = None
    work_coordinate_offset: tuple[float, float, float] | None = None
    pins: str = ""
    x_limit_pressed: bool | None = None
    y_limit_pressed: bool | None = None
    z_limit_pressed: bool | None = None
    probe_input_active: bool | None = None


@dataclass(slots=True)
class PenMotionSettings:
    axis: str = "Z"
    pen_up_position_mm: float = 20.0
    pen_down_position_mm: float = 28.0
    feed_rate_mm_min: float = 900.0
    pen_up_dwell_seconds: float = 0.0
    pen_down_dwell_seconds: float = 0.0


@dataclass(slots=True)
class BridgeStatusSnapshot:
    raw_payload: str
    bridge_ok: bool | None = None
    bridge_ip: str = ""
    last_command: str = ""
    recent_log: list[str] | None = None
    recent_log_entries: list[tuple[int, str]] | None = None
    grbl_status: GrblStatus | None = None
    servo_status: "ServoStatus | None" = None


@dataclass(slots=True)
class ServoStatus:
    pin: int | None = None
    min_angle: int | None = None
    max_angle: int | None = None
    up_angle: int | None = None
    down_angle: int | None = None
    default_angle: int | None = None
    step_angle: int | None = None
    current_angle: int | None = None
    attached: bool | None = None
    last_action: str = ""


@dataclass(slots=True)
class GcodeStreamResult:
    ok: bool
    total_commands: int
    completed_commands: int
    sent_commands: int
    message: str
    failed_command: str | None = None
    failed_command_index: int | None = None
    last_snapshot: BridgeStatusSnapshot | None = None
    active_base_url: str = ""


@dataclass(slots=True)
class GrblLinkTestResult:
    ok: bool
    message: str
    observed_lines: list[str]
    last_snapshot: BridgeStatusSnapshot | None = None


@dataclass(slots=True)
class StreamItem:
    kind: str
    command: str
    settle_seconds: float = 0.0
    source_line_count: int = 1
    start_command_index: int = 0


@dataclass(slots=True)
class IdleWaitResult:
    ok: bool
    message: str
    last_snapshot: BridgeStatusSnapshot | None = None
    active_settings: BridgeSettings | None = None


REALTIME_COMMANDS = {"?", "!", "~"}
GRBL_RX_BUFFER_BYTES = 128
GRBL_SAFE_STREAM_BUFFER_BYTES = 96
UNCERTAIN_STREAM_REPLAY_LIMIT = 3
UNCERTAIN_BATCH_REPLAY_LIMIT = 4
UNCERTAIN_BATCH_REPLAY_OVERLAP_COMMANDS = 3
STREAM_COMMAND_TIMEOUT_SECONDS = 2.0
MAX_CONSECUTIVE_DEGRADED_COMMAND_RECOVERIES = 2
TRANSIENT_BRIDGE_ERROR_HINTS = (
    "remotedisconnected",
    "remote end closed connection without response",
    "connection aborted",
    "max retries exceeded",
    "connecttimeout",
    "connect timeout",
    "read timed out",
    "readtimeout",
    "connection reset",
    "connection refused",
    "temporarily unavailable",
    "timed out",
)
BRIDGE_HTTP_HEADERS = {"Connection": "close"}
GRBL_ERROR_DESCRIPTIONS = {
    1: "expected command letter",
    2: "bad number format",
    3: "invalid statement",
    4: "negative value",
    5: "setting disabled",
    8: "command requires the controller to be Idle",
    9: "G-code locked until homing or unlock",
    10: "soft limit triggered",
    11: "line overflow",
    14: "line length exceeded",
    15: "travel exceeded",
    16: "invalid jog command",
    20: "unsupported command",
    21: "modal group violation",
    22: "undefined feed rate",
    23: "command value must be an integer",
    24: "axis command conflict",
    25: "repeated word in one block",
    26: "no axis words found where motion expected",
    28: "missing value for a word",
    33: "invalid target",
    34: "arc radius error",
    36: "unused words on this line",
}


def _is_transient_bridge_error(error: str | None) -> bool:
    if not error:
        return False
    normalized = error.strip().lower()
    return any(hint in normalized for hint in TRANSIENT_BRIDGE_ERROR_HINTS)


def build_jog_command(
    axis: str,
    distance_mm: float,
    feed_rate_mm_min: float,
) -> str:
    normalized_axis = axis.upper()
    if normalized_axis not in {"X", "Y", "Z"}:
        raise ValueError("Jog axis must be X, Y, or Z.")
    return f"$J=G91 G21 {normalized_axis}{distance_mm:.3f} F{feed_rate_mm_min:.1f}"


def build_absolute_move_command(
    axis: str,
    position_mm: float,
    feed_rate_mm_min: float,
    *,
    motion_code: str = "G1",
) -> str:
    normalized_axis = axis.upper()
    if normalized_axis not in {"X", "Y", "Z"}:
        raise ValueError("Absolute move axis must be X, Y, or Z.")

    normalized_motion_code = motion_code.upper()
    if normalized_motion_code not in {"G0", "G1"}:
        raise ValueError("Motion code must be G0 or G1.")
    if feed_rate_mm_min <= 0:
        raise ValueError("Feed rate must be greater than zero.")

    return (
        f"G90 G21 {normalized_motion_code} "
        f"{normalized_axis}{position_mm:.3f} F{feed_rate_mm_min:.1f}"
    )


def build_step_jog_command(
    axis: str,
    step_count: int,
    steps_per_mm: float,
    feed_rate_mm_min: float,
) -> str:
    if steps_per_mm <= 0:
        raise ValueError("Steps per mm must be greater than zero.")
    distance_mm = float(step_count) / float(steps_per_mm)
    return build_jog_command(axis, distance_mm, feed_rate_mm_min)


def send_bridge_command(
    settings: BridgeSettings,
    command: str,
    *,
    http_session: requests.Session | None = None,
) -> BridgeResponse:
    if not settings.command_url:
        return BridgeResponse(ok=False, response_text="", error="Command URL is empty.")

    try:
        request_client = requests if http_session is None else http_session
        response = request_client.post(
            settings.command_url,
            json={settings.command_field: command},
            headers=BRIDGE_HTTP_HEADERS,
            timeout=settings.timeout_seconds,
        )
    except requests.RequestException as exc:
        return BridgeResponse(ok=False, response_text="", error=str(exc))

    return BridgeResponse(
        ok=response.ok,
        response_text=response.text,
        status_code=response.status_code,
        error=None if response.ok else response.text,
    )


def send_bridge_realtime_command(
    settings: BridgeSettings,
    realtime_command: str,
    *,
    http_session: requests.Session | None = None,
) -> BridgeResponse:
    normalized = realtime_command.strip()
    if normalized not in REALTIME_COMMANDS:
        return BridgeResponse(
            ok=False,
            response_text="",
            error=f"Unsupported realtime command: {realtime_command}",
        )
    if not settings.realtime_url:
        return BridgeResponse(ok=False, response_text="", error="Realtime URL is empty.")

    try:
        request_client = requests if http_session is None else http_session
        response = request_client.post(
            settings.realtime_url,
            json={settings.realtime_field: normalized},
            headers=BRIDGE_HTTP_HEADERS,
            timeout=settings.timeout_seconds,
        )
    except requests.RequestException as exc:
        return BridgeResponse(ok=False, response_text="", error=str(exc))

    return BridgeResponse(
        ok=response.ok,
        response_text=response.text,
        status_code=response.status_code,
        error=None if response.ok else response.text,
    )


def send_grbl_command(
    settings: BridgeSettings,
    command: str,
    *,
    http_session: requests.Session | None = None,
) -> BridgeResponse:
    normalized = _sanitize_grbl_command(command)
    if not normalized:
        return BridgeResponse(ok=False, response_text="", error="Command is empty.")
    if normalized in REALTIME_COMMANDS:
        return send_bridge_realtime_command(settings, normalized, http_session=http_session)
    return send_bridge_command(settings, normalized, http_session=http_session)


def fetch_bridge_status(settings: BridgeSettings) -> tuple[BridgeResponse, BridgeStatusSnapshot | None]:
    return fetch_bridge_status_snapshot(settings, request_fresh_status=False)


def fetch_bridge_status_snapshot(
    settings: BridgeSettings,
    *,
    request_fresh_status: bool = False,
    status_settle_seconds: float = 0.12,
    http_session: requests.Session | None = None,
) -> tuple[BridgeResponse, BridgeStatusSnapshot | None]:
    if not settings.status_url:
        return BridgeResponse(ok=False, response_text="", error="Status URL is empty."), None

    if request_fresh_status:
        realtime_response = send_bridge_realtime_command(settings, "?", http_session=http_session)
        if realtime_response.ok and status_settle_seconds > 0.0:
            time.sleep(status_settle_seconds)

    try:
        request_client = requests if http_session is None else http_session
        response = request_client.get(
            settings.status_url,
            headers=BRIDGE_HTTP_HEADERS,
            timeout=settings.timeout_seconds,
        )
    except requests.RequestException as exc:
        return BridgeResponse(ok=False, response_text="", error=str(exc)), None

    bridge_response = BridgeResponse(
        ok=response.ok,
        response_text=response.text,
        status_code=response.status_code,
        error=None if response.ok else response.text,
    )
    if not response.ok:
        return bridge_response, None

    snapshot = parse_bridge_status(response.text)
    return bridge_response, snapshot


def clear_bridge_log(
    settings: BridgeSettings,
    *,
    http_session: requests.Session | None = None,
) -> BridgeResponse:
    if not settings.clear_log_url:
        return BridgeResponse(ok=False, response_text="", error="Clear-log URL is empty.")

    try:
        request_client = requests if http_session is None else http_session
        response = request_client.post(
            settings.clear_log_url,
            headers=BRIDGE_HTTP_HEADERS,
            timeout=settings.timeout_seconds,
        )
    except requests.RequestException as exc:
        return BridgeResponse(ok=False, response_text="", error=str(exc))

    return BridgeResponse(
        ok=response.ok,
        response_text=response.text,
        status_code=response.status_code,
        error=None if response.ok else response.text,
    )


def _fetch_bridge_status_snapshot_with_retries(
    settings: BridgeSettings,
    *,
    request_fresh_status: bool,
    status_settle_seconds: float = 0.12,
    http_session: requests.Session | None = None,
    retry_attempts: int = 3,
    retry_sleep_seconds: float = 0.35,
) -> tuple[BridgeResponse, BridgeStatusSnapshot | None]:
    last_response, last_snapshot = fetch_bridge_status_snapshot(
        settings,
        request_fresh_status=request_fresh_status,
        status_settle_seconds=status_settle_seconds,
        http_session=http_session,
    )
    if last_response.ok or not _is_transient_bridge_error(last_response.error):
        return last_response, last_snapshot

    for attempt_index in range(retry_attempts):
        time.sleep(retry_sleep_seconds * (attempt_index + 1))
        retry_response, retry_snapshot = fetch_bridge_status_snapshot(
            settings,
            request_fresh_status=request_fresh_status,
            status_settle_seconds=status_settle_seconds,
            http_session=None,
        )
        if retry_response.ok or not _is_transient_bridge_error(retry_response.error):
            return retry_response, retry_snapshot
        last_response, last_snapshot = retry_response, retry_snapshot

    return last_response, last_snapshot


def _clear_bridge_log_with_retries(
    settings: BridgeSettings,
    *,
    http_session: requests.Session | None = None,
    retry_attempts: int = 3,
    retry_sleep_seconds: float = 0.35,
) -> BridgeResponse:
    last_response = clear_bridge_log(settings, http_session=http_session)
    if last_response.ok or not _is_transient_bridge_error(last_response.error):
        return last_response

    for attempt_index in range(retry_attempts):
        time.sleep(retry_sleep_seconds * (attempt_index + 1))
        retry_response = clear_bridge_log(settings, http_session=None)
        if retry_response.ok or not _is_transient_bridge_error(retry_response.error):
            return retry_response
        last_response = retry_response

    return last_response


def _normalized_bridge_base_url_candidates(
    settings: BridgeSettings,
    bridge_base_url_candidates: list[str] | tuple[str, ...] | None,
) -> list[str]:
    candidates: list[str] = []
    for candidate in (settings.normalized_base_url, *(bridge_base_url_candidates or ())):
        if candidate is None:
            continue
        normalized = BridgeSettings(base_url=str(candidate)).normalized_base_url
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _fetch_bridge_status_with_recovery(
    settings: BridgeSettings,
    *,
    request_fresh_status: bool,
    status_settle_seconds: float = 0.12,
    http_session: requests.Session | None = None,
    retry_attempts: int = 3,
    retry_sleep_seconds: float = 0.35,
    bridge_base_url_candidates: list[str] | tuple[str, ...] | None = None,
    recovery_timeout_seconds: float = 0.0,
) -> tuple[BridgeSettings, BridgeResponse, BridgeStatusSnapshot | None]:
    status_settings = replace(
        settings,
        timeout_seconds=max(0.5, min(settings.timeout_seconds, 2.0)),
    )
    response, snapshot = _fetch_bridge_status_snapshot_with_retries(
        status_settings,
        request_fresh_status=request_fresh_status,
        status_settle_seconds=status_settle_seconds,
        http_session=http_session,
        retry_attempts=retry_attempts,
        retry_sleep_seconds=retry_sleep_seconds,
    )
    if response.ok or not _is_transient_bridge_error(response.error) or recovery_timeout_seconds <= 0:
        return settings, response, snapshot

    last_settings = settings
    last_response = response
    last_snapshot = snapshot
    deadline = time.time() + recovery_timeout_seconds
    probe_timeout_seconds = max(0.5, min(settings.timeout_seconds, 1.5))

    while time.time() < deadline:
        for base_url in _normalized_bridge_base_url_candidates(settings, bridge_base_url_candidates):
            probe_settings = replace(settings, base_url=base_url, timeout_seconds=probe_timeout_seconds)
            probe_response, probe_snapshot = _fetch_bridge_status_snapshot_with_retries(
                probe_settings,
                request_fresh_status=request_fresh_status,
                status_settle_seconds=status_settle_seconds,
                http_session=None,
                retry_attempts=1,
                retry_sleep_seconds=retry_sleep_seconds,
            )
            if probe_response.ok:
                recovered_settings = replace(settings, base_url=probe_settings.normalized_base_url)
                return recovered_settings, probe_response, probe_snapshot

            last_settings = replace(settings, base_url=probe_settings.normalized_base_url)
            last_response = probe_response
            last_snapshot = probe_snapshot
            if not _is_transient_bridge_error(probe_response.error):
                return last_settings, last_response, last_snapshot

        time.sleep(max(retry_sleep_seconds, 0.2))

    return last_settings, last_response, last_snapshot


def send_servo_named_position(
    settings: BridgeSettings,
    position_name: str,
    *,
    timeout_seconds: float | None = None,
    http_session: requests.Session | None = None,
) -> BridgeResponse:
    normalized = position_name.strip().lower()
    if normalized not in {"up", "down", "default"}:
        return BridgeResponse(
            ok=False,
            response_text="",
            error=f"Unsupported servo position: {position_name}",
        )
    if not settings.servo_move_url:
        return BridgeResponse(ok=False, response_text="", error="Servo move URL is empty.")

    try:
        request_client = requests if http_session is None else http_session
        response = request_client.post(
            settings.servo_move_url,
            json={"name": normalized},
            headers=BRIDGE_HTTP_HEADERS,
            timeout=settings.timeout_seconds if timeout_seconds is None else timeout_seconds,
        )
    except requests.RequestException as exc:
        return BridgeResponse(ok=False, response_text="", error=str(exc))

    return BridgeResponse(
        ok=response.ok,
        response_text=response.text,
        status_code=response.status_code,
        error=None if response.ok else response.text,
    )


def prepare_gcode_for_streaming(gcode_text: str) -> list[str]:
    commands: list[str] = []
    for raw_line in gcode_text.splitlines():
        line = _sanitize_grbl_command(_strip_gcode_comments(raw_line))
        if not line:
            continue
        commands.append(line)
    return _inline_feed_only_motion_commands(commands)


def strip_pen_control_commands(gcode_text: str) -> str:
    filtered_lines: list[str] = []
    skip_next_dwell = False

    for raw_line in gcode_text.splitlines():
        line = _strip_gcode_comments(raw_line).strip()
        if not line:
            continue

        if _is_bridge_local_command(line):
            skip_next_dwell = True
            continue

        if skip_next_dwell and _parse_dwell_seconds(line) is not None:
            skip_next_dwell = False
            continue

        skip_next_dwell = False
        filtered_lines.append(line)

    return "\n".join(filtered_lines) + ("\n" if filtered_lines else "")


def build_pen_position_command(
    pen_motion_settings: PenMotionSettings,
    position_name: str,
    *,
    include_modal_prefix: bool = True,
) -> str:
    normalized = position_name.strip().lower()
    if normalized == "up":
        position_mm = pen_motion_settings.pen_up_position_mm
    elif normalized == "down":
        position_mm = pen_motion_settings.pen_down_position_mm
    else:
        raise ValueError(f"Unsupported pen position: {position_name}")

    if include_modal_prefix:
        return build_absolute_move_command(
            pen_motion_settings.axis,
            position_mm,
            pen_motion_settings.feed_rate_mm_min,
        )

    return (
        f"G1 {pen_motion_settings.axis.upper()}{position_mm:.3f} "
        f"F{pen_motion_settings.feed_rate_mm_min:.1f}"
    )


def replace_pen_control_commands_with_axis_moves(
    gcode_text: str,
    pen_motion_settings: PenMotionSettings,
) -> str:
    rewritten_lines: list[str] = []
    raw_lines = gcode_text.splitlines()
    index = 0

    while index < len(raw_lines):
        raw_line = raw_lines[index]
        line = _strip_gcode_comments(raw_line).strip()
        if not line:
            index += 1
            continue

        pen_position = _bridge_local_command_position(line)
        if pen_position == "up":
            rewritten_lines.append(
                build_pen_position_command(
                    pen_motion_settings,
                    "up",
                    include_modal_prefix=False,
                )
            )
            index = _skip_attached_dwell(raw_lines, index)
            if pen_motion_settings.pen_up_dwell_seconds > 0:
                rewritten_lines.append(_format_dwell_command(pen_motion_settings.pen_up_dwell_seconds))
        elif pen_position == "down":
            rewritten_lines.append(
                build_pen_position_command(
                    pen_motion_settings,
                    "down",
                    include_modal_prefix=False,
                )
            )
            index = _skip_attached_dwell(raw_lines, index)
            if pen_motion_settings.pen_down_dwell_seconds > 0:
                rewritten_lines.append(_format_dwell_command(pen_motion_settings.pen_down_dwell_seconds))
        elif pen_position == "default":
            rewritten_lines.append(
                build_pen_position_command(
                    pen_motion_settings,
                    "up",
                    include_modal_prefix=False,
                )
            )
            index = _skip_attached_dwell(raw_lines, index)
            if pen_motion_settings.pen_up_dwell_seconds > 0:
                rewritten_lines.append(_format_dwell_command(pen_motion_settings.pen_up_dwell_seconds))
        else:
            rewritten_lines.append(line)
        index += 1

    return "\n".join(rewritten_lines) + ("\n" if rewritten_lines else "")

def _skip_attached_dwell(raw_lines: list[str], current_index: int) -> int:
    next_index = current_index + 1
    while next_index < len(raw_lines):
        candidate = _strip_gcode_comments(raw_lines[next_index]).strip()
        if not candidate:
            next_index += 1
            continue
        if _parse_dwell_seconds(candidate) is not None:
            return next_index
        break
    return current_index


def _format_dwell_command(seconds: float) -> str:
    return f"G4 P{max(float(seconds), 0.0):.3f}"


def _inline_feed_only_motion_commands(commands: list[str]) -> list[str]:
    normalized_commands: list[str] = []
    pending_motion_code: str | None = None
    pending_feed_word: str | None = None

    for command in commands:
        motion_feed = _extract_feed_only_motion_command(command)
        if motion_feed is not None:
            if pending_motion_code is not None and pending_feed_word is not None:
                normalized_commands.append(f"{pending_motion_code} {pending_feed_word}")
            pending_motion_code, pending_feed_word = motion_feed
            continue

        if pending_motion_code is not None and pending_feed_word is not None:
            if _can_attach_feed_word(command, pending_motion_code):
                normalized_commands.append(f"{command} {pending_feed_word}")
                pending_motion_code = None
                pending_feed_word = None
                continue

            normalized_commands.append(f"{pending_motion_code} {pending_feed_word}")
            pending_motion_code = None
            pending_feed_word = None

        normalized_commands.append(command)

    if pending_motion_code is not None and pending_feed_word is not None:
        normalized_commands.append(f"{pending_motion_code} {pending_feed_word}")

    return normalized_commands


def _extract_feed_only_motion_command(command: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"\s*(G0|G1)\s+(F[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*", command, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).upper(), match.group(2).upper()


def _can_attach_feed_word(command: str, motion_code: str) -> bool:
    upper_command = command.upper()
    if not re.match(rf"^\s*{motion_code}(?:\s|$)", upper_command):
        return False
    return re.search(r"(^|\s)F[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s|$)", upper_command) is None


def _prepare_stream_items(commands: list[str]) -> list[StreamItem]:
    items: list[StreamItem] = []
    index = 0
    while index < len(commands):
        command = commands[index]
        if _is_bridge_local_command(command):
            settle_seconds = 0.0
            source_line_count = 1
            if index + 1 < len(commands):
                dwell_seconds = _parse_dwell_seconds(commands[index + 1])
                if dwell_seconds is not None:
                    settle_seconds = dwell_seconds
                    source_line_count = 2
                    index += 1
            items.append(
                StreamItem(
                    kind="local_servo",
                    command=command,
                    settle_seconds=settle_seconds,
                    source_line_count=source_line_count,
                    start_command_index=index,
                )
            )
        else:
            items.append(StreamItem(kind="grbl", command=command, start_command_index=index))
        index += 1
    return items


def stream_gcode_to_bridge(
    settings: BridgeSettings,
    gcode_text: str,
    *,
    batch_line_limit: int = 8,
    batch_timeout_seconds: float = 12.0,
    pen_sync_timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.12,
    ready_timeout_seconds: float = 5.0,
    max_in_flight_commands: int = 6,
    inter_command_delay_seconds: float = 0.02,
    bridge_base_url_candidates: list[str] | tuple[str, ...] | None = None,
    bridge_recovery_timeout_seconds: float = 15.0,
    allow_uncertain_batch_replay: bool = True,
    cancel_check=None,
    progress_callback=None,
) -> GcodeStreamResult:
    streaming_settings = replace(
        settings,
        timeout_seconds=max(0.5, min(settings.timeout_seconds, STREAM_COMMAND_TIMEOUT_SECONDS)),
    )
    commands = prepare_gcode_for_streaming(gcode_text)
    items = _prepare_stream_items(commands)
    total_commands = len(commands)
    if total_commands == 0:
        return GcodeStreamResult(
            ok=False,
            total_commands=0,
            completed_commands=0,
            sent_commands=0,
            message="No GRBL commands were found in the generated G-code.",
            failed_command_index=0,
            active_base_url=streaming_settings.normalized_base_url,
        )

    completed_commands = 0
    sent_commands = 0
    last_snapshot: BridgeStatusSnapshot | None = None
    known_log_lines: list[str] = []
    known_log_entries: list[tuple[int, str]] = []
    consecutive_degraded_command_recoveries = 0

    def check_canceled() -> None:
        if cancel_check is not None:
            cancel_check()

    def degraded_slow_mode_result(item: StreamItem) -> GcodeStreamResult:
        return GcodeStreamResult(
            ok=False,
            total_commands=total_commands,
            completed_commands=completed_commands,
            sent_commands=sent_commands,
            message=(
                "Bridge command responses are timing out repeatedly, so drawing was paused "
                "instead of continuing in timeout-paced slow mode."
            ),
            failed_command=item.command,
            failed_command_index=item.start_command_index,
            last_snapshot=last_snapshot,
            active_base_url=streaming_settings.normalized_base_url,
        )

    def record_degraded_command_recovery(item: StreamItem) -> GcodeStreamResult | None:
        nonlocal consecutive_degraded_command_recoveries
        consecutive_degraded_command_recoveries += 1
        if consecutive_degraded_command_recoveries >= MAX_CONSECUTIVE_DEGRADED_COMMAND_RECOVERIES:
            return degraded_slow_mode_result(item)
        return None

    def record_clean_command_response() -> None:
        nonlocal consecutive_degraded_command_recoveries
        consecutive_degraded_command_recoveries = 0

    with requests.Session() as http_session:
        check_canceled()
        idle_message = _wait_for_grbl_idle(
            streaming_settings,
            total_commands=total_commands,
            completed_commands=0,
            sent_commands=0,
            timeout_seconds=ready_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            progress_callback=progress_callback,
            http_session=http_session,
            bridge_base_url_candidates=bridge_base_url_candidates,
            bridge_recovery_timeout_seconds=bridge_recovery_timeout_seconds,
        )
        if idle_message.active_settings is not None:
            streaming_settings = idle_message.active_settings
        if not idle_message.ok:
            return GcodeStreamResult(
                ok=False,
                total_commands=total_commands,
                completed_commands=0,
                sent_commands=0,
                message=idle_message.message,
                last_snapshot=idle_message.last_snapshot,
                active_base_url=streaming_settings.normalized_base_url,
            )
        last_snapshot = idle_message.last_snapshot
        known_log_lines = [] if last_snapshot is None or last_snapshot.recent_log is None else list(last_snapshot.recent_log)
        known_log_entries = (
            []
            if last_snapshot is None or last_snapshot.recent_log_entries is None
            else list(last_snapshot.recent_log_entries)
        )

        item_index = 0
        uncertain_batch_replay_counts: dict[int, int] = {}

        def plan_uncertain_batch_replay(
            *,
            batch_start_item_index: int,
            batch_items: list[StreamItem],
            ack_count: int,
            reason_message: str,
        ) -> int | None:
            nonlocal known_log_entries, known_log_lines, last_snapshot, streaming_settings

            if not batch_items:
                return None
            if not allow_uncertain_batch_replay:
                return None
            replay_count = uncertain_batch_replay_counts.get(batch_start_item_index, 0)
            if replay_count >= UNCERTAIN_BATCH_REPLAY_LIMIT:
                return None
            if not _is_stream_batch_replay_safe(batch_items):
                return None

            if progress_callback is not None:
                progress_callback(completed_commands, total_commands, reason_message)

            idle_message = _wait_for_grbl_idle(
                streaming_settings,
                total_commands=total_commands,
                completed_commands=completed_commands,
                sent_commands=sent_commands,
                timeout_seconds=max(
                    batch_timeout_seconds,
                    bridge_recovery_timeout_seconds,
                    streaming_settings.timeout_seconds,
                    5.0,
                ),
                poll_interval_seconds=poll_interval_seconds,
                reason_label="before replaying the last uncertain drawing lines",
                progress_callback=progress_callback,
                http_session=http_session,
                bridge_base_url_candidates=bridge_base_url_candidates,
                bridge_recovery_timeout_seconds=bridge_recovery_timeout_seconds,
            )
            if idle_message.active_settings is not None:
                streaming_settings = idle_message.active_settings
            last_snapshot = idle_message.last_snapshot or last_snapshot
            if not idle_message.ok:
                return None

            known_log_lines = (
                []
                if last_snapshot is None or last_snapshot.recent_log is None
                else list(last_snapshot.recent_log)
            )
            known_log_entries = (
                []
                if last_snapshot is None or last_snapshot.recent_log_entries is None
                else list(last_snapshot.recent_log_entries)
            )
            replay_offset = _uncertain_batch_replay_offset(ack_count, len(batch_items))
            uncertain_batch_replay_counts[batch_start_item_index] = replay_count + 1
            if progress_callback is not None:
                progress_callback(
                    completed_commands,
                    total_commands,
                    (
                        "Recovered bridge contact; replaying from the last confirmed overlap "
                        f"at command {batch_items[replay_offset].start_command_index + 1}."
                    ),
                )
            return batch_start_item_index + replay_offset

        while item_index < len(items):
            check_canceled()
            if items[item_index].kind == "local_servo":
                idle_message = _wait_for_grbl_idle(
                    streaming_settings,
                    total_commands=total_commands,
                    completed_commands=completed_commands,
                    sent_commands=sent_commands,
                    timeout_seconds=max(batch_timeout_seconds, pen_sync_timeout_seconds),
                    poll_interval_seconds=poll_interval_seconds,
                    reason_label="before the next pen command",
                    progress_callback=progress_callback,
                    http_session=http_session,
                    bridge_base_url_candidates=bridge_base_url_candidates,
                    bridge_recovery_timeout_seconds=bridge_recovery_timeout_seconds,
                )
                if idle_message.active_settings is not None:
                    streaming_settings = idle_message.active_settings
                if not idle_message.ok:
                    return GcodeStreamResult(
                        ok=False,
                        total_commands=total_commands,
                        completed_commands=completed_commands,
                        sent_commands=sent_commands,
                        message=idle_message.message,
                        failed_command=items[item_index].command,
                        failed_command_index=items[item_index].start_command_index,
                        last_snapshot=idle_message.last_snapshot,
                        active_base_url=streaming_settings.normalized_base_url,
                    )

                last_snapshot = idle_message.last_snapshot
                servo_position = _bridge_local_command_position(items[item_index].command)
                check_canceled()
                if servo_position is None:
                    response = send_grbl_command(
                        streaming_settings,
                        items[item_index].command,
                        http_session=http_session,
                    )
                else:
                    response = send_servo_named_position(
                        streaming_settings,
                        servo_position,
                        timeout_seconds=max(streaming_settings.timeout_seconds, 6.0),
                        http_session=http_session,
                    )
                if not response.ok:
                    return GcodeStreamResult(
                        ok=False,
                        total_commands=total_commands,
                        completed_commands=completed_commands,
                        sent_commands=sent_commands,
                        message=response.error or "A local servo command failed during drawing.",
                        failed_command=items[item_index].command,
                        failed_command_index=items[item_index].start_command_index,
                        last_snapshot=last_snapshot,
                        active_base_url=streaming_settings.normalized_base_url,
                    )

                sent_commands += items[item_index].source_line_count
                completed_commands += items[item_index].source_line_count
                if progress_callback is not None:
                    progress_callback(
                        completed_commands,
                        total_commands,
                        f"Executed pen command {completed_commands}/{total_commands}: {items[item_index].command}",
                    )
                if items[item_index].settle_seconds > 0:
                    time.sleep(items[item_index].settle_seconds)
                item_index += 1
                continue

            batch_start_item_index = item_index
            batch_items: list[StreamItem] = []
            batch_line_total = 0
            while item_index < len(items) and items[item_index].kind == "grbl" and len(batch_items) < batch_line_limit:
                batch_items.append(items[item_index])
                batch_line_total += items[item_index].source_line_count
                item_index += 1

            required_ack_count = len(batch_items)
            batch_replay_item_index: int | None = None

            effective_in_flight_limit = max(
                1,
                min(max_in_flight_commands, required_ack_count if required_ack_count > 0 else max_in_flight_commands),
            )
            effective_char_budget = max(GRBL_SAFE_STREAM_BUFFER_BYTES, 1)
            batch_sent_count = 0
            ack_count = 0
            last_ack_count = 0
            outstanding_lengths: list[int] = []
            outstanding_char_count = 0
            last_progress_at = time.time()
            deadline = time.time() + batch_timeout_seconds

            while time.time() < deadline:
                check_canceled()
                while batch_sent_count < len(batch_items):
                    next_item = batch_items[batch_sent_count]
                    unacked_ack_required = batch_sent_count - ack_count
                    next_wire_length = _command_wire_length(next_item.command)
                    command_recovered_from_transient = False

                    if unacked_ack_required >= effective_in_flight_limit:
                        break
                    if outstanding_char_count + next_wire_length > effective_char_budget and outstanding_lengths:
                        break

                    check_canceled()
                    response = send_grbl_command(
                        streaming_settings,
                        next_item.command,
                        http_session=http_session,
                    )
                    if not response.ok:
                        if (
                            _is_transient_bridge_error(response.error)
                            and unacked_ack_required == 0
                            and _is_low_risk_stream_retry_command(next_item.command)
                        ):
                            check_canceled()
                            command_recovered_from_transient = True
                            retry_response = send_grbl_command(
                                streaming_settings,
                                next_item.command,
                                http_session=None,
                            )
                            if retry_response.ok or not _is_transient_bridge_error(retry_response.error):
                                response = retry_response

                    if not response.ok:
                        if _is_transient_bridge_error(response.error):
                            batch_replay_item_index = plan_uncertain_batch_replay(
                                batch_start_item_index=batch_start_item_index,
                                batch_items=batch_items,
                                ack_count=ack_count,
                                reason_message=(
                                    "Bridge lost track of an in-flight drawing line. "
                                    "Waiting for Idle before replaying a small overlap."
                                ),
                            )
                            if batch_replay_item_index is not None:
                                break

                        # If the bridge dropped the HTTP connection while no other GRBL lines were in flight,
                        # give it a short grace period and check whether the line was actually accepted.
                        if _is_transient_bridge_error(response.error) and unacked_ack_required == 0:
                            recovery_window_seconds = max(
                                float(bridge_recovery_timeout_seconds),
                                streaming_settings.timeout_seconds,
                                2.0,
                            )
                            recovery_deadline = time.time() + recovery_window_seconds
                            deadline = max(deadline, recovery_deadline + max(batch_timeout_seconds, 2.0))
                            recovered = False
                            replay_attempts = 0
                            while time.time() < recovery_deadline:
                                streaming_settings, status_response, snapshot = _fetch_bridge_status_with_recovery(
                                    streaming_settings,
                                    request_fresh_status=False,
                                    http_session=http_session,
                                    retry_attempts=2,
                                    retry_sleep_seconds=max(poll_interval_seconds, 0.2),
                                    bridge_base_url_candidates=bridge_base_url_candidates,
                                    recovery_timeout_seconds=max(recovery_deadline - time.time(), 0.0),
                                )
                                if status_response.ok:
                                    last_snapshot = snapshot
                                    current_log_lines = [] if snapshot is None or snapshot.recent_log is None else snapshot.recent_log
                                    current_log_entries = (
                                        []
                                        if snapshot is None or snapshot.recent_log_entries is None
                                        else snapshot.recent_log_entries
                                    )
                                    new_log_lines = _diff_recent_logs(
                                        known_log_lines,
                                        current_log_lines,
                                        known_log_entries,
                                        current_log_entries,
                                    )
                                    known_log_lines = list(current_log_lines)
                                    known_log_entries = list(current_log_entries)
                                    recovery_state = None if snapshot is None or snapshot.grbl_status is None else snapshot.grbl_status.state
                                    recovered_ack_increment = _count_grbl_ack_lines(new_log_lines)
                                    error_line = _find_grbl_error_line(new_log_lines)
                                    reset_line = _find_grbl_reset_marker(new_log_lines)
                                    if recovered_ack_increment > 0:
                                        degraded_result = record_degraded_command_recovery(next_item)
                                        if degraded_result is not None:
                                            return degraded_result
                                        batch_sent_count += 1
                                        sent_commands += next_item.source_line_count
                                        ack_count = min(ack_count + recovered_ack_increment, batch_sent_count)
                                        last_ack_count = ack_count
                                        last_progress_at = time.time()
                                        recovered = True
                                        if progress_callback is not None:
                                            progress_callback(
                                                completed_commands,
                                                total_commands,
                                                (
                                                    "Bridge briefly disconnected, but the last line appears to have been accepted. "
                                                    f"Continuing with {sent_commands}/{total_commands} sent."
                                                ),
                                            )
                                        break

                                    if reset_line is not None:
                                        response = BridgeResponse(
                                            ok=False,
                                            response_text="",
                                            error=(
                                                "GRBL appears to have reset during drawing. "
                                                f"Detected startup marker: {reset_line}."
                                            ),
                                        )
                                        break

                                    if error_line is not None:
                                        response = BridgeResponse(
                                            ok=False,
                                            response_text="",
                                            error=f"GRBL reported an error while drawing: {_describe_grbl_error_line(error_line)}",
                                        )
                                        break

                                    if recovery_state == "Alarm":
                                        response = BridgeResponse(
                                            ok=False,
                                            response_text="",
                                            error="GRBL entered Alarm while recovering from a bridge timeout.",
                                        )
                                        break

                                    can_replay_uncertain_line = (
                                        _is_low_risk_stream_retry_command(next_item.command)
                                        or recovery_state == "Idle"
                                    )
                                    if can_replay_uncertain_line and replay_attempts < UNCERTAIN_STREAM_REPLAY_LIMIT:
                                        replay_attempts += 1
                                        if progress_callback is not None:
                                            progress_callback(
                                                completed_commands,
                                                total_commands,
                                                (
                                                    "Bridge recovered; replaying the last unconfirmed line "
                                                    f"from the last acknowledged point: {next_item.command}"
                                                ),
                                            )
                                        check_canceled()
                                        retry_response = send_grbl_command(
                                            streaming_settings,
                                            next_item.command,
                                            http_session=None,
                                        )
                                        if retry_response.ok:
                                            response = retry_response
                                            recovered = True
                                            break
                                        response = retry_response
                                        if not _is_transient_bridge_error(retry_response.error):
                                            break
                                time.sleep(max(poll_interval_seconds, 0.2))
                                check_canceled()

                            if recovered:
                                if response.ok:
                                    degraded_result = record_degraded_command_recovery(next_item)
                                    if degraded_result is not None:
                                        return degraded_result
                                    batch_sent_count += 1
                                    sent_commands += next_item.source_line_count
                                    outstanding_lengths.append(next_wire_length)
                                    outstanding_char_count += next_wire_length
                                    if progress_callback is not None:
                                        progress_callback(
                                            completed_commands,
                                            total_commands,
                                            (
                                                f"Sent {sent_commands}/{total_commands}: {next_item.command} "
                                                f"(in flight: {len(outstanding_lengths)} lines, {outstanding_char_count} chars)"
                                            ),
                                        )
                                    if inter_command_delay_seconds > 0:
                                        time.sleep(inter_command_delay_seconds)
                                continue

                        return GcodeStreamResult(
                            ok=False,
                            total_commands=total_commands,
                            completed_commands=completed_commands,
                            sent_commands=sent_commands,
                            message=response.error or "A bridge command failed during drawing.",
                            failed_command=next_item.command,
                            failed_command_index=next_item.start_command_index,
                            last_snapshot=last_snapshot,
                            active_base_url=streaming_settings.normalized_base_url,
                        )

                    if command_recovered_from_transient:
                        degraded_result = record_degraded_command_recovery(next_item)
                        if degraded_result is not None:
                            return degraded_result
                    else:
                        record_clean_command_response()

                    batch_sent_count += 1
                    sent_commands += next_item.source_line_count
                    outstanding_lengths.append(next_wire_length)
                    outstanding_char_count += next_wire_length
                    if progress_callback is not None:
                        progress_callback(
                            completed_commands,
                            total_commands,
                            (
                                f"Sent {sent_commands}/{total_commands}: {next_item.command} "
                                f"(in flight: {len(outstanding_lengths)} lines, {outstanding_char_count} chars)"
                            ),
                        )
                    if inter_command_delay_seconds > 0:
                        time.sleep(inter_command_delay_seconds)

                if batch_replay_item_index is not None:
                    break

                request_fresh_status = (
                    batch_sent_count >= len(batch_items)
                    and (time.time() - last_progress_at) >= max(poll_interval_seconds * 2.0, 0.35)
                )
                streaming_settings, status_response, snapshot = _fetch_bridge_status_with_recovery(
                    streaming_settings,
                    request_fresh_status=request_fresh_status,
                    http_session=http_session,
                    retry_attempts=3,
                    retry_sleep_seconds=max(poll_interval_seconds, 0.2),
                    bridge_base_url_candidates=bridge_base_url_candidates,
                    recovery_timeout_seconds=max(0.0, float(bridge_recovery_timeout_seconds)),
                )
                if not status_response.ok:
                    if _is_transient_bridge_error(status_response.error):
                        batch_replay_item_index = plan_uncertain_batch_replay(
                            batch_start_item_index=batch_start_item_index,
                            batch_items=batch_items,
                            ack_count=ack_count,
                            reason_message=(
                                "Bridge status timed out during a drawing batch. "
                                "Waiting for Idle before replaying a small overlap."
                            ),
                        )
                        if batch_replay_item_index is not None:
                            break

                    return GcodeStreamResult(
                        ok=False,
                        total_commands=total_commands,
                        completed_commands=completed_commands,
                        sent_commands=sent_commands,
                        message=status_response.error or "Could not read bridge status while drawing.",
                        failed_command=batch_items[-1].command if batch_items else None,
                        failed_command_index=(
                            batch_items[min(max(ack_count, 0), len(batch_items) - 1)].start_command_index
                            if batch_items
                            else completed_commands
                        ),
                        last_snapshot=last_snapshot,
                        active_base_url=streaming_settings.normalized_base_url,
                    )
                deadline = max(deadline, time.time() + max(batch_timeout_seconds, 2.0))

                last_snapshot = snapshot
                current_log_lines = [] if snapshot is None or snapshot.recent_log is None else snapshot.recent_log
                current_log_entries = (
                    []
                    if snapshot is None or snapshot.recent_log_entries is None
                    else snapshot.recent_log_entries
                )
                new_log_lines = _diff_recent_logs(
                    known_log_lines,
                    current_log_lines,
                    known_log_entries,
                    current_log_entries,
                )
                known_log_lines = list(current_log_lines)
                known_log_entries = list(current_log_entries)
                ack_count = min(ack_count + _count_grbl_ack_lines(new_log_lines), batch_sent_count)
                error_line = _find_grbl_error_line(new_log_lines)
                reset_line = _find_grbl_reset_marker(new_log_lines)
                state = None if snapshot is None or snapshot.grbl_status is None else snapshot.grbl_status.state

                if ack_count > last_ack_count:
                    for _ in range(ack_count - last_ack_count):
                        if not outstanding_lengths:
                            break
                        outstanding_char_count -= outstanding_lengths.pop(0)
                    last_ack_count = ack_count
                    last_progress_at = time.time()

                if reset_line is not None:
                    failed_item = batch_items[min(max(ack_count, 0), len(batch_items) - 1)] if batch_items else None
                    return GcodeStreamResult(
                        ok=False,
                        total_commands=total_commands,
                        completed_commands=completed_commands,
                        sent_commands=sent_commands,
                        message=(
                            "GRBL appears to have reset during drawing. "
                            f"Detected startup marker: {reset_line}. "
                            "This usually means the controller or bridge reset mid-job."
                        ),
                        failed_command=None if failed_item is None else failed_item.command,
                        failed_command_index=completed_commands if failed_item is None else failed_item.start_command_index,
                        last_snapshot=last_snapshot,
                        active_base_url=streaming_settings.normalized_base_url,
                    )

                if error_line is not None:
                    return GcodeStreamResult(
                        ok=False,
                        total_commands=total_commands,
                        completed_commands=completed_commands,
                        sent_commands=sent_commands,
                        message=f"GRBL reported an error while drawing: {_describe_grbl_error_line(error_line)}",
                        failed_command=batch_items[min(max(ack_count, 0), len(batch_items) - 1)].command if batch_items else None,
                        failed_command_index=(
                            batch_items[min(max(ack_count, 0), len(batch_items) - 1)].start_command_index
                            if batch_items
                            else completed_commands
                        ),
                        last_snapshot=last_snapshot,
                        active_base_url=streaming_settings.normalized_base_url,
                    )

                if state == "Alarm":
                    return GcodeStreamResult(
                        ok=False,
                        total_commands=total_commands,
                        completed_commands=completed_commands,
                        sent_commands=sent_commands,
                        message="GRBL entered Alarm while drawing.",
                        failed_command=batch_items[-1].command if batch_items else None,
                        failed_command_index=(
                            batch_items[min(max(ack_count, 0), len(batch_items) - 1)].start_command_index
                            if batch_items
                            else completed_commands
                        ),
                        last_snapshot=last_snapshot,
                        active_base_url=streaming_settings.normalized_base_url,
                    )

                if ack_count >= required_ack_count:
                    completed_commands += batch_line_total
                    if progress_callback is not None:
                        progress_callback(
                            completed_commands,
                            total_commands,
                            f"Completed {completed_commands}/{total_commands} commands",
                        )
                    break

                if batch_sent_count >= len(batch_items) and state == "Idle":
                    completed_commands += batch_line_total
                    if progress_callback is not None:
                        progress_callback(
                            completed_commands,
                            total_commands,
                            (
                                "Controller returned to Idle after the current batch. "
                                f"Assuming {completed_commands}/{total_commands} commands completed."
                            ),
                        )
                    break

                time.sleep(poll_interval_seconds)
            else:
                batch_replay_item_index = plan_uncertain_batch_replay(
                    batch_start_item_index=batch_start_item_index,
                    batch_items=batch_items,
                    ack_count=ack_count,
                    reason_message=(
                        "GRBL acknowledgement timed out during a drawing batch. "
                        "Waiting for Idle before replaying a small overlap."
                    ),
                )
                if batch_replay_item_index is not None:
                    item_index = batch_replay_item_index
                    continue

                return GcodeStreamResult(
                    ok=False,
                    total_commands=total_commands,
                    completed_commands=completed_commands,
                    sent_commands=sent_commands,
                    message="Timed out waiting for GRBL to acknowledge a batch of drawing commands.",
                    failed_command=batch_items[-1].command if batch_items else None,
                    failed_command_index=(
                        batch_items[min(max(ack_count, 0), len(batch_items) - 1)].start_command_index
                        if batch_items
                        else completed_commands
                    ),
                    last_snapshot=last_snapshot,
                    active_base_url=streaming_settings.normalized_base_url,
                )

            if batch_replay_item_index is not None:
                item_index = batch_replay_item_index
                continue

    return GcodeStreamResult(
        ok=True,
        total_commands=total_commands,
        completed_commands=completed_commands,
        sent_commands=sent_commands,
        message="Drawing stream completed successfully.",
        last_snapshot=last_snapshot,
        active_base_url=streaming_settings.normalized_base_url,
    )


def poll_grbl_status(settings: BridgeSettings) -> tuple[BridgeResponse, GrblStatus | None]:
    bridge_response, snapshot = fetch_bridge_status_snapshot(settings, request_fresh_status=True)
    return bridge_response, None if snapshot is None else snapshot.grbl_status


def run_grbl_link_test(
    settings: BridgeSettings,
    *,
    probe_command: str = "$I",
    timeout_seconds: float = 3.0,
    poll_interval_seconds: float = 0.12,
) -> GrblLinkTestResult:
    with requests.Session() as http_session:
        clear_response = clear_bridge_log(settings, http_session=http_session)
        if not clear_response.ok:
            return GrblLinkTestResult(
                ok=False,
                message=clear_response.error or "Could not clear the bridge log before the link test.",
                observed_lines=[],
                last_snapshot=None,
            )

        command_response = send_grbl_command(settings, probe_command, http_session=http_session)
        if not command_response.ok:
            return GrblLinkTestResult(
                ok=False,
                message=command_response.error or "Could not send the GRBL link-test command.",
                observed_lines=[],
                last_snapshot=None,
            )

        deadline = time.time() + timeout_seconds
        last_snapshot: BridgeStatusSnapshot | None = None

        while time.time() < deadline:
            status_response, snapshot = fetch_bridge_status_snapshot(
                settings,
                request_fresh_status=False,
                http_session=http_session,
            )
            if not status_response.ok:
                return GrblLinkTestResult(
                    ok=False,
                    message=status_response.error or "Could not read bridge status during the link test.",
                    observed_lines=[],
                    last_snapshot=last_snapshot,
                )

            last_snapshot = snapshot
            log_lines = [] if snapshot is None or snapshot.recent_log is None else snapshot.recent_log
            observed_lines = [line.strip() for line in log_lines if line.strip()]
            if _has_positive_grbl_link_evidence(observed_lines):
                return GrblLinkTestResult(
                    ok=True,
                    message="GRBL responded through the ESP32 bridge.",
                    observed_lines=observed_lines,
                    last_snapshot=last_snapshot,
                )

            time.sleep(poll_interval_seconds)

        observed_lines = [] if last_snapshot is None or last_snapshot.recent_log is None else [line.strip() for line in last_snapshot.recent_log if line.strip()]
        return GrblLinkTestResult(
            ok=False,
            message=(
                "No GRBL response was observed before the link-test timeout. "
                "The ESP32 HTTP server may be alive while the Uno/GRBL side is disconnected, reset, locked, or not replying."
            ),
            observed_lines=observed_lines,
            last_snapshot=last_snapshot,
        )


def _wait_for_grbl_idle(
    settings: BridgeSettings,
    *,
    total_commands: int,
    completed_commands: int,
    sent_commands: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
    reason_label: str = "before continuing",
    progress_callback=None,
    http_session: requests.Session | None = None,
    bridge_base_url_candidates: list[str] | tuple[str, ...] | None = None,
    bridge_recovery_timeout_seconds: float = 0.0,
) -> IdleWaitResult:
    deadline = time.time() + timeout_seconds
    last_snapshot: BridgeStatusSnapshot | None = None
    last_state = "Unknown"
    active_settings = settings

    while time.time() < deadline:
        active_settings, status_response, snapshot = _fetch_bridge_status_with_recovery(
            active_settings,
            request_fresh_status=True,
            http_session=http_session,
            retry_attempts=3,
            retry_sleep_seconds=max(poll_interval_seconds, 0.2),
            bridge_base_url_candidates=bridge_base_url_candidates,
            recovery_timeout_seconds=min(
                bridge_recovery_timeout_seconds,
                max(deadline - time.time(), 0.0),
            ),
        )
        if not status_response.ok:
            return IdleWaitResult(
                ok=False,
                message=status_response.error or "Could not read bridge status while waiting for Idle.",
                last_snapshot=last_snapshot,
                active_settings=active_settings,
            )

        last_snapshot = snapshot
        state = None if snapshot is None or snapshot.grbl_status is None else snapshot.grbl_status.state
        if state:
            last_state = state

        if state == "Idle":
            return IdleWaitResult(
                ok=True,
                message="GRBL is Idle.",
                last_snapshot=last_snapshot,
                active_settings=active_settings,
            )

        if state == "Alarm":
            return IdleWaitResult(
                ok=False,
                message="GRBL is in Alarm. Run $H to home the machine, or $X only if you intentionally need to clear the lock, and wait for Idle before drawing.",
                last_snapshot=last_snapshot,
                active_settings=active_settings,
            )

        if progress_callback is not None:
            progress_callback(
                completed_commands,
                total_commands,
                f"Waiting for controller to become Idle {reason_label}. Current state: {last_state} ({sent_commands} sent)",
            )
        time.sleep(poll_interval_seconds)

    if last_state == "Run":
        return IdleWaitResult(
            ok=False,
            message=(
                "GRBL was still in Run when the idle-wait timeout expired. "
                f"This usually means the machine was still finishing a long move {reason_label}. "
                "Increase the relevant wait timeout or lower the motion speeds if this keeps happening."
            ),
            last_snapshot=last_snapshot,
            active_settings=active_settings,
        )

    return IdleWaitResult(
        ok=False,
        message=f"GRBL did not become Idle before the timeout. Current state: {last_state}.",
        last_snapshot=last_snapshot,
        active_settings=active_settings,
    )


def parse_bridge_status(payload: str) -> BridgeStatusSnapshot | None:
    recent_log: list[str] = []
    recent_log_entries: list[tuple[int, str]] = []

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        status = parse_grbl_status(payload)
        return BridgeStatusSnapshot(
            raw_payload=payload,
            recent_log=[],
            grbl_status=status,
        )

    if not isinstance(parsed, dict):
        status = parse_grbl_status(payload)
        return BridgeStatusSnapshot(
            raw_payload=payload,
            recent_log=[],
            grbl_status=status,
        )

    for key in ("recent", "log", "recentLog", "recent_log", "responses"):
        value = parsed.get(key)
        if isinstance(value, list):
            recent_log = [str(item) for item in value]
            break
        if isinstance(value, str):
            recent_log = [line for line in value.splitlines() if line.strip()]
            break

    entries_value = parsed.get("recentLogEntries")
    if isinstance(entries_value, list):
        for item in entries_value:
            if not isinstance(item, dict):
                continue
            try:
                sequence = int(item.get("seq"))
            except (TypeError, ValueError):
                continue
            line = str(item.get("line", ""))
            recent_log_entries.append((sequence, line))
        if recent_log_entries:
            recent_log = [line for _, line in recent_log_entries]

    status = _extract_latest_grbl_status(payload, recent_log)
    return BridgeStatusSnapshot(
        raw_payload=payload,
        bridge_ok=_coerce_optional_bool(parsed.get("ok")),
        bridge_ip=_first_string(parsed, ("ip", "esp32Ip", "esp32_ip", "localIp", "local_ip")),
        last_command=_first_string(parsed, ("lastCommand", "last_command")),
        recent_log=recent_log,
        recent_log_entries=recent_log_entries,
        grbl_status=status,
        servo_status=_parse_servo_status(parsed.get("servo")),
    )


def parse_grbl_status(payload: str) -> GrblStatus | None:
    status_line = _extract_status_line(payload)
    if status_line is None:
        return None
    return parse_grbl_status_line(status_line)


def parse_grbl_status_line(status_line: str) -> GrblStatus | None:
    if status_line is None:
        return None

    content = status_line.strip()[1:-1]
    parts = content.split("|")
    state = parts[0] if parts else "Unknown"
    machine_position = None
    work_position = None
    work_coordinate_offset = None
    pins = ""

    for token in parts[1:]:
        if token.startswith("MPos:"):
            values = token.removeprefix("MPos:").split(",")
            if len(values) >= 3:
                try:
                    machine_position = (float(values[0]), float(values[1]), float(values[2]))
                except ValueError:
                    machine_position = None
        if token.startswith("WPos:"):
            values = token.removeprefix("WPos:").split(",")
            if len(values) >= 3:
                try:
                    work_position = (float(values[0]), float(values[1]), float(values[2]))
                except ValueError:
                    work_position = None
        if token.startswith("WCO:"):
            values = token.removeprefix("WCO:").split(",")
            if len(values) >= 3:
                try:
                    work_coordinate_offset = (float(values[0]), float(values[1]), float(values[2]))
                except ValueError:
                    work_coordinate_offset = None
        if token.startswith("Pn:"):
            pins = token.removeprefix("Pn:")

    if work_position is None and machine_position is not None and work_coordinate_offset is not None:
        work_position = tuple(
            machine_position[index] - work_coordinate_offset[index]
            for index in range(3)
        )

    return GrblStatus(
        raw=status_line,
        state=state,
        machine_position=machine_position,
        work_position=work_position,
        work_coordinate_offset=work_coordinate_offset,
        pins=pins,
        x_limit_pressed=("X" in pins),
        y_limit_pressed=("Y" in pins),
        z_limit_pressed=("Z" in pins),
        probe_input_active=("P" in pins),
    )


def _extract_status_line(payload: str) -> str | None:
    stripped = payload.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return stripped

    matches = re.findall(r"<[^>]+>", payload)
    if matches:
        return matches[-1]

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None

    return _find_status_line_in_json(parsed)


def _find_status_line_in_json(value) -> str | None:
    if isinstance(value, str):
        if value.startswith("<") and value.endswith(">"):
            return value
        return None
    if isinstance(value, dict):
        for nested in reversed(tuple(value.values())):
            found = _find_status_line_in_json(nested)
            if found:
                return found
        return None
    if isinstance(value, list):
        for nested in reversed(value):
            found = _find_status_line_in_json(nested)
            if found:
                return found
        return None
    return None


def _extract_latest_grbl_status(payload: str, recent_log: list[str]) -> GrblStatus | None:
    status_line = _extract_status_line_from_lines(recent_log)
    if status_line is not None:
        return parse_grbl_status_line(status_line)
    return parse_grbl_status(payload)


def _extract_status_line_from_lines(lines: list[str]) -> str | None:
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("<") and stripped.endswith(">"):
            return stripped
    return None


def _resolve_url(base_url: str, path_or_url: str) -> str:
    if not base_url:
        return ""

    value = path_or_url.strip()
    if not value:
        return ""
    if re.match(r"^https?://", value, flags=re.IGNORECASE):
        return value
    return urljoin(f"{base_url}/", value.lstrip("/"))


def _coerce_optional_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _first_string(data: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            return value
    return ""


def _parse_servo_status(value) -> ServoStatus | None:
    if not isinstance(value, dict):
        return None

    return ServoStatus(
        pin=_coerce_optional_int(value.get("pin")),
        min_angle=_coerce_optional_int(value.get("minAngle")),
        max_angle=_coerce_optional_int(value.get("maxAngle")),
        up_angle=_coerce_optional_int(value.get("upAngle")),
        down_angle=_coerce_optional_int(value.get("downAngle")),
        default_angle=_coerce_optional_int(value.get("defaultAngle")),
        step_angle=_coerce_optional_int(value.get("stepAngle")),
        current_angle=_coerce_optional_int(value.get("currentAngle")),
        attached=value.get("attached") if isinstance(value.get("attached"), bool) else None,
        last_action=value.get("lastAction") if isinstance(value.get("lastAction"), str) else "",
    )


def _coerce_optional_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _strip_gcode_comments(line: str) -> str:
    without_semicolon = line.split(";", 1)[0]
    return re.sub(r"\([^)]*\)", "", without_semicolon)


def _sanitize_grbl_command(command: str) -> str:
    ascii_only = command.encode("ascii", errors="ignore").decode("ascii")
    printable = "".join(character if character >= " " else " " for character in ascii_only)
    collapsed = " ".join(printable.strip().split())
    return collapsed


def _count_grbl_ack_lines(lines: list[str]) -> int:
    count = 0
    for line in lines:
        stripped = line.strip().lower()
        if stripped == "ok" or stripped.startswith("error:"):
            count += 1
    return count


def _diff_recent_log_lines(previous_lines: list[str], current_lines: list[str]) -> list[str]:
    if not current_lines:
        return []
    if not previous_lines:
        return current_lines
    if previous_lines == current_lines:
        return []

    max_overlap = min(len(previous_lines), len(current_lines))
    for overlap in range(max_overlap, 0, -1):
        if previous_lines[-overlap:] == current_lines[:overlap]:
            return current_lines[overlap:]

    return current_lines


def _diff_recent_log_entries(
    previous_entries: list[tuple[int, str]],
    current_entries: list[tuple[int, str]],
) -> list[str]:
    if not current_entries:
        return []
    if not previous_entries:
        return [line for _, line in current_entries]

    previous_max_sequence = max(sequence for sequence, _ in previous_entries)
    current_max_sequence = max(sequence for sequence, _ in current_entries)
    if current_max_sequence < previous_max_sequence:
        return [line for _, line in current_entries]

    return [line for sequence, line in current_entries if sequence > previous_max_sequence]


def _diff_recent_logs(
    previous_lines: list[str],
    current_lines: list[str],
    previous_entries: list[tuple[int, str]],
    current_entries: list[tuple[int, str]],
) -> list[str]:
    if current_entries:
        return _diff_recent_log_entries(previous_entries, current_entries)
    return _diff_recent_log_lines(previous_lines, current_lines)


def _find_grbl_error_line(lines: list[str]) -> str | None:
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("error:"):
            return stripped
    return None


def _describe_grbl_error_line(error_line: str) -> str:
    match = re.fullmatch(r"error:(\d+)", error_line.strip(), flags=re.IGNORECASE)
    if match is None:
        return error_line

    error_code = int(match.group(1))
    description = GRBL_ERROR_DESCRIPTIONS.get(error_code)
    if description is None:
        return error_line

    return f"{error_line} ({description})"


def _find_grbl_reset_marker(lines: list[str]) -> str | None:
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("[ver:") or lowered.startswith("grbl "):
            return stripped
    return None


def _is_stream_batch_replay_safe(batch_items: list[StreamItem]) -> bool:
    return all(_is_low_risk_stream_retry_command(item.command) for item in batch_items)


def _uncertain_batch_replay_offset(ack_count: int, batch_size: int) -> int:
    if batch_size <= 0:
        return 0
    confirmed_offset = min(max(ack_count, 0), batch_size - 1)
    return max(0, confirmed_offset - UNCERTAIN_BATCH_REPLAY_OVERLAP_COMMANDS)


def _is_low_risk_stream_retry_command(command: str) -> bool:
    normalized = _sanitize_grbl_command(command).upper()
    if not normalized:
        return False
    if normalized in REALTIME_COMMANDS:
        return False
    if normalized.startswith("$"):
        return False
    if "G91" in normalized:
        return False
    if re.search(r"(^|\s)G(?:10|28|30|53|92)(?:\s|$)", normalized):
        return False
    return True


def _is_bridge_local_command(command: str) -> bool:
    normalized = _strip_gcode_comments(command).strip().upper()
    return (
        normalized == "M5"
        or normalized.startswith("M5 ")
        or normalized == "M3"
        or normalized.startswith("M3 ")
        or normalized == "SERVO UP"
        or normalized == "SERVO DOWN"
        or normalized == "SERVO DEFAULT"
    )


def _bridge_local_command_position(command: str) -> str | None:
    normalized = _strip_gcode_comments(command).strip().upper()
    if normalized == "M5" or normalized.startswith("M5 ") or normalized == "SERVO UP":
        return "up"
    if normalized == "M3" or normalized.startswith("M3 ") or normalized == "SERVO DOWN":
        return "down"
    if normalized == "SERVO DEFAULT":
        return "default"
    return None


def _has_positive_grbl_link_evidence(lines: list[str]) -> bool:
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if stripped.startswith("[HTTP->GRBL"):
            continue
        if stripped.startswith("[HTTP]"):
            continue
        if lowered == "ok":
            return True
        if lowered.startswith("[ver:") or lowered.startswith("[opt:") or lowered.startswith("grbl "):
            return True
        if stripped.startswith("<") and stripped.endswith(">"):
            return True
    return False


def _parse_dwell_seconds(command: str) -> float | None:
    normalized = _strip_gcode_comments(command).strip().upper()
    if not normalized.startswith("G4"):
        return None

    for token in normalized.split()[1:]:
        if token.startswith("P") or token.startswith("S"):
            try:
                return float(token[1:])
            except ValueError:
                return None
    return None


def _command_wire_length(command: str) -> int:
    return len(command.encode("ascii", errors="ignore")) + 2
