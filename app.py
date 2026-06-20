from __future__ import annotations

import hashlib
import io
import json
import os
import time
from datetime import datetime
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.elements.image as st_image
from PIL import Image, ImageDraw, ImageOps

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

if not hasattr(st_image, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url as _image_to_url_v2
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def _compat_image_to_url(image, width, clamp, channels, output_format, image_id):
        return _image_to_url_v2(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    st_image.image_to_url = _compat_image_to_url

from streamlit_drawable_canvas import st_canvas

from photo_to_gcode.cura_backend import (
    CuraSettings,
    CuraSliceResult,
    build_cura_page_mask,
    build_cura_page_tone,
    slice_page_mask_with_cura,
)
from photo_to_gcode.gcode import generate_gcode
from photo_to_gcode.image_processing import (
    apply_erase_overlay_to_mask,
    mask_bounds_mm,
    mask_to_preview_image,
)
from photo_to_gcode.machine_control import (
    BridgeSettings,
    BridgeResponse,
    PenMotionSettings,
    build_pen_position_command,
    build_jog_command,
    clear_bridge_log,
    fetch_bridge_status_snapshot,
    prepare_gcode_for_streaming,
    replace_pen_control_commands_with_axis_moves,
    run_grbl_link_test,
    send_grbl_command,
    strip_pen_control_commands,
    stream_gcode_to_bridge,
)
from photo_to_gcode.models import PlannedDrawing, ProcessingSettings
from photo_to_gcode.openai_images import (
    DEFAULT_PLOTTER_AI_PROMPT,
    convert_image_to_plotter_friendly_ai,
)
from photo_to_gcode.planner import plan_drawing, plan_page_mask
from photo_to_gcode.preview import (
    render_toolpath_preview,
    render_toolpath_simulation_preview,
    render_vector_preview,
)
from photo_to_gcode.triangle_mesh import TriangleMeshSettings, plan_triangle_mesh_from_tone_map
from photo_to_gcode.toolpaths import calculate_path_metrics

PAPER_PRESETS: dict[str, tuple[float, float]] = {
    "Letter (8.5 x 11 in / 216 x 280 mm)": (215.9, 279.4),
    "Full Working Area (218 x 373 mm)": (218.0, 373.0),
}

DEFAULT_PAPER_PRESET = "Letter (8.5 x 11 in / 216 x 280 mm)"

DEFAULT_X_STEPS_PER_MM = 17400.0 / 218.0
DEFAULT_Y_STEPS_PER_MM = 30000.0 / 373.0
APP_VERSION = "V3.0"
DRAW_RESUME_STATE_DIR = Path(".draw_resume_state")
MANUAL_CONTROL_MAPPING_VERSION = 4
CURA_DEFAULTS_VERSION = 12
MACHINE_DEFAULTS_VERSION = 15
DEFAULT_MOTION_MAX_RATE_MM_MIN = 7200.0
DEFAULT_MOTION_ACCEL_MM_S2 = 240.0
DEFAULT_DRAW_SPEED_MM_PER_S = DEFAULT_MOTION_MAX_RATE_MM_MIN / 60.0
DEFAULT_CURA_PROCESSING_RESOLUTION_PPMM = 18.0
DEFAULT_CURA_LINE_WIDTH_MM = 0.050
DEFAULT_MIN_INFILL_SPACING_MM = 0.40
DEFAULT_PEN_UP_GAP_MM = 8.0
SECONDARY_PEN_UP_GAP_MM = 5.0
BRIDGE_DISCOVERY_CANDIDATES = (
    "http://10.0.0.90",
    "http://10.0.0.89",
    "http://esp32-grbl-bridge.local",
)
BRIDGE_DISCOVERY_INTERVAL_SECONDS = 20.0
BRIDGE_DISCOVERY_TIMEOUT_SECONDS = 1.2
BRIDGE_TRANSIENT_ERROR_HINTS = (
    "connection to",
    "connecttimeout",
    "connect timeout",
    "connection refused",
    "host is down",
    "host not found",
    "max retries exceeded",
    "name or service not known",
    "nodename nor servname provided",
    "read timed out",
    "remotedisconnected",
    "timed out",
)
_send_grbl_command_direct = send_grbl_command
MOTION_TUNING_PROFILES: dict[str, dict[str, float | str]] = {
    "balanced": {
        "label": "Balanced",
        "xy_max_rate_mm_min": 2500.0,
        "xy_accel_mm_s2": 80.0,
        "z_max_rate_mm_min": 1200.0,
        "z_accel_mm_s2": 20.0,
    },
    "snappy": {
        "label": "Snappy Plotter",
        "xy_max_rate_mm_min": DEFAULT_MOTION_MAX_RATE_MM_MIN,
        "xy_accel_mm_s2": DEFAULT_MOTION_ACCEL_MM_S2,
        "z_max_rate_mm_min": DEFAULT_MOTION_MAX_RATE_MM_MIN,
        "z_accel_mm_s2": DEFAULT_MOTION_ACCEL_MM_S2,
    },
    "fast_pen": {
        "label": "Fast Pen Lift",
        "xy_max_rate_mm_min": DEFAULT_MOTION_MAX_RATE_MM_MIN,
        "xy_accel_mm_s2": DEFAULT_MOTION_ACCEL_MM_S2,
        "z_max_rate_mm_min": DEFAULT_MOTION_MAX_RATE_MM_MIN,
        "z_accel_mm_s2": DEFAULT_MOTION_ACCEL_MM_S2,
    },
}
MANUAL_AXIS_TUNING_PRESETS = (
    {
        "id": "pen_lift",
        "default_label": "Pen Lift",
        "default_grbl_axis": "Z",
        "default_positive_move_amount_mm": 8.0,
        "default_negative_move_amount_mm": 28.0,
        "default_feed_rate_mm_min": DEFAULT_MOTION_MAX_RATE_MM_MIN,
        "default_direction_multiplier": -1,
        "max_travel_from_home_mm": 28.0,
        "behavior_note": (
            "`Pen Lift +` raises toward the limit switch. `Pen Lift -` lowers toward the paper. "
            "Automated drawing homes Z to the switch, then uses Z20.000 for pen-up and Z28.000 for pen-down."
        ),
        "description": "Default matches your rewiring: the CNC shield Z driver now raises and lowers the pen.",
    },
    {
        "id": "draw_x",
        "default_label": "Drawing X",
        "default_grbl_axis": "X",
        "default_move_amount_mm": 5.0,
        "default_feed_rate_mm_min": DEFAULT_MOTION_MAX_RATE_MM_MIN,
        "default_direction_multiplier": 1,
        "max_travel_from_home_mm": 219.0,
        "behavior_note": "`Drawing X -` moves left and away from the limit switch when `Positive button sends + distance` is selected.",
        "description": "Default matches your rewiring: the machine's horizontal X motion now lives on GRBL X.",
    },
    {
        "id": "draw_y",
        "default_label": "Drawing Y",
        "default_grbl_axis": "Y",
        "default_move_amount_mm": 5.0,
        "default_feed_rate_mm_min": DEFAULT_MOTION_MAX_RATE_MM_MIN,
        "default_direction_multiplier": 1,
        "max_travel_from_home_mm": 374.0,
        "behavior_note": "`Drawing Y -` moves downward and away from the limit switch when `Positive button sends + distance` is selected.",
        "description": "Your Y axis is still on the GRBL Y driver, so this one stays mapped straight through.",
    },
)
GRBL_AXIS_ORDER = ("X", "Y", "Z")
GRBL_INPUT_ORDER = ("X", "Y", "Z", "P")


def _apply_app_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.12), transparent 26%),
                linear-gradient(180deg, #0b1020 0%, #101828 38%, #0f172a 100%);
        }
        .stApp [data-testid="stHeader"] {
            background: rgba(11, 16, 32, 0.72);
        }
        .stApp section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(13, 18, 33, 0.98), rgba(17, 24, 39, 0.98));
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }
        .stApp [data-testid="stExpander"] {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.78);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.18);
        }
        .stApp div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 16px;
            padding: 0.6rem 0.8rem;
        }
        .stApp .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(125, 211, 252, 0.24);
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.96), rgba(15, 23, 42, 0.96));
            color: #e5eefc;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18);
        }
        .stApp .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #22c55e 0%, #0ea5e9 100%);
            color: #08111f;
            border: none;
            font-weight: 700;
        }
        .stApp .stTextInput input,
        .stApp .stNumberInput input,
        .stApp textarea {
            border-radius: 14px;
        }
        #control-rail-anchor {
            display: none;
        }
        div[data-testid="stVerticalBlock"]:has(#control-rail-anchor) {
            position: sticky;
            top: 5.5rem;
            align-self: flex-start;
        }
        .control-rail-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.85rem 1rem;
            border: 1px solid rgba(125, 211, 252, 0.2);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(13, 18, 33, 0.92));
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.18);
            margin-bottom: 0.8rem;
        }
        .control-rail-header strong {
            font-size: 1rem;
            color: #e5eefc;
            letter-spacing: 0.02em;
        }
        .control-rail-header span {
            color: #8fdcff;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _resume_payload_path(file_stem: str) -> Path:
    slug = hashlib.sha1(file_stem.encode("utf-8")).hexdigest()[:16]
    return DRAW_RESUME_STATE_DIR / f"{slug}.json"


def _command_context_hash(commands: list[str]) -> str:
    digest = hashlib.sha1()
    for command in commands:
        digest.update(command.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_persisted_resume_payload(file_stem: str, expected_command_hash: str) -> dict[str, object] | None:
    payload_path = _resume_payload_path(file_stem)
    if not payload_path.exists():
        return None

    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    if payload.get("command_hash") != expected_command_hash:
        try:
            payload_path.unlink()
        except OSError:
            pass
        return None

    commands = payload.get("commands")
    if not isinstance(commands, list) or not all(isinstance(command, str) for command in commands):
        return None

    return payload


def _store_persisted_resume_payload(file_stem: str, payload: dict[str, object]) -> None:
    DRAW_RESUME_STATE_DIR.mkdir(parents=True, exist_ok=True)
    _resume_payload_path(file_stem).write_text(json.dumps(payload), encoding="utf-8")


def _clear_persisted_resume_payload(file_stem: str) -> None:
    payload_path = _resume_payload_path(file_stem)
    try:
        payload_path.unlink()
    except (FileNotFoundError, OSError):
        return


def _open_uploaded_image(uploaded_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(uploaded_bytes))
    return ImageOps.exif_transpose(image)


def _image_bytes_signature(image_bytes: bytes) -> str:
    return hashlib.sha1(image_bytes).hexdigest()


def _configured_openai_api_key() -> str:
    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key:
        return env_key

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:  # noqa: BLE001
        secret_key = ""
    return str(secret_key).strip()


def _unique_desktop_export_path(filename: str) -> Path:
    desktop_dir = Path.home() / "Desktop"
    desktop_dir.mkdir(parents=True, exist_ok=True)

    candidate_name = Path(filename).name.strip() or "plotter_ai_image.png"
    candidate_path = desktop_dir / candidate_name
    if not candidate_path.exists():
        return candidate_path

    stem = Path(candidate_name).stem or "plotter_ai_image"
    suffix = Path(candidate_name).suffix or ".png"
    for index in range(2, 1000):
        numbered_path = desktop_dir / f"{stem} ({index}){suffix}"
        if not numbered_path.exists():
            return numbered_path

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return desktop_dir / f"{stem}-{timestamp}{suffix}"


def _save_image_bytes_to_desktop(image_bytes: bytes, filename: str) -> Path:
    export_path = _unique_desktop_export_path(filename)
    export_path.write_bytes(image_bytes)
    return export_path


def _sync_workflow_source_image(uploaded_file) -> tuple[bytes, str]:
    original_bytes = uploaded_file.getvalue()
    original_signature = _image_bytes_signature(original_bytes)
    if st.session_state.get("workflow_source_image_signature") != original_signature:
        st.session_state["workflow_source_image_signature"] = original_signature
        st.session_state["workflow_source_image_bytes"] = original_bytes
        st.session_state["workflow_source_image_name"] = str(uploaded_file.name)
        st.session_state["workflow_source_image_mime"] = str(uploaded_file.type or "image/png")
        st.session_state["workflow_active_image_bytes"] = original_bytes
        st.session_state["workflow_active_image_name"] = str(uploaded_file.name)
        st.session_state["workflow_active_image_kind"] = "original"
        st.session_state.pop("workflow_ai_image_bytes", None)
        st.session_state.pop("workflow_ai_image_name", None)
        st.session_state.pop("workflow_ai_revised_prompt", None)
        st.session_state.pop("workflow_ai_feedback", None)
        st.session_state.pop("workflow_ai_additional_comments", None)
    active_bytes = st.session_state.get("workflow_active_image_bytes", original_bytes)
    active_kind = str(st.session_state.get("workflow_active_image_kind", "original"))
    return bytes(active_bytes), active_kind


def _build_plotter_ai_prompt(additional_comments: str) -> str:
    cleaned_comments = " ".join(str(additional_comments).split())
    if not cleaned_comments:
        return DEFAULT_PLOTTER_AI_PROMPT
    return (
        f"{DEFAULT_PLOTTER_AI_PROMPT}\n\n"
        f"Additional user comments: {cleaned_comments}"
    )


@st.cache_data(show_spinner=False)
def build_plan(uploaded_bytes: bytes, settings_data: dict[str, float | int | str | bool]) -> PlannedDrawing:
    image = _open_uploaded_image(uploaded_bytes)
    settings = ProcessingSettings(**settings_data)
    return plan_drawing(image, settings)


@st.cache_data(show_spinner=False)
def build_cura_page_mask_data(
    uploaded_bytes: bytes,
    settings_data: dict[str, float | int | str | bool],
) -> np.ndarray:
    image = _open_uploaded_image(uploaded_bytes)
    settings = CuraSettings(**settings_data)
    return build_cura_page_mask(image, settings)


@st.cache_data(show_spinner=False)
def build_cura_page_tone_data(
    uploaded_bytes: bytes,
    settings_data: dict[str, float | int | str | bool],
) -> np.ndarray:
    image = _open_uploaded_image(uploaded_bytes)
    settings = CuraSettings(**settings_data)
    return build_cura_page_tone(image, settings)


@st.cache_data(show_spinner=False)
def build_vector_plan_from_page_mask(
    page_mask: np.ndarray,
    settings_data: dict[str, float | int | str | bool],
) -> PlannedDrawing:
    settings = ProcessingSettings(**settings_data)
    return plan_page_mask(page_mask, settings)


def build_native_settings(page_width_mm: float, page_height_mm: float) -> ProcessingSettings:
    st.sidebar.header("Processing")
    threshold = st.sidebar.slider("Black/white threshold", 0, 255, 165)
    invert_input = st.sidebar.checkbox(
        "Invert input",
        value=False,
        help="Turn this on if your source image is light on a dark background.",
    )

    st.sidebar.header("Stroke Planning")
    pen_width_mm = st.sidebar.number_input(
        "Virtual nozzle / pen width (mm)",
        min_value=0.02,
        max_value=1.20,
        value=DEFAULT_CURA_LINE_WIDTH_MM,
        step=0.001,
        format="%.3f",
        help="Planner-side stand-in for the actual mark width your pen leaves.",
    )
    min_feature_width_mm = st.sidebar.number_input(
        "Ignore lines thinner than (mm)",
        min_value=0.05,
        max_value=10.0,
        value=0.15,
        step=0.05,
        format="%.2f",
        help="This is now a detail-retention control, not the pen width. You can set it below 0.5 mm to keep finer source features in the preview.",
    )
    wall_count = st.sidebar.number_input(
        "Perimeter walls",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Like slicer wall count. A 1.0 mm stroke becomes two 0.5 mm passes when possible.",
    )
    thin_feature_mode = st.sidebar.checkbox(
        "Thin-feature mode",
        value=True,
        help="Treat narrow strokes separately using centerline tracing instead of walls + infill.",
    )
    thin_feature_max_width_mm = st.sidebar.number_input(
        "Thin feature max width (mm)",
        min_value=0.5,
        max_value=3.0,
        value=0.75,
        step=0.05,
        format="%.2f",
        help="Features narrower than this are candidates for centerline tracing.",
    )
    centerline_min_length_mm = st.sidebar.number_input(
        "Centerline minimum length (mm)",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=0.25,
        format="%.2f",
        help="Shorter thin-stroke fragments are discarded to reduce noisy travel moves.",
    )
    simplify_tolerance_px = st.sidebar.slider(
        "Path simplification (processing px)",
        0.0,
        6.0,
        0.75,
        0.25,
        help="Lower values preserve more shape detail before smoothing.",
    )
    curve_smoothing_passes = st.sidebar.slider(
        "Curve smoothing passes",
        0,
        4,
        2,
        1,
        help="Rounds jagged pixel contours into smoother curve-like paths.",
    )
    min_region_area_mm2 = st.sidebar.number_input(
        "Minimum dark region area (mm²)",
        min_value=0.0,
        max_value=100.0,
        value=0.75,
        step=0.25,
        format="%.2f",
    )

    st.sidebar.header("Fill Planning")
    fill_mode = st.sidebar.selectbox("Interior fill style", ("none", "zigzag"))
    fill_spacing_mm = st.sidebar.number_input(
        "Zigzag spacing (mm)",
        min_value=0.2,
        max_value=10.0,
        value=DEFAULT_MIN_INFILL_SPACING_MM,
        step=0.1,
        format="%.2f",
        help="The planner will not put infill lines closer together than the pen width.",
    )

    st.sidebar.header("Machine Output")
    feed_rate = st.sidebar.number_input(
        "Drawing feed rate (mm/min)",
        min_value=100,
        max_value=20000,
        value=1500,
        step=100,
    )
    pen_up_command = st.sidebar.text_input("Pen up command", "M5")
    pen_down_command = st.sidebar.text_input("Pen down command", "M3 S30")
    pen_pause_seconds = st.sidebar.number_input(
        "Pen settle pause (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=0.15,
        step=0.05,
        format="%.2f",
    )

    st.sidebar.header("Advanced")
    margin_mm = st.sidebar.slider("Margin (mm)", 0.0, 40.0, 10.0, 0.5)
    processing_resolution_ppmm = st.sidebar.slider(
        "Processing resolution (px/mm)",
        6.0,
        24.0,
        18.0,
        0.5,
        help="Higher values use more of your Mac's CPU and RAM, but preserve much finer detail in the page-space masks.",
    )
    curve_sample_step_mm = st.sidebar.number_input(
        "Curve sample step (mm)",
        min_value=0.1,
        max_value=2.0,
        value=0.25,
        step=0.05,
        format="%.2f",
        help="Smaller values follow smooth curves more closely but increase G-code size.",
    )

    return ProcessingSettings(
        page_width_mm=page_width_mm,
        page_height_mm=page_height_mm,
        margin_mm=margin_mm,
        threshold=threshold,
        invert_input=invert_input,
        pen_width_mm=pen_width_mm,
        min_feature_width_mm=min_feature_width_mm,
        min_region_area_mm2=min_region_area_mm2,
        wall_count=wall_count,
        thin_feature_mode=thin_feature_mode,
        thin_feature_max_width_mm=thin_feature_max_width_mm,
        centerline_min_length_mm=centerline_min_length_mm,
        simplify_tolerance_px=simplify_tolerance_px,
        curve_smoothing_passes=curve_smoothing_passes,
        curve_sample_step_mm=curve_sample_step_mm,
        fill_mode=fill_mode,
        fill_spacing_mm=fill_spacing_mm,
        processing_resolution_ppmm=processing_resolution_ppmm,
        feed_rate=feed_rate,
        pen_up_command=pen_up_command,
        pen_down_command=pen_down_command,
        pen_pause_seconds=pen_pause_seconds,
    )


def _ensure_cura_defaults() -> None:
    defaults = {
        "cura_threshold": 165,
        "cura_invert_input": False,
        "cura_margin_mm": 10.0,
        "cura_processing_resolution_ppmm": DEFAULT_CURA_PROCESSING_RESOLUTION_PPMM,
        "cura_placement_scale": 1.0,
        "cura_placement_rotation_degrees": 0.0,
        "cura_placement_offset_x_mm": 0.0,
        "cura_placement_offset_y_mm": 0.0,
        "cura_line_width_mm": DEFAULT_CURA_LINE_WIDTH_MM,
        "cura_wall_line_count": 1,
        "cura_infill_density_percent": 100,
        "cura_fill_strategy": "continuous_zigzag",
        "path_generation_mode": "vector_trace",
        "cura_fill_turn_split_angle_degrees": 20.0,
        "cura_continuous_fill_chunk_segments": 0,
        "cura_draw_speed_mm_per_s": DEFAULT_DRAW_SPEED_MM_PER_S,
        "cura_travel_speed_mm_per_s": DEFAULT_DRAW_SPEED_MM_PER_S,
        "cura_pen_up_command": "M5",
        "cura_pen_down_command": "M3 S30",
        "cura_pen_pause_seconds": 0.0,
        "cura_path_simplify_tolerance_mm": 0.08,
        "cura_min_segment_length_mm": 0.10,
        "cura_min_toolpath_length_mm": 0.10,
        "cura_coordinate_decimals": 3,
        "cura_editor_mode": "Move / Resize",
        "triangle_min_spacing_mm": 3.0,
        "triangle_max_spacing_mm": 10.0,
        "triangle_boundary_spacing_mm": 2.5,
        "draw_auto_home_after_finish": True,
    }

    defaults_version_changed = st.session_state.get("cura_defaults_version") != CURA_DEFAULTS_VERSION
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if defaults_version_changed:
        try:
            chunk_segments = int(st.session_state.get("cura_continuous_fill_chunk_segments", 0))
        except (TypeError, ValueError):
            chunk_segments = 0
        if chunk_segments == 24:
            st.session_state["cura_continuous_fill_chunk_segments"] = 0
        if abs(float(st.session_state.get("cura_draw_speed_mm_per_s", DEFAULT_DRAW_SPEED_MM_PER_S)) - 12.0) < 1e-9:
            st.session_state["cura_draw_speed_mm_per_s"] = DEFAULT_DRAW_SPEED_MM_PER_S
        if abs(float(st.session_state.get("cura_draw_speed_mm_per_s", DEFAULT_DRAW_SPEED_MM_PER_S)) - 90.0) < 1e-9:
            st.session_state["cura_draw_speed_mm_per_s"] = DEFAULT_DRAW_SPEED_MM_PER_S
        if abs(float(st.session_state.get("cura_draw_speed_mm_per_s", DEFAULT_DRAW_SPEED_MM_PER_S)) - 100.0) < 1e-9:
            st.session_state["cura_draw_speed_mm_per_s"] = DEFAULT_DRAW_SPEED_MM_PER_S
        if abs(float(st.session_state.get("cura_travel_speed_mm_per_s", DEFAULT_DRAW_SPEED_MM_PER_S)) - 25.0) < 1e-9:
            st.session_state["cura_travel_speed_mm_per_s"] = DEFAULT_DRAW_SPEED_MM_PER_S
        if abs(float(st.session_state.get("cura_travel_speed_mm_per_s", DEFAULT_DRAW_SPEED_MM_PER_S)) - 90.0) < 1e-9:
            st.session_state["cura_travel_speed_mm_per_s"] = DEFAULT_DRAW_SPEED_MM_PER_S
        if abs(float(st.session_state.get("cura_travel_speed_mm_per_s", DEFAULT_DRAW_SPEED_MM_PER_S)) - 100.0) < 1e-9:
            st.session_state["cura_travel_speed_mm_per_s"] = DEFAULT_DRAW_SPEED_MM_PER_S
        if abs(float(st.session_state.get("cura_pen_pause_seconds", 0.0)) - 0.02) < 1e-9:
            st.session_state["cura_pen_pause_seconds"] = 0.0
        if abs(float(st.session_state.get("cura_line_width_mm", DEFAULT_CURA_LINE_WIDTH_MM)) - 0.50) < 1e-9:
            st.session_state["cura_line_width_mm"] = DEFAULT_CURA_LINE_WIDTH_MM
        if abs(float(st.session_state.get("cura_line_width_mm", DEFAULT_CURA_LINE_WIDTH_MM)) - 0.20) < 1e-9:
            st.session_state["cura_line_width_mm"] = DEFAULT_CURA_LINE_WIDTH_MM
        if abs(float(st.session_state.get("cura_line_width_mm", DEFAULT_CURA_LINE_WIDTH_MM)) - 0.10) < 1e-9:
            st.session_state["cura_line_width_mm"] = DEFAULT_CURA_LINE_WIDTH_MM
        if abs(float(st.session_state.get("cura_line_width_mm", DEFAULT_CURA_LINE_WIDTH_MM)) - 0.08) < 1e-9:
            st.session_state["cura_line_width_mm"] = DEFAULT_CURA_LINE_WIDTH_MM
        if abs(float(st.session_state.get("cura_line_width_mm", DEFAULT_CURA_LINE_WIDTH_MM)) - 0.072) < 1e-9:
            st.session_state["cura_line_width_mm"] = DEFAULT_CURA_LINE_WIDTH_MM
        if abs(float(st.session_state.get("cura_processing_resolution_ppmm", 12.0)) - 6.0) < 1e-9:
            st.session_state["cura_processing_resolution_ppmm"] = 12.0
        if abs(float(st.session_state.get("cura_processing_resolution_ppmm", 12.0)) - 10.0) < 1e-9:
            st.session_state["cura_processing_resolution_ppmm"] = DEFAULT_CURA_PROCESSING_RESOLUTION_PPMM
        if abs(float(st.session_state.get("cura_processing_resolution_ppmm", 12.0)) - 12.0) < 1e-9:
            st.session_state["cura_processing_resolution_ppmm"] = DEFAULT_CURA_PROCESSING_RESOLUTION_PPMM
        if abs(float(st.session_state.get("cura_path_simplify_tolerance_mm", 0.08)) - 0.15) < 1e-9:
            st.session_state["cura_path_simplify_tolerance_mm"] = 0.08
        if abs(float(st.session_state.get("cura_min_segment_length_mm", 0.10)) - 0.50) < 1e-9:
            st.session_state["cura_min_segment_length_mm"] = 0.10
        if abs(float(st.session_state.get("cura_min_toolpath_length_mm", 0.10)) - 0.75) < 1e-9:
            st.session_state["cura_min_toolpath_length_mm"] = 0.10
        if int(st.session_state.get("cura_coordinate_decimals", 3)) == 2:
            st.session_state["cura_coordinate_decimals"] = 3
        if str(st.session_state.get("path_generation_mode", "")).strip().lower() not in {
            "vector_trace",
            "cura_slice",
            "triangle_mesh",
        }:
            st.session_state["path_generation_mode"] = "vector_trace"

    st.session_state["cura_defaults_version"] = CURA_DEFAULTS_VERSION


def _current_cura_settings(page_width_mm: float, page_height_mm: float) -> CuraSettings:
    _ensure_cura_defaults()
    return CuraSettings(
        page_width_mm=page_width_mm,
        page_height_mm=page_height_mm,
        margin_mm=float(st.session_state["cura_margin_mm"]),
        threshold=int(st.session_state["cura_threshold"]),
        invert_input=bool(st.session_state["cura_invert_input"]),
        placement_scale=float(st.session_state["cura_placement_scale"]),
        placement_rotation_degrees=float(st.session_state["cura_placement_rotation_degrees"]),
        placement_offset_x_mm=float(st.session_state["cura_placement_offset_x_mm"]),
        placement_offset_y_mm=float(st.session_state["cura_placement_offset_y_mm"]),
        processing_resolution_ppmm=float(st.session_state["cura_processing_resolution_ppmm"]),
        feature_height_mm=0.2,
        line_width_mm=float(st.session_state["cura_line_width_mm"]),
        wall_line_count=int(st.session_state["cura_wall_line_count"]),
        infill_density_percent=int(st.session_state["cura_infill_density_percent"]),
        draw_speed_mm_per_s=float(st.session_state["cura_draw_speed_mm_per_s"]),
        travel_speed_mm_per_s=float(st.session_state["cura_travel_speed_mm_per_s"]),
        pen_up_command=str(st.session_state["cura_pen_up_command"]),
        pen_down_command=str(st.session_state["cura_pen_down_command"]),
        pen_pause_seconds=float(st.session_state["cura_pen_pause_seconds"]),
        plotter_fill_mode=str(st.session_state["cura_fill_strategy"]),
        fill_turn_split_angle_degrees=float(st.session_state["cura_fill_turn_split_angle_degrees"]),
        continuous_fill_chunk_segments=int(st.session_state["cura_continuous_fill_chunk_segments"]),
        path_simplify_tolerance_mm=float(st.session_state["cura_path_simplify_tolerance_mm"]),
        min_segment_length_mm=float(st.session_state["cura_min_segment_length_mm"]),
        min_toolpath_length_mm=float(st.session_state["cura_min_toolpath_length_mm"]),
        coordinate_decimals=int(st.session_state["cura_coordinate_decimals"]),
    )


def _build_vector_settings_from_page_mask(
    page_mask: np.ndarray,
    cura_settings: CuraSettings,
) -> ProcessingSettings:
    mask_height_px, mask_width_px = page_mask.shape
    pixels_per_mm_x = mask_width_px / cura_settings.page_width_mm if cura_settings.page_width_mm > 0 else 0.0
    pixels_per_mm_y = mask_height_px / cura_settings.page_height_mm if cura_settings.page_height_mm > 0 else 0.0
    pixels_per_mm = max(0.1, min(pixels_per_mm_x, pixels_per_mm_y))

    infill_density = max(0, int(cura_settings.infill_density_percent))
    fill_mode = "zigzag" if infill_density > 0 else "none"
    if infill_density <= 0:
        fill_spacing_mm = max(cura_settings.line_width_mm, DEFAULT_MIN_INFILL_SPACING_MM)
    else:
        fill_spacing_mm = max(
            cura_settings.line_width_mm,
            cura_settings.line_width_mm * (100.0 / max(float(infill_density), 1.0)),
            DEFAULT_MIN_INFILL_SPACING_MM,
        )

    return ProcessingSettings(
        page_width_mm=cura_settings.page_width_mm,
        page_height_mm=cura_settings.page_height_mm,
        margin_mm=0.0,
        threshold=int(cura_settings.threshold),
        invert_input=bool(cura_settings.invert_input),
        pen_width_mm=float(cura_settings.line_width_mm),
        min_feature_width_mm=0.05,
        min_region_area_mm2=0.05,
        wall_count=max(1, int(cura_settings.wall_line_count)),
        thin_feature_mode=True,
        thin_feature_max_width_mm=max(0.75, float(cura_settings.line_width_mm) * 1.5),
        centerline_min_length_mm=0.35,
        simplify_tolerance_px=0.50,
        curve_smoothing_passes=2,
        curve_sample_step_mm=0.25,
        fill_mode=fill_mode,
        fill_spacing_mm=fill_spacing_mm,
        processing_resolution_ppmm=pixels_per_mm,
        potrace_turdsize=2,
        potrace_alphamax=1.0,
        potrace_opttolerance=0.2,
        feed_rate=max(60, int(round(float(cura_settings.draw_speed_mm_per_s) * 60.0))),
        pen_up_command=str(cura_settings.pen_up_command),
        pen_down_command=str(cura_settings.pen_down_command),
        pen_pause_seconds=float(cura_settings.pen_pause_seconds),
    )


def _build_triangle_mesh_settings_from_page_tone(
    page_tone_map: np.ndarray,
    cura_settings: CuraSettings,
) -> TriangleMeshSettings:
    mask_height_px, mask_width_px = page_tone_map.shape
    pixels_per_mm_x = mask_width_px / cura_settings.page_width_mm if cura_settings.page_width_mm > 0 else 0.0
    pixels_per_mm_y = mask_height_px / cura_settings.page_height_mm if cura_settings.page_height_mm > 0 else 0.0
    pixels_per_mm = max(0.1, min(pixels_per_mm_x, pixels_per_mm_y))

    return TriangleMeshSettings(
        page_width_mm=cura_settings.page_width_mm,
        page_height_mm=cura_settings.page_height_mm,
        processing_resolution_ppmm=pixels_per_mm,
        threshold=int(cura_settings.threshold),
        invert_input=bool(cura_settings.invert_input),
        min_spacing_mm=float(st.session_state["triangle_min_spacing_mm"]),
        max_spacing_mm=float(st.session_state["triangle_max_spacing_mm"]),
        boundary_spacing_mm=float(st.session_state["triangle_boundary_spacing_mm"]),
    )


def _build_gcode_processing_settings_from_cura(
    cura_settings: CuraSettings,
    *,
    fill_mode: str = "none",
    fill_spacing_mm: float | None = None,
) -> ProcessingSettings:
    return ProcessingSettings(
        page_width_mm=cura_settings.page_width_mm,
        page_height_mm=cura_settings.page_height_mm,
        margin_mm=0.0,
        threshold=int(cura_settings.threshold),
        invert_input=bool(cura_settings.invert_input),
        pen_width_mm=float(cura_settings.line_width_mm),
        min_feature_width_mm=0.05,
        min_region_area_mm2=0.05,
        wall_count=max(1, int(cura_settings.wall_line_count)),
        thin_feature_mode=True,
        thin_feature_max_width_mm=max(0.75, float(cura_settings.line_width_mm) * 1.5),
        centerline_min_length_mm=0.35,
        simplify_tolerance_px=0.0,
        curve_smoothing_passes=2,
        curve_sample_step_mm=0.25,
        fill_mode=fill_mode,
        fill_spacing_mm=max(float(cura_settings.line_width_mm), fill_spacing_mm or float(cura_settings.line_width_mm)),
        processing_resolution_ppmm=float(cura_settings.processing_resolution_ppmm),
        feed_rate=max(60, int(round(float(cura_settings.draw_speed_mm_per_s) * 60.0))),
        pen_up_command=str(cura_settings.pen_up_command),
        pen_down_command=str(cura_settings.pen_down_command),
        pen_pause_seconds=float(cura_settings.pen_pause_seconds),
    )


def _render_cura_workflow_settings() -> None:
    _ensure_cura_defaults()

    quick_cols = st.columns([1.25, 2.75], gap="large")
    with quick_cols[0]:
        st.session_state["cura_threshold"] = int(
            st.slider(
                "Black/white threshold",
                0,
                255,
                int(st.session_state["cura_threshold"]),
                key="workflow_cura_threshold_quick",
            )
        )
    with quick_cols[1]:
        st.caption("Path generation mode")
        current_generation_mode = str(st.session_state.get("path_generation_mode", "vector_trace"))
        mode_button_cols = st.columns(3)
        mode_buttons = (
            ("vector_trace", "Vector Trace"),
            ("triangle_mesh", "Triangle Mesh"),
            ("cura_slice", "Cura Slice"),
        )
        for index, (mode_value, mode_label) in enumerate(mode_buttons):
            with mode_button_cols[index]:
                if st.button(
                    mode_label,
                    key=f"workflow_path_mode_button::{mode_value}",
                    type="primary" if current_generation_mode == mode_value else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["path_generation_mode"] = mode_value
                    st.rerun()

    with st.expander("Path Mode", expanded=False):
        fill_mode_label = st.selectbox(
            "Infill plotting mode",
            (
                "Continuous zigzag fill",
                "Pen lift / drop each fill line",
            ),
            index=0 if st.session_state["cura_fill_strategy"] == "continuous_zigzag" else 1,
            help=(
                "Use continuous zigzag for faster fills, or switch to the older lift/drop behavior "
                "to test whether the dense zigzag region is the thing causing failures."
            ),
            key="workflow_fill_mode_label",
        )
        st.session_state["cura_fill_strategy"] = (
            "continuous_zigzag"
            if fill_mode_label == "Continuous zigzag fill"
            else "pen_lift_fill"
        )

    if st.session_state.get("path_generation_mode") == "triangle_mesh":
        with st.expander("Triangle Mesh", expanded=False):
            triangle_cols = st.columns(3)
            with triangle_cols[0]:
                st.session_state["triangle_min_spacing_mm"] = float(
                    st.number_input(
                        "Dark-region spacing (mm)",
                        min_value=0.8,
                        max_value=20.0,
                        value=float(st.session_state["triangle_min_spacing_mm"]),
                        step=0.1,
                        format="%.1f",
                        help="Triangles get tighter than this in the darkest parts of the image.",
                        key="workflow_triangle_min_spacing_mm",
                    )
                )
            with triangle_cols[1]:
                st.session_state["triangle_max_spacing_mm"] = float(
                    st.number_input(
                        "Light-region spacing (mm)",
                        min_value=1.0,
                        max_value=40.0,
                        value=float(st.session_state["triangle_max_spacing_mm"]),
                        step=0.5,
                        format="%.1f",
                        help="Lighter regions keep a wider mesh so the drawing stays airy and faster to plot.",
                        key="workflow_triangle_max_spacing_mm",
                    )
                )
            with triangle_cols[2]:
                st.session_state["triangle_boundary_spacing_mm"] = float(
                    st.number_input(
                        "Boundary spacing (mm)",
                        min_value=0.5,
                        max_value=20.0,
                        value=float(st.session_state["triangle_boundary_spacing_mm"]),
                        step=0.1,
                        format="%.1f",
                        help="Adds extra anchor points around the silhouette so outer edges stay recognizable.",
                        key="workflow_triangle_boundary_spacing_mm",
                    )
                )

    with st.expander("Stroke + Speed", expanded=False):
        stroke_cols = st.columns(5)
        with stroke_cols[0]:
            st.session_state["cura_line_width_mm"] = st.slider(
                "Virtual nozzle / pen width (mm)",
                0.05,
                1.20,
                float(st.session_state["cura_line_width_mm"]),
                0.001,
                help="This is the slicer-side stand-in for your pen width.",
                key="workflow_cura_line_width_mm",
            )
        with stroke_cols[1]:
            st.session_state["cura_wall_line_count"] = int(
                st.number_input(
                    "Wall line count",
                    min_value=1,
                    max_value=4,
                    value=int(st.session_state["cura_wall_line_count"]),
                    step=1,
                    key="workflow_cura_wall_line_count",
                )
            )
        with stroke_cols[2]:
            st.session_state["cura_infill_density_percent"] = int(
                st.slider(
                    "Fill density (%)",
                    0,
                    100,
                    int(st.session_state["cura_infill_density_percent"]),
                    5,
                    key="workflow_cura_infill_density_percent",
                )
            )
        with stroke_cols[3]:
            st.session_state["cura_draw_speed_mm_per_s"] = float(
                st.number_input(
                    "Drawing speed (mm/s)",
                    min_value=1.0,
                    max_value=200.0,
                    value=float(st.session_state["cura_draw_speed_mm_per_s"]),
                    step=1.0,
                    format="%.1f",
                    key="workflow_cura_draw_speed_mm_per_s",
                )
            )
        with stroke_cols[4]:
            st.session_state["cura_travel_speed_mm_per_s"] = float(
                st.number_input(
                    "Travel speed (mm/s)",
                    min_value=1.0,
                    max_value=300.0,
                    value=float(st.session_state["cura_travel_speed_mm_per_s"]),
                    step=1.0,
                    format="%.1f",
                    key="workflow_cura_travel_speed_mm_per_s",
                )
            )

    with st.expander("Image Processing", expanded=False):
        processing_cols = st.columns(3)
        with processing_cols[0]:
            st.session_state["cura_invert_input"] = st.checkbox(
                "Invert input",
                value=bool(st.session_state["cura_invert_input"]),
                help="Turn this on if your source image is light on a dark background.",
                key="workflow_cura_invert_input",
            )
        with processing_cols[1]:
            st.session_state["cura_margin_mm"] = float(
                st.slider(
                    "Margin (mm)",
                    0.0,
                    40.0,
                    float(st.session_state["cura_margin_mm"]),
                    0.5,
                    key="workflow_cura_margin_mm",
                )
            )
        with processing_cols[2]:
            st.session_state["cura_processing_resolution_ppmm"] = float(
                st.slider(
                    "Mask resolution (px/mm)",
                    2.0,
                    18.0,
                    float(st.session_state["cura_processing_resolution_ppmm"]),
                    0.5,
                    help="Higher values preserve more detail but increase STL size and slicing time.",
                    key="workflow_cura_processing_resolution_ppmm",
                )
            )

    with st.expander("Advanced Path Cleanup", expanded=False):
        advanced_cols_top = st.columns(4)
        with advanced_cols_top[0]:
            st.session_state["cura_pen_pause_seconds"] = float(
                st.number_input(
                    "Pen settle pause (seconds)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(st.session_state["cura_pen_pause_seconds"]),
                    step=0.05,
                    format="%.2f",
                    key="workflow_cura_pen_pause_seconds",
                )
            )
        with advanced_cols_top[1]:
            st.session_state["cura_fill_turn_split_angle_degrees"] = float(
                st.number_input(
                    "Fill turn split angle (degrees)",
                    min_value=5.0,
                    max_value=180.0,
                    value=float(st.session_state["cura_fill_turn_split_angle_degrees"]),
                    step=5.0,
                    format="%.1f",
                    key="workflow_cura_fill_turn_split_angle_degrees",
                )
            )
        with advanced_cols_top[2]:
            st.session_state["cura_continuous_fill_chunk_segments"] = int(
                st.number_input(
                    "Continuous fill chunk size (segments)",
                    min_value=0,
                    max_value=500,
                    value=int(st.session_state["cura_continuous_fill_chunk_segments"]),
                    step=4,
                    key="workflow_cura_continuous_fill_chunk_segments",
                )
            )
        with advanced_cols_top[3]:
            st.session_state["cura_coordinate_decimals"] = int(
                st.number_input(
                    "Coordinate decimals",
                    min_value=0,
                    max_value=4,
                    value=int(st.session_state["cura_coordinate_decimals"]),
                    step=1,
                    help="Lower precision shortens each G-code line and is useful for isolating parser/transport issues.",
                    key="workflow_cura_coordinate_decimals",
                )
            )

        advanced_cols_bottom = st.columns(3)
        with advanced_cols_bottom[0]:
            st.session_state["cura_path_simplify_tolerance_mm"] = float(
                st.number_input(
                    "Path simplification (mm)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(st.session_state["cura_path_simplify_tolerance_mm"]),
                    step=0.05,
                    format="%.2f",
                    key="workflow_cura_path_simplify_tolerance_mm",
                )
            )
        with advanced_cols_bottom[1]:
            st.session_state["cura_min_segment_length_mm"] = float(
                st.number_input(
                    "Minimum segment length (mm)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(st.session_state["cura_min_segment_length_mm"]),
                    step=0.05,
                    format="%.2f",
                    key="workflow_cura_min_segment_length_mm",
                )
            )
        with advanced_cols_bottom[2]:
            st.session_state["cura_min_toolpath_length_mm"] = float(
                st.number_input(
                    "Minimum whole toolpath length (mm)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(st.session_state["cura_min_toolpath_length_mm"]),
                    step=0.05,
                    format="%.2f",
                    key="workflow_cura_min_toolpath_length_mm",
                )
            )


def _ensure_machine_defaults() -> None:
    defaults_version_changed = st.session_state.get("machine_defaults_version") != MACHINE_DEFAULTS_VERSION
    if st.session_state.get("bridge_base_url") in (None, "", "http://esp32.local"):
        st.session_state["bridge_base_url"] = "http://10.0.0.90"
    if "bridge_command_path" not in st.session_state:
        st.session_state["bridge_command_path"] = "/command"
    if "bridge_realtime_path" not in st.session_state:
        st.session_state["bridge_realtime_path"] = "/realtime"
    if "bridge_status_path" not in st.session_state:
        st.session_state["bridge_status_path"] = "/status"
    if "bridge_clear_log_path" not in st.session_state:
        st.session_state["bridge_clear_log_path"] = "/clear-log"
    if "bridge_timeout" not in st.session_state:
        st.session_state["bridge_timeout"] = 8.0
    if "x_steps_per_mm" not in st.session_state:
        st.session_state["x_steps_per_mm"] = DEFAULT_X_STEPS_PER_MM
    if "y_steps_per_mm" not in st.session_state:
        st.session_state["y_steps_per_mm"] = DEFAULT_Y_STEPS_PER_MM
    if "jog_step_count" not in st.session_state:
        st.session_state["jog_step_count"] = 80
    if "jog_feed_rate" not in st.session_state:
        st.session_state["jog_feed_rate"] = 300.0
    if "diagnostic_xy_max_rate_mm_min" not in st.session_state:
        st.session_state["diagnostic_xy_max_rate_mm_min"] = DEFAULT_MOTION_MAX_RATE_MM_MIN
    if "diagnostic_xy_accel_mm_s2" not in st.session_state:
        st.session_state["diagnostic_xy_accel_mm_s2"] = DEFAULT_MOTION_ACCEL_MM_S2
    if "diagnostic_z_max_rate_mm_min" not in st.session_state:
        st.session_state["diagnostic_z_max_rate_mm_min"] = DEFAULT_MOTION_MAX_RATE_MM_MIN
    if "diagnostic_z_accel_mm_s2" not in st.session_state:
        st.session_state["diagnostic_z_accel_mm_s2"] = DEFAULT_MOTION_ACCEL_MM_S2
    if "motion_tuning_mode" not in st.session_state:
        st.session_state["motion_tuning_mode"] = "turbo"
    if "stepper_power_mode" not in st.session_state:
        st.session_state["stepper_power_mode"] = "deenergized"
    if "pen_reference_ready" not in st.session_state:
        st.session_state["pen_reference_ready"] = False
    if "grbl_homing_pulloff_mm" not in st.session_state:
        st.session_state["grbl_homing_pulloff_mm"] = 3.0
    if "pen_height_calibration_step_mm" not in st.session_state:
        st.session_state["pen_height_calibration_step_mm"] = 1.0

    valid_grbl_axes = {"X", "Y", "Z"}
    mapping_version_changed = st.session_state.get("manual_control_mapping_version") != MANUAL_CONTROL_MAPPING_VERSION
    for preset in MANUAL_AXIS_TUNING_PRESETS:
        axis_id = str(preset["id"])
        defaults = _manual_axis_defaults(preset)

        label_key = _manual_axis_state_key(axis_id, "label")
        grbl_axis_key = _manual_axis_state_key(axis_id, "grbl_axis")
        feed_rate_key = _manual_axis_state_key(axis_id, "feed_rate_mm_min")
        direction_key = _manual_axis_state_key(axis_id, "direction_multiplier")

        if mapping_version_changed or label_key not in st.session_state:
            st.session_state[label_key] = str(defaults["label"])
        if (
            mapping_version_changed
            or grbl_axis_key not in st.session_state
            or str(st.session_state[grbl_axis_key]).upper() not in valid_grbl_axes
        ):
            st.session_state[grbl_axis_key] = str(defaults["grbl_axis"])
        if mapping_version_changed or feed_rate_key not in st.session_state:
            st.session_state[feed_rate_key] = float(defaults["feed_rate_mm_min"])
        try:
            direction_value = int(st.session_state.get(direction_key, defaults["direction_multiplier"]))
        except (TypeError, ValueError):
            direction_value = int(defaults["direction_multiplier"])
        if mapping_version_changed or direction_key not in st.session_state or direction_value not in {-1, 1}:
            st.session_state[direction_key] = int(defaults["direction_multiplier"])

        if "move_amount_mm" in defaults:
            move_amount_key = _manual_axis_state_key(axis_id, "move_amount_mm")
            if mapping_version_changed or move_amount_key not in st.session_state:
                st.session_state[move_amount_key] = float(defaults["move_amount_mm"])
        if "positive_move_amount_mm" in defaults:
            positive_move_key = _manual_axis_state_key(axis_id, "positive_move_amount_mm")
            if mapping_version_changed or positive_move_key not in st.session_state:
                st.session_state[positive_move_key] = float(defaults["positive_move_amount_mm"])
        if "negative_move_amount_mm" in defaults:
            negative_move_key = _manual_axis_state_key(axis_id, "negative_move_amount_mm")
            if mapping_version_changed or negative_move_key not in st.session_state:
                st.session_state[negative_move_key] = float(defaults["negative_move_amount_mm"])

    if "pen_height_saved_up_position_mm" not in st.session_state:
        st.session_state["pen_height_saved_up_position_mm"] = 20.0
    if "pen_height_saved_down_position_mm" not in st.session_state:
        st.session_state["pen_height_saved_down_position_mm"] = 23.0
    if "pen_up_dwell_seconds" not in st.session_state:
        st.session_state["pen_up_dwell_seconds"] = 0.0
    if "pen_down_dwell_seconds" not in st.session_state:
        st.session_state["pen_down_dwell_seconds"] = 0.0
    if "pen_height_auto_clearance_mm" not in st.session_state:
        st.session_state["pen_height_auto_clearance_mm"] = DEFAULT_PEN_UP_GAP_MM
    if "pen_height_workflow_confirmed" not in st.session_state:
        st.session_state["pen_height_workflow_confirmed"] = False
    if "pen_height_workflow_feedback" not in st.session_state:
        st.session_state["pen_height_workflow_feedback"] = ""

    if defaults_version_changed:
        pen_lift_feed_key = _manual_axis_state_key("pen_lift", "feed_rate_mm_min")
        if float(st.session_state.get(pen_lift_feed_key, DEFAULT_MOTION_MAX_RATE_MM_MIN)) in {
            900.0,
            1800.0,
            2000.0,
            6000.0,
        }:
            st.session_state[pen_lift_feed_key] = DEFAULT_MOTION_MAX_RATE_MM_MIN
        for axis_id in ("draw_x", "draw_y"):
            draw_axis_feed_key = _manual_axis_state_key(axis_id, "feed_rate_mm_min")
            if float(st.session_state.get(draw_axis_feed_key, DEFAULT_MOTION_MAX_RATE_MM_MIN)) in {
                900.0,
                6000.0,
            }:
                st.session_state[draw_axis_feed_key] = DEFAULT_MOTION_MAX_RATE_MM_MIN
        if float(st.session_state.get("diagnostic_xy_max_rate_mm_min", DEFAULT_MOTION_MAX_RATE_MM_MIN)) in {
            2500.0,
            5400.0,
            6000.0,
        }:
            st.session_state["diagnostic_xy_max_rate_mm_min"] = DEFAULT_MOTION_MAX_RATE_MM_MIN
        if float(st.session_state.get("diagnostic_xy_accel_mm_s2", DEFAULT_MOTION_ACCEL_MM_S2)) in {
            80.0,
            160.0,
            200.0,
        }:
            st.session_state["diagnostic_xy_accel_mm_s2"] = DEFAULT_MOTION_ACCEL_MM_S2
        if float(st.session_state.get("diagnostic_z_max_rate_mm_min", DEFAULT_MOTION_MAX_RATE_MM_MIN)) in {
            900.0,
            1200.0,
            1800.0,
            2400.0,
            6000.0,
        }:
            st.session_state["diagnostic_z_max_rate_mm_min"] = DEFAULT_MOTION_MAX_RATE_MM_MIN
        if float(st.session_state.get("diagnostic_z_accel_mm_s2", DEFAULT_MOTION_ACCEL_MM_S2)) in {
            10.0,
            20.0,
            35.0,
            60.0,
            200.0,
        }:
            st.session_state["diagnostic_z_accel_mm_s2"] = DEFAULT_MOTION_ACCEL_MM_S2
        if abs(float(st.session_state.get("pen_down_dwell_seconds", 0.0)) - 0.02) < 1e-9:
            st.session_state["pen_down_dwell_seconds"] = 0.0
        if abs(float(st.session_state.get("pen_height_saved_up_position_mm", 20.0)) - 19.0) < 1e-9:
            st.session_state["pen_height_saved_up_position_mm"] = 20.0
        if abs(float(st.session_state.get("pen_height_saved_down_position_mm", 23.0)) - 22.0) < 1e-9:
            st.session_state["pen_height_saved_down_position_mm"] = 23.0
        if abs(float(st.session_state.get("bridge_timeout", 8.0)) - 2.0) < 1e-9:
            st.session_state["bridge_timeout"] = 8.0
        if abs(float(st.session_state.get("bridge_timeout", 8.0)) - 6.0) < 1e-9:
            st.session_state["bridge_timeout"] = 8.0
        if str(st.session_state.get("motion_tuning_mode", "")).strip().lower() not in {"chill", "turbo"}:
            st.session_state["motion_tuning_mode"] = "turbo"
        if str(st.session_state.get("stepper_power_mode", "")).strip().lower() not in {"energized", "deenergized"}:
            st.session_state["stepper_power_mode"] = "deenergized"
        if str(st.session_state.get("bridge_base_url", "")).strip().rstrip("/") in {
            "http://esp32-grbl-bridge.local",
            "http://10.0.0.89",
        }:
            st.session_state["bridge_base_url"] = "http://10.0.0.90"
        current_pen_gap = float(st.session_state.get("pen_height_auto_clearance_mm", DEFAULT_PEN_UP_GAP_MM))
        if any(abs(current_pen_gap - legacy_gap) < 1e-9 for legacy_gap in (3.0, 5.0)):
            st.session_state["pen_height_auto_clearance_mm"] = DEFAULT_PEN_UP_GAP_MM
        current_pen_gap_input = float(
            st.session_state.get("pen_height_auto_clearance_mm_input", DEFAULT_PEN_UP_GAP_MM)
        )
        if any(abs(current_pen_gap_input - legacy_gap) < 1e-9 for legacy_gap in (3.0, 5.0)):
            st.session_state["pen_height_auto_clearance_mm_input"] = DEFAULT_PEN_UP_GAP_MM

    st.session_state["manual_control_mapping_version"] = MANUAL_CONTROL_MAPPING_VERSION
    st.session_state["machine_defaults_version"] = MACHINE_DEFAULTS_VERSION


def _bridge_discovery_candidates(current_base_url: str | None = None) -> list[str]:
    candidates: list[str] = []
    for candidate in (*BRIDGE_DISCOVERY_CANDIDATES, current_base_url):
        if candidate is None:
            continue
        normalized = BridgeSettings(base_url=str(candidate)).normalized_base_url
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _probe_bridge_base_url(current_base_url: str | None = None) -> str | None:
    candidates = _bridge_discovery_candidates(current_base_url)
    for candidate in candidates:
        probe_settings = BridgeSettings(
            base_url=candidate,
            command_path=st.session_state.get("bridge_command_path", "/command"),
            realtime_path=st.session_state.get("bridge_realtime_path", "/realtime"),
            status_path=st.session_state.get("bridge_status_path", "/status"),
            clear_log_path=st.session_state.get("bridge_clear_log_path", "/clear-log"),
            timeout_seconds=BRIDGE_DISCOVERY_TIMEOUT_SECONDS,
        )
        response, snapshot = fetch_bridge_status_snapshot(probe_settings, request_fresh_status=False)
        if response.ok and snapshot is not None:
            return probe_settings.normalized_base_url
    return None


def _auto_detect_bridge_base_url(*, force: bool = False) -> bool:
    now = time.monotonic()
    last_checked = float(st.session_state.get("bridge_discovery_checked_at", 0.0))
    if not force and now - last_checked < BRIDGE_DISCOVERY_INTERVAL_SECONDS:
        return False

    current_base_url = str(st.session_state.get("bridge_base_url", "")).strip()
    candidates = _bridge_discovery_candidates(current_base_url)
    st.session_state["bridge_discovery_checked_at"] = now

    detected_url = _probe_bridge_base_url(current_base_url)
    if detected_url is not None:
        st.session_state["bridge_base_url"] = detected_url
        st.session_state["bridge_discovery_message"] = f"Detected ESP32 bridge at {detected_url}."
        st.session_state["bridge_discovery_ok"] = True
        return True

    st.session_state["bridge_discovery_message"] = (
        "Could not auto-detect the ESP32 bridge at "
        + ", ".join(_bridge_discovery_candidates(current_base_url))
        + "."
    )
    st.session_state["bridge_discovery_ok"] = False
    return False


def _looks_like_bridge_connection_failure(response: BridgeResponse) -> bool:
    if response.ok:
        return False
    error_text = (response.error or response.response_text or "").strip().lower()
    return any(hint in error_text for hint in BRIDGE_TRANSIENT_ERROR_HINTS)


def send_grbl_command(
    settings: BridgeSettings,
    command: str,
    *,
    http_session=None,
) -> BridgeResponse:
    response = _send_grbl_command_direct(settings, command, http_session=http_session)
    if response.ok or http_session is not None or not _looks_like_bridge_connection_failure(response):
        return response

    original_url = settings.normalized_base_url
    detected_url = _probe_bridge_base_url(original_url)
    if detected_url is None:
        return response

    recovered_settings = BridgeSettings(
        base_url=detected_url,
        command_path=settings.command_path,
        realtime_path=settings.realtime_path,
        status_path=settings.status_path,
        clear_log_path=settings.clear_log_path,
        servo_move_path=settings.servo_move_path,
        servo_status_path=settings.servo_status_path,
        command_field=settings.command_field,
        realtime_field=settings.realtime_field,
        timeout_seconds=settings.timeout_seconds,
    )
    if recovered_settings.normalized_base_url == original_url:
        return response

    retry_response = _send_grbl_command_direct(recovered_settings, command)
    if retry_response.ok:
        st.session_state["bridge_pending_base_url"] = recovered_settings.normalized_base_url
        st.session_state["bridge_discovery_message"] = (
            f"Recovered ESP32 bridge at {recovered_settings.normalized_base_url}."
        )
        st.session_state["bridge_discovery_ok"] = True
        st.session_state["machine_last_status_response"] = BridgeResponse(
            ok=True,
            response_text=f"Recovered bridge connection at {recovered_settings.normalized_base_url}.",
        )
    return retry_response


def _build_bridge_settings_from_session() -> BridgeSettings:
    _ensure_machine_defaults()
    return BridgeSettings(
        base_url=st.session_state["bridge_base_url"],
        command_path=st.session_state["bridge_command_path"],
        realtime_path=st.session_state["bridge_realtime_path"],
        status_path=st.session_state["bridge_status_path"],
        clear_log_path=st.session_state["bridge_clear_log_path"],
        timeout_seconds=float(st.session_state["bridge_timeout"]),
    )


def _manual_axis_preset(axis_id: str) -> dict[str, object]:
    for preset in MANUAL_AXIS_TUNING_PRESETS:
        if str(preset["id"]) == axis_id:
            return preset
    raise KeyError(f"Unknown manual axis preset: {axis_id}")


def _pen_motion_defaults_from_manual_axis() -> tuple[str, float, float, float]:
    pen_preset = _manual_axis_preset("pen_lift")
    pen_axis = _manual_axis_grbl_axis("pen_lift", str(pen_preset["default_grbl_axis"]))
    pen_up_travel_mm = _manual_axis_move_amount("pen_lift", pen_preset, 1)
    pen_down_position_mm = _manual_axis_move_amount("pen_lift", pen_preset, -1)
    pen_up_position_mm = max(0.0, pen_down_position_mm - pen_up_travel_mm)
    feed_rate_mm_min = max(
        float(
            st.session_state.get(
                _manual_axis_state_key("pen_lift", "feed_rate_mm_min"),
                float(pen_preset["default_feed_rate_mm_min"]),
            )
        ),
        1.0,
    )
    return pen_axis, pen_up_position_mm, pen_down_position_mm, feed_rate_mm_min


def _current_pen_motion_settings() -> PenMotionSettings:
    pen_axis, default_pen_up_position_mm, default_pen_down_position_mm, feed_rate_mm_min = (
        _pen_motion_defaults_from_manual_axis()
    )
    pen_up_position_mm = float(
        st.session_state.get("pen_height_saved_up_position_mm", default_pen_up_position_mm)
    )
    pen_down_position_mm = float(
        st.session_state.get("pen_height_saved_down_position_mm", default_pen_down_position_mm)
    )
    down_direction_sign = _pen_down_work_direction_sign()
    if (pen_down_position_mm - pen_up_position_mm) * down_direction_sign <= 0:
        pen_down_position_mm = pen_up_position_mm + (down_direction_sign * 0.001)
    return PenMotionSettings(
        axis=pen_axis,
        pen_up_position_mm=pen_up_position_mm,
        pen_down_position_mm=pen_down_position_mm,
        feed_rate_mm_min=feed_rate_mm_min,
        pen_up_dwell_seconds=max(float(st.session_state.get("pen_up_dwell_seconds", 0.0)), 0.0),
        pen_down_dwell_seconds=max(float(st.session_state.get("pen_down_dwell_seconds", 0.02)), 0.0),
    )


def _current_grbl_homing_pulloff_mm() -> float:
    return max(float(st.session_state.get("grbl_homing_pulloff_mm", 3.0)), 0.0)


def _grbl_axis_position(status, axis: str) -> float | None:
    if status is None:
        return None
    normalized_axis = axis.upper()
    if normalized_axis not in GRBL_AXIS_ORDER:
        return None
    if status.work_position is not None:
        return float(status.work_position[GRBL_AXIS_ORDER.index(normalized_axis)])
    if status.machine_position is not None:
        return float(status.machine_position[GRBL_AXIS_ORDER.index(normalized_axis)])
    return None


def _grbl_axis_work_position(status, axis: str) -> float | None:
    if status is None or status.work_position is None:
        return None
    normalized_axis = axis.upper()
    if normalized_axis not in GRBL_AXIS_ORDER:
        return None
    return float(status.work_position[GRBL_AXIS_ORDER.index(normalized_axis)])


def _tracked_pen_work_position_mm(status=None) -> float | None:
    pen_motion_settings = _current_pen_motion_settings()
    live_work_position_mm = _grbl_axis_work_position(status, pen_motion_settings.axis)
    if live_work_position_mm is not None:
        st.session_state["pen_calibration_work_position_mm"] = float(live_work_position_mm)
        return float(live_work_position_mm)

    tracked_value = st.session_state.get("pen_calibration_work_position_mm")
    if tracked_value is None:
        return None
    try:
        return float(tracked_value)
    except (TypeError, ValueError):
        return None


def _set_tracked_pen_work_position_mm(position_mm: float) -> None:
    st.session_state["pen_calibration_work_position_mm"] = float(position_mm)


def _pen_depth_from_home_mm(position_mm: float | None) -> float | None:
    if position_mm is None:
        return None
    return abs(float(position_mm))


def _pen_down_work_direction_sign() -> float:
    direction_multiplier = _manual_axis_direction_multiplier("pen_lift")
    return -1.0 if direction_multiplier > 0 else 1.0


def _clear_pen_calibration(message: str | None = None) -> None:
    st.session_state["pen_height_workflow_confirmed"] = False
    if message is not None:
        st.session_state["pen_height_workflow_feedback"] = message


def _update_pen_height_calibration(
    *,
    pen_up_position_mm: float | None = None,
    pen_down_position_mm: float | None = None,
) -> tuple[bool, str, dict[str, float] | None]:
    current_pen_settings = _current_pen_motion_settings()
    next_pen_up = (
        current_pen_settings.pen_up_position_mm
        if pen_up_position_mm is None
        else float(pen_up_position_mm)
    )
    next_pen_down = (
        current_pen_settings.pen_down_position_mm
        if pen_down_position_mm is None
        else float(pen_down_position_mm)
    )

    down_direction_sign = _pen_down_work_direction_sign()
    stroke_travel_mm = (next_pen_down - next_pen_up) * down_direction_sign
    if stroke_travel_mm <= 0:
        return False, "Pen-down must be farther toward the paper than pen-up.", None

    return (
        True,
        (
            f"Saved pen-up at {next_pen_up:.3f} mm and pen-down at {next_pen_down:.3f} mm "
            f"for GRBL {current_pen_settings.axis.upper()} ({stroke_travel_mm:.3f} mm lift)."
        ),
        {
            "pen_height_saved_up_position_mm": next_pen_up,
            "pen_height_saved_down_position_mm": next_pen_down,
        },
    )


def _apply_pending_pen_height_calibration() -> None:
    pending_update = st.session_state.pop("pending_pen_height_calibration", None)
    if not isinstance(pending_update, dict):
        return

    pen_up_position_mm = pending_update.get("pen_height_saved_up_position_mm")
    pen_down_position_mm = pending_update.get("pen_height_saved_down_position_mm")
    if pen_up_position_mm is not None:
        st.session_state["pen_height_saved_up_position_mm"] = float(pen_up_position_mm)
    if pen_down_position_mm is not None:
        st.session_state["pen_height_saved_down_position_mm"] = float(pen_down_position_mm)

    # Backward compatibility for any older pending payload still stored in a live session.
    positive_move_amount_mm = pending_update.get("positive_move_amount_mm")
    negative_move_amount_mm = pending_update.get("negative_move_amount_mm")
    if pen_up_position_mm is None and pen_down_position_mm is None:
        if positive_move_amount_mm is not None:
            st.session_state[_manual_axis_state_key("pen_lift", "positive_move_amount_mm")] = float(positive_move_amount_mm)
        if negative_move_amount_mm is not None:
            st.session_state[_manual_axis_state_key("pen_lift", "negative_move_amount_mm")] = float(negative_move_amount_mm)


def _queue_pen_height_feedback(kind: str, message: str) -> None:
    st.session_state["pen_height_calibration_feedback"] = {"kind": kind, "message": message}


def _load_motion_tuning_profile(profile_name: str) -> str:
    profile = MOTION_TUNING_PROFILES[profile_name]
    st.session_state["diagnostic_xy_max_rate_mm_min"] = float(profile["xy_max_rate_mm_min"])
    st.session_state["diagnostic_xy_accel_mm_s2"] = float(profile["xy_accel_mm_s2"])
    st.session_state["diagnostic_z_max_rate_mm_min"] = float(profile["z_max_rate_mm_min"])
    st.session_state["diagnostic_z_accel_mm_s2"] = float(profile["z_accel_mm_s2"])
    return str(profile["label"])


def _motion_tuning_commands() -> list[tuple[str, str]]:
    return [
        ("Set X max rate", f"$110={float(st.session_state['diagnostic_xy_max_rate_mm_min']):.0f}"),
        ("Set Y max rate", f"$111={float(st.session_state['diagnostic_xy_max_rate_mm_min']):.0f}"),
        ("Set Z max rate", f"$112={float(st.session_state['diagnostic_z_max_rate_mm_min']):.0f}"),
        ("Set X acceleration", f"$120={float(st.session_state['diagnostic_xy_accel_mm_s2']):.0f}"),
        ("Set Y acceleration", f"$121={float(st.session_state['diagnostic_xy_accel_mm_s2']):.0f}"),
        ("Set Z acceleration", f"$122={float(st.session_state['diagnostic_z_accel_mm_s2']):.0f}"),
    ]


def _send_stepper_power_sequence(
    bridge_settings: BridgeSettings,
    commands: tuple[tuple[str, str], ...],
) -> bool:
    last_response = None
    for label, command in commands:
        last_response = send_grbl_command(bridge_settings, command)
        _store_machine_command_response(last_response, label)
        if not last_response.ok:
            return False
    if last_response is not None and last_response.ok:
        _refresh_machine_snapshot(bridge_settings, request_fresh_status=True)
        return True
    return False


def _apply_motion_tuning_profile_to_grbl(
    bridge_settings: BridgeSettings,
    profile_name: str,
) -> bool:
    if profile_name not in MOTION_TUNING_PROFILES:
        return False

    _load_motion_tuning_profile(profile_name)
    last_response = None
    for label, command in _motion_tuning_commands():
        last_response = send_grbl_command(bridge_settings, command)
        _store_machine_command_response(last_response, label)
        if not last_response.ok:
            return False

    _refresh_machine_snapshot(bridge_settings, request_fresh_status=True)
    return True


def _apply_stepper_power_mode(
    bridge_settings: BridgeSettings,
    mode: str,
) -> bool:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "energized":
        response = send_grbl_command(bridge_settings, "$1=255")
        _store_machine_command_response(response, "Energize steppers")
        if response.ok:
            st.session_state["stepper_power_mode"] = "energized"
            _refresh_machine_snapshot(bridge_settings, request_fresh_status=True)
            return True
        return False

    if normalized_mode == "deenergized":
        hold_response = send_grbl_command(bridge_settings, "!")
        _store_machine_command_response(hold_response, "Feed hold before de-energize")
        response = send_grbl_command(bridge_settings, "$1=0")
        _store_machine_command_response(response, "De-energize steppers")
        if response.ok:
            st.session_state["stepper_power_mode"] = "deenergized"
            time.sleep(0.15)
            _refresh_machine_snapshot(bridge_settings, request_fresh_status=True)
            return True
        return False

    return False


def _send_pen_motion_command(
    bridge_settings: BridgeSettings,
    pen_motion_settings: PenMotionSettings,
    position_name: str,
    response_label: str,
):
    command = build_pen_position_command(pen_motion_settings, position_name)
    response = send_grbl_command(bridge_settings, command)
    _store_machine_command_response(response, response_label)
    if response.ok:
        _refresh_machine_snapshot_until_settled(bridge_settings)
    return response


def _sync_pen_reference_coordinate(
    bridge_settings: BridgeSettings,
    pen_motion_settings: PenMotionSettings,
) -> bool:
    homing_pulloff_mm = _current_grbl_homing_pulloff_mm()
    sync_commands = (
        ("Absolute positioning", "G90"),
        (
            "Sync pen home reference",
            f"G92 {pen_motion_settings.axis}{homing_pulloff_mm:.3f}",
        ),
    )

    for label, command in sync_commands:
        response = send_grbl_command(bridge_settings, command)
        _store_machine_command_response(response, label)
        if not response.ok:
            st.session_state["pen_reference_ready"] = False
            return False

    _refresh_machine_snapshot(bridge_settings, request_fresh_status=True)
    st.session_state["pen_reference_ready"] = True
    _set_tracked_pen_work_position_mm(homing_pulloff_mm)
    return True


def _sync_draw_page_coordinates(
    bridge_settings: BridgeSettings,
    *,
    page_width_mm: float,
    page_height_mm: float,
    pen_motion_settings: PenMotionSettings,
) -> bool:
    homing_pulloff_mm = _current_grbl_homing_pulloff_mm()
    sync_commands = (
        ("Absolute positioning", "G90"),
        (
            "Sync page coordinates",
            (
                f"G92 X{page_width_mm:.3f} "
                f"Y{page_height_mm:.3f} "
                f"{pen_motion_settings.axis}{homing_pulloff_mm:.3f}"
            ),
        ),
    )

    for label, command in sync_commands:
        response = send_grbl_command(bridge_settings, command)
        _store_machine_command_response(response, label)
        if not response.ok:
            st.session_state["pen_reference_ready"] = False
            return False

    _refresh_machine_snapshot(bridge_settings, request_fresh_status=True)
    st.session_state["pen_reference_ready"] = True
    _set_tracked_pen_work_position_mm(homing_pulloff_mm)
    return True


def _render_pre_slice_pen_calibration(*, page_width_mm: float, page_height_mm: float) -> bool:
    _ensure_machine_defaults()
    bridge_settings = _build_bridge_settings_from_session()
    pen_motion_settings = _current_pen_motion_settings()
    calibration_confirmed = bool(st.session_state.get("pen_height_workflow_confirmed", False))
    current_clearance_mm = float(st.session_state.get("pen_height_auto_clearance_mm", DEFAULT_PEN_UP_GAP_MM))

    st.subheader("1. Calibrate Pen Height")
    st.caption(
        "Do this before generating G-code: home the machine, lower the pen until it just touches the paper, "
        "choose the lift gap, then confirm. The draw will start by lifting that exact gap before moving."
    )

    status = None
    snapshot = st.session_state.get("machine_snapshot")
    if snapshot is not None:
        status = snapshot.grbl_status
    tracked_z_mm = _tracked_pen_work_position_mm(status)

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Bridge", bridge_settings.normalized_base_url or "Not set")
    with metric_cols[1]:
        st.metric("Tracked pen Z", "Home first" if tracked_z_mm is None else f"{tracked_z_mm:.3f} mm")
    with metric_cols[2]:
        st.metric("Pen-up gap", f"{current_clearance_mm:.1f} mm")

    button_cols = st.columns([1.0, 0.9, 0.75, 0.75, 0.75, 0.75, 0.85, 0.85, 1.45])
    with button_cols[0]:
        home_all_requested = st.button("Home All", key="pre_slice_home_all", use_container_width=True)
    with button_cols[1]:
        home_pen_requested = st.button("Home Pen", key="pre_slice_home_pen", use_container_width=True)
    with button_cols[2]:
        down_10_requested = st.button("Down 10", key="pre_slice_pen_down_10", use_container_width=True)
    with button_cols[3]:
        down_5_requested = st.button("Down 5", key="pre_slice_pen_down_5", use_container_width=True)
    with button_cols[4]:
        down_1_requested = st.button("Down 1", key="pre_slice_pen_down_1", use_container_width=True)
    with button_cols[5]:
        up_1_requested = st.button("Up 1", key="pre_slice_pen_up_1", use_container_width=True)
    with button_cols[6]:
        gap_8_requested = st.button(
            "Up Gap 8",
            key="pre_slice_pen_gap_8",
            type="primary" if abs(current_clearance_mm - DEFAULT_PEN_UP_GAP_MM) < 1e-9 else "secondary",
            use_container_width=True,
        )
    with button_cols[7]:
        gap_5_requested = st.button(
            "Up Gap 5",
            key="pre_slice_pen_gap_5",
            type="primary" if abs(current_clearance_mm - SECONDARY_PEN_UP_GAP_MM) < 1e-9 else "secondary",
            use_container_width=True,
        )
    with button_cols[8]:
        confirm_requested = st.button(
            "Confirm Pen Down Calibration",
            key="pre_slice_confirm_pen_down",
            use_container_width=True,
        )

    if home_all_requested:
        _clear_pen_calibration("Homed all axes. Lower the pen to paper contact, choose a gap, then confirm.")
        response = send_grbl_command(bridge_settings, "$H")
        _store_machine_command_response(response, "Home all axes")
        if response.ok:
            _refresh_machine_snapshot_until_settled(bridge_settings)
            _sync_draw_page_coordinates(
                bridge_settings,
                page_width_mm=page_width_mm,
                page_height_mm=page_height_mm,
                pen_motion_settings=pen_motion_settings,
            )
        else:
            _clear_pen_calibration("Home All failed. Check the GRBL log/bridge connection, then try again.")
        st.rerun()

    if home_pen_requested:
        _clear_pen_calibration("Homed the pen axis. Lower the pen to paper contact, choose a gap, then confirm.")
        response = send_grbl_command(bridge_settings, f"$H{pen_motion_settings.axis.upper()}")
        _store_machine_command_response(response, f"Home pen axis ({pen_motion_settings.axis.upper()})")
        if response.ok:
            _refresh_machine_snapshot_until_settled(bridge_settings)
            _sync_draw_page_coordinates(
                bridge_settings,
                page_width_mm=page_width_mm,
                page_height_mm=page_height_mm,
                pen_motion_settings=pen_motion_settings,
            )
        else:
            _clear_pen_calibration("Home Pen failed. Check the GRBL log/bridge connection, then try again.")
        st.rerun()

    requested_jog_mm = 0.0
    requested_jog_label = ""
    if down_10_requested:
        requested_jog_mm = 10.0
        requested_jog_label = "Pen down 10 mm"
    elif down_5_requested:
        requested_jog_mm = 5.0
        requested_jog_label = "Pen down 5 mm"
    elif down_1_requested:
        requested_jog_mm = 1.0
        requested_jog_label = "Pen down 1 mm"
    elif up_1_requested:
        requested_jog_mm = -1.0
        requested_jog_label = "Pen up 1 mm"

    if requested_jog_mm:
        if _tracked_pen_work_position_mm(status) is None:
            _clear_pen_calibration("Home All first so the app can track the pen Z coordinate reliably.")
            st.rerun()

        direction_multiplier = _manual_axis_direction_multiplier("pen_lift")
        signed_distance_mm = -requested_jog_mm * direction_multiplier
        jog_command = build_jog_command(
            pen_motion_settings.axis,
            signed_distance_mm,
            pen_motion_settings.feed_rate_mm_min,
        )
        _clear_pen_calibration("Pen position changed. Confirm the pen-down point before generating G-code.")
        response = send_grbl_command(bridge_settings, jog_command)
        _store_machine_command_response(response, requested_jog_label)
        if response.ok:
            current_work_position = _tracked_pen_work_position_mm(status)
            if current_work_position is not None:
                _set_tracked_pen_work_position_mm(current_work_position + signed_distance_mm)
            _refresh_machine_snapshot_until_settled(bridge_settings)
        else:
            _clear_pen_calibration(f"{requested_jog_label} failed. Check the GRBL log, then try again.")
        st.rerun()

    if gap_8_requested or gap_5_requested:
        selected_gap_mm = DEFAULT_PEN_UP_GAP_MM if gap_8_requested else SECONDARY_PEN_UP_GAP_MM
        st.session_state["pen_height_auto_clearance_mm"] = selected_gap_mm
        _clear_pen_calibration(
            f"Pen-up clearance set to {selected_gap_mm:.1f} mm. Confirm the current pen-down touch point."
        )
        st.rerun()

    if confirm_requested:
        if not bool(st.session_state.get("pen_reference_ready", False)):
            _clear_pen_calibration("Home All first so calibration uses a fresh drawing reference.")
            st.rerun()

        _refresh_machine_snapshot_until_settled(
            bridge_settings,
            timeout_seconds=5.0,
            poll_interval_seconds=0.15,
        )
        latest_snapshot = st.session_state.get("machine_snapshot")
        latest_status = None if latest_snapshot is None else latest_snapshot.grbl_status
        pen_down_position_mm = _tracked_pen_work_position_mm(latest_status)
        if pen_down_position_mm is None:
            _clear_pen_calibration(
                "Could not read a reliable pen work-coordinate. Click Home All, then jog down using these buttons."
            )
            st.rerun()

        clearance_mm = float(st.session_state.get("pen_height_auto_clearance_mm", DEFAULT_PEN_UP_GAP_MM))
        down_direction_sign = _pen_down_work_direction_sign()
        ok, message, pending_update = _update_pen_height_calibration(
            pen_up_position_mm=float(pen_down_position_mm) - (down_direction_sign * clearance_mm),
            pen_down_position_mm=float(pen_down_position_mm),
        )
        if ok and isinstance(pending_update, dict):
            st.session_state["pen_height_saved_up_position_mm"] = float(
                pending_update["pen_height_saved_up_position_mm"]
            )
            st.session_state["pen_height_saved_down_position_mm"] = float(
                pending_update["pen_height_saved_down_position_mm"]
            )
            st.session_state["pen_height_workflow_confirmed"] = True
            st.session_state["pen_height_workflow_feedback"] = (
                f"{message} The pen is currently at the down position; drawing will lift "
                f"{clearance_mm:.1f} mm before the first travel move."
            )
        else:
            st.session_state["pen_height_workflow_confirmed"] = False
            st.session_state["pen_height_workflow_feedback"] = message
        st.rerun()

    feedback_message = str(st.session_state.get("pen_height_workflow_feedback", "")).strip()
    if calibration_confirmed:
        st.success(
            feedback_message
            or "Pen-down calibration is confirmed. You can generate G-code when the artwork is ready."
        )
    else:
        st.warning(
            feedback_message
            or "Generate G-code is locked until the pen-down point is confirmed."
        )

    return bool(st.session_state.get("pen_height_workflow_confirmed", False))


def main() -> None:
    st.set_page_config(page_title="Photo to G-code", layout="wide")
    _apply_app_theme()
    title_cols = st.columns([5.0, 1.0])
    with title_cols[0]:
        st.title("Photo to G-code")
    with title_cols[1]:
        st.markdown(
            f"""
            <div style="display:flex;justify-content:flex-end;margin-top:0.7rem;">
              <div style="border:1px solid rgba(246,174,45,0.45);border-radius:10px;padding:0.35rem 0.7rem;color:#f6ae2d;font-weight:900;background:rgba(15,23,42,0.72);">
                {APP_VERSION}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Upload artwork, place it on the page, generate the real plotter path, then home, calibrate, and draw."
    )
    upload_cols = st.columns([1.2, 2.6, 1.2])
    with upload_cols[1]:
        uploaded_file = st.file_uploader(
            "Add Image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="visible",
            key="workflow_uploaded_image",
        )

    render_cura_mode(uploaded_file)


def render_cura_mode(uploaded_file) -> None:
    if uploaded_file is None:
        st.markdown(
            """
            <div style="max-width:720px;margin:2.5rem auto 0 auto;padding:2rem 2.25rem;border:1px solid rgba(148,163,184,0.18);border-radius:24px;background:rgba(15,23,42,0.78);box-shadow:0 20px 60px rgba(0,0,0,0.22);text-align:center;">
              <div style="font-size:1.5rem;font-weight:700;color:#e5eefc;margin-bottom:0.6rem;">Add Image</div>
              <div style="color:#9fb0c8;font-size:1rem;">Start by choosing one picture. The next screen will let you place it on the paper, generate the actual plotter path, then set up the machine and draw.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    uploaded_bytes, active_image_kind = _sync_workflow_source_image(uploaded_file)
    active_image_name = str(st.session_state.get("workflow_active_image_name", uploaded_file.name))
    ai_image_available = "workflow_ai_image_bytes" in st.session_state

    ai_cols = st.columns([1.1, 2.0, 1.1, 1.3, 2.3], gap="small")
    openai_api_key = _configured_openai_api_key()
    with ai_cols[0]:
        convert_to_ai_requested = st.button("Convert to AI", key="workflow_convert_to_ai", use_container_width=True)
    with ai_cols[1]:
        ai_additional_comments = st.text_input(
            "Additional comments?",
            key="workflow_ai_additional_comments",
            placeholder="e.g. add a dog next to the car",
        )
    with ai_cols[2]:
        revert_to_original_requested = st.button(
            "Use Original",
            key="workflow_use_original_image",
            use_container_width=True,
            disabled=(active_image_kind == "original"),
        )
    with ai_cols[3]:
        save_ai_to_desktop_requested = st.button(
            "Save AI to Desktop",
            key="workflow_save_ai_to_desktop",
            use_container_width=True,
            disabled=not ai_image_available,
        )
    with ai_cols[4]:
        source_label = "AI refined" if active_image_kind == "ai" else "Original upload"
        st.caption(
            f"Active source: `{source_label}` (`{active_image_name}`). "
            "Convert to AI uses OpenAI image editing with a plotter-friendly prompt plus your extra comments. "
            f"OpenAI key: `{'detected' if openai_api_key else 'missing'}`."
        )

    ai_feedback = st.session_state.get("workflow_ai_feedback")
    if isinstance(ai_feedback, dict):
        feedback_message = str(ai_feedback.get("message", "")).strip()
        if feedback_message:
            if str(ai_feedback.get("kind", "info")) == "error":
                st.error(feedback_message)
            else:
                st.success(feedback_message)

    if convert_to_ai_requested:
        if not openai_api_key:
            st.session_state["workflow_ai_feedback"] = {
                "kind": "error",
                "message": "Set OPENAI_API_KEY in your environment or .streamlit/secrets.toml before using Convert to AI.",
            }
            st.rerun()

        with st.spinner("Sending the image to OpenAI to create a cleaner plotter-friendly version..."):
            ai_result = convert_image_to_plotter_friendly_ai(
                st.session_state["workflow_source_image_bytes"]
                if "workflow_source_image_bytes" in st.session_state
                else uploaded_file.getvalue(),
                api_key=openai_api_key,
                prompt=_build_plotter_ai_prompt(ai_additional_comments),
            )
        if ai_result.ok and ai_result.image_bytes is not None:
            file_stem = Path(uploaded_file.name).stem
            ai_name = f"{file_stem}_ai.png"
            st.session_state["workflow_ai_image_bytes"] = ai_result.image_bytes
            st.session_state["workflow_ai_image_name"] = ai_name
            st.session_state["workflow_ai_revised_prompt"] = ai_result.revised_prompt
            st.session_state["workflow_active_image_bytes"] = ai_result.image_bytes
            st.session_state["workflow_active_image_name"] = ai_name
            st.session_state["workflow_active_image_kind"] = "ai"
            st.session_state["workflow_ai_feedback"] = {
                "kind": "success",
                "message": "AI conversion finished. You are now editing the AI-refined image.",
            }
        else:
            st.session_state["workflow_ai_feedback"] = {
                "kind": "error",
                "message": ai_result.error or "OpenAI image conversion failed.",
            }
        st.rerun()

    if revert_to_original_requested:
        st.session_state["workflow_active_image_bytes"] = uploaded_file.getvalue()
        st.session_state["workflow_active_image_name"] = str(uploaded_file.name)
        st.session_state["workflow_active_image_kind"] = "original"
        st.rerun()

    if save_ai_to_desktop_requested:
        ai_bytes = st.session_state.get("workflow_ai_image_bytes")
        ai_name = str(st.session_state.get("workflow_ai_image_name", "plotter_ai_image.png"))
        if isinstance(ai_bytes, (bytes, bytearray)) and bytes(ai_bytes):
            try:
                export_path = _save_image_bytes_to_desktop(bytes(ai_bytes), ai_name)
            except OSError as exc:
                st.session_state["workflow_ai_feedback"] = {
                    "kind": "error",
                    "message": f"Could not save the AI image to Desktop: {exc}",
                }
            else:
                st.session_state["workflow_ai_feedback"] = {
                    "kind": "success",
                    "message": f"Saved AI image to `{export_path}`.",
                }
        else:
            st.session_state["workflow_ai_feedback"] = {
                "kind": "error",
                "message": "No AI-generated image is available to save yet.",
            }
        st.rerun()

    if ai_image_available and active_image_kind == "ai":
        revised_prompt = st.session_state.get("workflow_ai_revised_prompt")
        if isinstance(revised_prompt, str) and revised_prompt.strip():
            with st.expander("AI revised prompt", expanded=False):
                st.code(revised_prompt, language="text")

    preset_cols = st.columns([1.6, 3.4], gap="small")
    preset_names = tuple(PAPER_PRESETS)
    preset_index = preset_names.index(DEFAULT_PAPER_PRESET)
    with preset_cols[0]:
        preset_name = st.selectbox("Paper size", preset_names, index=preset_index, key="workflow_paper_preset")
    page_width_mm, page_height_mm = PAPER_PRESETS[preset_name]
    with preset_cols[1]:
        st.caption(f"Working page: `{page_width_mm:.1f} mm x {page_height_mm:.1f} mm`")

    pen_calibration_ready = _render_pre_slice_pen_calibration(
        page_width_mm=page_width_mm,
        page_height_mm=page_height_mm,
    )

    settings = _current_cura_settings(page_width_mm, page_height_mm)
    generation_mode = str(st.session_state.get("path_generation_mode", "vector_trace"))

    settings_data = asdict(settings)
    with st.spinner("Preparing the page-space mask for Cura..."):
        base_page_mask = build_cura_page_mask_data(uploaded_bytes, settings_data)
    base_page_tone = None
    if generation_mode == "triangle_mesh":
        with st.spinner("Preparing the tonal page map for the triangle mesh..."):
            base_page_tone = build_cura_page_tone_data(uploaded_bytes, settings_data)

    editor_signature = _cura_signature(uploaded_bytes, settings_data)
    edited_mask_key = f"cura_edited_mask::{editor_signature}"
    editor_visible_key = f"cura_editor_visible::{editor_signature}"
    existing_result = st.session_state.get("plotter_result")
    if editor_visible_key not in st.session_state:
        st.session_state[editor_visible_key] = existing_result is None

    edited_mask = st.session_state.get(edited_mask_key, base_page_mask)
    if existing_result is not None:
        editor_toggle_cols = st.columns([1.4, 4.6])
        with editor_toggle_cols[0]:
            if not st.session_state[editor_visible_key]:
                if st.button("Adjust Artwork", key=f"show_cura_editor::{editor_signature}"):
                    st.session_state[editor_visible_key] = True
                    st.rerun()
            else:
                if st.button("Hide Editor", key=f"hide_cura_editor::{editor_signature}"):
                    st.session_state[editor_visible_key] = False
                    st.rerun()
        with editor_toggle_cols[1]:
            st.caption(
                "Keeping the live page editor hidden after slicing reduces Safari/browser memory usage. "
                "Reopen it only when you want to move, resize, or erase the artwork again."
            )

    if st.session_state[editor_visible_key]:
        with st.container():
            edited_mask = _render_cura_artwork_editor(
                uploaded_bytes=uploaded_bytes,
                base_page_mask=base_page_mask,
                settings=settings,
                settings_data=settings_data,
            )
    mask_signature = _cura_mask_signature(edited_mask, settings_data)

    _render_cura_workflow_settings()

    generate_cols = st.columns([1.8, 4.2])
    generate_label = "Generate G-Code"
    with generate_cols[0]:
        generate_requested = st.button(
            generate_label,
            type="primary",
            disabled=not pen_calibration_ready,
            use_container_width=True,
        )
    with generate_cols[1]:
        if not pen_calibration_ready:
            st.caption("Calibrate and confirm the pen-down point above before generating G-code.")
        elif generation_mode == "vector_trace":
            st.caption("This will generate direct vector-based plotter paths from the edited page mask.")
        elif generation_mode == "triangle_mesh":
            st.caption(
                "This will build a connected triangle mesh from the page image, with darker regions sampled more densely than lighter ones."
            )
        else:
            st.caption("This will send the edited page mask through the Cura-based slicing pipeline and then convert it into plotter G-code.")

    if generate_requested:
        result_signature = f"{mask_signature}:{generation_mode}"
        if generation_mode == "vector_trace":
            with st.spinner("Tracing the edited page mask into vector plotter paths..."):
                vector_settings = _build_vector_settings_from_page_mask(edited_mask, settings)
                st.session_state["plotter_result_signature"] = result_signature
                st.session_state["plotter_result_kind"] = generation_mode
                st.session_state["plotter_result"] = build_vector_plan_from_page_mask(
                    edited_mask,
                    asdict(vector_settings),
                )
                st.session_state["plotter_result_gcode"] = generate_gcode(
                    st.session_state["plotter_result"].toolpaths,
                    vector_settings,
                )
                st.session_state["plotter_result_settings"] = asdict(vector_settings)
                st.session_state["plotter_simulator_signature"] = result_signature
        elif generation_mode == "triangle_mesh":
            with st.spinner("Building the adaptive triangle mesh and connecting it into plotter paths..."):
                if base_page_tone is None:
                    base_page_tone = build_cura_page_tone_data(uploaded_bytes, settings_data)
                triangle_tone_map = base_page_tone.copy()
                erased_pixels = np.logical_and(base_page_mask > 0, edited_mask == 0)
                triangle_tone_map[erased_pixels] = 255
                triangle_settings = _build_triangle_mesh_settings_from_page_tone(triangle_tone_map, settings)
                gcode_settings = _build_gcode_processing_settings_from_cura(
                    settings,
                    fill_mode="none",
                    fill_spacing_mm=float(st.session_state["triangle_max_spacing_mm"]),
                )
                st.session_state["plotter_result_signature"] = result_signature
                st.session_state["plotter_result_kind"] = generation_mode
                st.session_state["plotter_result"] = plan_triangle_mesh_from_tone_map(
                    triangle_tone_map,
                    triangle_settings,
                )
                st.session_state["plotter_result_gcode"] = generate_gcode(
                    st.session_state["plotter_result"].toolpaths,
                    gcode_settings,
                )
                st.session_state["plotter_result_settings"] = asdict(gcode_settings)
                st.session_state["plotter_simulator_signature"] = result_signature
        else:
            with st.spinner("Converting the edited mask to STL, slicing with CuraEngine, and post-processing for pen plotting..."):
                st.session_state["plotter_result_signature"] = result_signature
                st.session_state["plotter_result_kind"] = generation_mode
                st.session_state["plotter_result"] = slice_page_mask_with_cura(edited_mask, settings)
                st.session_state["plotter_result_gcode"] = st.session_state["plotter_result"].plotter_gcode
                st.session_state["plotter_result_settings"] = asdict(settings)
                st.session_state["plotter_simulator_signature"] = result_signature

    result = None
    result_signature = f"{mask_signature}:{generation_mode}"
    if st.session_state.get("plotter_result_signature") == result_signature:
        result = st.session_state.get("plotter_result")

    if result is None:
        st.caption("Generate G-code after the artwork placement and settings look right.")
        return

    if generation_mode in {"vector_trace", "triangle_mesh"}:
        _render_vector_trace_result(
            uploaded_file=uploaded_file,
            plan=result,
            processing_settings=ProcessingSettings(**st.session_state["plotter_result_settings"]),
            page_width_mm=settings.page_width_mm,
            page_height_mm=settings.page_height_mm,
            signature=result_signature,
            gcode_text=str(st.session_state.get("plotter_result_gcode", "")),
            machine_suffix="triangle_mesh" if generation_mode == "triangle_mesh" else "vector_trace",
        )
        return

    metrics = calculate_path_metrics(result.toolpaths)
    preview = render_toolpath_preview(
        result.toolpaths,
        page_width_mm=settings.page_width_mm,
        page_height_mm=settings.page_height_mm,
        line_width_mm=settings.line_width_mm,
    )
    st.subheader("Generated Plotter Paths")
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    with metric_col_1:
        st.metric("Drawable paths", metrics["path_count"])
    with metric_col_2:
        st.metric("Drawing distance", f'{metrics["draw_distance_mm"]:.1f} mm')
    with metric_col_3:
        st.metric("Travel distance", f'{metrics["travel_distance_mm"]:.1f} mm')
    st.image(
        preview,
        caption="Actual plotter paths that will be sent to the machine.",
        use_container_width=True,
    )
    file_stem = Path(uploaded_file.name).stem
    with st.expander("Control Center", expanded=True):
        render_machine_control_panel()
    _render_draw_on_machine_section(
        result.plotter_gcode,
        f"{file_stem}_cura_plotter",
        page_width_mm=settings.page_width_mm,
        page_height_mm=settings.page_height_mm,
    )


def _render_vector_trace_result(
    *,
    uploaded_file,
    plan: PlannedDrawing,
    processing_settings: ProcessingSettings,
    page_width_mm: float,
    page_height_mm: float,
    signature: str,
    gcode_text: str,
    machine_suffix: str = "vector_trace",
) -> None:
    metrics = calculate_path_metrics(plan.toolpaths)
    path_preview = render_toolpath_preview(
        plan.toolpaths,
        page_width_mm=page_width_mm,
        page_height_mm=page_height_mm,
        line_width_mm=processing_settings.pen_width_mm,
    )
    st.subheader("Generated Plotter Paths")
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    with metric_col_1:
        st.metric("Drawable paths", metrics["path_count"])
    with metric_col_2:
        st.metric("Drawing distance", f'{metrics["draw_distance_mm"]:.1f} mm')
    with metric_col_3:
        st.metric("Travel distance", f'{metrics["travel_distance_mm"]:.1f} mm')
    st.image(
        path_preview,
        caption="Actual plotter paths that will be sent to the machine.",
        use_container_width=True,
    )
    file_stem = Path(uploaded_file.name).stem
    with st.expander("Control Center", expanded=True):
        render_machine_control_panel()
    _render_draw_on_machine_section(
        gcode_text,
        f"{file_stem}_{machine_suffix}",
        page_width_mm=page_width_mm,
        page_height_mm=page_height_mm,
    )


def _render_cura_artwork_editor(
    *,
    uploaded_bytes: bytes,
    base_page_mask: np.ndarray,
    settings: CuraSettings,
    settings_data: dict[str, float | int | str | bool],
) -> np.ndarray:
    editor_signature = _cura_signature(uploaded_bytes, settings_data)
    edited_mask_key = f"cura_edited_mask::{editor_signature}"
    erase_nonce_key = f"cura_erase_nonce::{editor_signature}"
    grid_visible_key = f"cura_editor_grid_visible::{editor_signature}"
    if erase_nonce_key not in st.session_state:
        st.session_state[erase_nonce_key] = 0
    if grid_visible_key not in st.session_state:
        st.session_state[grid_visible_key] = False

    working_mask = st.session_state.get(edited_mask_key, base_page_mask)
    preview_image = _build_page_editor_background(
        working_mask,
        show_grid=bool(st.session_state.get(grid_visible_key, False)),
    )
    current_bounds = mask_bounds_mm(
        working_mask,
        page_width_mm=settings.page_width_mm,
        page_height_mm=settings.page_height_mm,
    )

    st.subheader("Artwork Layout")
    st.caption(
        "Drag the box to move the artwork, resize it from the corners, then switch to erase mode to remove unwanted black regions."
    )
    if current_bounds is not None:
        bounds_width_mm = current_bounds[2] - current_bounds[0]
        bounds_height_mm = current_bounds[3] - current_bounds[1]
        bounds_col_1, bounds_col_2, bounds_col_3 = st.columns(3)
        with bounds_col_1:
            st.metric("Bounding box width", f"{bounds_width_mm / 10.0:.2f} cm")
        with bounds_col_2:
            st.metric("Bounding box height", f"{bounds_height_mm / 10.0:.2f} cm")
        with bounds_col_3:
            st.metric("Rotation", f"{float(st.session_state['cura_placement_rotation_degrees']):.0f} deg")

    mode_cols = st.columns([1.2, 1.1, 0.9, 0.9, 1.0, 2.2])
    with mode_cols[0]:
        if st.button("Move / Resize", key=f"editor_move::{editor_signature}"):
            st.session_state["cura_editor_mode"] = "Move / Resize"
    with mode_cols[1]:
        if st.button("Erase", key=f"editor_erase::{editor_signature}"):
            st.session_state["cura_editor_mode"] = "Erase"
    with mode_cols[2]:
        if st.button(
            "Grid",
            key=f"editor_grid::{editor_signature}",
            type="primary" if st.session_state.get(grid_visible_key, False) else "secondary",
        ):
            st.session_state[grid_visible_key] = not bool(st.session_state.get(grid_visible_key, False))
            st.rerun()
    with mode_cols[3]:
        if st.button("Done", key=f"editor_done::{editor_signature}"):
            st.session_state["cura_editor_mode"] = "Move / Resize"
    with mode_cols[4]:
        if st.button("Reset Size", key=f"editor_reset_placement::{editor_signature}"):
            st.session_state["cura_placement_scale"] = 1.0
            st.session_state["cura_placement_rotation_degrees"] = 0.0
            st.session_state["cura_placement_offset_x_mm"] = 0.0
            st.session_state["cura_placement_offset_y_mm"] = 0.0
            st.session_state["cura_skip_next_layout_update"] = True
            st.rerun()
    with mode_cols[5]:
        if st.session_state.get("cura_editor_mode", "Move / Resize") == "Erase" and st.button(
            "Clear Erase",
            key=f"editor_clear_erase::{editor_signature}",
        ):
            st.session_state.pop(edited_mask_key, None)
            st.session_state[erase_nonce_key] += 1
            st.rerun()

    grid_status = "on" if st.session_state.get(grid_visible_key, False) else "off"
    st.caption(f"Active tool: `{st.session_state.get('cura_editor_mode', 'Move / Resize')}` · Grid: `{grid_status}`")

    rotate_cols = st.columns([1.1, 1.5, 1.1, 3.3])
    with rotate_cols[0]:
        if st.button("Rotate -90", key=f"editor_rotate_left::{editor_signature}"):
            next_rotation = float(st.session_state.get("cura_placement_rotation_degrees", 0.0)) - 90.0
            while next_rotation <= -180.0:
                next_rotation += 360.0
            while next_rotation > 180.0:
                next_rotation -= 360.0
            st.session_state["cura_placement_rotation_degrees"] = next_rotation
            st.session_state["cura_skip_next_layout_update"] = True
            st.rerun()
    with rotate_cols[1]:
        st.number_input(
            "Rotation (deg)",
            min_value=-180.0,
            max_value=180.0,
            step=1.0,
            format="%.0f",
            key="cura_placement_rotation_degrees",
            help="Use this when you want the artwork to turn to match the page. The editor keeps move/resize on the footprint while rotation is controlled here.",
        )
    with rotate_cols[2]:
        if st.button("Rotate +90", key=f"editor_rotate_right::{editor_signature}"):
            next_rotation = float(st.session_state.get("cura_placement_rotation_degrees", 0.0)) + 90.0
            while next_rotation <= -180.0:
                next_rotation += 360.0
            while next_rotation > 180.0:
                next_rotation -= 360.0
            st.session_state["cura_placement_rotation_degrees"] = next_rotation
            st.session_state["cura_skip_next_layout_update"] = True
            st.rerun()
    with rotate_cols[3]:
        st.caption(
            "Rotation is now driven by the page-placement settings directly, so it will persist when you slice or regenerate the path."
        )

    if st.session_state.get("cura_editor_mode", "Move / Resize") == "Move / Resize":
        st.info("Move the blue box or drag a corner. The artwork rerenders in place after each adjustment.")
        if current_bounds is not None:
            layout_canvas = st_canvas(
                fill_color="rgba(59, 130, 246, 0.08)",
                stroke_width=2,
                stroke_color="#3b82f6",
                background_image=preview_image,
                update_streamlit=True,
                height=preview_image.height,
                width=preview_image.width,
                drawing_mode="transform",
                initial_drawing=_build_layout_rect_drawing(
                    current_bounds,
                    preview_width_px=preview_image.width,
                    preview_height_px=preview_image.height,
                    page_width_mm=settings.page_width_mm,
                    page_height_mm=settings.page_height_mm,
                ),
                display_toolbar=False,
                key=f"cura-layout::{editor_signature}",
            )
            if (
                layout_canvas is not None
                and layout_canvas.json_data is not None
                and current_bounds is not None
            ):
                if st.session_state.pop("cura_skip_next_layout_update", False):
                    st.caption("Rotation updated. Move or resize the box now if you want to reposition the rotated artwork.")
                    return working_mask
                new_bounds = _extract_bounds_from_layout_canvas(
                    layout_canvas.json_data,
                    preview_width_px=preview_image.width,
                    preview_height_px=preview_image.height,
                    page_width_mm=settings.page_width_mm,
                    page_height_mm=settings.page_height_mm,
                )
                if new_bounds is not None and _bounds_changed(current_bounds, new_bounds):
                    _apply_cura_placement_update(current_bounds, new_bounds)
                    st.rerun()
        else:
            st.image(preview_image, use_container_width=True)
            st.warning("The artwork is currently off the page. Use Reset Size to bring it back into view.")
        st.caption("The blue box tracks the slice footprint on the page. Resizing preserves the source image aspect ratio when it rerenders.")
        return working_mask

    brush_width_px = st.slider(
        "Erase brush size",
        min_value=4,
        max_value=80,
        value=18,
        step=2,
        key=f"erase_brush_size::{editor_signature}",
        help="Only affects the red erase brush in this page preview.",
    )
    erase_canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.18)",
        stroke_width=brush_width_px,
        stroke_color="#ff0000",
        background_image=preview_image,
        update_streamlit=True,
        height=preview_image.height,
        width=preview_image.width,
        drawing_mode="freedraw",
        display_toolbar=False,
        key=f"cura-erase::{editor_signature}::{st.session_state[erase_nonce_key]}",
    )
    edited_mask, erase_overlay = apply_erase_overlay_to_mask(
        working_mask,
        None if erase_canvas is None else erase_canvas.image_data,
    )
    st.session_state[edited_mask_key] = edited_mask

    edit_metrics = st.columns(3)
    original_dark_pixels = int(np.count_nonzero(working_mask))
    edited_dark_pixels = int(np.count_nonzero(edited_mask))
    removed_dark_pixels = original_dark_pixels - edited_dark_pixels
    with edit_metrics[0]:
        st.metric("Dark pixels kept", f"{edited_dark_pixels:,}")
    with edit_metrics[1]:
        st.metric("Dark pixels removed", f"{removed_dark_pixels:,}")
    with edit_metrics[2]:
        st.metric("Mask resolution", f"{edited_mask.shape[1]} x {edited_mask.shape[0]}")

    erase_preview_cols = st.columns(2)
    with erase_preview_cols[0]:
        st.image(
            mask_to_preview_image(edited_mask),
            caption="Mask that will be sliced",
            use_container_width=True,
            clamp=True,
        )
    with erase_preview_cols[1]:
        st.image(
            mask_to_preview_image(erase_overlay),
            caption="Erase strokes only",
            use_container_width=True,
            clamp=True,
        )

    return edited_mask


def render_native_mode(uploaded_file, settings: ProcessingSettings) -> None:
    if uploaded_file is None:
        st.info("Choose an image in the sidebar to plan the drawing paths.")
        st.markdown(
            """
            This backend uses the custom vector planner:

            - The source image is scaled into page space first.
            - The virtual nozzle / pen width controls how far walls and infill are inset during path planning.
            - Thick shapes get Potrace-style smooth vector tracing plus perimeter walls and optional infill.
            - Thin strokes can be routed through centerline tracing instead of being treated like tiny filled shapes.
            - G-code is generated only after you review the planned paths.
            """
        )
        return

    uploaded_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(uploaded_bytes))
    with st.spinner("Planning paths for the current image and settings..."):
        plan = build_plan(uploaded_bytes, asdict(settings))

    metrics = calculate_path_metrics(plan.toolpaths)
    centerline_paths = [toolpath for toolpath in plan.toolpaths if toolpath.kind == "centerline"]
    path_preview = render_toolpath_preview(
        plan.toolpaths,
        page_width_mm=settings.page_width_mm,
        page_height_mm=settings.page_height_mm,
        line_width_mm=settings.pen_width_mm,
    )
    vector_preview = render_vector_preview(
        plan.vector_loops,
        centerline_paths=centerline_paths,
        page_width_mm=settings.page_width_mm,
        page_height_mm=settings.page_height_mm,
    )

    st.info(
        "This runs locally on your Mac and uses your own CPU and RAM. There is no hard-coded processing limit here; "
        "runtime mostly depends on the page resolution, how much detail you keep, and whether fill is enabled."
    )

    st.subheader("Image and Mask Preview")
    preview_col_1, preview_col_2, preview_col_3, preview_col_4, preview_col_5 = st.columns(5)
    with preview_col_1:
        st.image(image, caption="Original image", use_container_width=True)
    with preview_col_2:
        st.image(
            plan.threshold_mask,
            caption="Thresholded in page space",
            use_container_width=True,
            clamp=True,
        )
    with preview_col_3:
        st.image(
            plan.processed_mask,
            caption="After detail filtering",
            use_container_width=True,
            clamp=True,
        )
    with preview_col_4:
        st.image(
            plan.shape_mask,
            caption="Thick shapes for vector tracing",
            use_container_width=True,
            clamp=True,
        )
    with preview_col_5:
        st.image(
            plan.thin_mask,
            caption="Thin features for centerlines",
            use_container_width=True,
            clamp=True,
        )

    st.subheader("Path Summary")
    metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)
    with metric_col_1:
        st.metric("Total paths", metrics["path_count"])
    with metric_col_2:
        st.metric("Perimeter paths", metrics["perimeter_paths"])
    with metric_col_3:
        st.metric("Fill paths", metrics["fill_paths"])
    with metric_col_4:
        st.metric("Centerline paths", metrics["centerline_paths"])
    with metric_col_5:
        st.metric("Drawing distance", f'{metrics["draw_distance_mm"]:.1f} mm')

    st.subheader("Vector vs Toolpath Preview")
    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.image(
            vector_preview,
            caption="Smooth filled vector preview. Orange lines show thin-feature centerlines.",
            use_container_width=True,
        )
    with preview_right:
        st.image(
            path_preview,
            caption="Actual plotter paths. Black = perimeters, blue = infill, orange = centerlines.",
            use_container_width=True,
        )
    st.caption(f'Travel distance: {metrics["travel_distance_mm"]:.1f} mm')

    with st.expander("Interior fill mask"):
        st.image(
            plan.fill_mask,
            caption="Remaining interior after perimeter walls, used for zigzag infill",
            use_container_width=True,
            clamp=True,
        )

    if not plan.toolpaths:
        st.warning(
            "No drawable paths were produced. Try lowering the minimum feature width, minimum region area, "
            "or black/white threshold."
        )
        return

    st.subheader("G-code Approval")
    st.write(
        "Review the path preview first. When it looks right, click the button below to generate "
        "downloadable G-code for the current preview."
    )
    gcode_signature = _native_gcode_signature(uploaded_bytes, asdict(settings))
    if st.button("Generate G-code From Current Preview", type="primary"):
        gcode = generate_gcode(plan.toolpaths, settings)
        file_stem = Path(uploaded_file.name).stem
        st.session_state["native_gcode_signature"] = gcode_signature
        st.session_state["native_gcode"] = gcode
        st.session_state["native_gcode_file_stem"] = file_stem
        st.download_button(
            "Download G-code",
            data=gcode,
            file_name=f"{file_stem}.gcode",
            mime="text/plain",
        )
        st.code(gcode[:6000], language="gcode")
        _render_draw_on_machine_section(
            gcode,
            file_stem,
            page_width_mm=settings.page_width_mm,
            page_height_mm=settings.page_height_mm,
        )
        return

    if st.session_state.get("native_gcode_signature") == gcode_signature:
        gcode = st.session_state.get("native_gcode", "")
        file_stem = st.session_state.get("native_gcode_file_stem", Path(uploaded_file.name).stem)
        st.download_button(
            "Download G-code",
            data=gcode,
            file_name=f"{file_stem}.gcode",
            mime="text/plain",
        )
        st.code(gcode[:6000], language="gcode")
        _render_draw_on_machine_section(
            gcode,
            file_stem,
            page_width_mm=settings.page_width_mm,
            page_height_mm=settings.page_height_mm,
        )

    with st.expander("How these controls map to your drawing"):
        st.markdown(
            """
            - `Perimeter walls`: works like slicer wall count. The planner tries to place that many 0.5 mm perimeter passes before starting infill.
            - `Ignore lines thinner than`: is now a detail-retention floor. You can drop it below 0.5 mm if you want to preserve finer source features.
            - `Thin-feature mode`: sends narrow strokes through skeleton/centerline tracing so they can be plotted as single lines instead of little blobs.
            - `Thin feature max width`: defines where the planner switches from centerline logic to perimeter/infill logic.
            - `Centerline minimum length`: removes tiny centerline fragments that would add a lot of travel without much visible value.
            - `Curve smoothing passes`: rounds jagged raster contours so circles and car fenders look less grainy.
            - `Zigzag spacing`: controls the distance between interior fill lines.
            - `Processing resolution`: trades speed for detail. Higher values use more local CPU/RAM but preserve much more detail.
            """
        )


def _render_draw_on_machine_section(
    gcode_text: str,
    file_stem: str,
    *,
    page_width_mm: float,
    page_height_mm: float,
) -> None:
    _ensure_machine_defaults()
    resume_payload_key = f"draw_resume_payload::{file_stem}"
    draw_pen_confirmed_key = f"draw_pen_confirmed::{file_stem}"
    draw_pen_feedback_key = f"draw_pen_feedback::{file_stem}"
    if draw_pen_confirmed_key not in st.session_state:
        st.session_state[draw_pen_confirmed_key] = False

    pen_motion_settings = _current_pen_motion_settings()
    homing_pulloff_mm = _current_grbl_homing_pulloff_mm()
    streaming_gcode_text = (
        f"G90\n"
        f"G92 X{page_width_mm:.3f} Y{page_height_mm:.3f}\n"
        "M5\n"
        f"{gcode_text}"
    )

    bridge_settings = _build_bridge_settings_from_session()
    st.subheader("Draw On Machine")
    st.caption(
        "This sends the currently approved drawing to the ESP32 bridge in small batches and waits for GRBL acknowledgements before sending more."
    )

    draw_info_col_1, draw_info_col_2 = st.columns(2)
    with draw_info_col_1:
        st.metric("Bridge target", bridge_settings.normalized_base_url or "Not set")
    with draw_info_col_2:
        st.metric("Pen mode", f"{pen_motion_settings.axis} stepper")

    st.info(
        "Home the machine, confirm the pen moves correctly, make sure the paper is loaded, and wait until the controller state is `Idle` before starting the draw."
    )
    st.caption(
        f"This draw always begins with `G92 X{page_width_mm:.3f} Y{page_height_mm:.3f}` so the homed X+/Y+ corner behaves like the page's max corner."
    )
    st.caption(
        (
            f"After {pen_motion_settings.axis} homes to the upper limit switch and pulls off by "
            f"`{homing_pulloff_mm:.3f} mm`, automated pen motion uses "
            f"`{pen_motion_settings.axis}{pen_motion_settings.pen_up_position_mm:.3f}` for pen-up "
            f"and `{pen_motion_settings.axis}{pen_motion_settings.pen_down_position_mm:.3f}` for pen-down "
            f"at `{pen_motion_settings.feed_rate_mm_min:.1f} mm/min`, with "
            f"`{pen_motion_settings.pen_up_dwell_seconds:.3f}s` pen-up dwell and "
            f"`{pen_motion_settings.pen_down_dwell_seconds:.3f}s` pen-down dwell."
        )
    )
    draw_quick_cols = st.columns([1.25, 2.8], gap="large")
    with draw_quick_cols[0]:
        st.session_state["draw_auto_home_after_finish"] = st.checkbox(
            "Auto-home after draw",
            value=bool(st.session_state.get("draw_auto_home_after_finish", True)),
            key=f"draw_auto_home_after_finish::{file_stem}",
            help="When the draw finishes successfully, home all axes and resync the page coordinates automatically.",
        )
    with draw_quick_cols[1]:
        st.caption(
            f"Current pen-up clearance choice: `{float(st.session_state.get('pen_height_auto_clearance_mm', DEFAULT_PEN_UP_GAP_MM)):.1f} mm`. "
            "Change this in the pre-slice pen calibration panel before generating G-code."
        )

    with st.expander("Advanced Drawing Controls", expanded=False):
        draw_batch_size_key = f"draw_batch_size::{file_stem}"
        draw_in_flight_key = f"draw_in_flight::{file_stem}"
        draw_send_spacing_key = f"draw_send_spacing_ms::{file_stem}"
        if draw_batch_size_key not in st.session_state:
            st.session_state[draw_batch_size_key] = 24
        elif int(st.session_state.get(draw_batch_size_key, 24)) == 1:
            st.session_state[draw_batch_size_key] = 24
        if draw_in_flight_key not in st.session_state:
            st.session_state[draw_in_flight_key] = 1
        elif int(st.session_state.get(draw_in_flight_key, 1)) == 2:
            st.session_state[draw_in_flight_key] = 1
        if draw_send_spacing_key not in st.session_state:
            st.session_state[draw_send_spacing_key] = 6
        elif int(st.session_state.get(draw_send_spacing_key, 6)) in {4, 8}:
            st.session_state[draw_send_spacing_key] = 6

        disable_pen_for_debug = st.checkbox(
            "Disable pen up/down during draw (debug)",
            value=False,
            key=f"draw_disable_pen::{file_stem}",
            help=(
                "Use this to test whether draw failures are coming from the pen-lift motion path. "
                "The machine will trace the XY motion path without moving the Z pen axis."
            ),
        )

        utility_cols = st.columns(2)
        with utility_cols[0]:
            draw_link_test_requested = st.button(
                "Check GRBL Link",
                key=f"draw_link_test::{file_stem}",
                use_container_width=True,
            )
        with utility_cols[1]:
            draw_go_home_requested = st.button(
                "Page Home",
                key=f"draw_go_home::{file_stem}",
                use_container_width=True,
            )

        stream_settings_col_1, stream_settings_col_2, stream_settings_col_3, stream_settings_col_4 = st.columns(4)
        with stream_settings_col_1:
            stream_batch_size = st.number_input(
                "Queue window size",
                min_value=1,
                max_value=128,
                value=24,
                step=1,
                key=draw_batch_size_key,
                help=(
                    "How many lines the app groups together before resetting its acknowledgement window. "
                    "This is not the same as how many lines are blasted to GRBL at once."
                ),
            )
        with stream_settings_col_2:
            stream_max_in_flight = st.number_input(
                "Max in-flight GRBL lines",
                min_value=1,
                max_value=16,
                value=1,
                step=1,
                key=draw_in_flight_key,
                help=(
                    "How many GRBL lines may be outstanding before the app waits for more acknowledgements. "
                    "Increase this for smoother motion; lower it if you see GRBL errors."
                ),
            )
        with stream_settings_col_3:
            stream_send_spacing_ms = st.number_input(
                "Send spacing (ms)",
                min_value=0,
                max_value=500,
                value=6,
                step=1,
                key=draw_send_spacing_key,
                help="Short delay between line sends to avoid overrunning the ESP32/Uno serial path.",
            )
        with stream_settings_col_4:
            stream_batch_timeout = st.number_input(
                "Batch ack timeout (seconds)",
                min_value=2.0,
                max_value=60.0,
                value=30.0,
                step=1.0,
                format="%.1f",
                key=f"draw_batch_timeout::{file_stem}",
                help="How long to wait for GRBL to acknowledge the current batch before stopping the draw.",
            )
        stream_recovery_timeout = st.number_input(
            "Bridge recovery timeout (seconds)",
            min_value=0.0,
            max_value=60.0,
            value=15.0,
            step=1.0,
            format="%.1f",
            key=f"draw_bridge_recovery_timeout::{file_stem}",
            help=(
                "How long the app should keep trying to reconnect to the ESP32 bridge after a transient "
                "Wi-Fi/HTTP disconnect before it stops and stores a resume point."
            ),
        )

        st.caption(
            "This now defaults to a conservative low-pressure stream: a larger acknowledgement window, only one GRBL line in flight at a time, and modest send spacing. That keeps the ESP32 from getting hammered while also reducing the chance of serial corruption."
        )
        diagnostic_cols = st.columns(5)
        with diagnostic_cols[0]:
            diagnostic_xy_max_rate = st.number_input(
                "XY max rate (mm/min)",
                min_value=100.0,
                max_value=10000.0,
                value=float(st.session_state["diagnostic_xy_max_rate_mm_min"]),
                step=100.0,
                format="%.0f",
                key=f"draw_diag_xy_rate::{file_stem}",
                help="Applied to both $110 and $111.",
            )
        with diagnostic_cols[1]:
            diagnostic_xy_accel = st.number_input(
                "XY acceleration (mm/s^2)",
                min_value=5.0,
                max_value=1000.0,
                value=float(st.session_state["diagnostic_xy_accel_mm_s2"]),
                step=5.0,
                format="%.0f",
                key=f"draw_diag_xy_accel::{file_stem}",
                help="Applied to both $120 and $121.",
            )
        with diagnostic_cols[2]:
            diagnostic_z_max_rate = st.number_input(
                "Z max rate (mm/min)",
                min_value=100.0,
                max_value=10000.0,
                value=float(st.session_state["diagnostic_z_max_rate_mm_min"]),
                step=100.0,
                format="%.0f",
                key=f"draw_diag_z_rate::{file_stem}",
                help="Applied to $112 for pen up/down speed.",
            )
        with diagnostic_cols[3]:
            diagnostic_z_accel = st.number_input(
                "Z acceleration (mm/s^2)",
                min_value=5.0,
                max_value=1000.0,
                value=float(st.session_state["diagnostic_z_accel_mm_s2"]),
                step=5.0,
                format="%.0f",
                key=f"draw_diag_z_accel::{file_stem}",
                help="Applied to $122 for pen lift responsiveness.",
            )
        with diagnostic_cols[4]:
            apply_safe_motion_profile = st.button(
                "Apply Motion Profile",
                key=f"draw_apply_motion_profile::{file_stem}",
                help="Sends $110/$111/$112/$120/$121/$122 to the Uno through the ESP32 bridge.",
            )

    pen_motion_settings = _current_pen_motion_settings()
    if disable_pen_for_debug:
        streaming_gcode_text = strip_pen_control_commands(streaming_gcode_text)
    else:
        streaming_gcode_text = replace_pen_control_commands_with_axis_moves(
            streaming_gcode_text,
            pen_motion_settings,
        )

    commands = prepare_gcode_for_streaming(streaming_gcode_text)
    if not commands:
        st.warning("No machine commands were found in this G-code.")
        return
    command_hash = _command_context_hash(commands)
    persisted_resume_payload = _load_persisted_resume_payload(file_stem, command_hash)
    if persisted_resume_payload is not None:
        st.session_state[resume_payload_key] = persisted_resume_payload

    st.caption(f"Executable GRBL lines: `{len(commands)}`")

    pen_ready_for_draw = bool(st.session_state.get("pen_height_workflow_confirmed", False))
    draw_home_requested = False
    draw_pen_home_requested = False
    draw_pen_down_10_requested = False
    draw_pen_down_5_requested = False
    draw_pen_down_1_requested = False
    draw_pen_up_step_requested = False
    draw_confirm_pen_requested = False

    workflow_cols = st.columns([1.0, 4.0], gap="large")
    with workflow_cols[0]:
        draw_requested = st.button(
            "Draw",
            key=f"draw_machine::{file_stem}",
            type="primary",
            disabled=not pen_ready_for_draw,
            use_container_width=True,
        )
    with workflow_cols[1]:
        if pen_ready_for_draw:
            st.success("Pen calibration is locked in. Draw will lift to pen-up before the first travel move.")
        else:
            st.warning("Draw is locked until you confirm pen calibration before generating G-code.")

    resume_payload = st.session_state.get(resume_payload_key)
    resume_rewind = 0
    resume_requested = False
    if isinstance(resume_payload, dict) and resume_payload.get("commands"):
        st.warning(
            "Resume only works if you have not re-homed, jogged, powered down, or otherwise changed the machine position since the stop."
        )
        resume_cols = st.columns([1.2, 1.0, 4.0])
        with resume_cols[0]:
            resume_requested = st.button("Resume From Failure", key=f"draw_resume_failed::{file_stem}")
        with resume_cols[1]:
            resume_rewind = int(
                st.number_input(
                    "Resume rewind (lines)",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=1,
                    key=f"draw_resume_rewind::{file_stem}",
                    help="Optionally back up a few commands so the resumed stroke slightly overlaps the already-drawn path.",
                )
            )
        with resume_cols[2]:
            resume_line_number = int(resume_payload.get("resume_index", 0)) + 1
            st.caption(f"Stored resume point: line {resume_line_number}")

    if draw_link_test_requested:
        link_test = run_grbl_link_test(bridge_settings)
        if link_test.ok:
            st.success(link_test.message)
        else:
            st.error(link_test.message)
        if link_test.observed_lines:
            with st.expander("GRBL Link Test Log", expanded=not link_test.ok):
                st.code("\n".join(link_test.observed_lines[-20:]), language="text")

    if apply_safe_motion_profile:
        motion_profile_commands = [
            ("Set X max rate", f"$110={float(diagnostic_xy_max_rate):.0f}"),
            ("Set Y max rate", f"$111={float(diagnostic_xy_max_rate):.0f}"),
            ("Set Z max rate", f"$112={float(diagnostic_z_max_rate):.0f}"),
            ("Set X acceleration", f"$120={float(diagnostic_xy_accel):.0f}"),
            ("Set Y acceleration", f"$121={float(diagnostic_xy_accel):.0f}"),
            ("Set Z acceleration", f"$122={float(diagnostic_z_accel):.0f}"),
        ]
        last_response = None
        for label, command in motion_profile_commands:
            last_response = send_grbl_command(bridge_settings, command)
            _store_machine_command_response(last_response, label)
            if not last_response.ok:
                break
        if last_response is not None and last_response.ok:
            st.session_state["diagnostic_xy_max_rate_mm_min"] = float(diagnostic_xy_max_rate)
            st.session_state["diagnostic_xy_accel_mm_s2"] = float(diagnostic_xy_accel)
            st.session_state["diagnostic_z_max_rate_mm_min"] = float(diagnostic_z_max_rate)
            st.session_state["diagnostic_z_accel_mm_s2"] = float(diagnostic_z_accel)
            _refresh_machine_snapshot(bridge_settings, request_fresh_status=True)

    def _invalidate_draw_pen_confirmation(message: str | None = None) -> None:
        st.session_state[draw_pen_confirmed_key] = False
        if message is not None:
            st.session_state[draw_pen_feedback_key] = message

    if draw_go_home_requested:
        pen_response = _send_pen_motion_command(
            bridge_settings,
            pen_motion_settings,
            "up",
            "Move pen to drawing-clearance height",
        )
        if pen_response.ok:
            preamble_response = send_grbl_command(bridge_settings, "G90")
            _store_machine_command_response(preamble_response, "Absolute positioning")
            if preamble_response.ok:
                move_response = send_grbl_command(
                    bridge_settings,
                    f"G0 X{page_width_mm:.3f} Y{page_height_mm:.3f}",
                )
                _store_machine_command_response(move_response, "Go to page home corner")
                if move_response.ok:
                    _refresh_machine_snapshot_until_settled(bridge_settings)

    if draw_home_requested:
        _invalidate_draw_pen_confirmation("Pen calibration cleared. Reconfirm it after homing.")
        response = send_grbl_command(bridge_settings, "$H")
        _store_machine_command_response(response, "Home all axes")
        if response.ok:
            _refresh_machine_snapshot_until_settled(bridge_settings)
            _sync_draw_page_coordinates(
                bridge_settings,
                page_width_mm=page_width_mm,
                page_height_mm=page_height_mm,
                pen_motion_settings=pen_motion_settings,
            )
            st.rerun()

    if draw_pen_home_requested:
        _invalidate_draw_pen_confirmation("Pen calibration cleared. Reconfirm it after homing the pen.")
        response = send_grbl_command(bridge_settings, f"$H{pen_motion_settings.axis.upper()}")
        _store_machine_command_response(response, f"Home pen axis ({pen_motion_settings.axis.upper()})")
        if response.ok:
            _refresh_machine_snapshot_until_settled(bridge_settings)
            _sync_draw_page_coordinates(
                bridge_settings,
                page_width_mm=page_width_mm,
                page_height_mm=page_height_mm,
                pen_motion_settings=pen_motion_settings,
            )
            st.rerun()

    requested_pen_jog_mm = 0.0
    requested_pen_jog_label = ""
    if draw_pen_down_10_requested:
        requested_pen_jog_mm = 10.0
        requested_pen_jog_label = "Pen down 10 mm"
    elif draw_pen_down_5_requested:
        requested_pen_jog_mm = 5.0
        requested_pen_jog_label = "Pen down 5 mm"
    elif draw_pen_down_1_requested:
        requested_pen_jog_mm = 1.0
        requested_pen_jog_label = "Pen down 1 mm"
    elif draw_pen_up_step_requested:
        requested_pen_jog_mm = -1.0
        requested_pen_jog_label = "Pen up 1 mm"

    if requested_pen_jog_mm:
        direction_multiplier = _manual_axis_direction_multiplier("pen_lift")
        signed_distance_mm = -requested_pen_jog_mm * direction_multiplier
        jog_command = build_jog_command(
            pen_motion_settings.axis,
            signed_distance_mm,
            pen_motion_settings.feed_rate_mm_min,
        )
        _invalidate_draw_pen_confirmation("Pen calibration changed. Reconfirm it before drawing.")
        response = send_grbl_command(bridge_settings, jog_command)
        _store_machine_command_response(response, requested_pen_jog_label)
        if response.ok:
            _refresh_machine_snapshot_until_settled(bridge_settings)
            st.rerun()

    if draw_confirm_pen_requested:
        if not bool(st.session_state.get("pen_reference_ready", False)):
            st.session_state[draw_pen_feedback_key] = "Home All first so the pen calibration uses the current synced drawing reference."
            st.session_state[draw_pen_confirmed_key] = False
            st.rerun()

        _refresh_machine_snapshot_until_settled(
            bridge_settings,
            timeout_seconds=5.0,
            poll_interval_seconds=0.15,
        )
        latest_snapshot = st.session_state.get("machine_snapshot")
        latest_status = None if latest_snapshot is None else latest_snapshot.grbl_status
        latest_pen_position_mm = _grbl_axis_position(latest_status, pen_motion_settings.axis)
        latest_pen_depth_mm = _pen_depth_from_home_mm(latest_pen_position_mm)
        if latest_pen_depth_mm is None:
            st.session_state[draw_pen_feedback_key] = "Could not read the live pen position from GRBL before saving the calibration."
            st.session_state[draw_pen_confirmed_key] = False
            st.rerun()

        clearance_mm = float(st.session_state.get("pen_height_auto_clearance_mm", DEFAULT_PEN_UP_GAP_MM))
        pen_down_position_mm = float(latest_pen_position_mm)
        down_direction_sign = _pen_down_work_direction_sign()
        ok, message, pending_update = _update_pen_height_calibration(
            pen_up_position_mm=pen_down_position_mm - (down_direction_sign * clearance_mm),
            pen_down_position_mm=pen_down_position_mm,
        )
        if ok and isinstance(pending_update, dict):
            st.session_state["pen_height_saved_up_position_mm"] = float(
                pending_update["pen_height_saved_up_position_mm"]
            )
            st.session_state["pen_height_saved_down_position_mm"] = float(
                pending_update["pen_height_saved_down_position_mm"]
            )
            st.session_state[draw_pen_confirmed_key] = True
            st.session_state[draw_pen_feedback_key] = (
                f"{message} Auto-up gap fixed at {clearance_mm:.3f} mm. Draw is now unlocked."
            )
        else:
            st.session_state[draw_pen_confirmed_key] = False
            st.session_state[draw_pen_feedback_key] = message
        st.rerun()

    def _run_stream_job(
        *,
        stream_text: str,
        command_context: list[str],
        payload_key: str,
    ) -> None:
        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()

        def _update_progress(completed: int, total: int, message: str) -> None:
            ratio = 0.0 if total <= 0 else min(max(completed / total, 0.0), 1.0)
            progress_bar.progress(ratio)
            status_placeholder.info(message)

        active_bridge_settings = bridge_settings
        result = stream_gcode_to_bridge(
            active_bridge_settings,
            stream_text,
            batch_line_limit=int(stream_batch_size),
            batch_timeout_seconds=float(stream_batch_timeout),
            max_in_flight_commands=int(stream_max_in_flight),
            inter_command_delay_seconds=float(stream_send_spacing_ms) / 1000.0,
            bridge_base_url_candidates=_bridge_discovery_candidates(active_bridge_settings.normalized_base_url),
            bridge_recovery_timeout_seconds=float(stream_recovery_timeout),
            progress_callback=_update_progress,
        )
        if result.active_base_url and result.active_base_url != active_bridge_settings.normalized_base_url:
            st.session_state["bridge_pending_base_url"] = result.active_base_url
            active_bridge_settings = replace(active_bridge_settings, base_url=result.active_base_url)
            st.info(f"Recovered ESP32 bridge at `{result.active_base_url}`.")

        if result.ok:
            progress_bar.progress(1.0)
            status_placeholder.success(result.message)
            st.success(f"Finished streaming {result.completed_commands} commands to the machine.")
            st.session_state.pop(payload_key, None)
            _clear_persisted_resume_payload(file_stem)
            if bool(st.session_state.get("draw_auto_home_after_finish", True)):
                status_placeholder.info("Draw finished. Auto-homing all axes...")
                st.session_state[draw_pen_confirmed_key] = False
                st.session_state[draw_pen_feedback_key] = (
                    "Draw finished and the machine auto-homed. Reconfirm the pen-down calibration before the next draw."
                )
                _clear_pen_calibration(
                    "Draw finished and the machine auto-homed. Reconfirm the pen-down calibration before generating the next draw."
                )
                home_response = send_grbl_command(active_bridge_settings, "$H")
                _store_machine_command_response(home_response, "Auto-home after draw")
                if home_response.ok:
                    _refresh_machine_snapshot_until_settled(active_bridge_settings)
                    _sync_draw_page_coordinates(
                        active_bridge_settings,
                        page_width_mm=page_width_mm,
                        page_height_mm=page_height_mm,
                        pen_motion_settings=pen_motion_settings,
                    )
                    status_placeholder.success("Draw finished and the machine auto-homed.")
                else:
                    status_placeholder.warning("Draw finished, but the auto-home step failed.")
                    st.warning("The draw completed, but the machine could not auto-home afterward.")
        else:
            status_placeholder.error(result.message)
            st.error(
                f"Drawing stopped after {result.completed_commands} confirmed commands out of {result.total_commands}."
            )
            if result.failed_command:
                st.code(result.failed_command, language="gcode")
            failure_anchor = min(
                max(
                    result.failed_command_index
                    if result.failed_command_index is not None
                    else result.completed_commands,
                    0,
                ),
                len(command_context),
            )
            st.session_state[payload_key] = {
                "commands": command_context,
                "resume_index": failure_anchor,
                "command_hash": _command_context_hash(command_context),
            }
            _store_persisted_resume_payload(file_stem, st.session_state[payload_key])
            failure_context_start = max(0, failure_anchor - 6)
            failure_context_end = min(len(command_context), failure_anchor + 6)
            if failure_context_end > failure_context_start:
                context_lines = []
                for command_index in range(failure_context_start, failure_context_end):
                    marker = ">>" if command_context[command_index] == result.failed_command else "  "
                    context_lines.append(f"{marker} {command_index + 1:05d}: {command_context[command_index]}")
                with st.expander("Commands around failure", expanded=True):
                    st.code("\n".join(context_lines), language="text")

        _refresh_machine_snapshot(active_bridge_settings, request_fresh_status=True)

    if draw_requested:
        _run_stream_job(
            stream_text=streaming_gcode_text,
            command_context=commands,
            payload_key=resume_payload_key,
        )

    if resume_requested and isinstance(resume_payload, dict) and resume_payload.get("commands"):
        resume_commands = list(resume_payload["commands"])
        start_index = max(0, int(resume_payload.get("resume_index", 0)) - resume_rewind)
        resume_stream_text = "G90\n" + "\n".join(resume_commands[start_index:]) + "\n"
        _run_stream_job(
            stream_text=resume_stream_text,
            command_context=resume_commands,
            payload_key=resume_payload_key,
        )


def _build_page_editor_background(
    mask: np.ndarray,
    *,
    max_width_px: int = 950,
    max_height_px: int = 760,
    show_grid: bool = False,
) -> Image.Image:
    page_image = Image.fromarray(mask_to_preview_image(mask)).convert("RGB")
    if show_grid:
        _draw_page_alignment_grid(page_image)
    border_draw = ImageDraw.Draw(page_image)
    border_draw.rectangle(
        (0, 0, page_image.width - 1, page_image.height - 1),
        outline=(190, 190, 190),
        width=max(1, int(round(min(page_image.size) / 220))),
    )
    scale = min(
        max_width_px / page_image.width,
        max_height_px / page_image.height,
        1.0,
    )
    if scale < 1.0:
        page_image = page_image.resize(
            (
                max(1, int(round(page_image.width * scale))),
                max(1, int(round(page_image.height * scale))),
            ),
            Image.Resampling.NEAREST,
        )
    return page_image


def _draw_page_alignment_grid(page_image: Image.Image) -> None:
    draw = ImageDraw.Draw(page_image, "RGBA")
    width_px, height_px = page_image.size
    line_width_px = max(1, int(round(min(width_px, height_px) / 260)))

    for index in range(1, 4):
        x_pos = round(width_px * index / 4)
        draw.line(
            [(x_pos, 0), (x_pos, height_px - 1)],
            fill=(18, 106, 85, 150),
            width=line_width_px,
        )

    for index in range(1, 6):
        y_pos = round(height_px * index / 6)
        draw.line(
            [(0, y_pos), (width_px - 1, y_pos)],
            fill=(239, 123, 69, 135),
            width=line_width_px,
        )


def _build_layout_rect_drawing(
    bounds_mm: tuple[float, float, float, float],
    *,
    preview_width_px: int,
    preview_height_px: int,
    page_width_mm: float,
    page_height_mm: float,
) -> dict[str, object]:
    x_min_mm, y_min_mm, x_max_mm, y_max_mm = bounds_mm
    scale_x = preview_width_px / page_width_mm
    scale_y = preview_height_px / page_height_mm
    left = x_min_mm * scale_x
    top = (page_height_mm - y_max_mm) * scale_y
    width = max((x_max_mm - x_min_mm) * scale_x, 1.0)
    height = max((y_max_mm - y_min_mm) * scale_y, 1.0)
    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "rect",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "fill": "rgba(59, 130, 246, 0.08)",
                "stroke": "#3b82f6",
                "strokeWidth": 2,
                "transparentCorners": False,
                "cornerColor": "#2563eb",
                "cornerStrokeColor": "#2563eb",
                "lockRotation": True,
                "hasRotatingPoint": False,
            }
        ],
    }


def _extract_bounds_from_layout_canvas(
    canvas_json: dict[str, object],
    *,
    preview_width_px: int,
    preview_height_px: int,
    page_width_mm: float,
    page_height_mm: float,
) -> tuple[float, float, float, float] | None:
    objects = canvas_json.get("objects")
    if not isinstance(objects, list) or not objects:
        return None
    rect = objects[0]
    if not isinstance(rect, dict):
        return None
    scale_x = page_width_mm / preview_width_px
    scale_y = page_height_mm / preview_height_px
    left = float(rect.get("left", 0.0))
    top = float(rect.get("top", 0.0))
    width = float(rect.get("width", 0.0)) * float(rect.get("scaleX", 1.0))
    height = float(rect.get("height", 0.0)) * float(rect.get("scaleY", 1.0))
    if width <= 0 or height <= 0:
        return None
    x_min_mm = max(0.0, min(page_width_mm, left * scale_x))
    x_max_mm = max(0.0, min(page_width_mm, (left + width) * scale_x))
    y_max_mm = max(0.0, min(page_height_mm, page_height_mm - (top * scale_y)))
    y_min_mm = max(0.0, min(page_height_mm, page_height_mm - ((top + height) * scale_y)))
    if x_max_mm <= x_min_mm or y_max_mm <= y_min_mm:
        return None
    return x_min_mm, y_min_mm, x_max_mm, y_max_mm


def _bounds_changed(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    *,
    tolerance_mm: float = 0.6,
) -> bool:
    return any(abs(first[index] - second[index]) > tolerance_mm for index in range(4))


def _apply_cura_placement_update(
    current_bounds: tuple[float, float, float, float],
    new_bounds: tuple[float, float, float, float],
) -> None:
    current_width = max(current_bounds[2] - current_bounds[0], 1e-6)
    current_height = max(current_bounds[3] - current_bounds[1], 1e-6)
    new_width = max(new_bounds[2] - new_bounds[0], 1e-6)
    new_height = max(new_bounds[3] - new_bounds[1], 1e-6)
    width_scale = new_width / current_width
    height_scale = new_height / current_height
    scale_factor = max(0.05, min(5.0, (width_scale + height_scale) / 2.0))

    current_center_x = (current_bounds[0] + current_bounds[2]) / 2.0
    current_center_y = (current_bounds[1] + current_bounds[3]) / 2.0
    new_center_x = (new_bounds[0] + new_bounds[2]) / 2.0
    new_center_y = (new_bounds[1] + new_bounds[3]) / 2.0

    st.session_state["cura_placement_scale"] = max(
        0.05,
        min(6.0, float(st.session_state["cura_placement_scale"]) * scale_factor),
    )
    st.session_state["cura_placement_offset_x_mm"] = float(
        st.session_state["cura_placement_offset_x_mm"]
    ) + (new_center_x - current_center_x)
    st.session_state["cura_placement_offset_y_mm"] = float(
        st.session_state["cura_placement_offset_y_mm"]
    ) + (new_center_y - current_center_y)


def render_machine_control_panel() -> None:
    _ensure_machine_defaults()
    pending_base_url = st.session_state.pop("bridge_pending_base_url", None)
    if pending_base_url:
        st.session_state["bridge_base_url"] = pending_base_url
    if st.session_state.pop("bridge_auto_detect_requested", False):
        _auto_detect_bridge_base_url(force=True)
    else:
        _auto_detect_bridge_base_url()
    _apply_pending_pen_height_calibration()
    st.markdown("#### Control Center")
    settings_col_1, settings_col_2, settings_col_3 = st.columns([2.2, 1.0, 0.9])
    with settings_col_1:
        st.text_input(
            "ESP32 address",
            key="bridge_base_url",
            help="Use the ESP32 bridge IP or hostname. The app can auto-detect `.89`, `.90`, or `esp32-grbl-bridge.local`.",
        )
    with settings_col_2:
        st.number_input(
            "HTTP timeout (seconds)",
            min_value=0.5,
            max_value=20.0,
            step=0.5,
            format="%.1f",
            key="bridge_timeout",
        )
    with settings_col_3:
        if st.button("Auto-detect", key="bridge_auto_detect_button", use_container_width=True):
            st.session_state["bridge_auto_detect_requested"] = True
            st.rerun()

    discovery_message = st.session_state.get("bridge_discovery_message")
    if discovery_message:
        if st.session_state.get("bridge_discovery_ok"):
            st.caption(str(discovery_message))
        else:
            st.warning(str(discovery_message))

    bridge_settings = _build_bridge_settings_from_session()
    snapshot = st.session_state.get("machine_snapshot")
    pen_motion_settings = _current_pen_motion_settings()

    command_to_send: str | None = None
    command_label: str | None = None
    motion_tuning_mode_requested: str | None = None

    top_action_cols = st.columns([1.0, 1.0, 1.35], gap="small")
    with top_action_cols[0]:
        if st.button("Settings ($$)", key="machine_panel_settings_minimal", use_container_width=True):
            command_to_send = "$$"
            command_label = "Settings dump"
    with top_action_cols[1]:
        if st.button("Home All", key="machine_panel_home_all_minimal", use_container_width=True):
            command_to_send = "$H"
            command_label = "Home all axes"
    current_power_mode = str(st.session_state.get("stepper_power_mode", "deenergized")).strip().lower()
    power_button_label = "Energize Steppers" if current_power_mode != "energized" else "De-energize Steppers"
    with top_action_cols[2]:
        power_button_clicked = st.button(power_button_label, key="machine_panel_power_toggle", use_container_width=True)

    power_state_label = "Energized" if current_power_mode == "energized" else "De-energized"
    st.caption(f"Stepper hold mode: `{power_state_label}`")

    with st.expander("Motion Tuning", expanded=False):
        motion_tuning_mode_requested = _render_motion_tuning_group()

    with st.expander("GRBL Console", expanded=True):
        manual_command_submitted = False
        with st.form("machine_manual_command_form_minimal", clear_on_submit=False):
            raw_command = st.text_input(
                "Send manual command",
                value=st.session_state.get("raw_grbl_command", ""),
                key="raw_grbl_command",
                help="Examples: `$X`, `$HZ`, `?`, `G0 X10 Y10`",
            )
            manual_command_submitted = st.form_submit_button("Send", type="primary")

        if manual_command_submitted:
            command_to_send = raw_command.strip()
            command_label = "Manual command"

        recent_log = [] if snapshot is None or snapshot.recent_log is None else snapshot.recent_log
        updated_at = st.session_state.get("machine_snapshot_updated_at")
        if updated_at:
            st.caption(f"Last GRBL log refresh: {updated_at}")
        st.text_area(
            "GRBL log",
            value="\n".join(recent_log[-80:]) if recent_log else "(No GRBL log entries returned yet.)",
            height=260,
            disabled=True,
        )

    if power_button_clicked:
        requested_mode = "energized" if current_power_mode != "energized" else "deenergized"
        _apply_stepper_power_mode(bridge_settings, requested_mode)
        snapshot = st.session_state.get("machine_snapshot")

    if motion_tuning_mode_requested is not None:
        requested_profile = "balanced" if motion_tuning_mode_requested == "chill" else "fast_pen"
        if _apply_motion_tuning_profile_to_grbl(bridge_settings, requested_profile):
            st.session_state["motion_tuning_mode"] = motion_tuning_mode_requested
        snapshot = st.session_state.get("machine_snapshot")

    if command_to_send:
        response = send_grbl_command(bridge_settings, command_to_send)
        _store_machine_command_response(response, command_label or "Command")
        if response.ok:
            normalized_command = command_to_send.strip().upper()
            if normalized_command.startswith("$J=") or normalized_command.startswith("G"):
                st.session_state["stepper_power_mode"] = "energized"
            if command_to_send == "$H":
                st.session_state["stepper_power_mode"] = "energized"
                _refresh_machine_snapshot_until_settled(bridge_settings)
                _sync_pen_reference_coordinate(bridge_settings, pen_motion_settings)
            elif normalized_command.startswith("$J=") or normalized_command.startswith("G0") or normalized_command.startswith("G1"):
                _refresh_machine_snapshot_until_settled(bridge_settings)
            else:
                _refresh_machine_snapshot(
                    bridge_settings,
                    request_fresh_status=command_to_send != "?",
                )
            snapshot = st.session_state.get("machine_snapshot")

    response = st.session_state.get("machine_last_command_response")
    status_response = st.session_state.get("machine_last_status_response")
    if response is not None and not response.ok:
        st.error(response.error or "The bridge command failed.")
    if status_response is not None and not status_response.ok:
        st.error(status_response.error or "Fetching `/status` failed.")


def _refresh_machine_snapshot(
    bridge_settings: BridgeSettings,
    *,
    request_fresh_status: bool = False,
) -> None:
    response, snapshot = fetch_bridge_status_snapshot(
        bridge_settings,
        request_fresh_status=request_fresh_status,
    )
    st.session_state["machine_last_status_response"] = response
    if snapshot is not None:
        st.session_state["machine_snapshot"] = snapshot
        st.session_state["machine_snapshot_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _refresh_machine_snapshot_until_settled(
    bridge_settings: BridgeSettings,
    *,
    timeout_seconds: float = 25.0,
    poll_interval_seconds: float = 0.25,
) -> None:
    deadline = time.time() + timeout_seconds
    while True:
        _refresh_machine_snapshot(bridge_settings, request_fresh_status=True)
        snapshot = st.session_state.get("machine_snapshot")
        state = None if snapshot is None or snapshot.grbl_status is None else snapshot.grbl_status.state
        if state in {"Idle", "Alarm"}:
            return
        if time.time() >= deadline:
            return
        time.sleep(poll_interval_seconds)


def _capture_limit_activity(
    bridge_settings: BridgeSettings,
    *,
    capture_seconds: float = 2.0,
    poll_interval_seconds: float = 0.1,
) -> tuple[dict[str, bool], str]:
    seen = {axis: False for axis in GRBL_INPUT_ORDER}
    deadline = time.time() + capture_seconds
    last_error = ""

    while time.time() < deadline:
        response, snapshot = fetch_bridge_status_snapshot(
            bridge_settings,
            request_fresh_status=True,
            status_settle_seconds=0.05,
        )
        st.session_state["machine_last_status_response"] = response
        if snapshot is not None:
            st.session_state["machine_snapshot"] = snapshot
            st.session_state["machine_snapshot_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = snapshot.grbl_status
            if status is not None:
                seen["X"] = seen["X"] or bool(status.x_limit_pressed)
                seen["Y"] = seen["Y"] or bool(status.y_limit_pressed)
                seen["Z"] = seen["Z"] or bool(status.z_limit_pressed)
                seen["P"] = seen["P"] or bool(status.probe_input_active)
        elif not response.ok and response.error:
            last_error = response.error

        time.sleep(poll_interval_seconds)

    return seen, last_error


def _store_machine_command_response(response, command_label: str) -> None:
    st.session_state["machine_last_command_response"] = response
    st.session_state["machine_last_command_label"] = command_label


def _manual_axis_state_key(axis_id: str, field: str) -> str:
    return f"manual_axis::{axis_id}::{field}"


def _manual_axis_defaults(preset: dict[str, object]) -> dict[str, object]:
    defaults: dict[str, object] = {
        "label": str(preset["default_label"]),
        "grbl_axis": str(preset["default_grbl_axis"]),
        "feed_rate_mm_min": float(preset["default_feed_rate_mm_min"]),
        "direction_multiplier": int(preset["default_direction_multiplier"]),
    }
    if "default_move_amount_mm" in preset:
        defaults["move_amount_mm"] = float(preset["default_move_amount_mm"])
    if "default_positive_move_amount_mm" in preset:
        defaults["positive_move_amount_mm"] = float(preset["default_positive_move_amount_mm"])
    if "default_negative_move_amount_mm" in preset:
        defaults["negative_move_amount_mm"] = float(preset["default_negative_move_amount_mm"])
    return defaults


def _reset_manual_axis_to_defaults(axis_id: str, preset: dict[str, object]) -> None:
    for field, value in _manual_axis_defaults(preset).items():
        st.session_state[_manual_axis_state_key(axis_id, field)] = value


def _manual_axis_label(axis_id: str, fallback_label: str) -> str:
    label = str(st.session_state.get(_manual_axis_state_key(axis_id, "label"), fallback_label)).strip()
    return label or fallback_label


def _manual_axis_grbl_axis(axis_id: str, fallback_axis: str) -> str:
    grbl_axis = str(st.session_state.get(_manual_axis_state_key(axis_id, "grbl_axis"), fallback_axis)).strip().upper()
    if grbl_axis not in {"X", "Y", "Z"}:
        return fallback_axis
    return grbl_axis


def _manual_axis_direction_multiplier(axis_id: str) -> int:
    value = st.session_state.get(_manual_axis_state_key(axis_id, "direction_multiplier"), 1)
    try:
        return -1 if int(value) < 0 else 1
    except (TypeError, ValueError):
        return 1


def _manual_axis_move_amount(axis_id: str, preset: dict[str, object], button_direction: int) -> float:
    if button_direction > 0 and "default_positive_move_amount_mm" in preset:
        field = "positive_move_amount_mm"
    elif button_direction < 0 and "default_negative_move_amount_mm" in preset:
        field = "negative_move_amount_mm"
    else:
        field = "move_amount_mm"

    default_value = float(_manual_axis_defaults(preset).get(field, 1.0))
    return max(float(st.session_state.get(_manual_axis_state_key(axis_id, field), default_value)), 0.001)


def _manual_axis_value_matches_default(axis_id: str, preset: dict[str, object], field: str) -> bool:
    defaults = _manual_axis_defaults(preset)
    if field not in defaults:
        return True
    current_value = st.session_state.get(_manual_axis_state_key(axis_id, field), defaults[field])
    default_value = defaults[field]
    if isinstance(default_value, float):
        return abs(float(current_value) - default_value) < 1e-9
    return current_value == default_value


def _manual_axis_default_status_line(label: str, value: str, is_default: bool) -> None:
    color = "#16a34a" if is_default else "#dc2626"
    state_text = "Default" if is_default else "Modified"
    st.markdown(
        (
            f"<div style='font-size:0.92rem; margin-bottom:0.2rem;'>"
            f"<span style='font-weight:600;'>{label}:</span> "
            f"<span style='color:{color}; font-weight:700;'>{value}</span> "
            f"<span style='color:{color};'>({state_text})</span>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def _manual_axis_labels_by_grbl_axis() -> dict[str, str]:
    labels_by_axis: dict[str, str] = {}
    for preset in MANUAL_AXIS_TUNING_PRESETS:
        axis_id = str(preset["id"])
        label = _manual_axis_label(axis_id, str(preset["default_label"]))
        grbl_axis = _manual_axis_grbl_axis(axis_id, str(preset["default_grbl_axis"]))
        labels_by_axis[grbl_axis] = label
    return labels_by_axis


def _friendly_limit_label(grbl_axis: str) -> str:
    if grbl_axis == "P":
        return "Probe Input"
    labels_by_axis = _manual_axis_labels_by_grbl_axis()
    return labels_by_axis.get(grbl_axis, f"GRBL {grbl_axis}")


def _build_manual_axis_command(
    axis_id: str,
    preset: dict[str, object],
    button_direction: int,
) -> tuple[str, str]:
    default_label = str(preset["default_label"])
    default_grbl_axis = str(preset["default_grbl_axis"])
    label = _manual_axis_label(axis_id, default_label)
    grbl_axis = _manual_axis_grbl_axis(axis_id, default_grbl_axis)
    move_amount_mm = _manual_axis_move_amount(axis_id, preset, button_direction)
    feed_rate_mm_min = max(
        float(st.session_state.get(_manual_axis_state_key(axis_id, "feed_rate_mm_min"), float(preset["default_feed_rate_mm_min"]))),
        1.0,
    )
    direction_multiplier = _manual_axis_direction_multiplier(axis_id)
    signed_distance_mm = move_amount_mm * direction_multiplier * button_direction
    direction_label = "+" if signed_distance_mm >= 0 else "-"
    return (
        build_jog_command(grbl_axis, signed_distance_mm, feed_rate_mm_min),
        f"Jog {label} ({grbl_axis}{direction_label}{abs(move_amount_mm):.3f} mm)",
    )


def _render_compact_axis_control_cards() -> tuple[str | None, str | None, str | None]:
    command_to_send: str | None = None
    command_label: str | None = None
    pen_motion_action: str | None = None
    pen_motion_settings = _current_pen_motion_settings()

    card_columns = st.columns(len(MANUAL_AXIS_TUNING_PRESETS))
    for column, preset in zip(card_columns, MANUAL_AXIS_TUNING_PRESETS):
        axis_id = str(preset["id"])
        default_label = str(preset["default_label"])
        default_grbl_axis = str(preset["default_grbl_axis"])
        label = _manual_axis_label(axis_id, default_label)
        grbl_axis = _manual_axis_grbl_axis(axis_id, default_grbl_axis)
        feed_rate_mm_min = float(
            st.session_state.get(
                _manual_axis_state_key(axis_id, "feed_rate_mm_min"),
                float(preset["default_feed_rate_mm_min"]),
            )
        )

        with column:
            st.markdown(f"**{label}**")
            st.caption(f"GRBL `{grbl_axis}`")

            if axis_id == "pen_lift":
                st.caption(
                    f"Raise to `{grbl_axis}{pen_motion_settings.pen_up_position_mm:.3f}` or "
                    f"draw at `{grbl_axis}{pen_motion_settings.pen_down_position_mm:.3f}`."
                )
                pen_cols = st.columns(3)
                with pen_cols[0]:
                    if st.button("Home", key=f"compact_axis_home::{axis_id}"):
                        pen_motion_action = "home_pen"
                with pen_cols[1]:
                    if st.button("Raise", key=f"compact_axis_raise::{axis_id}"):
                        pen_motion_action = "pen_up"
                with pen_cols[2]:
                    if st.button("Lower", key=f"compact_axis_lower::{axis_id}"):
                        pen_motion_action = "pen_down"
                st.caption(f"{feed_rate_mm_min:.0f} mm/min")
                continue

            home_cols = st.columns(3)
            with home_cols[0]:
                if st.button("Home", key=f"compact_axis_home::{axis_id}"):
                    command_to_send = f"$H{grbl_axis}"
                    command_label = f"Home {label}"
            with home_cols[1]:
                if st.button("-", key=f"compact_axis_minus::{axis_id}"):
                    command_to_send, command_label = _build_manual_axis_command(axis_id, preset, -1)
            with home_cols[2]:
                if st.button("+", key=f"compact_axis_plus::{axis_id}"):
                    command_to_send, command_label = _build_manual_axis_command(axis_id, preset, 1)

            minus_amount_mm = _manual_axis_move_amount(axis_id, preset, -1)
            plus_amount_mm = _manual_axis_move_amount(axis_id, preset, 1)
            if abs(minus_amount_mm - plus_amount_mm) < 1e-9:
                st.caption(f"{plus_amount_mm:.1f} mm per jog at {feed_rate_mm_min:.0f} mm/min")
            else:
                st.caption(
                    f"- {minus_amount_mm:.1f} mm | + {plus_amount_mm:.1f} mm at {feed_rate_mm_min:.0f} mm/min"
                )

    return command_to_send, command_label, pen_motion_action


def _render_pen_height_calibration_group(
    snapshot,
    bridge_settings: BridgeSettings,
) -> tuple[str | None, str | None, str | None]:
    command_to_send: str | None = None
    command_label: str | None = None
    pen_motion_action: str | None = None

    pen_motion_settings = _current_pen_motion_settings()
    status = None if snapshot is None else snapshot.grbl_status
    live_pen_position_mm = _grbl_axis_position(status, pen_motion_settings.axis)
    live_pen_depth_mm = _pen_depth_from_home_mm(live_pen_position_mm)
    calibration_step_mm = max(float(st.session_state.get("pen_height_calibration_step_mm", 1.0)), 0.1)
    pen_reference_ready = bool(st.session_state.get("pen_reference_ready", False))

    st.write("Pen Height Calibration")
    st.caption(
        "Home the pen, nudge it down until it just touches the paper, then save that as the down point. The app will automatically keep pen-up above it by the clearance you choose."
    )
    if not pen_reference_ready:
        st.warning("Home Pen or Home All + Sync first. The saved pen heights only make sense after the Z reference has been freshly synced.")

    feedback = st.session_state.pop("pen_height_calibration_feedback", None)
    if isinstance(feedback, dict):
        message = str(feedback.get("message", "")).strip()
        if message:
            if str(feedback.get("kind", "info")) == "error":
                st.error(message)
            else:
                st.success(message)

    info_col_1, info_col_2, info_col_3 = st.columns(3)
    with info_col_1:
        st.metric(
            f"Live {pen_motion_settings.axis} depth from home",
            "Unknown" if live_pen_depth_mm is None else f"{live_pen_depth_mm:.3f} mm",
        )
    with info_col_2:
        st.metric("Saved pen-up point", f"{pen_motion_settings.pen_up_position_mm:.3f} mm")
    with info_col_3:
        st.metric("Saved pen-down point", f"{pen_motion_settings.pen_down_position_mm:.3f} mm")

    st.caption(
        f"Current pen stroke travel: {pen_motion_settings.pen_down_position_mm - pen_motion_settings.pen_up_position_mm:.3f} mm."
    )
    if live_pen_position_mm is not None:
        st.caption(
            f"Raw synced GRBL coordinate: `{live_pen_position_mm:.3f} mm`. Calibration now saves the absolute depth from home so the sign no longer matters."
        )
    clearance_col, home_col, set_col = st.columns([1.2, 1.0, 2.2])
    with clearance_col:
        st.session_state["pen_height_auto_clearance_mm"] = float(
            st.number_input(
                "Pen-up gap (mm)",
                min_value=0.5,
                max_value=10.0,
                value=float(st.session_state.get("pen_height_auto_clearance_mm", DEFAULT_PEN_UP_GAP_MM)),
                step=0.1,
                format="%.1f",
                key="pen_height_auto_clearance_mm_input",
            )
        )
    with home_col:
        if st.button("Home Pen", key="pen_height_home_pen"):
            pen_motion_action = "home_pen"
    with set_col:
        auto_clearance_mm = float(st.session_state.get("pen_height_auto_clearance_mm", DEFAULT_PEN_UP_GAP_MM))
        if st.button(
            f"Set Down + {auto_clearance_mm:.1f}mm Auto Up",
            key="pen_height_set_down_and_up",
            disabled=live_pen_depth_mm is None,
        ):
            if not pen_reference_ready:
                _queue_pen_height_feedback(
                    "error",
                    "Home Pen or Home All + Sync first so the pen calibration uses the current synced drawing reference.",
                )
                st.rerun()
            _refresh_machine_snapshot_until_settled(
                bridge_settings,
                timeout_seconds=5.0,
                poll_interval_seconds=0.15,
            )
            latest_snapshot = st.session_state.get("machine_snapshot")
            latest_status = None if latest_snapshot is None else latest_snapshot.grbl_status
            latest_pen_position_mm = _grbl_axis_position(latest_status, pen_motion_settings.axis)
            latest_pen_depth_mm = _pen_depth_from_home_mm(latest_pen_position_mm)
            if latest_pen_depth_mm is None:
                _queue_pen_height_feedback(
                    "error",
                    "Could not read the live pen position from GRBL before saving the calibration.",
                )
                st.rerun()
            pen_down_position_mm = float(latest_pen_position_mm)
            down_direction_sign = _pen_down_work_direction_sign()
            ok, message, pending_update = _update_pen_height_calibration(
                pen_up_position_mm=pen_down_position_mm - (down_direction_sign * auto_clearance_mm),
                pen_down_position_mm=pen_down_position_mm,
            )
            if ok:
                st.session_state["pending_pen_height_calibration"] = pending_update
                _queue_pen_height_feedback(
                    "success",
                    f"{message} Auto-up gap fixed at {auto_clearance_mm:.3f} mm.",
                )
                st.rerun()
            else:
                _queue_pen_height_feedback("error", message)
                st.rerun()

    down_cols = st.columns(3)
    for column, step_mm in zip(down_cols, (10.0, 5.0, 1.0)):
        with column:
            if st.button(f"Down {int(step_mm)}", key=f"pen_height_down_{int(step_mm)}"):
                command_to_send = build_jog_command(
                    pen_motion_settings.axis,
                    step_mm,
                    pen_motion_settings.feed_rate_mm_min,
                )
                command_label = f"Nudge {pen_motion_settings.axis} down {step_mm:.0f} mm"

    up_cols = st.columns(3)
    for column, step_mm in zip(up_cols, (10.0, 5.0, 1.0)):
        with column:
            if st.button(f"Up {int(step_mm)}", key=f"pen_height_up_{int(step_mm)}"):
                command_to_send = build_jog_command(
                    pen_motion_settings.axis,
                    -step_mm,
                    pen_motion_settings.feed_rate_mm_min,
                )
                command_label = f"Nudge {pen_motion_settings.axis} up {step_mm:.0f} mm"

    st.caption(
        f"Calibration nudges use GRBL `{pen_motion_settings.axis}` at {pen_motion_settings.feed_rate_mm_min:.0f} mm/min."
    )

    return command_to_send, command_label, pen_motion_action


def _render_motion_tuning_group() -> str | None:
    st.write("Motion Tuning")
    st.caption("`Chill` uses the balanced profile. `Turbo` uses the snappy XY profile plus the fast pen-lift profile.")

    current_mode = str(st.session_state.get("motion_tuning_mode", "turbo")).strip().lower()
    if current_mode not in {"chill", "turbo"}:
        current_mode = "turbo"

    mode_cols = st.columns(2)
    with mode_cols[0]:
        if st.button(
            "Chill",
            key="motion_mode_chill",
            type="primary" if current_mode == "chill" else "secondary",
            use_container_width=True,
        ):
            return "chill"
    with mode_cols[1]:
        if st.button(
            "Turbo",
            key="motion_mode_turbo",
            type="primary" if current_mode == "turbo" else "secondary",
            use_container_width=True,
        ):
            return "turbo"

    active_label = "Turbo" if current_mode == "turbo" else "Chill"
    st.caption(f"Current motion mode: `{active_label}`")
    return None


def _render_manual_axis_tuning_controls() -> tuple[str | None, str | None]:
    st.write("Manual Axis Tuning")
    st.caption(
        "Set which GRBL axis each motor is wired to, choose how far each click should move, and flip the `+` button direction whenever a motor spins the wrong way."
    )
    st.info(
        "These defaults match your current rewiring: drawing X on GRBL `X`, drawing Y on GRBL `Y`, and pen lift on GRBL `Z`."
    )

    command_to_send: str | None = None
    command_label: str | None = None

    axis_columns = st.columns(len(MANUAL_AXIS_TUNING_PRESETS))
    for column, preset in zip(axis_columns, MANUAL_AXIS_TUNING_PRESETS):
        axis_id = str(preset["id"])
        default_label = str(preset["default_label"])
        default_grbl_axis = str(preset["default_grbl_axis"])
        defaults = _manual_axis_defaults(preset)

        with column:
            if axis_id == "pen_lift" and st.button("Reset Pen Lift to Default", key="manual_axis_reset_pen_lift_defaults"):
                _reset_manual_axis_to_defaults(axis_id, preset)
                st.rerun()

            st.markdown(f"**{default_label}**")
            st.caption(str(preset["description"]))
            if "behavior_note" in preset:
                st.caption(str(preset["behavior_note"]))
            st.text_input("Display label", key=_manual_axis_state_key(axis_id, "label"))
            st.selectbox(
                "GRBL axis",
                options=("X", "Y", "Z"),
                key=_manual_axis_state_key(axis_id, "grbl_axis"),
            )
            st.selectbox(
                "Positive button sends",
                options=(1, -1),
                format_func=lambda value: "+ distance" if value > 0 else "- distance",
                key=_manual_axis_state_key(axis_id, "direction_multiplier"),
            )
            if "positive_move_amount_mm" in defaults:
                st.number_input(
                    "Move amount for + (mm)",
                    min_value=0.001,
                    max_value=500.0,
                    step=0.5,
                    format="%.3f",
                    key=_manual_axis_state_key(axis_id, "positive_move_amount_mm"),
                )
            if "negative_move_amount_mm" in defaults:
                st.number_input(
                    "Move amount for - (mm)",
                    min_value=0.001,
                    max_value=500.0,
                    step=0.5,
                    format="%.3f",
                    key=_manual_axis_state_key(axis_id, "negative_move_amount_mm"),
                )
            if "move_amount_mm" in defaults:
                st.number_input(
                    "Move amount (mm)",
                    min_value=0.001,
                    max_value=500.0,
                    step=0.5,
                    format="%.3f",
                    key=_manual_axis_state_key(axis_id, "move_amount_mm"),
                )
            st.number_input(
                "Move speed (mm/min)",
                min_value=1.0,
                max_value=10000.0,
                step=10.0,
                format="%.1f",
                key=_manual_axis_state_key(axis_id, "feed_rate_mm_min"),
            )

            live_label = _manual_axis_label(axis_id, default_label)
            grbl_axis = _manual_axis_grbl_axis(axis_id, default_grbl_axis)
            plus_move_amount_mm = _manual_axis_move_amount(axis_id, preset, 1)
            minus_move_amount_mm = _manual_axis_move_amount(axis_id, preset, -1)
            positive_distance_mm = plus_move_amount_mm * _manual_axis_direction_multiplier(axis_id)
            negative_distance_mm = minus_move_amount_mm * _manual_axis_direction_multiplier(axis_id) * -1
            st.caption(
                f"`+` sends `{grbl_axis}{positive_distance_mm:+.3f}` and `-` sends `{grbl_axis}{negative_distance_mm:+.3f}`."
            )

            if axis_id == "pen_lift":
                current_pen_settings = _current_pen_motion_settings()
                _manual_axis_default_status_line(
                    "Positive button sends",
                    "+ distance" if _manual_axis_direction_multiplier(axis_id) > 0 else "- distance",
                    _manual_axis_value_matches_default(axis_id, preset, "direction_multiplier"),
                )
                _manual_axis_default_status_line(
                    "Pen Lift + travel",
                    f"{plus_move_amount_mm:.3f} mm",
                    _manual_axis_value_matches_default(axis_id, preset, "positive_move_amount_mm"),
                )
                _manual_axis_default_status_line(
                    "Pen Lift - travel",
                    f"{minus_move_amount_mm:.3f} mm",
                    _manual_axis_value_matches_default(axis_id, preset, "negative_move_amount_mm"),
                )
                _manual_axis_default_status_line(
                    "Move speed",
                    f"{float(st.session_state[_manual_axis_state_key(axis_id, 'feed_rate_mm_min')]):.1f} mm/min",
                    _manual_axis_value_matches_default(axis_id, preset, "feed_rate_mm_min"),
                )
                st.caption(
                    (
                        f"Automated drawing assumes homed `{grbl_axis}` is the upper reference, then uses "
                        f"`{grbl_axis}{current_pen_settings.pen_up_position_mm:.3f}` for pen-up and "
                        f"`{grbl_axis}{current_pen_settings.pen_down_position_mm:.3f}` for pen-down."
                    )
                )
                st.caption(
                    "The `Pen Lift + / -` buttons below are raw setup jogs. Use `Operational Pen Positions` for normal plotting moves."
                )
            else:
                _manual_axis_default_status_line(
                    "Positive button sends",
                    "+ distance" if _manual_axis_direction_multiplier(axis_id) > 0 else "- distance",
                    _manual_axis_value_matches_default(axis_id, preset, "direction_multiplier"),
                )
                _manual_axis_default_status_line(
                    "Move speed",
                    f"{float(st.session_state[_manual_axis_state_key(axis_id, 'feed_rate_mm_min')]):.1f} mm/min",
                    _manual_axis_value_matches_default(axis_id, preset, "feed_rate_mm_min"),
                )
                if "max_travel_from_home_mm" in preset:
                    st.caption(f"Travel from home to far end: {float(preset['max_travel_from_home_mm']):.1f} mm")

            move_button_cols = st.columns(2)
            with move_button_cols[0]:
                if st.button(f"{live_label} -", key=f"manual_axis_move_minus::{axis_id}"):
                    command_to_send, command_label = _build_manual_axis_command(
                        axis_id,
                        preset,
                        -1,
                    )
            with move_button_cols[1]:
                if st.button(f"{live_label} +", key=f"manual_axis_move_plus::{axis_id}"):
                    command_to_send, command_label = _build_manual_axis_command(
                        axis_id,
                        preset,
                        1,
                    )

    return command_to_send, command_label


def _build_mask_editor_background(mask: np.ndarray) -> Image.Image:
    preview_array = mask_to_preview_image(mask)
    background = Image.fromarray(preview_array).convert("RGB")
    max_width_px = 900
    max_height_px = 1100
    scale = min(
        max_width_px / background.width,
        max_height_px / background.height,
        1.0,
    )
    if scale < 1.0:
        background = background.resize(
            (
                max(1, int(round(background.width * scale))),
                max(1, int(round(background.height * scale))),
            ),
            Image.Resampling.NEAREST,
        )
    return background


def _cura_signature(
    uploaded_bytes: bytes,
    settings_data: dict[str, float | int | str | bool],
) -> str:
    payload = {
        "image_sha1": hashlib.sha1(uploaded_bytes).hexdigest(),
        "settings": settings_data,
    }
    return json.dumps(payload, sort_keys=True)


def _native_gcode_signature(
    uploaded_bytes: bytes,
    settings_data: dict[str, float | int | str | bool],
) -> str:
    payload = {
        "image_sha1": hashlib.sha1(uploaded_bytes).hexdigest(),
        "settings": settings_data,
        "mode": "native",
    }
    return json.dumps(payload, sort_keys=True)


def _cura_mask_signature(
    page_mask: np.ndarray,
    settings_data: dict[str, float | int | str | bool],
) -> str:
    payload = {
        "mask_sha1": hashlib.sha1(page_mask.tobytes()).hexdigest(),
        "mask_shape": page_mask.shape,
        "settings": settings_data,
    }
    return json.dumps(payload, sort_keys=True)


def _status_label(value: bool | None) -> str:
    if value is True:
        return "Pressed"
    if value is False:
        return "Open"
    return "Unknown"


def _bool_label(value: bool | None) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"


if __name__ == "__main__":
    main()
