from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from photo_to_gcode.cura_mesh import write_mask_stl
from photo_to_gcode.cura_postprocess import (
    build_plotter_gcode,
    extract_toolpaths_from_cura,
    prepare_plotter_toolpaths,
    toolpath_bounds_mm,
    translate_toolpaths,
)
from photo_to_gcode.image_processing import (
    build_binary_mask,
    build_page_mask,
    build_page_tone_map,
    mask_bounds_mm,
    mask_to_preview_image,
    pil_to_rgb_array,
    to_grayscale,
)
from photo_to_gcode.models import Toolpath

APP_BUNDLE_PATHS = (
    Path("/Applications/UltiMaker Cura.app"),
    Path("/Applications/Cura.app"),
)

PROFILE_ROOT = Path(__file__).with_name("cura_profile")
PROFILE_DEFINITION = PROFILE_ROOT / "definitions" / "pen_plotter.def.json"
PROFILE_EXTRUDER = PROFILE_ROOT / "extruders" / "pen_plotter_extruder_0.def.json"


@dataclass(slots=True)
class CuraSettings:
    page_width_mm: float
    page_height_mm: float
    margin_mm: float = 10.0
    threshold: int = 160
    invert_input: bool = False
    placement_scale: float = 1.0
    placement_rotation_degrees: float = 0.0
    placement_offset_x_mm: float = 0.0
    placement_offset_y_mm: float = 0.0
    processing_resolution_ppmm: float = 6.0
    feature_height_mm: float = 0.2
    line_width_mm: float = 0.2
    wall_line_count: int = 1
    infill_density_percent: int = 100
    draw_speed_mm_per_s: float = 90.0
    travel_speed_mm_per_s: float = 90.0
    pen_up_command: str = "M5"
    pen_down_command: str = "M3 S30"
    pen_pause_seconds: float = 0.0
    plotter_fill_mode: str = "continuous_zigzag"
    fill_turn_split_angle_degrees: float = 35.0
    continuous_fill_chunk_segments: int = 0
    path_simplify_tolerance_mm: float = 0.0
    min_segment_length_mm: float = 0.0
    min_toolpath_length_mm: float = 0.75
    coordinate_decimals: int = 3


@dataclass(slots=True)
class CuraSliceResult:
    threshold_mask_preview: np.ndarray
    raw_cura_gcode: str
    plotter_gcode: str
    toolpaths: list[Toolpath]
    engine_path: str
    triangle_count: int


def build_cura_page_mask(image: Image.Image, settings: CuraSettings) -> np.ndarray:
    image_rgb = pil_to_rgb_array(image)
    grayscale = to_grayscale(image_rgb)
    binary_mask = build_binary_mask(
        grayscale,
        threshold=settings.threshold,
        invert_input=settings.invert_input,
    )
    return build_page_mask(
        binary_mask,
        page_width_mm=settings.page_width_mm,
        page_height_mm=settings.page_height_mm,
        margin_mm=settings.margin_mm,
        pixels_per_mm=settings.processing_resolution_ppmm,
        scale_multiplier=settings.placement_scale,
        rotation_degrees=settings.placement_rotation_degrees,
        offset_x_mm=settings.placement_offset_x_mm,
        offset_y_mm=settings.placement_offset_y_mm,
    )


def build_cura_page_tone(image: Image.Image, settings: CuraSettings) -> np.ndarray:
    image_rgb = pil_to_rgb_array(image)
    grayscale = to_grayscale(image_rgb)
    return build_page_tone_map(
        grayscale,
        page_width_mm=settings.page_width_mm,
        page_height_mm=settings.page_height_mm,
        margin_mm=settings.margin_mm,
        pixels_per_mm=settings.processing_resolution_ppmm,
        scale_multiplier=settings.placement_scale,
        rotation_degrees=settings.placement_rotation_degrees,
        offset_x_mm=settings.placement_offset_x_mm,
        offset_y_mm=settings.placement_offset_y_mm,
    )


def build_cura_preview_mask(image: Image.Image, settings: CuraSettings) -> np.ndarray:
    return mask_to_preview_image(build_cura_page_mask(image, settings))


def slice_image_with_cura(image: Image.Image, settings: CuraSettings) -> CuraSliceResult:
    page_mask = build_cura_page_mask(image, settings)
    return slice_page_mask_with_cura(page_mask, settings)


def slice_page_mask_with_cura(page_mask: np.ndarray, settings: CuraSettings) -> CuraSliceResult:
    engine_path = find_cura_engine()
    target_bounds = mask_bounds_mm(
        page_mask,
        page_width_mm=settings.page_width_mm,
        page_height_mm=settings.page_height_mm,
    )

    with tempfile.TemporaryDirectory(prefix="photo-to-gcode-cura-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        model_path = temp_dir_path / "input_mask.stl"
        gcode_path = temp_dir_path / "cura_output.gcode"
        triangle_count = write_mask_stl(
            page_mask,
            model_path,
            page_width_mm=settings.page_width_mm,
            page_height_mm=settings.page_height_mm,
            feature_height_mm=settings.feature_height_mm,
            center_on_page=True,
        )
        raw_cura_gcode = _slice_model_with_cura(engine_path, model_path, gcode_path, settings)

    toolpaths = extract_toolpaths_from_cura(raw_cura_gcode)
    toolpaths = _align_toolpaths_to_mask_bounds(toolpaths, target_bounds)
    toolpaths = prepare_plotter_toolpaths(
        toolpaths,
        fill_mode=settings.plotter_fill_mode,
        fill_turn_split_angle_degrees=settings.fill_turn_split_angle_degrees,
        continuous_fill_chunk_segments=settings.continuous_fill_chunk_segments,
        path_simplify_tolerance_mm=settings.path_simplify_tolerance_mm,
        min_segment_length_mm=settings.min_segment_length_mm,
        min_toolpath_length_mm=settings.min_toolpath_length_mm,
    )
    plotter_gcode = build_plotter_gcode(
        toolpaths,
        pen_up_command=settings.pen_up_command,
        pen_down_command=settings.pen_down_command,
        pen_pause_seconds=settings.pen_pause_seconds,
        draw_speed_mm_per_s=settings.draw_speed_mm_per_s,
        coordinate_decimals=settings.coordinate_decimals,
        toolpaths_already_prepared=True,
    )

    return CuraSliceResult(
        threshold_mask_preview=mask_to_preview_image(page_mask),
        raw_cura_gcode=raw_cura_gcode,
        plotter_gcode=plotter_gcode,
        toolpaths=toolpaths,
        engine_path=str(engine_path),
        triangle_count=triangle_count,
    )


def find_cura_engine() -> Path:
    for bundle_path in APP_BUNDLE_PATHS:
        candidate = bundle_path / "Contents" / "MacOS" / "CuraEngine"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "CuraEngine was not found. Install UltiMaker Cura locally, or adjust the search path in "
        "photo_to_gcode/cura_backend.py."
    )


def _slice_model_with_cura(
    engine_path: Path,
    model_path: Path,
    output_path: Path,
    settings: CuraSettings,
) -> str:
    command = [
        str(engine_path),
        "slice",
        "-j",
        str(PROFILE_DEFINITION),
        "-s",
        f"machine_width={settings.page_width_mm}",
        "-s",
        f"machine_depth={settings.page_height_mm}",
        "-s",
        "machine_height=5",
        "-s",
        "machine_extruder_count=1",
        "-s",
        f"machine_nozzle_size={settings.line_width_mm}",
        "-s",
        f"line_width={settings.line_width_mm}",
        "-s",
        f"wall_line_width_0={settings.line_width_mm}",
        "-s",
        f"wall_line_width_x={settings.line_width_mm}",
        "-s",
        f"infill_line_width={settings.line_width_mm}",
        "-s",
        f"layer_height={settings.feature_height_mm}",
        "-s",
        f"layer_height_0={settings.feature_height_mm}",
        "-s",
        f"wall_line_count={settings.wall_line_count}",
        "-s",
        "top_layers=0",
        "-s",
        "bottom_layers=0",
        "-s",
        "roofing_layer_count=0",
        "-s",
        "adhesion_type=none",
        "-s",
        "support_enable=false",
        "-s",
        "retraction_enable=false",
        "-s",
        "material_diameter=1.75",
        "-s",
        f"infill_sparse_density={settings.infill_density_percent}",
        "-s",
        f"speed_print={settings.draw_speed_mm_per_s}",
        "-s",
        f"speed_travel={settings.travel_speed_mm_per_s}",
        "-s",
        "cool_min_temperature=0",
        "-s",
        "roofing_monotonic=false",
        "-o",
        str(output_path),
        "-l",
        str(model_path),
    ]

    environment = os.environ.copy()
    environment["CURA_ENGINE_SEARCH_PATH"] = os.pathsep.join(_search_paths())
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "CuraEngine slicing failed.\n\n"
            f"Command: {' '.join(command)}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )

    return output_path.read_text(encoding="utf-8", errors="replace")


def _search_paths() -> list[str]:
    search_paths = [
        str(PROFILE_ROOT / "definitions"),
        str(PROFILE_ROOT / "extruders"),
    ]
    for bundle_path in APP_BUNDLE_PATHS:
        resources_root = bundle_path / "Contents" / "Resources" / "share" / "cura" / "resources"
        definitions = resources_root / "definitions"
        extruders = resources_root / "extruders"
        if definitions.exists():
            search_paths.append(str(definitions))
        if extruders.exists():
            search_paths.append(str(extruders))
    return search_paths


def _align_toolpaths_to_mask_bounds(
    toolpaths: list[Toolpath],
    target_bounds: tuple[float, float, float, float] | None,
) -> list[Toolpath]:
    if not toolpaths or target_bounds is None:
        return toolpaths

    current_bounds = toolpath_bounds_mm(toolpaths)
    if current_bounds is None:
        return toolpaths

    current_center_x = (current_bounds[0] + current_bounds[2]) / 2.0
    current_center_y = (current_bounds[1] + current_bounds[3]) / 2.0
    target_center_x = (target_bounds[0] + target_bounds[2]) / 2.0
    target_center_y = (target_bounds[1] + target_bounds[3]) / 2.0
    return translate_toolpaths(
        toolpaths,
        delta_x_mm=target_center_x - current_center_x,
        delta_y_mm=target_center_y - current_center_y,
    )
