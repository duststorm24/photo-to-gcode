from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]
PixelPoint = tuple[int, int]
Contour = list[PixelPoint]


@dataclass(slots=True)
class Toolpath:
    points: list[Point]
    closed: bool = False
    kind: str = "perimeter"


@dataclass(slots=True)
class VectorLoop:
    points: list[Point]
    is_hole: bool = False


@dataclass(slots=True)
class ProcessingSettings:
    page_width_mm: float
    page_height_mm: float
    margin_mm: float = 10.0
    threshold: int = 160
    invert_input: bool = False
    pen_width_mm: float = 0.5
    min_feature_width_mm: float = 0.15
    min_region_area_mm2: float = 0.75
    wall_count: int = 2
    thin_feature_mode: bool = True
    thin_feature_max_width_mm: float = 0.75
    centerline_min_length_mm: float = 1.0
    simplify_tolerance_px: float = 0.75
    curve_smoothing_passes: int = 2
    curve_sample_step_mm: float = 0.25
    fill_mode: str = "zigzag"
    fill_spacing_mm: float = 0.5
    processing_resolution_ppmm: float = 18.0
    potrace_turdsize: int = 2
    potrace_alphamax: float = 1.0
    potrace_opttolerance: float = 0.2
    feed_rate: int = 1500
    pen_up_command: str = "M5"
    pen_down_command: str = "M3 S30"
    pen_pause_seconds: float = 0.15


@dataclass(slots=True)
class PlannedDrawing:
    threshold_mask: np.ndarray
    processed_mask: np.ndarray
    shape_mask: np.ndarray
    thin_mask: np.ndarray
    fill_mask: np.ndarray
    toolpaths: list[Toolpath]
    vector_loops: list[VectorLoop]
    pixels_per_mm: float
