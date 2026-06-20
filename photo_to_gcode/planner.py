from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from photo_to_gcode.image_processing import (
    build_binary_mask,
    build_page_mask,
    erode_mask,
    filter_mask_by_min_width,
    filter_small_regions,
    mask_to_preview_image,
    pil_to_rgb_array,
    split_shape_and_thin_masks,
    to_grayscale,
)
from photo_to_gcode.models import PlannedDrawing, ProcessingSettings
from photo_to_gcode.thin_features import trace_centerlines
from photo_to_gcode.toolpaths import (
    build_hatch_toolpaths,
    merge_continuous_fill_toolpaths,
    order_toolpaths,
    simplify_toolpaths,
)
from photo_to_gcode.vector_trace import trace_mask_to_vector_loops, vector_loops_to_toolpaths


def plan_drawing(image: Image.Image, settings: ProcessingSettings) -> PlannedDrawing:
    image_rgb = pil_to_rgb_array(image)
    grayscale = to_grayscale(image_rgb)
    binary_mask = build_binary_mask(
        grayscale,
        threshold=settings.threshold,
        invert_input=settings.invert_input,
    )
    threshold_mask = build_page_mask(
        binary_mask,
        page_width_mm=settings.page_width_mm,
        page_height_mm=settings.page_height_mm,
        margin_mm=settings.margin_mm,
        pixels_per_mm=settings.processing_resolution_ppmm,
    )
    return plan_page_mask(threshold_mask, settings)


def plan_page_mask(
    threshold_mask: np.ndarray,
    settings: ProcessingSettings,
) -> PlannedDrawing:
    effective_pen_width_mm = max(0.01, float(settings.pen_width_mm))
    effective_min_feature_width_mm = settings.min_feature_width_mm
    effective_fill_spacing_mm = max(settings.fill_spacing_mm, effective_pen_width_mm)
    threshold_mask = threshold_mask.copy()
    processed_mask = filter_mask_by_min_width(
        threshold_mask,
        min_feature_width_mm=effective_min_feature_width_mm,
        pixels_per_mm=settings.processing_resolution_ppmm,
    )
    processed_mask = filter_small_regions(
        processed_mask,
        min_region_area_mm2=settings.min_region_area_mm2,
        pixels_per_mm=settings.processing_resolution_ppmm,
    )

    if settings.thin_feature_mode:
        shape_mask, thin_mask = split_shape_and_thin_masks(
            processed_mask,
            thin_feature_max_width_mm=settings.thin_feature_max_width_mm,
            pixels_per_mm=settings.processing_resolution_ppmm,
        )
    else:
        shape_mask = processed_mask.copy()
        thin_mask = cv2.subtract(processed_mask, shape_mask)

    shape_mask = filter_small_regions(
        shape_mask,
        min_region_area_mm2=settings.min_region_area_mm2,
        pixels_per_mm=settings.processing_resolution_ppmm,
    )
    thin_mask = filter_small_regions(
        thin_mask,
        min_region_area_mm2=settings.min_region_area_mm2 / 2.0,
        pixels_per_mm=settings.processing_resolution_ppmm,
    )

    vector_loops = trace_mask_to_vector_loops(
        shape_mask,
        pixels_per_mm=settings.processing_resolution_ppmm,
        sample_step_mm=settings.curve_sample_step_mm,
        min_region_area_mm2=settings.min_region_area_mm2,
        turdsize=settings.potrace_turdsize,
        alphamax=settings.potrace_alphamax,
        opttolerance=settings.potrace_opttolerance,
    )

    perimeter_paths = []
    current_shape_mask = shape_mask.copy()
    for _ in range(settings.wall_count):
        if cv2.countNonZero(current_shape_mask) == 0:
            break

        shell_loops = trace_mask_to_vector_loops(
            current_shape_mask,
            pixels_per_mm=settings.processing_resolution_ppmm,
            sample_step_mm=settings.curve_sample_step_mm,
            min_region_area_mm2=settings.min_region_area_mm2,
            turdsize=settings.potrace_turdsize,
            alphamax=settings.potrace_alphamax,
            opttolerance=settings.potrace_opttolerance,
        )
        perimeter_paths.extend(vector_loops_to_toolpaths(shell_loops, kind="perimeter"))
        next_shape_mask = erode_mask(current_shape_mask, effective_pen_width_mm, settings.processing_resolution_ppmm)
        if cv2.countNonZero(next_shape_mask) == 0:
            current_shape_mask = next_shape_mask
            break
        current_shape_mask = next_shape_mask

    pixel_simplify_tolerance_mm = 0.0
    if settings.processing_resolution_ppmm > 0:
        pixel_simplify_tolerance_mm = settings.simplify_tolerance_px / settings.processing_resolution_ppmm

    perimeter_simplify_tolerance_mm = max(
        0.08,
        settings.curve_sample_step_mm * 0.50,
        pixel_simplify_tolerance_mm,
    )
    perimeter_min_segment_length_mm = max(0.05, perimeter_simplify_tolerance_mm * 0.8)

    perimeter_paths = simplify_toolpaths(
        perimeter_paths,
        tolerance_mm=perimeter_simplify_tolerance_mm,
        min_segment_length_mm=perimeter_min_segment_length_mm,
    )

    infill_mask = current_shape_mask
    fill_paths = []
    if settings.fill_mode == "zigzag" and cv2.countNonZero(infill_mask) > 0:
        fill_paths = build_hatch_toolpaths(
            infill_mask,
            pixels_per_mm=settings.processing_resolution_ppmm,
            spacing_mm=effective_fill_spacing_mm,
            axis="horizontal",
            segment_extension_mm=effective_pen_width_mm * 0.10,
        )

    centerline_paths = []
    if settings.thin_feature_mode and cv2.countNonZero(thin_mask) > 0:
        centerline_paths = trace_centerlines(
            thin_mask,
            pixels_per_mm=settings.processing_resolution_ppmm,
            curve_smoothing_passes=settings.curve_smoothing_passes,
            curve_sample_step_mm=settings.curve_sample_step_mm,
            min_length_mm=settings.centerline_min_length_mm,
        )
        centerline_simplify_tolerance_mm = max(
            0.06,
            settings.curve_sample_step_mm * 0.45,
            pixel_simplify_tolerance_mm,
        )
        centerline_paths = simplify_toolpaths(
            centerline_paths,
            tolerance_mm=centerline_simplify_tolerance_mm,
            min_segment_length_mm=max(0.05, centerline_simplify_tolerance_mm * 0.9),
        )

    ordered_fill_paths = order_toolpaths(fill_paths)
    if settings.fill_mode == "zigzag":
        ordered_fill_paths = merge_continuous_fill_toolpaths(
            ordered_fill_paths,
            max_connector_gap_mm=max(1.0, effective_fill_spacing_mm * 3.0),
        )
    ordered_line_paths = order_toolpaths(perimeter_paths + centerline_paths)
    ordered_toolpaths = ordered_fill_paths + ordered_line_paths

    return PlannedDrawing(
        threshold_mask=mask_to_preview_image(threshold_mask),
        processed_mask=mask_to_preview_image(processed_mask),
        shape_mask=mask_to_preview_image(shape_mask),
        thin_mask=mask_to_preview_image(thin_mask),
        fill_mask=mask_to_preview_image(infill_mask),
        toolpaths=ordered_toolpaths,
        vector_loops=vector_loops,
        pixels_per_mm=settings.processing_resolution_ppmm,
    )
