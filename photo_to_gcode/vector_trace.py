from __future__ import annotations

from math import ceil, dist

import cv2
import numpy as np
import potrace

from photo_to_gcode.geometry import deduplicate_points, interpolate, pixel_to_mm, signed_area
from photo_to_gcode.models import Toolpath, VectorLoop


def trace_mask_to_vector_loops(
    mask: np.ndarray,
    pixels_per_mm: float,
    sample_step_mm: float,
    min_region_area_mm2: float,
    turdsize: int,
    alphamax: float,
    opttolerance: float,
) -> list[VectorLoop]:
    if cv2.countNonZero(mask) == 0:
        return []

    page_height_px = mask.shape[0]
    min_area_px = min_region_area_mm2 * pixels_per_mm * pixels_per_mm
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    loops: list[VectorLoop] = []

    for label_index in range(1, component_count):
        if stats[label_index, cv2.CC_STAT_AREA] < min_area_px:
            continue

        x_pos = stats[label_index, cv2.CC_STAT_LEFT]
        y_pos = stats[label_index, cv2.CC_STAT_TOP]
        width = stats[label_index, cv2.CC_STAT_WIDTH]
        height = stats[label_index, cv2.CC_STAT_HEIGHT]
        padding_px = 3
        x0 = max(0, x_pos - padding_px)
        y0 = max(0, y_pos - padding_px)
        x1 = min(mask.shape[1], x_pos + width + padding_px)
        y1 = min(mask.shape[0], y_pos + height + padding_px)

        component_mask = np.where(labels[y0:y1, x0:x1] == label_index, 255, 0).astype(np.uint8)
        traced = potrace.Bitmap(component_mask).trace(
            turdsize=turdsize,
            alphamax=alphamax,
            opticurve=True,
            opttolerance=opttolerance,
        )

        for curve in traced:
            sampled_local_points = _sample_curve(curve, sample_step_mm * pixels_per_mm)
            if len(sampled_local_points) < 3:
                continue
            if _touches_crop_border(sampled_local_points, component_mask.shape[1], component_mask.shape[0]):
                continue

            sampled_global_points = [
                (point[0] + x0, point[1] + y0) for point in sampled_local_points
            ]
            points_mm = [
                pixel_to_mm(point, page_height_px=page_height_px, pixels_per_mm=pixels_per_mm)
                for point in sampled_global_points
            ]
            points_mm = deduplicate_points(points_mm)
            if len(points_mm) < 3:
                continue

            loops.append(
                VectorLoop(
                    points=points_mm,
                    is_hole=signed_area(sampled_local_points) > 0,
                )
            )

    return loops


def vector_loops_to_toolpaths(loops: list[VectorLoop], kind: str = "perimeter") -> list[Toolpath]:
    toolpaths: list[Toolpath] = []
    for loop in loops:
        if len(loop.points) < 3:
            continue

        points = loop.points[:]
        if points[0] != points[-1]:
            points.append(points[0])
        toolpaths.append(Toolpath(points=points, closed=True, kind=kind))
    return toolpaths


def _sample_curve(curve: potrace.Curve, sample_step_px: float) -> list[tuple[float, float]]:
    sample_step_px = max(sample_step_px, 1.0)
    points = [(curve.start_point.x, curve.start_point.y)]
    current = points[0]

    for segment in curve.segments:
        end_point = (segment.end_point.x, segment.end_point.y)
        if segment.is_corner:
            corner_point = (segment.c.x, segment.c.y)
            corner_length = dist(current, corner_point) + dist(corner_point, end_point)
            sample_count = max(2, int(ceil(corner_length / sample_step_px)))
            for index in range(1, sample_count + 1):
                ratio = index / sample_count
                if ratio <= 0.5:
                    local_ratio = ratio * 2.0
                    points.append(interpolate(current, corner_point, local_ratio))
                else:
                    local_ratio = (ratio - 0.5) * 2.0
                    points.append(interpolate(corner_point, end_point, local_ratio))
        else:
            control_1 = (segment.c1.x, segment.c1.y)
            control_2 = (segment.c2.x, segment.c2.y)
            control_length = (
                dist(current, control_1)
                + dist(control_1, control_2)
                + dist(control_2, end_point)
            )
            sample_count = max(4, int(ceil(control_length / sample_step_px)))
            for index in range(1, sample_count + 1):
                ratio = index / sample_count
                points.append(_sample_cubic_bezier(current, control_1, control_2, end_point, ratio))
        current = end_point

    return deduplicate_points([(round(x_pos, 3), round(y_pos, 3)) for x_pos, y_pos in points])


def _sample_cubic_bezier(
    start_point: tuple[float, float],
    control_1: tuple[float, float],
    control_2: tuple[float, float],
    end_point: tuple[float, float],
    ratio: float,
) -> tuple[float, float]:
    inverse_ratio = 1.0 - ratio
    return (
        (inverse_ratio**3 * start_point[0])
        + (3 * inverse_ratio**2 * ratio * control_1[0])
        + (3 * inverse_ratio * ratio**2 * control_2[0])
        + (ratio**3 * end_point[0]),
        (inverse_ratio**3 * start_point[1])
        + (3 * inverse_ratio**2 * ratio * control_1[1])
        + (3 * inverse_ratio * ratio**2 * control_2[1])
        + (ratio**3 * end_point[1]),
    )


def _touches_crop_border(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    tolerance_px: float = 0.75,
) -> bool:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        min(xs) <= tolerance_px
        or min(ys) <= tolerance_px
        or max(xs) >= (width - 1 - tolerance_px)
        or max(ys) >= (height - 1 - tolerance_px)
    )
