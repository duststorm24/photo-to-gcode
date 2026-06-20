from __future__ import annotations

from dataclasses import dataclass
from math import atan2, dist

import cv2
import numpy as np

from photo_to_gcode.geometry import pixel_to_mm
from photo_to_gcode.image_processing import mask_to_preview_image
from photo_to_gcode.models import PlannedDrawing, Toolpath
from photo_to_gcode.toolpaths import order_toolpaths, simplify_toolpaths


@dataclass(slots=True)
class TriangleMeshSettings:
    page_width_mm: float
    page_height_mm: float
    processing_resolution_ppmm: float
    threshold: int = 165
    invert_input: bool = False
    min_spacing_mm: float = 3.0
    max_spacing_mm: float = 10.0
    boundary_spacing_mm: float = 2.5


def plan_triangle_mesh_from_tone_map(
    page_tone_map: np.ndarray,
    settings: TriangleMeshSettings,
) -> PlannedDrawing:
    tone_map = np.clip(page_tone_map.astype(np.uint8), 0, 255)
    tone_float = tone_map.astype(np.float32) / 255.0
    darkness = tone_float if settings.invert_input else (1.0 - tone_float)

    blurred = cv2.GaussianBlur(darkness, (0, 0), sigmaX=1.2, sigmaY=1.2)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_peak = float(np.max(gradient_magnitude))
    if gradient_peak > 1e-6:
        edge_strength = gradient_magnitude / gradient_peak
    else:
        edge_strength = np.zeros_like(gradient_magnitude, dtype=np.float32)

    importance = np.clip((darkness * 0.75) + (edge_strength * 0.65), 0.0, 1.0)
    threshold_darkness = (
        float(settings.threshold) / 255.0 if settings.invert_input else (255.0 - float(settings.threshold)) / 255.0
    )
    active_cutoff = max(0.05, threshold_darkness * 0.35)
    active_mask = np.where(np.maximum(darkness, edge_strength * 0.85) >= active_cutoff, 255, 0).astype(np.uint8)

    if cv2.countNonZero(active_mask) == 0:
        active_mask = np.where(darkness > 0.02, 255, 0).astype(np.uint8)

    sampled_points = _sample_adaptive_points(active_mask, importance, settings)
    toolpaths = _build_triangle_edge_toolpaths(sampled_points, active_mask, settings.processing_resolution_ppmm)
    toolpaths = simplify_toolpaths(
        toolpaths,
        tolerance_mm=max(0.18, settings.min_spacing_mm * 0.16),
        min_segment_length_mm=max(0.12, settings.min_spacing_mm * 0.10),
    )
    toolpaths = order_toolpaths(toolpaths)

    preview_mask = mask_to_preview_image(active_mask)
    return PlannedDrawing(
        threshold_mask=preview_mask,
        processed_mask=preview_mask,
        shape_mask=preview_mask,
        thin_mask=mask_to_preview_image(np.zeros_like(active_mask)),
        fill_mask=preview_mask,
        toolpaths=toolpaths,
        vector_loops=[],
        pixels_per_mm=settings.processing_resolution_ppmm,
    )


def _sample_adaptive_points(
    active_mask: np.ndarray,
    importance: np.ndarray,
    settings: TriangleMeshSettings,
) -> list[tuple[int, int]]:
    height_px, width_px = active_mask.shape
    points: set[tuple[int, int]] = set()

    boundary_step_px = max(2, int(round(settings.boundary_spacing_mm * settings.processing_resolution_ppmm)))
    contours, _ = cv2.findContours(active_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        contour_points = contour[:, 0, :]
        if len(contour_points) == 0:
            continue
        for index in range(0, len(contour_points), boundary_step_px):
            x_pos, y_pos = contour_points[index]
            points.add((int(x_pos), int(y_pos)))

    min_spacing_mm = max(1.0, min(settings.min_spacing_mm, settings.max_spacing_mm))
    max_spacing_mm = max(min_spacing_mm, settings.max_spacing_mm)
    mid_spacing_mm = (min_spacing_mm + max_spacing_mm) / 2.0
    layers = (
        (max_spacing_mm, (0.50, 0.50), 0.08),
        (mid_spacing_mm, (0.25, 0.75), 0.32),
        (min_spacing_mm, (0.75, 0.25), 0.60),
    )

    for spacing_mm, (offset_x_ratio, offset_y_ratio), importance_threshold in layers:
        step_px = max(2, int(round(spacing_mm * settings.processing_resolution_ppmm)))
        start_x = int(round(step_px * offset_x_ratio))
        start_y = int(round(step_px * offset_y_ratio))
        for y_pos in range(start_y, height_px, step_px):
            for x_pos in range(start_x, width_px, step_px):
                if active_mask[y_pos, x_pos] == 0:
                    continue
                if float(importance[y_pos, x_pos]) < importance_threshold:
                    continue
                points.add((int(x_pos), int(y_pos)))

    if len(points) < 3:
        ys, xs = np.where(active_mask > 0)
        for x_pos, y_pos in zip(xs[:: max(1, len(xs) // 200 or 1)], ys[:: max(1, len(ys) // 200 or 1)]):
            points.add((int(x_pos), int(y_pos)))

    points.add((0, 0))
    points.add((width_px - 1, 0))
    points.add((0, height_px - 1))
    points.add((width_px - 1, height_px - 1))
    return list(points)


def _build_triangle_edge_toolpaths(
    sampled_points_px: list[tuple[int, int]],
    active_mask: np.ndarray,
    pixels_per_mm: float,
) -> list[Toolpath]:
    height_px, width_px = active_mask.shape
    if len(sampled_points_px) < 3:
        return []

    subdiv = cv2.Subdiv2D((0, 0, width_px, height_px))
    for x_pos, y_pos in sampled_points_px:
        try:
            subdiv.insert((float(x_pos), float(y_pos)))
        except cv2.error:
            continue

    edge_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for triangle in subdiv.getTriangleList():
        points = [
            (float(triangle[0]), float(triangle[1])),
            (float(triangle[2]), float(triangle[3])),
            (float(triangle[4]), float(triangle[5])),
        ]
        if any(x_pos < 0 or y_pos < 0 or x_pos >= width_px or y_pos >= height_px for x_pos, y_pos in points):
            continue

        centroid_x = int(round((points[0][0] + points[1][0] + points[2][0]) / 3.0))
        centroid_y = int(round((points[0][1] + points[1][1] + points[2][1]) / 3.0))
        if not _mask_contains(active_mask, centroid_x, centroid_y):
            continue

        for first, second in ((points[0], points[1]), (points[1], points[2]), (points[2], points[0])):
            mid_x = int(round((first[0] + second[0]) / 2.0))
            mid_y = int(round((first[1] + second[1]) / 2.0))
            if not _mask_contains(active_mask, mid_x, mid_y):
                continue

            start = (int(round(first[0])), int(round(first[1])))
            end = (int(round(second[0])), int(round(second[1])))
            if start == end:
                continue
            edge_set.add(tuple(sorted((start, end))))

    if not edge_set:
        return []

    return _chain_edges_to_toolpaths(edge_set, page_height_px=height_px, pixels_per_mm=pixels_per_mm)


def _chain_edges_to_toolpaths(
    edge_set: set[tuple[tuple[int, int], tuple[int, int]]],
    *,
    page_height_px: int,
    pixels_per_mm: float,
) -> list[Toolpath]:
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for first, second in edge_set:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)

    remaining = set(edge_set)
    toolpaths: list[Toolpath] = []

    while remaining:
        current_edge = next(iter(remaining))
        candidate_starts = [node for node in current_edge if _remaining_degree(node, remaining) <= 1]
        start = candidate_starts[0] if candidate_starts else current_edge[0]
        current = start
        previous: tuple[int, int] | None = None
        path_points = [start]

        while True:
            candidate_neighbors = [
                neighbor for neighbor in adjacency.get(current, set()) if _normalized_edge(current, neighbor) in remaining
            ]
            if not candidate_neighbors:
                break

            next_node = _choose_next_neighbor(current, previous, candidate_neighbors)
            remaining.remove(_normalized_edge(current, next_node))
            path_points.append(next_node)
            previous, current = current, next_node

        if len(path_points) >= 2:
            mm_points = [pixel_to_mm(point, page_height_px, pixels_per_mm) for point in path_points]
            toolpaths.append(Toolpath(points=mm_points, closed=False, kind="perimeter"))

    return toolpaths


def _remaining_degree(
    node: tuple[int, int],
    remaining: set[tuple[tuple[int, int], tuple[int, int]]],
) -> int:
    degree = 0
    for first, second in remaining:
        if first == node or second == node:
            degree += 1
    return degree


def _normalized_edge(
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    return (first, second) if first <= second else (second, first)


def _choose_next_neighbor(
    current: tuple[int, int],
    previous: tuple[int, int] | None,
    candidate_neighbors: list[tuple[int, int]],
) -> tuple[int, int]:
    if previous is None or len(candidate_neighbors) == 1:
        return max(candidate_neighbors, key=lambda neighbor: dist(current, neighbor))

    incoming_angle = atan2(current[1] - previous[1], current[0] - previous[0])

    def _turn_score(neighbor: tuple[int, int]) -> tuple[float, float]:
        outgoing_angle = atan2(neighbor[1] - current[1], neighbor[0] - current[0])
        angle_delta = abs(outgoing_angle - incoming_angle)
        while angle_delta > np.pi:
            angle_delta -= 2.0 * np.pi
        return abs(angle_delta), -dist(current, neighbor)

    return min(candidate_neighbors, key=_turn_score)


def _mask_contains(mask: np.ndarray, x_pos: int, y_pos: int) -> bool:
    if x_pos < 0 or y_pos < 0 or y_pos >= mask.shape[0] or x_pos >= mask.shape[1]:
        return False
    return bool(mask[y_pos, x_pos] > 0)
