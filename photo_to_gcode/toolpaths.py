from __future__ import annotations

from math import acos, dist

import numpy as np

from photo_to_gcode.geometry import pixel_to_mm
from photo_to_gcode.models import Toolpath


def build_hatch_toolpaths(
    mask: np.ndarray,
    pixels_per_mm: float,
    spacing_mm: float,
    axis: str,
    segment_extension_mm: float = 0.0,
) -> list[Toolpath]:
    if spacing_mm <= 0:
        return []

    step_px = max(1, int(round(spacing_mm * pixels_per_mm)))
    page_height_px, page_width_px = mask.shape
    page_width_mm = max(0.0, (page_width_px - 1) / pixels_per_mm)
    page_height_mm = max(0.0, page_height_px / pixels_per_mm)
    extension_mm = max(0.0, float(segment_extension_mm))
    toolpaths: list[Toolpath] = []

    if axis == "horizontal":
        for row_index, y_pos in enumerate(range(0, page_height_px, step_px)):
            segments = _segments_from_line(mask[y_pos, :])
            if row_index % 2 == 1:
                segments.reverse()

            for segment_index, (start_x, end_x) in enumerate(segments):
                start_point = pixel_to_mm((start_x, y_pos), page_height_px, pixels_per_mm)
                end_point = pixel_to_mm((end_x, y_pos), page_height_px, pixels_per_mm)
                segment_points = [
                    (max(0.0, start_point[0] - extension_mm), start_point[1]),
                    (min(page_width_mm, end_point[0] + extension_mm), end_point[1]),
                ]
                if (row_index + segment_index) % 2 == 1:
                    segment_points.reverse()
                toolpaths.append(Toolpath(points=segment_points, closed=False, kind="fill"))

    if axis == "vertical":
        for column_index, x_pos in enumerate(range(0, page_width_px, step_px)):
            segments = _segments_from_line(mask[:, x_pos])
            if column_index % 2 == 1:
                segments.reverse()

            for segment_index, (start_y, end_y) in enumerate(segments):
                start_point = pixel_to_mm((x_pos, start_y), page_height_px, pixels_per_mm)
                end_point = pixel_to_mm((x_pos, end_y), page_height_px, pixels_per_mm)
                segment_points = [
                    (start_point[0], min(page_height_mm, start_point[1] + extension_mm)),
                    (end_point[0], max(0.0, end_point[1] - extension_mm)),
                ]
                if (column_index + segment_index) % 2 == 1:
                    segment_points.reverse()
                toolpaths.append(Toolpath(points=segment_points, closed=False, kind="fill"))

    return toolpaths


def calculate_path_metrics(toolpaths: list[Toolpath]) -> dict[str, float | int]:
    draw_distance_mm = 0.0
    travel_distance_mm = 0.0
    previous_end: tuple[float, float] | None = None
    perimeter_paths = 0
    fill_paths = 0
    centerline_paths = 0

    for toolpath in toolpaths:
        if len(toolpath.points) < 2:
            continue

        if toolpath.kind == "fill":
            fill_paths += 1
        elif toolpath.kind == "centerline":
            centerline_paths += 1
        else:
            perimeter_paths += 1

        start = toolpath.points[0]
        end = toolpath.points[-1]
        if previous_end is not None:
            travel_distance_mm += dist(previous_end, start)

        for first, second in zip(toolpath.points, toolpath.points[1:]):
            draw_distance_mm += dist(first, second)

        previous_end = end

    return {
        "path_count": len(toolpaths),
        "perimeter_paths": perimeter_paths,
        "fill_paths": fill_paths,
        "centerline_paths": centerline_paths,
        "draw_distance_mm": draw_distance_mm,
        "travel_distance_mm": travel_distance_mm,
    }


def merge_continuous_fill_toolpaths(
    toolpaths: list[Toolpath],
    max_connector_gap_mm: float,
) -> list[Toolpath]:
    if max_connector_gap_mm <= 0 or len(toolpaths) <= 1:
        return toolpaths

    merged: list[Toolpath] = []
    current = _clone_toolpath(toolpaths[0])

    for candidate in toolpaths[1:]:
        candidate_copy = _clone_toolpath(candidate)
        if _can_merge_fill_pair(current, candidate_copy, max_connector_gap_mm):
            if current.points and candidate_copy.points and current.points[-1] == candidate_copy.points[0]:
                current.points.extend(candidate_copy.points[1:])
            else:
                current.points.extend(candidate_copy.points)
            continue

        merged.append(current)
        current = candidate_copy

    merged.append(current)
    return merged


def split_fill_toolpaths_at_turns(
    toolpaths: list[Toolpath],
    angle_threshold_degrees: float,
) -> list[Toolpath]:
    if angle_threshold_degrees <= 0:
        return toolpaths

    split_toolpaths: list[Toolpath] = []
    for toolpath in toolpaths:
        if toolpath.kind != "fill" or toolpath.closed or len(toolpath.points) < 3:
            split_toolpaths.append(toolpath)
            continue

        current_points = [toolpath.points[0]]
        for point_index in range(1, len(toolpath.points) - 1):
            current_point = toolpath.points[point_index]
            current_points.append(current_point)

            previous_point = toolpath.points[point_index - 1]
            next_point = toolpath.points[point_index + 1]
            turn_angle = _turn_angle_degrees(previous_point, current_point, next_point)

            if turn_angle >= angle_threshold_degrees and len(current_points) >= 2:
                split_toolpaths.append(
                    Toolpath(points=current_points[:], closed=False, kind=toolpath.kind)
                )
                current_points = [current_point]

        current_points.append(toolpath.points[-1])
        if len(current_points) >= 2:
            split_toolpaths.append(
                Toolpath(points=current_points, closed=False, kind=toolpath.kind)
            )

    return split_toolpaths


def split_fill_toolpaths_by_segment_count(
    toolpaths: list[Toolpath],
    max_segments_per_toolpath: int,
) -> list[Toolpath]:
    if max_segments_per_toolpath <= 0:
        return toolpaths

    split_toolpaths: list[Toolpath] = []
    for toolpath in toolpaths:
        if toolpath.kind != "fill" or toolpath.closed or len(toolpath.points) <= max_segments_per_toolpath + 1:
            split_toolpaths.append(toolpath)
            continue

        segment_start = 0
        last_index = len(toolpath.points) - 1
        while segment_start < last_index:
            segment_end = min(segment_start + max_segments_per_toolpath, last_index)
            chunk_points = toolpath.points[segment_start : segment_end + 1]
            if len(chunk_points) >= 2:
                split_toolpaths.append(
                    Toolpath(points=chunk_points, closed=False, kind=toolpath.kind)
                )
            segment_start = segment_end

    return split_toolpaths


def order_toolpaths(toolpaths: list[Toolpath]) -> list[Toolpath]:
    if len(toolpaths) <= 1:
        return toolpaths

    remaining = [Toolpath(points=toolpath.points[:], closed=toolpath.closed, kind=toolpath.kind) for toolpath in toolpaths]
    ordered: list[Toolpath] = []
    current_end: tuple[float, float] | None = None

    while remaining:
        if current_end is None:
            next_index = min(range(len(remaining)), key=lambda index: _path_sort_key(remaining[index]))
            chosen = remaining.pop(next_index)
            chosen = _normalize_first_path(chosen)
        else:
            best_index = 0
            best_distance = float("inf")
            best_path = remaining[0]
            for index, candidate in enumerate(remaining):
                oriented_candidate, candidate_distance = _orient_path(candidate, current_end)
                if candidate_distance < best_distance:
                    best_distance = candidate_distance
                    best_index = index
                    best_path = oriented_candidate
            remaining.pop(best_index)
            chosen = best_path

        ordered.append(chosen)
        current_end = chosen.points[-1]

    return ordered


def simplify_toolpaths(
    toolpaths: list[Toolpath],
    *,
    tolerance_mm: float = 0.0,
    min_segment_length_mm: float = 0.0,
) -> list[Toolpath]:
    if tolerance_mm <= 0 and min_segment_length_mm <= 0:
        return toolpaths

    simplified: list[Toolpath] = []
    for toolpath in toolpaths:
        simplified_points = _simplify_points(
            toolpath.points,
            tolerance_mm=tolerance_mm,
            min_segment_length_mm=min_segment_length_mm,
            closed=toolpath.closed,
        )
        if len(simplified_points) < 2:
            continue
        simplified.append(
            Toolpath(points=simplified_points, closed=toolpath.closed, kind=toolpath.kind)
        )
    return simplified


def _clone_toolpath(toolpath: Toolpath) -> Toolpath:
    return Toolpath(points=toolpath.points[:], closed=toolpath.closed, kind=toolpath.kind)


def _simplify_points(
    points: list[tuple[float, float]],
    *,
    tolerance_mm: float,
    min_segment_length_mm: float,
    closed: bool,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points[:]

    working_points = points[:]
    if closed and len(working_points) >= 2 and working_points[0] == working_points[-1]:
        working_points = working_points[:-1]

    if min_segment_length_mm > 0:
        filtered_points = [working_points[0]]
        for point in working_points[1:]:
            if dist(filtered_points[-1], point) >= min_segment_length_mm:
                filtered_points.append(point)
        if len(filtered_points) == 1 and len(working_points) > 1:
            filtered_points.append(working_points[-1])
        working_points = filtered_points

    if tolerance_mm > 0 and len(working_points) >= 3:
        working_points = _rdp_points(working_points, tolerance_mm)

    if min_segment_length_mm > 0 and len(working_points) >= 2:
        filtered_points = [working_points[0]]
        for index, point in enumerate(working_points[1:], start=1):
            is_last = index == len(working_points) - 1
            if is_last or dist(filtered_points[-1], point) >= min_segment_length_mm:
                filtered_points.append(point)
        working_points = filtered_points

    if closed and working_points:
        working_points.append(working_points[0])
    return working_points


def _segments_from_line(line: np.ndarray) -> list[tuple[int, int]]:
    active_indices = np.where(line > 0)[0]
    if active_indices.size == 0:
        return []

    splits = np.where(np.diff(active_indices) > 1)[0] + 1
    runs = np.split(active_indices, splits)
    return [(int(run[0]), int(run[-1])) for run in runs if run.size >= 2]


def _can_merge_fill_pair(
    first: Toolpath,
    second: Toolpath,
    max_connector_gap_mm: float,
) -> bool:
    if first.kind != "fill" or second.kind != "fill":
        return False
    if first.closed or second.closed:
        return False
    if len(first.points) < 2 or len(second.points) < 2:
        return False
    return dist(first.points[-1], second.points[0]) <= max_connector_gap_mm


def _rdp_points(
    points: list[tuple[float, float]],
    tolerance_mm: float,
) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points[:]

    max_distance = -1.0
    split_index = -1
    start = points[0]
    end = points[-1]
    for index in range(1, len(points) - 1):
        distance_to_line = _point_line_distance(points[index], start, end)
        if distance_to_line > max_distance:
            max_distance = distance_to_line
            split_index = index

    if max_distance <= tolerance_mm or split_index < 0:
        return [start, end]

    left = _rdp_points(points[: split_index + 1], tolerance_mm)
    right = _rdp_points(points[split_index:], tolerance_mm)
    return left[:-1] + right


def _point_line_distance(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    if line_start == line_end:
        return dist(point, line_start)

    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    numerator = abs(((y2 - y1) * px) - ((x2 - x1) * py) + (x2 * y1) - (y2 * x1))
    denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return numerator / denominator


def _turn_angle_degrees(
    previous_point: tuple[float, float],
    current_point: tuple[float, float],
    next_point: tuple[float, float],
) -> float:
    incoming = (
        current_point[0] - previous_point[0],
        current_point[1] - previous_point[1],
    )
    outgoing = (
        next_point[0] - current_point[0],
        next_point[1] - current_point[1],
    )
    incoming_length = (incoming[0] ** 2 + incoming[1] ** 2) ** 0.5
    outgoing_length = (outgoing[0] ** 2 + outgoing[1] ** 2) ** 0.5
    if incoming_length <= 1e-9 or outgoing_length <= 1e-9:
        return 0.0

    cosine = (
        (incoming[0] * outgoing[0]) + (incoming[1] * outgoing[1])
    ) / (incoming_length * outgoing_length)
    cosine = max(-1.0, min(1.0, cosine))
    return acos(cosine) * (180.0 / np.pi)


def _path_sort_key(toolpath: Toolpath) -> tuple[float, float]:
    normalized = _normalize_first_path(toolpath)
    return normalized.points[0][1], normalized.points[0][0]


def _orient_path(
    toolpath: Toolpath,
    current_end: tuple[float, float],
) -> tuple[Toolpath, float]:
    if not toolpath.points:
        return toolpath, float("inf")

    if toolpath.closed:
        rotated = _rotate_closed_path_to_nearest(toolpath, current_end)
        return rotated, dist(current_end, rotated.points[0])

    start_distance = dist(current_end, toolpath.points[0])
    end_distance = dist(current_end, toolpath.points[-1])
    if end_distance < start_distance:
        reversed_points = list(reversed(toolpath.points))
        return Toolpath(points=reversed_points, closed=False, kind=toolpath.kind), end_distance
    return toolpath, start_distance


def _normalize_first_path(toolpath: Toolpath) -> Toolpath:
    if not toolpath.closed:
        return toolpath
    return _rotate_closed_path_to_nearest(toolpath, toolpath.points[0])


def _rotate_closed_path_to_nearest(
    toolpath: Toolpath,
    target: tuple[float, float],
) -> Toolpath:
    body = toolpath.points[:-1] if toolpath.points[0] == toolpath.points[-1] else toolpath.points[:]
    if not body:
        return toolpath

    best_index = min(range(len(body)), key=lambda index: dist(body[index], target))
    rotated = body[best_index:] + body[:best_index]
    rotated.append(rotated[0])
    return Toolpath(points=rotated, closed=True, kind=toolpath.kind)
