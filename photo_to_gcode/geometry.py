from __future__ import annotations

from itertools import pairwise
from math import dist


def pixel_to_mm(
    point: tuple[float, float],
    page_height_px: int,
    pixels_per_mm: float,
) -> tuple[float, float]:
    x_pos, y_pos = point
    x_mm = round(x_pos / pixels_per_mm, 3)
    y_mm = round((page_height_px - y_pos) / pixels_per_mm, 3)
    return x_mm, y_mm


def deduplicate_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []

    deduplicated = [points[0]]
    for point in points[1:]:
        if point != deduplicated[-1]:
            deduplicated.append(point)
    return deduplicated


def signed_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for first, second in zip(points, points[1:] + points[:1]):
        area += (first[0] * second[1]) - (second[0] * first[1])
    return area / 2.0


def smooth_path(
    points: list[tuple[float, float]],
    closed: bool,
    passes: int,
    sample_step_mm: float,
) -> list[tuple[float, float]]:
    working_points = points[:-1] if closed and points[0] == points[-1] else points[:]
    if len(working_points) < 3:
        return points

    for _ in range(passes):
        working_points = _chaikin_subdivide(working_points, closed=closed)
        if len(working_points) < 3:
            break

    if sample_step_mm > 0:
        working_points = _resample_path(
            working_points,
            closed=closed,
            sample_step_mm=sample_step_mm,
        )

    return working_points


def interpolate(
    first: tuple[float, float],
    second: tuple[float, float],
    ratio: float,
) -> tuple[float, float]:
    inverse_ratio = 1.0 - ratio
    return (
        (first[0] * inverse_ratio) + (second[0] * ratio),
        (first[1] * inverse_ratio) + (second[1] * ratio),
    )


def _chaikin_subdivide(
    points: list[tuple[float, float]],
    closed: bool,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points

    subdivided: list[tuple[float, float]] = []
    if closed:
        pairs = list(zip(points, points[1:] + points[:1]))
        for first, second in pairs:
            subdivided.append(interpolate(first, second, 0.25))
            subdivided.append(interpolate(first, second, 0.75))
        return subdivided

    subdivided.append(points[0])
    for first, second in pairwise(points):
        subdivided.append(interpolate(first, second, 0.25))
        subdivided.append(interpolate(first, second, 0.75))
    subdivided.append(points[-1])
    return subdivided


def _resample_path(
    points: list[tuple[float, float]],
    closed: bool,
    sample_step_mm: float,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points

    source_points = points + [points[0]] if closed else points[:]
    resampled = [source_points[0]]
    distance_since_last = 0.0

    for start, end in pairwise(source_points):
        segment_length = dist(start, end)
        if segment_length == 0:
            continue

        traveled = sample_step_mm - distance_since_last
        while traveled < segment_length:
            ratio = traveled / segment_length
            resampled.append(interpolate(start, end, ratio))
            traveled += sample_step_mm
        distance_since_last = max(0.0, traveled - segment_length)

    if not closed and resampled[-1] != source_points[-1]:
        resampled.append(source_points[-1])
    if closed and len(resampled) > 1 and resampled[-1] == resampled[0]:
        resampled.pop()

    return deduplicate_points(
        [(round(x_pos, 3), round(y_pos, 3)) for x_pos, y_pos in resampled]
    )
