from __future__ import annotations

from skimage.morphology import skeletonize

from photo_to_gcode.geometry import deduplicate_points, pixel_to_mm, smooth_path
from photo_to_gcode.models import Toolpath

PixelNode = tuple[int, int]


def trace_centerlines(
    mask,
    pixels_per_mm: float,
    curve_smoothing_passes: int,
    curve_sample_step_mm: float,
    min_length_mm: float,
) -> list[Toolpath]:
    skeleton = skeletonize(mask > 0)
    if skeleton.sum() == 0:
        return []

    points = {(int(y_pos), int(x_pos)) for y_pos, x_pos in zip(*skeleton.nonzero())}
    neighbors = {point: _neighbors(point, points) for point in points}
    nodes = {point for point, point_neighbors in neighbors.items() if len(point_neighbors) != 2}
    visited_edges: set[tuple[PixelNode, PixelNode]] = set()
    pixel_paths: list[list[PixelNode]] = []

    for node in nodes:
        for neighbor in neighbors[node]:
            edge = _edge_key(node, neighbor)
            if edge in visited_edges:
                continue
            pixel_paths.append(_trace_path(node, neighbor, neighbors, nodes, visited_edges))

    for point, point_neighbors in neighbors.items():
        for neighbor in point_neighbors:
            edge = _edge_key(point, neighbor)
            if edge in visited_edges:
                continue
            pixel_paths.append(_trace_cycle(point, neighbor, neighbors, visited_edges))

    page_height_px = mask.shape[0]
    toolpaths: list[Toolpath] = []
    for pixel_path in pixel_paths:
        if len(pixel_path) < 2:
            continue

        closed = pixel_path[0] == pixel_path[-1]
        points_mm = [
            pixel_to_mm(
                (x_pos + 0.5, y_pos + 0.5),
                page_height_px=page_height_px,
                pixels_per_mm=pixels_per_mm,
            )
            for y_pos, x_pos in pixel_path
        ]
        points_mm = deduplicate_points(points_mm)
        if len(points_mm) < 2:
            continue

        if curve_smoothing_passes > 0 and len(points_mm) >= 3:
            points_mm = smooth_path(
                points_mm,
                closed=closed,
                passes=max(1, curve_smoothing_passes),
                sample_step_mm=curve_sample_step_mm,
            )

        if closed and points_mm[0] != points_mm[-1]:
            points_mm.append(points_mm[0])

        if _path_length(points_mm) < min_length_mm:
            continue

        toolpaths.append(Toolpath(points=points_mm, closed=closed, kind="centerline"))

    return toolpaths


def _trace_path(
    start_node: PixelNode,
    next_node: PixelNode,
    neighbors: dict[PixelNode, list[PixelNode]],
    nodes: set[PixelNode],
    visited_edges: set[tuple[PixelNode, PixelNode]],
) -> list[PixelNode]:
    path = [start_node]
    previous = start_node
    current = next_node
    visited_edges.add(_edge_key(start_node, next_node))

    while True:
        path.append(current)
        if current in nodes and current != start_node:
            break

        next_options = [neighbor for neighbor in neighbors[current] if neighbor != previous]
        if not next_options:
            break

        upcoming = next_options[0]
        edge = _edge_key(current, upcoming)
        if edge in visited_edges:
            break

        visited_edges.add(edge)
        previous, current = current, upcoming

    return path


def _trace_cycle(
    start_node: PixelNode,
    next_node: PixelNode,
    neighbors: dict[PixelNode, list[PixelNode]],
    visited_edges: set[tuple[PixelNode, PixelNode]],
) -> list[PixelNode]:
    path = [start_node]
    previous = start_node
    current = next_node
    visited_edges.add(_edge_key(start_node, next_node))

    while True:
        path.append(current)
        next_options = [neighbor for neighbor in neighbors[current] if neighbor != previous]
        if not next_options:
            break

        upcoming = next_options[0]
        edge = _edge_key(current, upcoming)
        if edge in visited_edges:
            if upcoming == start_node:
                path.append(upcoming)
            break

        visited_edges.add(edge)
        previous, current = current, upcoming

    return path


def _neighbors(point: PixelNode, points: set[PixelNode]) -> list[PixelNode]:
    y_pos, x_pos = point
    neighbors: list[PixelNode] = []
    for y_offset in (-1, 0, 1):
        for x_offset in (-1, 0, 1):
            if y_offset == 0 and x_offset == 0:
                continue
            neighbor = (y_pos + y_offset, x_pos + x_offset)
            if neighbor in points:
                neighbors.append(neighbor)
    return neighbors


def _edge_key(first: PixelNode, second: PixelNode) -> tuple[PixelNode, PixelNode]:
    return (first, second) if first <= second else (second, first)


def _path_length(points: list[tuple[float, float]]) -> float:
    length = 0.0
    for first, second in zip(points, points[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length += (dx * dx + dy * dy) ** 0.5
    return length
