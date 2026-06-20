from __future__ import annotations

from math import dist

from photo_to_gcode.models import Toolpath
from photo_to_gcode.toolpaths import (
    merge_continuous_fill_toolpaths,
    order_toolpaths,
    simplify_toolpaths,
    split_fill_toolpaths_by_segment_count,
    split_fill_toolpaths_at_turns,
)

COMMENT_KIND_MAP = {
    "WALL-OUTER": "perimeter",
    "WALL-INNER": "perimeter",
    "SKIN": "fill",
    "FILL": "fill",
}


def _dedupe_consecutive_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []

    deduped = [points[0]]
    for point in points[1:]:
        if point != deduped[-1]:
            deduped.append(point)
    return deduped


def convert_cura_gcode_to_plotter(
    raw_gcode: str,
    pen_up_command: str,
    pen_down_command: str,
    pen_pause_seconds: float,
    draw_speed_mm_per_s: float,
    fill_mode: str = "continuous_zigzag",
    fill_turn_split_angle_degrees: float = 35.0,
    continuous_fill_chunk_segments: int = 0,
    path_simplify_tolerance_mm: float = 0.0,
    min_segment_length_mm: float = 0.0,
    coordinate_decimals: int = 3,
) -> tuple[str, list[Toolpath]]:
    toolpaths = extract_toolpaths_from_cura(raw_gcode)
    plotter_gcode = build_plotter_gcode(
        toolpaths,
        pen_up_command=pen_up_command,
        pen_down_command=pen_down_command,
        pen_pause_seconds=pen_pause_seconds,
        draw_speed_mm_per_s=draw_speed_mm_per_s,
        fill_mode=fill_mode,
        fill_turn_split_angle_degrees=fill_turn_split_angle_degrees,
        continuous_fill_chunk_segments=continuous_fill_chunk_segments,
        path_simplify_tolerance_mm=path_simplify_tolerance_mm,
        min_segment_length_mm=min_segment_length_mm,
        coordinate_decimals=coordinate_decimals,
    )
    return plotter_gcode, toolpaths


def build_plotter_gcode(
    toolpaths: list[Toolpath],
    pen_up_command: str,
    pen_down_command: str,
    pen_pause_seconds: float,
    draw_speed_mm_per_s: float,
    fill_mode: str = "continuous_zigzag",
    fill_turn_split_angle_degrees: float = 35.0,
    continuous_fill_chunk_segments: int = 0,
    path_simplify_tolerance_mm: float = 0.0,
    min_segment_length_mm: float = 0.0,
    min_toolpath_length_mm: float = 0.0,
    coordinate_decimals: int = 3,
    toolpaths_already_prepared: bool = False,
) -> str:
    if not toolpaths_already_prepared:
        toolpaths = prepare_plotter_toolpaths(
            toolpaths,
            fill_mode=fill_mode,
            fill_turn_split_angle_degrees=fill_turn_split_angle_degrees,
            continuous_fill_chunk_segments=continuous_fill_chunk_segments,
            path_simplify_tolerance_mm=path_simplify_tolerance_mm,
            min_segment_length_mm=min_segment_length_mm,
            min_toolpath_length_mm=min_toolpath_length_mm,
        )
    draw_feed_rate_mm_min = max(1, int(round(draw_speed_mm_per_s * 60.0)))
    coordinate_decimals = max(0, min(4, int(coordinate_decimals)))

    lines = [
        "; Post-processed from CuraEngine output for pen plotting",
        "G21 ; millimeters",
        "G90 ; absolute positioning",
        pen_up_command,
    ]

    if pen_pause_seconds > 0:
        lines.append(f"G4 P{pen_pause_seconds:.2f}")

    for toolpath in toolpaths:
        toolpath_points = _dedupe_consecutive_points(toolpath.points)
        if len(toolpath_points) < 2:
            continue

        start_x, start_y = toolpath_points[0]
        lines.append(_format_xy_command("G0", start_x, start_y, coordinate_decimals))
        lines.append(pen_down_command)
        if pen_pause_seconds > 0:
            lines.append(f"G4 P{pen_pause_seconds:.2f}")

        draw_points = toolpath_points[1:]
        first_x, first_y = draw_points[0]
        lines.append(
            _format_xy_command("G1", first_x, first_y, coordinate_decimals)
            + f" F{draw_feed_rate_mm_min}"
        )
        for x_pos, y_pos in draw_points[1:]:
            lines.append(_format_xy_command("G1", x_pos, y_pos, coordinate_decimals))

        lines.append(pen_up_command)
        if pen_pause_seconds > 0:
            lines.append(f"G4 P{pen_pause_seconds:.2f}")

    lines.append(pen_up_command)
    return "\n".join(lines) + "\n"


def prepare_plotter_toolpaths(
    toolpaths: list[Toolpath],
    *,
    fill_mode: str = "continuous_zigzag",
    fill_turn_split_angle_degrees: float = 35.0,
    continuous_fill_chunk_segments: int = 0,
    path_simplify_tolerance_mm: float = 0.0,
    min_segment_length_mm: float = 0.0,
    min_toolpath_length_mm: float = 0.0,
) -> list[Toolpath]:
    toolpaths = _order_fill_before_lines(toolpaths)
    if fill_mode == "continuous_zigzag":
        toolpaths = merge_continuous_fill_toolpaths(toolpaths, max_connector_gap_mm=1.5)
        toolpaths = split_fill_toolpaths_by_segment_count(
            toolpaths,
            max_segments_per_toolpath=continuous_fill_chunk_segments,
        )
    elif fill_mode == "pen_lift_fill":
        toolpaths = split_fill_toolpaths_at_turns(
            toolpaths,
            angle_threshold_degrees=fill_turn_split_angle_degrees,
        )
    fill_paths = [toolpath for toolpath in toolpaths if toolpath.kind == "fill"]
    line_paths = [toolpath for toolpath in toolpaths if toolpath.kind != "fill"]
    line_paths = simplify_toolpaths(
        line_paths,
        tolerance_mm=path_simplify_tolerance_mm,
        min_segment_length_mm=min_segment_length_mm,
    )
    line_paths = _filter_short_toolpaths(line_paths, min_toolpath_length_mm)
    return _order_fill_before_lines(fill_paths + line_paths)


def _format_xy_command(prefix: str, x_pos: float, y_pos: float, decimals: int) -> str:
    return f"{prefix} X{x_pos:.{decimals}f} Y{y_pos:.{decimals}f}"


def _filter_short_toolpaths(
    toolpaths: list[Toolpath],
    min_toolpath_length_mm: float,
) -> list[Toolpath]:
    if min_toolpath_length_mm <= 0:
        return toolpaths

    filtered: list[Toolpath] = []
    for toolpath in toolpaths:
        if len(toolpath.points) < 2:
            continue
        total_length_mm = sum(
            dist(first, second) for first, second in zip(toolpath.points, toolpath.points[1:])
        )
        if total_length_mm >= min_toolpath_length_mm:
            filtered.append(toolpath)
    return filtered


def _order_fill_before_lines(toolpaths: list[Toolpath]) -> list[Toolpath]:
    fill_paths = [toolpath for toolpath in toolpaths if toolpath.kind == "fill"]
    line_paths = [toolpath for toolpath in toolpaths if toolpath.kind != "fill"]
    return order_toolpaths(fill_paths) + order_toolpaths(line_paths)


def extract_toolpaths_from_cura(raw_gcode: str) -> list[Toolpath]:
    toolpaths: list[Toolpath] = []
    active_points: list[tuple[float, float]] = []
    active_kind = "perimeter"
    current_kind = "perimeter"
    absolute_positioning = True
    absolute_extrusion = True
    current_x = 0.0
    current_y = 0.0
    current_e = 0.0

    def flush_active() -> None:
        nonlocal active_points
        if len(active_points) >= 2:
            toolpaths.append(Toolpath(points=active_points, closed=False, kind=active_kind))
        active_points = []

    for raw_line in raw_gcode.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        if stripped.startswith(";TYPE:"):
            flush_active()
            current_kind = COMMENT_KIND_MAP.get(stripped.removeprefix(";TYPE:"), "perimeter")
            continue

        line = stripped.split(";", 1)[0].strip()
        if not line:
            continue

        tokens = line.split()
        command = tokens[0].upper()

        if command == "G90":
            absolute_positioning = True
            continue
        if command == "G91":
            absolute_positioning = False
            continue
        if command == "M82":
            absolute_extrusion = True
            continue
        if command == "M83":
            absolute_extrusion = False
            continue
        if command == "G92":
            parameters = _parse_parameters(tokens[1:])
            if "X" in parameters:
                current_x = parameters["X"]
            if "Y" in parameters:
                current_y = parameters["Y"]
            if "E" in parameters:
                current_e = parameters["E"]
            continue
        if command not in {"G0", "G1"}:
            continue

        parameters = _parse_parameters(tokens[1:])
        next_x = current_x
        next_y = current_y
        if "X" in parameters:
            next_x = parameters["X"] if absolute_positioning else current_x + parameters["X"]
        if "Y" in parameters:
            next_y = parameters["Y"] if absolute_positioning else current_y + parameters["Y"]

        next_e = current_e
        extruding = False
        if "E" in parameters:
            if absolute_extrusion:
                next_e = parameters["E"]
                extruding = next_e > current_e + 1e-9
            else:
                next_e = current_e + parameters["E"]
                extruding = parameters["E"] > 1e-9

        moved_xy = (abs(next_x - current_x) > 1e-9) or (abs(next_y - current_y) > 1e-9)
        if moved_xy and extruding:
            start_point = (current_x, current_y)
            end_point = (next_x, next_y)
            if not active_points:
                active_points = [start_point, end_point]
                active_kind = current_kind
            elif active_kind != current_kind or not _points_match(active_points[-1], start_point):
                flush_active()
                active_points = [start_point, end_point]
                active_kind = current_kind
            else:
                active_points.append(end_point)
        elif moved_xy:
            flush_active()

        current_x = next_x
        current_y = next_y
        current_e = next_e

    flush_active()
    return toolpaths


def translate_toolpaths(
    toolpaths: list[Toolpath],
    delta_x_mm: float,
    delta_y_mm: float,
) -> list[Toolpath]:
    translated: list[Toolpath] = []
    for toolpath in toolpaths:
        translated_points = [
            (x_pos + delta_x_mm, y_pos + delta_y_mm)
            for x_pos, y_pos in toolpath.points
        ]
        translated.append(
            Toolpath(points=translated_points, closed=toolpath.closed, kind=toolpath.kind)
        )
    return translated


def toolpath_bounds_mm(
    toolpaths: list[Toolpath],
) -> tuple[float, float, float, float] | None:
    xs = [x_pos for toolpath in toolpaths for x_pos, _ in toolpath.points]
    ys = [y_pos for toolpath in toolpaths for _, y_pos in toolpath.points]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _parse_parameters(tokens: list[str]) -> dict[str, float]:
    parameters: dict[str, float] = {}
    for token in tokens:
        if len(token) < 2:
            continue
        key = token[0].upper()
        try:
            parameters[key] = float(token[1:])
        except ValueError:
            continue
    return parameters


def _points_match(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return abs(first[0] - second[0]) <= 1e-9 and abs(first[1] - second[1]) <= 1e-9
