from __future__ import annotations

from PIL import Image, ImageDraw

from photo_to_gcode.geometry import signed_area
from photo_to_gcode.models import Toolpath, VectorLoop

PATH_COLORS = {
    "fill": (78, 131, 177),
    "perimeter": (20, 20, 20),
    "centerline": (201, 87, 53),
}


def render_toolpath_preview(
    toolpaths: list[Toolpath],
    page_width_mm: float,
    page_height_mm: float,
    preview_width_px: int = 900,
    preview_height_px: int = 650,
    line_width_mm: float = 0.20,
    oversample_factor: int = 3,
) -> Image.Image:
    oversample_factor = max(1, int(oversample_factor))
    render_width_px = preview_width_px * oversample_factor
    render_height_px = preview_height_px * oversample_factor
    image = Image.new("RGB", (render_width_px, render_height_px), "white")
    draw = ImageDraw.Draw(image)
    transform = _page_transform(render_width_px, render_height_px, page_width_mm, page_height_mm)

    _draw_page_frame(draw, transform, oversample_factor=oversample_factor)

    for toolpath in toolpaths:
        if len(toolpath.points) < 2:
            continue

        preview_points = [_transform_point(point, transform) for point in toolpath.points]
        color = PATH_COLORS.get(toolpath.kind, PATH_COLORS["perimeter"])
        line_width = _toolpath_line_width_px(
            toolpath,
            transform,
            line_width_mm=line_width_mm,
        )
        draw.line(preview_points, fill=color, width=line_width)

    if oversample_factor > 1:
        image = image.resize((preview_width_px, preview_height_px), Image.Resampling.LANCZOS)
    return image


def render_toolpath_simulation_preview(
    toolpaths: list[Toolpath],
    page_width_mm: float,
    page_height_mm: float,
    progress_ratio: float,
    preview_width_px: int = 900,
    preview_height_px: int = 650,
    line_width_mm: float = 0.20,
    oversample_factor: int = 3,
) -> Image.Image:
    oversample_factor = max(1, int(oversample_factor))
    render_width_px = preview_width_px * oversample_factor
    render_height_px = preview_height_px * oversample_factor
    image = Image.new("RGB", (render_width_px, render_height_px), "white")
    draw = ImageDraw.Draw(image)
    transform = _page_transform(render_width_px, render_height_px, page_width_mm, page_height_mm)

    _draw_page_frame(draw, transform, oversample_factor=oversample_factor)

    segments = _flatten_segments(toolpaths)
    if not segments:
        if oversample_factor > 1:
            image = image.resize((preview_width_px, preview_height_px), Image.Resampling.LANCZOS)
        return image

    completed_count = max(0, min(len(segments), int(round(len(segments) * progress_ratio))))
    for index, (start, end, kind) in enumerate(segments):
        preview_points = [_transform_point(start, transform), _transform_point(end, transform)]
        if index < completed_count:
            color = (20, 20, 20)
            width = _line_width_px_from_kind(kind, transform, line_width_mm=line_width_mm)
        else:
            color = (190, 205, 220) if kind == "fill" else (215, 215, 215)
            width = max(1, _line_width_px_from_kind(kind, transform, line_width_mm=line_width_mm) // 2)
        draw.line(preview_points, fill=color, width=width)

    if completed_count <= 0:
        tip_point = segments[0][0]
    else:
        tip_point = segments[min(completed_count - 1, len(segments) - 1)][1]
    preview_tip = _transform_point(tip_point, transform)
    tip_radius = max(3, oversample_factor * 2)
    draw.ellipse(
        (
            preview_tip[0] - tip_radius,
            preview_tip[1] - tip_radius,
            preview_tip[0] + tip_radius,
            preview_tip[1] + tip_radius,
        ),
        fill=(220, 65, 54),
        outline=(90, 20, 14),
    )

    if oversample_factor > 1:
        image = image.resize((preview_width_px, preview_height_px), Image.Resampling.LANCZOS)
    return image


def render_vector_preview(
    vector_loops: list[VectorLoop],
    centerline_paths: list[Toolpath],
    page_width_mm: float,
    page_height_mm: float,
    preview_width_px: int = 900,
    preview_height_px: int = 650,
) -> Image.Image:
    image = Image.new("RGB", (preview_width_px, preview_height_px), "white")
    draw = ImageDraw.Draw(image)
    transform = _page_transform(preview_width_px, preview_height_px, page_width_mm, page_height_mm)

    _draw_page_frame(draw, transform)

    sorted_loops = sorted(vector_loops, key=lambda loop: abs(signed_area(loop.points)), reverse=True)
    for loop in sorted_loops:
        if len(loop.points) < 3:
            continue

        preview_points = [_transform_point(point, transform) for point in loop.points]
        fill_color = "white" if loop.is_hole else (32, 32, 32)
        outline_color = (160, 160, 160) if loop.is_hole else (48, 48, 48)
        draw.polygon(preview_points, fill=fill_color, outline=outline_color)

    for toolpath in centerline_paths:
        if len(toolpath.points) < 2:
            continue
        preview_points = [_transform_point(point, transform) for point in toolpath.points]
        draw.line(preview_points, fill=PATH_COLORS["centerline"], width=2)

    return image


def _page_transform(
    preview_width_px: int,
    preview_height_px: int,
    page_width_mm: float,
    page_height_mm: float,
) -> dict[str, float]:
    padding_px = 28
    scale = min(
        (preview_width_px - (padding_px * 2)) / page_width_mm,
        (preview_height_px - (padding_px * 2)) / page_height_mm,
    )
    page_pixel_width = page_width_mm * scale
    page_pixel_height = page_height_mm * scale
    offset_x = (preview_width_px - page_pixel_width) / 2.0
    offset_y = (preview_height_px - page_pixel_height) / 2.0
    return {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "page_width_px": page_pixel_width,
        "page_height_px": page_pixel_height,
        "page_height_mm": page_height_mm,
    }


def _draw_page_frame(
    draw: ImageDraw.ImageDraw,
    transform: dict[str, float],
    *,
    oversample_factor: int = 1,
) -> None:
    draw.rectangle(
        (
            transform["offset_x"],
            transform["offset_y"],
            transform["offset_x"] + transform["page_width_px"],
            transform["offset_y"] + transform["page_height_px"],
        ),
        outline=(180, 180, 180),
        width=max(1, oversample_factor),
    )


def _transform_point(point: tuple[float, float], transform: dict[str, float]) -> tuple[float, float]:
    x_mm, y_mm = point
    return (
        transform["offset_x"] + (x_mm * transform["scale"]),
        transform["offset_y"] + ((transform["page_height_mm"] - y_mm) * transform["scale"]),
    )


def _flatten_segments(
    toolpaths: list[Toolpath],
) -> list[tuple[tuple[float, float], tuple[float, float], str]]:
    segments: list[tuple[tuple[float, float], tuple[float, float], str]] = []
    for toolpath in toolpaths:
        if len(toolpath.points) < 2:
            continue
        for start, end in zip(toolpath.points, toolpath.points[1:]):
            segments.append((start, end, toolpath.kind))
    return segments


def _toolpath_line_width_px(
    toolpath: Toolpath,
    transform: dict[str, float],
    *,
    line_width_mm: float,
) -> int:
    return _line_width_px_from_kind(toolpath.kind, transform, line_width_mm=line_width_mm)


def _line_width_px_from_kind(
    kind: str,
    transform: dict[str, float],
    *,
    line_width_mm: float,
) -> int:
    mm_width = max(0.03, float(line_width_mm))
    if kind == "centerline":
        mm_width *= 0.9
    elif kind == "fill":
        mm_width *= 0.85
    width_px = int(round(mm_width * transform["scale"]))
    return max(1, width_px)
