from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_mask_stl(
    mask: np.ndarray,
    output_path: str | Path,
    page_width_mm: float,
    page_height_mm: float,
    feature_height_mm: float,
    center_on_page: bool = True,
) -> int:
    solid = mask > 0
    page_height_px, page_width_px = solid.shape
    cell_width_mm = page_width_mm / page_width_px
    cell_height_mm = page_height_mm / page_height_px
    triangle_count = _count_triangles(solid)

    output = Path(output_path)
    with output.open("wb") as handle:
        header = b"Photo to G-code Cura mask STL"
        handle.write(header.ljust(80, b"\0"))
        handle.write(struct.pack("<I", triangle_count))

        for row_index in range(page_height_px):
            for column_index in range(page_width_px):
                if not solid[row_index, column_index]:
                    continue

                x0 = column_index * cell_width_mm
                x1 = x0 + cell_width_mm
                y1 = page_height_mm - (row_index * cell_height_mm)
                y0 = y1 - cell_height_mm
                z0 = 0.0
                z1 = feature_height_mm

                if center_on_page:
                    x0 -= page_width_mm / 2.0
                    x1 -= page_width_mm / 2.0
                    y0 -= page_height_mm / 2.0
                    y1 -= page_height_mm / 2.0

                _write_quad(
                    handle,
                    (x0, y0, z1),
                    (x1, y0, z1),
                    (x1, y1, z1),
                    (x0, y1, z1),
                )
                _write_quad(
                    handle,
                    (x0, y0, z0),
                    (x0, y1, z0),
                    (x1, y1, z0),
                    (x1, y0, z0),
                )

                if column_index == 0 or not solid[row_index, column_index - 1]:
                    _write_quad(
                        handle,
                        (x0, y0, z0),
                        (x0, y0, z1),
                        (x0, y1, z1),
                        (x0, y1, z0),
                    )
                if column_index == page_width_px - 1 or not solid[row_index, column_index + 1]:
                    _write_quad(
                        handle,
                        (x1, y0, z0),
                        (x1, y1, z0),
                        (x1, y1, z1),
                        (x1, y0, z1),
                    )
                if row_index == 0 or not solid[row_index - 1, column_index]:
                    _write_quad(
                        handle,
                        (x0, y1, z0),
                        (x0, y1, z1),
                        (x1, y1, z1),
                        (x1, y1, z0),
                    )
                if row_index == page_height_px - 1 or not solid[row_index + 1, column_index]:
                    _write_quad(
                        handle,
                        (x0, y0, z0),
                        (x1, y0, z0),
                        (x1, y0, z1),
                        (x0, y0, z1),
                    )

    return triangle_count


def _count_triangles(solid: np.ndarray) -> int:
    page_height_px, page_width_px = solid.shape
    triangle_count = 0

    for row_index in range(page_height_px):
        for column_index in range(page_width_px):
            if not solid[row_index, column_index]:
                continue

            triangle_count += 4
            if column_index == 0 or not solid[row_index, column_index - 1]:
                triangle_count += 2
            if column_index == page_width_px - 1 or not solid[row_index, column_index + 1]:
                triangle_count += 2
            if row_index == 0 or not solid[row_index - 1, column_index]:
                triangle_count += 2
            if row_index == page_height_px - 1 or not solid[row_index + 1, column_index]:
                triangle_count += 2

    return triangle_count


def _write_quad(
    handle,
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
    fourth: tuple[float, float, float],
) -> None:
    _write_triangle(handle, first, second, third)
    _write_triangle(handle, first, third, fourth)


def _write_triangle(
    handle,
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> None:
    handle.write(
        struct.pack(
            "<12fH",
            0.0,
            0.0,
            0.0,
            first[0],
            first[1],
            first[2],
            second[0],
            second[1],
            second[2],
            third[0],
            third[1],
            third[2],
            0,
        )
    )
