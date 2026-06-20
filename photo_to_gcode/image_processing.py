from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from photo_to_gcode.models import Contour


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    rgba_image = image.convert("RGBA")
    white_background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white_background, rgba_image)
    return np.array(composited.convert("RGB"))


def to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def build_binary_mask(
    grayscale_image: np.ndarray,
    threshold: int,
    invert_input: bool = False,
) -> np.ndarray:
    threshold_mode = cv2.THRESH_BINARY if invert_input else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(grayscale_image, threshold, 255, threshold_mode)
    return mask


def build_page_mask(
    binary_mask: np.ndarray,
    page_width_mm: float,
    page_height_mm: float,
    margin_mm: float,
    pixels_per_mm: float,
    scale_multiplier: float = 1.0,
    rotation_degrees: float = 0.0,
    offset_x_mm: float = 0.0,
    offset_y_mm: float = 0.0,
) -> np.ndarray:
    normalized_rotation = float(rotation_degrees)
    if abs(normalized_rotation) > 1e-6:
        rotated = Image.fromarray(binary_mask).rotate(
            -normalized_rotation,
            expand=True,
            fillcolor=0,
            resample=Image.Resampling.NEAREST,
        )
        binary_mask = np.array(rotated, dtype=np.uint8)

    page_width_px = max(1, int(round(page_width_mm * pixels_per_mm)))
    page_height_px = max(1, int(round(page_height_mm * pixels_per_mm)))
    usable_width_px = max(1, int(round((page_width_mm - (margin_mm * 2.0)) * pixels_per_mm)))
    usable_height_px = max(1, int(round((page_height_mm - (margin_mm * 2.0)) * pixels_per_mm)))

    source_height, source_width = binary_mask.shape
    fit_scale = min(usable_width_px / source_width, usable_height_px / source_height)
    scale = fit_scale * max(scale_multiplier, 0.01)
    scaled_width = max(1, int(round(source_width * scale)))
    scaled_height = max(1, int(round(source_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled_mask = cv2.resize(binary_mask, (scaled_width, scaled_height), interpolation=interpolation)
    _, scaled_mask = cv2.threshold(scaled_mask, 127, 255, cv2.THRESH_BINARY)

    page_mask = np.zeros((page_height_px, page_width_px), dtype=np.uint8)
    offset_x = int(round((page_width_px - scaled_width) / 2.0))
    offset_y = int(round((page_height_px - scaled_height) / 2.0))
    offset_x += int(round(offset_x_mm * pixels_per_mm))
    offset_y -= int(round(offset_y_mm * pixels_per_mm))

    destination_x0 = max(0, offset_x)
    destination_y0 = max(0, offset_y)
    destination_x1 = min(page_width_px, offset_x + scaled_width)
    destination_y1 = min(page_height_px, offset_y + scaled_height)
    if destination_x0 >= destination_x1 or destination_y0 >= destination_y1:
        return page_mask

    source_x0 = max(0, -offset_x)
    source_y0 = max(0, -offset_y)
    source_x1 = source_x0 + (destination_x1 - destination_x0)
    source_y1 = source_y0 + (destination_y1 - destination_y0)
    page_mask[destination_y0:destination_y1, destination_x0:destination_x1] = scaled_mask[source_y0:source_y1, source_x0:source_x1]
    return page_mask


def build_page_tone_map(
    grayscale_image: np.ndarray,
    page_width_mm: float,
    page_height_mm: float,
    margin_mm: float,
    pixels_per_mm: float,
    scale_multiplier: float = 1.0,
    rotation_degrees: float = 0.0,
    offset_x_mm: float = 0.0,
    offset_y_mm: float = 0.0,
) -> np.ndarray:
    normalized_rotation = float(rotation_degrees)
    if abs(normalized_rotation) > 1e-6:
        rotated = Image.fromarray(grayscale_image).rotate(
            -normalized_rotation,
            expand=True,
            fillcolor=255,
            resample=Image.Resampling.BICUBIC,
        )
        grayscale_image = np.array(rotated, dtype=np.uint8)

    page_width_px = max(1, int(round(page_width_mm * pixels_per_mm)))
    page_height_px = max(1, int(round(page_height_mm * pixels_per_mm)))
    usable_width_px = max(1, int(round((page_width_mm - (margin_mm * 2.0)) * pixels_per_mm)))
    usable_height_px = max(1, int(round((page_height_mm - (margin_mm * 2.0)) * pixels_per_mm)))

    source_height, source_width = grayscale_image.shape
    fit_scale = min(usable_width_px / source_width, usable_height_px / source_height)
    scale = fit_scale * max(scale_multiplier, 0.01)
    scaled_width = max(1, int(round(source_width * scale)))
    scaled_height = max(1, int(round(source_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled_image = cv2.resize(grayscale_image, (scaled_width, scaled_height), interpolation=interpolation)

    page_tone = np.full((page_height_px, page_width_px), 255, dtype=np.uint8)
    offset_x = int(round((page_width_px - scaled_width) / 2.0))
    offset_y = int(round((page_height_px - scaled_height) / 2.0))
    offset_x += int(round(offset_x_mm * pixels_per_mm))
    offset_y -= int(round(offset_y_mm * pixels_per_mm))

    destination_x0 = max(0, offset_x)
    destination_y0 = max(0, offset_y)
    destination_x1 = min(page_width_px, offset_x + scaled_width)
    destination_y1 = min(page_height_px, offset_y + scaled_height)
    if destination_x0 >= destination_x1 or destination_y0 >= destination_y1:
        return page_tone

    source_x0 = max(0, -offset_x)
    source_y0 = max(0, -offset_y)
    source_x1 = source_x0 + (destination_x1 - destination_x0)
    source_y1 = source_y0 + (destination_y1 - destination_y0)
    page_tone[destination_y0:destination_y1, destination_x0:destination_x1] = scaled_image[
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    return page_tone


def filter_mask_by_min_width(
    mask: np.ndarray,
    min_feature_width_mm: float,
    pixels_per_mm: float,
) -> np.ndarray:
    if min_feature_width_mm <= 0:
        return mask.copy()

    feature_width_px = min_feature_width_mm * pixels_per_mm
    if feature_width_px <= 1.0:
        return mask.copy()

    kernel_radius_px = max(1, int(round(feature_width_px / 2.0)))
    kernel = _ellipse_kernel(kernel_radius_px)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def filter_small_regions(
    mask: np.ndarray,
    min_region_area_mm2: float,
    pixels_per_mm: float,
) -> np.ndarray:
    if min_region_area_mm2 <= 0:
        return mask.copy()

    min_area_px = min_region_area_mm2 * pixels_per_mm * pixels_per_mm
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)

    for label_index in range(1, component_count):
        area = stats[label_index, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            filtered[labels == label_index] = 255

    return filtered


def build_fill_mask(
    mask: np.ndarray,
    min_fill_width_mm: float,
    pixels_per_mm: float,
) -> np.ndarray:
    if min_fill_width_mm <= 0:
        return mask.copy()

    min_radius_px = (min_fill_width_mm * pixels_per_mm) / 2.0
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return np.where(distance >= min_radius_px, 255, 0).astype(np.uint8)


def extract_contours(
    mask: np.ndarray,
    min_contour_area_px: float,
    simplify_tolerance_px: float,
) -> list[Contour]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    filtered_contours: list[Contour] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area_px:
            continue

        simplified = contour
        if simplify_tolerance_px > 0:
            simplified = cv2.approxPolyDP(contour, simplify_tolerance_px, True)

        points = [(int(point[0][0]), int(point[0][1])) for point in simplified]
        if len(points) >= 2:
            filtered_contours.append(points)

    return sorted(filtered_contours, key=_contour_sort_key)


def erode_mask(mask: np.ndarray, offset_mm: float, pixels_per_mm: float) -> np.ndarray:
    if offset_mm <= 0:
        return mask.copy()

    kernel_radius_px = max(1, int(round(offset_mm * pixels_per_mm)))
    kernel = _ellipse_kernel(kernel_radius_px)
    return cv2.erode(mask, kernel, iterations=1)


def dilate_mask(mask: np.ndarray, offset_mm: float, pixels_per_mm: float) -> np.ndarray:
    if offset_mm <= 0:
        return mask.copy()

    kernel_radius_px = max(1, int(round(offset_mm * pixels_per_mm)))
    kernel = _ellipse_kernel(kernel_radius_px)
    return cv2.dilate(mask, kernel, iterations=1)


def split_shape_and_thin_masks(
    mask: np.ndarray,
    thin_feature_max_width_mm: float,
    pixels_per_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    if thin_feature_max_width_mm <= 0:
        return mask.copy(), np.zeros_like(mask)

    erosion_offset_mm = thin_feature_max_width_mm / 2.0
    thick_seed = erode_mask(mask, erosion_offset_mm, pixels_per_mm)
    if cv2.countNonZero(thick_seed) == 0:
        return np.zeros_like(mask), mask.copy()

    shape_mask = dilate_mask(thick_seed, erosion_offset_mm, pixels_per_mm)
    thin_mask = cv2.subtract(mask, shape_mask)
    return shape_mask, thin_mask


def mask_to_preview_image(mask: np.ndarray) -> np.ndarray:
    return 255 - mask


def mask_bounds_mm(
    mask: np.ndarray,
    page_width_mm: float,
    page_height_mm: float,
) -> tuple[float, float, float, float] | None:
    active_rows, active_columns = np.where(mask > 0)
    if active_rows.size == 0 or active_columns.size == 0:
        return None

    cell_width_mm = page_width_mm / mask.shape[1]
    cell_height_mm = page_height_mm / mask.shape[0]
    x_min = float(active_columns.min()) * cell_width_mm
    x_max = float(active_columns.max() + 1) * cell_width_mm
    y_min = page_height_mm - (float(active_rows.max() + 1) * cell_height_mm)
    y_max = page_height_mm - (float(active_rows.min()) * cell_height_mm)
    return x_min, y_min, x_max, y_max


def apply_erase_overlay_to_mask(
    mask: np.ndarray,
    overlay_rgba: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if overlay_rgba is None:
        return mask.copy(), np.zeros_like(mask)

    rgb_overlay = overlay_rgba[:, :, :3]
    erase_mask_small = np.where(
        (rgb_overlay[:, :, 0] > 180)
        & (rgb_overlay[:, :, 1] < 120)
        & (rgb_overlay[:, :, 2] < 120),
        255,
        0,
    ).astype(np.uint8)
    if erase_mask_small.max() == 0:
        return mask.copy(), np.zeros_like(mask)

    erase_mask = cv2.resize(
        erase_mask_small,
        (mask.shape[1], mask.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    edited_mask = mask.copy()
    edited_mask[erase_mask > 0] = 0
    return edited_mask, erase_mask


def _ellipse_kernel(radius_px: int) -> np.ndarray:
    diameter = (radius_px * 2) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def _contour_sort_key(contour: Contour) -> tuple[int, int]:
    xs = [point[0] for point in contour]
    ys = [point[1] for point in contour]
    return min(ys), min(xs)
