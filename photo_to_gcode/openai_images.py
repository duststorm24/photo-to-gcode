from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass

import requests
from PIL import Image, ImageOps


DEFAULT_PLOTTER_AI_PROMPT = (
    "Create a plotter-friendly black-and-white illustration from this source image. "
    "Preserve the main subject, pose, proportions, and recognizable design details, but simplify the scene into crisp, "
    "high-contrast graphic shapes. Remove distracting background clutter, photographic grain, and tiny speckles. "
    "Use clean bold outlines, readable interior linework, simplified shadows, and a white background. "
    "The final result should feel like premium pen-plotter-ready line art or poster art: no gradients, no halftones, "
    "no painterly texture, no color, and minimal micro-details that would create noisy toolpaths."
)


@dataclass(slots=True)
class OpenAIImageEditResult:
    ok: bool
    image_bytes: bytes | None = None
    error: str | None = None
    revised_prompt: str | None = None


def convert_image_to_plotter_friendly_ai(
    source_image_bytes: bytes,
    *,
    api_key: str | None = None,
    prompt: str = DEFAULT_PLOTTER_AI_PROMPT,
    timeout_seconds: float = 180.0,
) -> OpenAIImageEditResult:
    resolved_api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not resolved_api_key:
        return OpenAIImageEditResult(
            ok=False,
            error="OPENAI_API_KEY is not set. Add it to your environment before using Convert to AI.",
        )

    try:
        normalized_png_bytes, size = _normalize_image_to_png(source_image_bytes)
    except Exception as exc:  # noqa: BLE001
        return OpenAIImageEditResult(ok=False, error=f"Could not read the uploaded image for AI conversion: {exc}")

    try:
        response = requests.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {resolved_api_key}"},
            data={
                "model": "gpt-image-1.5",
                "prompt": prompt,
                "size": size,
                "output_format": "png",
                "background": "opaque",
                "input_fidelity": "high",
                "n": "1",
            },
            files={"image": ("source.png", normalized_png_bytes, "image/png")},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        return OpenAIImageEditResult(ok=False, error=f"OpenAI image edit request failed: {exc}")

    if not response.ok:
        return OpenAIImageEditResult(ok=False, error=_format_openai_error(response))

    try:
        payload = response.json()
    except json.JSONDecodeError:
        return OpenAIImageEditResult(ok=False, error="OpenAI returned a non-JSON response.")

    image_data = payload.get("data")
    if not isinstance(image_data, list) or not image_data:
        return OpenAIImageEditResult(ok=False, error="OpenAI did not return any image data.")

    first_item = image_data[0]
    if not isinstance(first_item, dict):
        return OpenAIImageEditResult(ok=False, error="OpenAI returned an unexpected image payload.")

    b64_json = first_item.get("b64_json")
    if not isinstance(b64_json, str) or not b64_json:
        return OpenAIImageEditResult(ok=False, error="OpenAI did not return a base64 PNG image.")

    try:
        image_bytes = base64.b64decode(b64_json)
    except Exception as exc:  # noqa: BLE001
        return OpenAIImageEditResult(ok=False, error=f"Could not decode the AI image output: {exc}")

    revised_prompt = first_item.get("revised_prompt")
    return OpenAIImageEditResult(
        ok=True,
        image_bytes=image_bytes,
        revised_prompt=revised_prompt if isinstance(revised_prompt, str) else None,
    )


def _normalize_image_to_png(source_image_bytes: bytes) -> tuple[bytes, str]:
    image = Image.open(io.BytesIO(source_image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")

    width, height = image.size
    if width > height * 1.15:
        size = "1536x1024"
    elif height > width * 1.15:
        size = "1024x1536"
    else:
        size = "1024x1024"
    return output.getvalue(), size


def _format_openai_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return f"OpenAI image edit failed with HTTP {response.status_code}: {response.text[:300]}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return f"OpenAI image edit failed: {message.strip()}"
    return f"OpenAI image edit failed with HTTP {response.status_code}."
