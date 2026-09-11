"""Prepare images for vision API calls.

Phone photos are often 8–12 MP. Gemini quality plateaus well below that,
so we downscale and convert to RGB before upload. This cuts request payload
size and vision-token cost without a noticeable quality drop.
"""

from __future__ import annotations

from typing import Iterable

import PIL.Image

from app.logger import log_debug

# Longest edge sent to the model. 1280 is inside Gemini's recommended
# 768–1568 band and keeps typical JPEGs well under 1 MB after convert.
MAX_IMAGE_SIDE = 1280


def prepare_image_for_api(path: str, max_side: int = MAX_IMAGE_SIDE) -> PIL.Image.Image:
    """Open an image, convert to RGB, and downscale if larger than max_side.

    The returned image should be closed by the caller (it is a context manager).
    """
    image = PIL.Image.open(path)
    try:
        if image.mode != "RGB":
            converted = image.convert("RGB")
            image.close()
            image = converted
        width, height = image.size
        longest = max(width, height)
        if longest > max_side:
            image.thumbnail((max_side, max_side), PIL.Image.Resampling.LANCZOS)
            log_debug(
                f"[image_prep] Downscaled {path} from {width}x{height} to {image.size[0]}x{image.size[1]}"
            )
        return image
    except Exception:
        image.close()
        raise


def prepare_images_for_api(paths: Iterable[str], max_side: int = MAX_IMAGE_SIDE) -> list[PIL.Image.Image]:
    """Open and prepare multiple images. Close all of them if any open fails."""
    images: list[PIL.Image.Image] = []
    try:
        for path in paths:
            images.append(prepare_image_for_api(path, max_side=max_side))
        return images
    except Exception:
        for image in images:
            image.close()
        raise


def close_images(images: Iterable[PIL.Image.Image | None]) -> None:
    """Close a sequence of PIL images, ignoring already-closed handles."""
    for image in images:
        if image is None:
            continue
        try:
            image.close()
        except Exception:
            pass
