"""
Ideogram 4.0 prompt engineering module.

Implements the official Ideogram 4.0 JSON caption schema for structured
prompt generation with color palette conditioning and bounding box layout control.

Reference: https://github.com/ideogram-os/ideogram4/blob/main/docs/prompting.md
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.logger import log_debug

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for AI to generate Ideogram4 JSON
# ---------------------------------------------------------------------------

_IDEOGRAM4_SYSTEM_PROMPT = """\
You are an expert prompt engineer for Ideogram 4.0, a state-of-the-art text-to-image model \
that uses structured JSON captions for optimal results.

Your job is to convert the user's natural language prompt into a valid Ideogram 4.0 JSON caption \
following the official schema.

STRICT RULES:
1. Output ONLY valid JSON — no markdown, no explanations, no text outside the JSON object
2. Follow the exact schema structure with proper key ordering
3. Hex colors must be uppercase #RRGBB format (e.g., #FF6B35, not #ff6b35)
4. Bounding boxes use normalized 0-1000 coordinates: [y_min, x_min, y_max, x_max]
5. background must come before elements in compositional_deconstruction
6. For photographic output, use "photo" field. For illustrations, use "art_style" field. Never both.
7. Key order in style_description:
   - Photo: aesthetics, lighting, photo, medium, color_palette
   - Non-photo: aesthetics, lighting, medium, art_style, color_palette
8. color_palette is optional but strongly recommended
9. Include high_level_description (1-2 sentences) for better results
10. NEVER name or reference a real, identifiable individual (public figures, celebrities, \
athletes, politicians, etc.) even if the input mentions one (e.g. "reminiscent of X", \
"looks like Y"). Most image models reject or degrade prompts naming real people. Instead, \
translate any such reference into purely physical/generic descriptive traits (hair color \
and style, build, complexion, clothing, expression) with no proper name attached. This \
applies to every field: high_level_description, style_description, and every element desc.

SCHEMA:
{
  "high_level_description": "string (1-2 sentence summary)",
  "style_description": {
    "aesthetics": "string (e.g., 'moody, cinematic, desaturated')",
    "lighting": "string (e.g., 'golden hour, rim light, dramatic shadows')",
    "photo": "string (e.g., '35mm, f/1.4, bokeh') OR art_style: 'string'",
    "medium": "string (e.g., 'photograph', 'illustration', '3d_render')",
    "color_palette": ["#RRGGBB", ...] (up to 16 colors, uppercase hex)
  },
  "compositional_deconstruction": {
    "background": "string (environment description)",
    "elements": [
      {
        "type": "obj" OR "text",
        "bbox": [y_min, x_min, y_max, x_max] (optional, 0-1000 coordinates),
        "desc": "string (detailed element description)",
        "text": "string (required for type:text only)",
        "color_palette": ["#RRGGBB", ...] (optional, up to 5 colors per element)
      }
    ]
  }
}

If the user doesn't specify layout details, estimate reasonable bounding boxes based on \
typical composition rules:
- Main subject: center-lower area (y: 400-900, x: 200-800)
- Headers/titles: top area (y: 50-200, x: 100-900)
- Side elements: left/right edges (y: 200-800, x: 50-200 or 800-950)
- Background: no bbox needed

Generate the JSON now. Ensure it is valid and follows all formatting rules.
"""


_STORYBOARD_LAYOUT_GRIDS: dict[str, tuple[int, int]] = {
    "horizontal_2": (1, 2),
    "horizontal_3": (1, 3),
    "vertical_2": (2, 1),
    "vertical_3": (3, 1),
    "grid_2x2": (2, 2),
    "grid_3x2": (2, 3),
}

_PANEL_COUNT_TO_LAYOUT: dict[int, str] = {
    2: "horizontal_2",
    3: "horizontal_3",
    4: "grid_2x2",
    6: "grid_3x2",
}


def _compute_grid_bboxes(rows: int, cols: int, gutter: int = 15) -> list[list[int]]:
    """
    Compute non-overlapping panel bboxes on a 0-1000 canvas, in reading order
    (left-to-right, top-to-bottom), separated by a fixed gutter.

    Returns bboxes as [y_min, x_min, y_max, x_max].
    """
    cell_h = (1000 - gutter * (rows + 1)) / rows
    cell_w = (1000 - gutter * (cols + 1)) / cols

    bboxes: list[list[int]] = []
    for r in range(rows):
        y_min = int(round(gutter + r * (cell_h + gutter)))
        y_max = int(round(y_min + cell_h))
        for c in range(cols):
            x_min = int(round(gutter + c * (cell_w + gutter)))
            x_max = int(round(x_min + cell_w))
            bboxes.append([y_min, x_min, y_max, x_max])
    return bboxes


_IDEOGRAM4_STORYBOARD_SYSTEM_PROMPT = """\
You are an expert prompt engineer for Ideogram 4.0, generating a SINGLE-IMAGE COMIC-STYLE \
STORYBOARD using the structured JSON caption schema. The final image is ONE composite image \
divided into multiple panels (like a comic strip), each panel depicting a different beat of \
the same story, in sequence.

STRICT RULES:
1. Output ONLY valid JSON — no markdown, no explanations, no text outside the JSON object.
2. The JSON follows the standard Ideogram 4.0 schema: high_level_description, \
style_description, compositional_deconstruction (background, elements).
3. Each entry in "elements" represents ONE PANEL of the storyboard. type must be "obj".
4. You MUST use the EXACT bbox values provided below for each panel, in the EXACT order given \
— do not invent, resize, or reorder bboxes.
5. Each panel's "desc" must describe a DIFFERENT, sequential beat of the story (clear \
narrative progression from panel 1 to the last panel) — do not repeat the same moment twice.
6. Keep the SAME character identity, appearance, clothing, and art style consistent across \
every panel — only action, expression, camera angle, and environment should change between \
panels to show progression.
7. "background" should describe the shared storyboard framing: clean panel gutters/borders \
separating each panel (e.g. "white gutters with thin black comic-panel borders separating \
each panel"), NOT the content of any single panel.
8. Hex colors must be uppercase #RRGGBB. color_palette optional, up to 16 image-level / 5 \
per-element.
9. high_level_description must summarize the overall story arc across all panels (1-2 \
sentences).
10. NEVER name or reference a real, identifiable individual (public figures, celebrities, \
athletes, politicians, etc.) even if the input mentions one. Translate any such reference \
into purely physical/generic descriptive traits (hair color/style, build, complexion, \
clothing, expression) with no proper name attached. Applies to every field.

SCHEMA (elements array MUST have exactly one entry per panel, in the given bbox order):
{
  "high_level_description": "string (1-2 sentence overall story summary)",
  "style_description": {
    "aesthetics": "string",
    "lighting": "string",
    "photo": "string (or art_style if illustrated)",
    "medium": "string",
    "color_palette": ["#RRGGBB", ...]
  },
  "compositional_deconstruction": {
    "background": "string (shared panel-gutter/border framing description)",
    "elements": [
      {"type": "obj", "bbox": [y_min, x_min, y_max, x_max], "desc": "panel 1 story beat..."},
      {"type": "obj", "bbox": [y_min, x_min, y_max, x_max], "desc": "panel 2 story beat..."}
    ]
  }
}

Generate the JSON now, using exactly the panel bboxes provided in the input, in order.
"""


def build_ideogram4_storyboard_prompt(
    user_story: str,
    panel_count: int = 4,
    layout: str | None = None,
    style: str = "none",
    cinematography: str = "none",
    lighting: str = "none",
    image_description: str | None = None,
) -> str:
    """
    Build a meta-prompt that produces a single Ideogram 4.0 JSON caption representing a
    multi-panel comic-style storyboard (one image, N panels, each a sequential story beat).

    Args:
        user_story: The user's story/scene idea to break into sequential beats.
        panel_count: Desired number of panels (2, 3, 4, or 6). Ignored if layout is given.
        layout: Explicit layout key from _STORYBOARD_LAYOUT_GRIDS. Overrides panel_count.
        style, cinematography, lighting: Same as build_ideogram4_json_prompt.
        image_description: Optional reference image analysis for character consistency.

    Returns:
        Meta-prompt string to send to the model provider.
    """
    layout_key = layout if layout in _STORYBOARD_LAYOUT_GRIDS else _PANEL_COUNT_TO_LAYOUT.get(
        panel_count, "grid_2x2"
    )
    rows, cols = _STORYBOARD_LAYOUT_GRIDS[layout_key]
    bboxes = _compute_grid_bboxes(rows, cols)

    context_parts = [
        f"Story/scene idea to break into {len(bboxes)} sequential panels: {user_story}",
        f"Panel layout: {layout_key} ({rows} row(s) x {cols} column(s))",
        f"Use these EXACT panel bboxes, in this EXACT order (panel 1 first): {bboxes}",
    ]

    if style and style.lower() not in ("none", "auto", "", "ideogram4_storyboard"):
        context_parts.append(f"Style preference: {style}")
    if cinematography and cinematography.lower() not in ("none", "auto", ""):
        context_parts.append(f"Cinematography hint: {cinematography}")
    if lighting and lighting.lower() not in ("none", "auto", ""):
        context_parts.append(f"Lighting preference: {lighting}")
    if image_description:
        context_parts.append(
            f"Reference character/subject appearance to keep consistent across all panels: "
            f"{image_description}"
        )

    context = "\n".join(context_parts)
    full_prompt = f"{_IDEOGRAM4_STORYBOARD_SYSTEM_PROMPT}\n\nINPUT:\n{context}\n\nOUTPUT:"

    log_debug(
        f"[ideogram4_storyboard] Built prompt (layout={layout_key}, panels={len(bboxes)}, "
        f"length={len(full_prompt)})"
    )
    return full_prompt


def build_ideogram4_json_prompt(
    user_prompt: str,
    style: str = "none",
    cinematography: str = "none",
    lighting: str = "none",
    image_description: str | None = None,
    prompt_type: str = "Image",
) -> str:
    """
    Generate an Ideogram 4.0 JSON prompt from user input.

    Args:
        user_prompt: The user's natural language prompt
        style: Selected style (may influence aesthetics)
        cinematography: Cinematography settings (may influence lighting/camera)
        lighting: Lighting preference
        image_description: Optional image analysis from reference
        prompt_type: Type of prompt (Image, VEO, etc.)

    Returns:
        JSON string following Ideogram 4.0 schema
    """
    # Build context for the AI
    context_parts = [f"User prompt: {user_prompt}"]

    if style and style.lower() not in ("none", "auto", ""):
        context_parts.append(f"Style preference: {style}")

    if cinematography and cinematography.lower() not in ("none", "auto", ""):
        context_parts.append(f"Cinematography hint: {cinematography}")

    if lighting and lighting.lower() not in ("none", "auto", ""):
        context_parts.append(f"Lighting preference: {lighting}")

    if image_description:
        context_parts.append(f"Reference image analysis: {image_description}")

    if prompt_type:
        context_parts.append(f"Output type: {prompt_type}")

    context = "\n".join(context_parts)

    # For now, we'll return a structured prompt that the AI should convert to JSON
    # In a full implementation, this would call the model provider
    full_prompt = f"{_IDEOGRAM4_SYSTEM_PROMPT}\n\nINPUT:\n{context}\n\nOUTPUT:"
    
    log_debug(f"[ideogram4] Generated prompt request (length: {len(full_prompt)})")
    
    # Note: This returns the prompt for the AI to convert to JSON
    # The actual JSON generation would happen in the model provider call
    return full_prompt


def validate_ideogram4_json(json_str: str) -> tuple[bool, list[str]]:
    """
    Validate an Ideogram 4.0 JSON prompt against the schema.

    Args:
        json_str: JSON string to validate

    Returns:
        (is_valid, list of error messages)
    """
    errors = []

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    # Check required fields
    if "compositional_deconstruction" not in data:
        errors.append("Missing required field: compositional_deconstruction")

    comp = data.get("compositional_deconstruction", {})

    if "background" not in comp:
        errors.append("Missing required field: compositional_deconstruction.background")

    if "elements" not in comp:
        errors.append("Missing required field: compositional_deconstruction.elements")

    # Check style_description structure if present
    if "style_description" in data:
        style = data["style_description"]
        
        # Must have exactly one of photo or art_style
        has_photo = "photo" in style
        has_art_style = "art_style" in style
        
        if has_photo and has_art_style:
            errors.append("Cannot have both 'photo' and 'art_style' in style_description")
        
        if not has_photo and not has_art_style:
            errors.append("style_description must contain either 'photo' or 'art_style'")

        # Check required fields for style_description
        if "aesthetics" not in style:
            errors.append("Missing required field: style_description.aesthetics")
        if "lighting" not in style:
            errors.append("Missing required field: style_description.lighting")
        if "medium" not in style:
            errors.append("Missing required field: style_description.medium")

    # Validate color palette format (uppercase hex)
    def validate_palette(palette, context=""):
        if not isinstance(palette, list):
            return
        
        for i, color in enumerate(palette):
            if not isinstance(color, str):
                errors.append(f"{context} color {i}: not a string")
                continue
            
            if not color.startswith("#"):
                errors.append(f"{context} color {i}: missing # prefix")
                continue
            
            if len(color) != 7:
                errors.append(f"{context} color {i}: must be 7 characters (#RRGGBB)")
                continue
            
            # Check if hex is uppercase
            if color != color.upper():
                errors.append(f"{context} color {i}: must be uppercase (#RRGGBB), got {color}")

    # Validate image-level color palette
    if "style_description" in data and "color_palette" in data["style_description"]:
        if len(data["style_description"]["color_palette"]) > 16:
            errors.append("style_description.color_palette: maximum 16 colors")
        validate_palette(data["style_description"]["color_palette"], "image-level")

    # Validate element color palettes
    if "elements" in comp:
        for i, element in enumerate(comp["elements"]):
            if "color_palette" in element:
                if len(element["color_palette"]) > 5:
                    errors.append(f"element {i}: color_palette maximum 5 colors")
                validate_palette(element["color_palette"], f"element {i}")

    # Validate bounding boxes
    if "elements" in comp:
        for i, element in enumerate(comp["elements"]):
            if "bbox" in element:
                bbox = element["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"element {i}: bbox must be [y_min, x_min, y_max, x_max]")
                    continue
                
                for coord in bbox:
                    if not isinstance(coord, (int, float)):
                        errors.append(f"element {i}: bbox coordinates must be numbers")
                        continue
                    
                    if not (0 <= coord <= 1000):
                        errors.append(f"element {i}: bbox coordinates must be 0-1000, got {coord}")

    return (len(errors) == 0, errors)


def validate_ideogram4_storyboard_json(
    json_str: str, expected_panels: int
) -> tuple[bool, list[str]]:
    """
    Validate a storyboard Ideogram 4.0 JSON: base schema plus an exact panel count check.

    Args:
        json_str: JSON string to validate
        expected_panels: Number of panels the layout should have produced

    Returns:
        (is_valid, list of error messages)
    """
    is_valid, errors = validate_ideogram4_json(json_str)

    try:
        data = json.loads(json_str)
        elements = data.get("compositional_deconstruction", {}).get("elements", [])
        if len(elements) != expected_panels:
            errors.append(
                f"Expected {expected_panels} panels, got {len(elements)}"
            )
            is_valid = False
        for i, element in enumerate(elements):
            if "bbox" not in element:
                errors.append(f"panel {i}: missing bbox (required for storyboard panels)")
                is_valid = False
    except json.JSONDecodeError:
        pass  # Already captured by validate_ideogram4_json

    return (is_valid and len(errors) == 0, errors)


def format_ideogram4_json(data: dict) -> str:
    """
    Format a dict as Ideogram 4.0 JSON with proper separators.

    Args:
        data: Dictionary containing Ideogram4 JSON structure

    Returns:
        Formatted JSON string
    """
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
