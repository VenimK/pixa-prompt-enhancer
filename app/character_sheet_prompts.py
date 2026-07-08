"""
LTX-2.3 Director character/reference sheet prompt engineering module.

Generates prompts for creating reference sheets compatible with:
  - LTX-2.3 IC-LoRA "Reference Sheet Control" (single composite image
    inventorying characters/props/location)
  - Licon-MSR (Multiple Subject Reference) — up to 5 separate reference
    images fed to the LTX Director node, addressed via @character1..3
    tags in the sub-prompt.

Two distinct outputs are supported:
  1. A *reference-sheet image* prompt — a single composite turnaround
     sheet (front/left/right/back + close-ups) suitable for feeding to
     an image model, which is then used as the IC-LoRA reference input.
  2. A *short two-line character description* used inside the LTX
     Director node's per-character text field so the model can recognise
     the character from the reference sheet/image in the latent space.

References:
  - https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients
  - https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference
  - "LTX - Character sheet with LTX Director's V2" (community workflow)
"""

from __future__ import annotations

import logging

from app.logger import log_debug

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout presets for the reference-sheet IMAGE prompt
# ---------------------------------------------------------------------------

_LAYOUT_4_COLUMN = """\
Arrange into four vertical columns, each representing one viewing angle. Each \
column contains a full-body view on top and a matching close-up portrait \
directly beneath it.

Columns (left to right):
Column 1: front view (full-body above, front portrait below).
Column 2: left profile (full-body character facing left) with portrait facing \
left below.
Column 3: right profile (full-body character facing right) with portrait \
facing right below.
Column 4: back view, with matching portrait below.

Maintain even spacing and framing around the character portraits. Clean \
silhouette, consistent alignment, and clean panel separation."""

_LAYOUT_2_COLUMN = """\
Arrange into two vertical columns. Left column: front full-body view on top, \
front close-up portrait below. Right column: back full-body view on top, back \
close-up portrait below. Maintain even spacing, consistent alignment, and \
clean panel separation."""

_LAYOUT_GRID = """\
Arrange as a composite reference grid with labelled rows for Setting, \
Character (multiple turnaround angles: front, side, back, 3/4 view), Props, \
and Logo/Branding if applicable. Each row should clearly separate its \
elements with consistent lighting and a plain neutral background throughout."""

_LAYOUT_PRESETS = {
    "4_column": _LAYOUT_4_COLUMN,
    "2_column": _LAYOUT_2_COLUMN,
    "grid": _LAYOUT_GRID,
}


# ---------------------------------------------------------------------------
# System prompt — reference sheet IMAGE generation
# ---------------------------------------------------------------------------

_CHARACTER_SHEET_SYSTEM_PROMPT = """\
You are an expert prompt engineer creating a professional character/reference \
sheet prompt for an image generation model. This reference sheet will be used \
as conditioning input for LTX-2.3 IC-LoRA (Reference Sheet Control) and \
Licon-MSR video generation, so consistency and clarity are critical.

RULES:
- Plain, neutral, evenly-lit background so the subject reads cleanly for \
  downstream video conditioning.
- Describe the character/subject with precise, consistent details (age, hair, \
  clothing, colors, distinguishing features, materials) that must remain \
  identical across every angle/panel.
- If props or a setting are part of the scene, include a small dedicated \
  panel/row for each, described with the same level of visual precision.
- Do NOT introduce new characters, props, or text/logos not mentioned by the \
  user.
- Output a single, clear paragraph (or short set of paragraphs) — no code \
  fences, no JSON, no bullet-point prefixes in the final prompt.
- Always begin with: "Create a professional character reference sheet for \
  the character. Plain background."

CRITICAL: Preserve the user's core subject/character description exactly. \
Only add sheet-layout and consistency instructions around it.
"""


def build_character_sheet_image_prompt(
    character_description: str,
    layout: str = "4_column",
    props_description: str | None = None,
    setting_description: str | None = None,
    reference_image_description: str | None = None,
) -> str:
    """
    Build a meta-prompt (to send to the model provider) that produces a
    reference-sheet IMAGE prompt — the composite turnaround sheet used as
    IC-LoRA/Licon-MSR conditioning input.

    Args:
        character_description: Text description of the character, OR a
            short summary if a reference image is also provided.
        layout: One of '4_column', '2_column', 'grid'.
        props_description: Optional description of props to include.
        setting_description: Optional description of setting/environment.
        reference_image_description: Optional analysis of an uploaded
            reference image (from /analyze-image) to keep consistent.

    Returns:
        Meta-prompt string to send to the model provider.
    """
    layout_block = _LAYOUT_PRESETS.get(layout, _LAYOUT_4_COLUMN)

    context_parts = [f"Character description: {character_description}"]

    if reference_image_description:
        context_parts.append(
            f"Reference image analysis (match this appearance exactly): "
            f"{reference_image_description}"
        )

    if props_description:
        context_parts.append(f"Props to include in a dedicated panel: {props_description}")

    if setting_description:
        context_parts.append(f"Setting/location to include in a dedicated panel: {setting_description}")

    context = "\n".join(context_parts)

    full_prompt = (
        f"{_CHARACTER_SHEET_SYSTEM_PROMPT}\n\n"
        f"LAYOUT INSTRUCTIONS:\n{layout_block}\n\n"
        f"INPUT:\n{context}\n\n"
        f"Generate the complete character reference sheet prompt now."
    )

    log_debug(f"[character_sheet] Built image prompt (layout={layout}, length={len(full_prompt)})")
    return full_prompt


# ---------------------------------------------------------------------------
# System prompt — 2-line Director character description
# ---------------------------------------------------------------------------

_DIRECTOR_DESCRIPTION_SYSTEM_PROMPT = """\
You are writing a concise two-line character description for the LTX-2.3 \
Director node's per-character text field (used with a Licon-MSR reference \
sheet). This description lets the model recognise and pick the character \
from the reference sheet/image within its latent space when referenced via \
@character1, @character2, or @character3 in a scene sub-prompt.

RULES:
- Exactly two short lines (or one to two sentences), no more.
- Focus on the most visually distinctive, unambiguous identifying details: \
  hair color/style, clothing/colors, distinguishing accessories or features.
- Do NOT describe pose, action, or setting — identity only.
- No markdown, no headers, no quotes around the output.
"""


def build_director_character_description(
    character_description: str,
    reference_image_description: str | None = None,
) -> str:
    """
    Build a meta-prompt that produces the short 2-line identity description
    for one LTX Director @characterN text field.
    """
    context = f"Character description: {character_description}"
    if reference_image_description:
        context += f"\nReference image analysis: {reference_image_description}"

    full_prompt = (
        f"{_DIRECTOR_DESCRIPTION_SYSTEM_PROMPT}\n\nINPUT:\n{context}\n\n"
        f"Generate the two-line identity description now."
    )
    log_debug(f"[character_sheet] Built director description prompt (length={len(full_prompt)})")
    return full_prompt


# ---------------------------------------------------------------------------
# Scene sub-prompt helper — weaves @characterN tags into an LTX2.3 scene
# ---------------------------------------------------------------------------

def build_scene_subprompt_with_references(
    scene_prompt: str,
    character_tags: list[str],
) -> str:
    """
    Prefix/annotate a scene prompt with @characterN reference usage notes so
    the resulting text is ready to paste into the LTX Director sub-prompt
    field. character_tags is an ordered list like ['@character1', '@character2'].

    This does not call the model — it is a lightweight formatting helper.
    """
    if not character_tags:
        return scene_prompt

    tags_str = ", ".join(character_tags)
    note = (
        f"[References: {tags_str} — draw visual identity for these characters "
        f"from the attached reference sheet(s).]\n"
    )
    return note + scene_prompt
