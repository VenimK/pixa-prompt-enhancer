"""
LTX-2.3 prompt engineering module.

Encodes the official LTX-2.3 prompting guide structure and produces
cinematic, single-paragraph prompts optimised for the model's redesigned
text connector.  Supports image-to-video, text-to-video, and audio-synced modes.

Reference: https://ltx.io/model/model-blog/ltx-2-3-prompt-guide
"""

from __future__ import annotations

import logging
from typing import Any

from app.logger import log_debug

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default negative prompt — common LTX artefacts to suppress
# ---------------------------------------------------------------------------

LTX23_NEGATIVE_PROMPT = (
    "judder, jitter, flicker, warped anatomy, distorted face, extra limbs, "
    "text overlay, watermark, logo, low fps, blurry, out of focus, "
    "abrupt cuts, scene jumps, inconsistent lighting, morphing artefacts, "
    "static frame, frozen motion, jerky camera, duplicate subjects"
)

# ---------------------------------------------------------------------------
# System prompt — encodes the official 6-element structure
# ---------------------------------------------------------------------------

_LTX23_SYSTEM_PROMPT = """\
You are an expert video-prompt writer for the LTX-2.3 video generation model.

Your job is to take the user's idea and produce a single, flowing, cinematic \
paragraph that LTX-2.3 will turn into a high-fidelity video clip.

Follow these rules strictly:

STRUCTURE — weave ALL six elements into one paragraph in this order:
1. Establish the Shot — shot scale, lens, genre cues (e.g. "Close-up, shallow depth of field").
2. Set the Scene — lighting, colour palette, textures, atmosphere.
3. Describe the Action — present-progressive verbs ("is walking", "turns slowly"), \
   chronological flow with temporal connectors ("as", "then", "while").
4. Define Characters — age, hair, clothing, distinguishing features. Express \
   emotion through physical cues, never abstract labels.
5. Camera Movement — specify only if the user requests it or the scene demands it. \
   Describe how subjects appear after the movement. Default to static frame.
6. Audio Layer — describe the complete soundscape alongside the actions, NOT at the end. \
   Be specific ("soft footsteps on tile", "distant city traffic") not vague ("ambient sound"). \
   If dialogue is present, place exact words in quotation marks.

FORMAT:
- Single concise paragraph in natural English.
- NO titles, headings, prefaces, sections, code fences, bullet lists, or Markdown.
- Never start with "The scene opens…" or "The video starts…" — begin directly.
- Never ask questions or add disclaimers.
- Use present tense throughout.
- Use restrained, natural language — avoid dramatic adjectives.
- Match detail density to shot scale: close-ups need more detail than wide shots.
- For longer clips, provide enough detail to fill the duration — short prompts produce rushed motion.

DIALOGUE RULES:
- Break long speech into short phrases with acting directions between them.
- Example: He speaks softly, "I remember…" pauses, glancing to the side, then \
  continues with a cracking voice, "something I never understood."
- DO NOT invent dialogue unless the user mentions speech, talking, or singing.
- DO NOT modify the user's provided dialogue except to fix obvious typos.

CAMERA RULES:
- DO NOT invent camera movement unless the user requests it.
- If unspecified, default to a static frame.

CRITICAL:
- Preserve the user's core idea exactly — subject, action, intent.
- DO NOT add extra subjects, props, text, or signage the user did not ask for.
- Your output will be fed directly to LTX-2.3. Quality and accuracy are paramount.
"""

# ---------------------------------------------------------------------------
# Mode-specific instruction blocks
# ---------------------------------------------------------------------------

_MODE_I2V = """\
MODE: Image-to-Video.
A reference image defines the first frame. Focus your prompt ONLY on motion \
and change from that still frame. DO NOT re-describe static elements already \
visible in the image. Describe: how the subject moves, how the camera follows, \
what sounds emerge, and the transition from stillness to motion."""

_MODE_T2V = """\
MODE: Text-to-Video.
No reference image exists — you must describe everything from scratch. Include \
full visual detail: subject appearance, environment, lighting, camera, and audio. \
The model generates the entire scene from your words alone."""

_MODE_AUDIO_SYNC = """\
MODE: Audio-Synced Video.
An audio track anchors the temporal structure. Describe the visual interpretation \
of that audio — what scenes, subjects, and camera work should accompany the \
soundtrack. Synchronise visual beats to audio beats. Weave audio descriptions \
naturally into the action, not as a separate block."""


# ---------------------------------------------------------------------------
# Helpers — convert structured audio data to natural-language cues
# ---------------------------------------------------------------------------

def _audio_to_prose(ac: dict) -> str:
    """Convert structured audio_characteristics dict into natural-language cues."""
    if not ac:
        return ""

    parts: list[str] = []

    # Vocal performance
    vocal = ac.get("vocal_style", "")
    vocal_conf = ac.get("vocal_confidence", 0)
    if vocal == "singing":
        vr = ac.get("vocal_range", "medium")
        parts.append(f"singing with a {vr} vocal range")
        if vocal_conf > 0.8:
            parts.append("confident, expressive vocal delivery with precise lip movement")
    elif vocal == "spoken":
        parts.append("speaking with clear, articulate lip movement")
        if vocal_conf > 0.8:
            parts.append("natural speech cadence and measured diction")
    elif vocal == "melodic_speech":
        parts.append("delivering melodic speech with rhythmic cadence")

    # Tempo & rhythm
    tempo = ac.get("tempo", "")
    bpm = ac.get("tempo_bpm")
    if tempo == "fast":
        bpm_str = f" ({bpm:.0f} BPM)" if bpm else ""
        parts.append(f"energetic movement matching the fast tempo{bpm_str}")
    elif tempo == "slow":
        bpm_str = f" ({bpm:.0f} BPM)" if bpm else ""
        parts.append(f"gentle, measured motion timed to the slow tempo{bpm_str}")

    # Beat & syncopation
    if ac.get("beat_strength") == "strong":
        parts.append("body movement subtly synchronised to strong rhythmic beats")
    if ac.get("syncopation") == "high":
        parts.append("off-beat gestural accents")

    # Energy & danceability
    energy = ac.get("energy_level", "")
    dance = ac.get("danceability", 0)
    if energy in ("high", "very_high"):
        parts.append("high-energy, dynamic physical presence")
    elif energy in ("low", "very_low"):
        parts.append("restrained, minimal movement")
    if isinstance(dance, (int, float)) and dance > 0.7:
        parts.append("rhythmic swaying and dance-like motion")

    # Mood
    mood = ac.get("mood", "")
    _mood_map = {
        "calm": "serene, peaceful expression and soft gestures",
        "energetic": "vibrant expression and lively gestures",
        "emotional": "expressive face and emotive body language",
        "contemplative": "thoughtful gaze with measured movement",
    }
    if mood in _mood_map:
        parts.append(_mood_map[mood])

    # Emotional arc
    arc = ac.get("emotional_arc", "")
    if arc == "building":
        parts.append("intensity gradually building throughout the clip")
    elif arc == "fading":
        parts.append("energy gently tapering toward the end")

    # Genre hint — only if useful
    genre = ac.get("genre", "")
    _genre_cues = {
        "rock": "rock-influenced head movement and shoulder energy",
        "pop": "polished, choreographed gestural rhythm",
        "electronic": "precise, rhythmic micro-movements",
        "jazz": "smooth, improvisational body sway",
        "classical": "elegant, refined gestural flow",
        "folk": "grounded, organic motion",
        "hip_hop": "confident, rhythmic street-style movement",
        "metal": "intense, powerful physical energy",
    }
    if genre in _genre_cues:
        parts.append(_genre_cues[genre])

    if not parts:
        return ""

    return "Audio context: " + ", ".join(parts) + "."


def _style_to_prose(ltx2_style: str | None) -> str:
    """Convert an LTX2 style key into a natural-language style hint."""
    if not ltx2_style or ltx2_style in ("auto", ""):
        return ""

    _map = {
        "music_video": "Music-video aesthetic with dynamic performance energy and rhythmic visual rhythm.",
        "concert": "Live concert atmosphere with stage lighting and authentic performance energy.",
        "dance": "Dance-focused visual with choreographed movement and rhythmic body expression.",
        "lip_sync": "Precise lip-sync focus — emphasis on mouth movement, facial expression, and vocal delivery.",
        "acoustic": "Intimate acoustic session — subtle movement, emotional closeness, warm tones.",
        "cinematic": "Cinematic film quality — dramatic lighting, composed framing, narrative atmosphere.",
        "dramatic": "Dramatic intensity — theatrical movement, powerful presence, high emotional stakes.",
        "documentary": "Documentary naturalism — candid, authentic moments with observational perspective.",
        "vintage": "Vintage film look — subtle grain, classic colour grading, nostalgic warmth.",
        "noir": "Film-noir aesthetic — high-contrast lighting, deep shadows, mysterious atmosphere.",
        "artistic": "Artistic visual expression — creative composition, experimental framing.",
        "surreal": "Surreal dreamscape — unconventional composition, fantastical elements.",
        "abstract": "Abstract visuals — geometric patterns, non-literal imagery, conceptual movement.",
        "dreamy": "Dreamy atmosphere — soft focus, ethereal light, gentle motion.",
        "psychedelic": "Psychedelic visuals — vivid saturation, fluid motion, kaleidoscopic colour.",
        "cyberpunk": "Cyberpunk setting — neon lighting, futuristic tech, urban dystopia.",
        "vaporwave": "Vaporwave aesthetic — pastel hues, retro-digital elements, nostalgic calm.",
        "lofi": "Lo-fi mood — cozy, intimate atmosphere, warm soft lighting, relaxed pace.",
        "retro": "Retro 80s/90s vibe — bold colours, vintage tech, period styling.",
        "futuristic": "Futuristic design — sleek surfaces, clean lines, advanced technology.",
    }
    hint = _map.get(ltx2_style, "")
    return f"Style direction: {hint}" if hint else ""


def _movement_to_prose(
    movement_level: str,
    ac: dict,
) -> str:
    """Derive a movement-level hint from explicit selection or audio data."""
    level = movement_level
    if level == "auto":
        # Auto-detect from structured audio
        tempo = ac.get("tempo", "")
        energy = ac.get("energy_level", "")
        vocal = ac.get("vocal_style", "")
        dance = ac.get("danceability", 0)
        if tempo == "fast" and vocal == "singing":
            level = "dynamic"
        elif energy in ("high", "very_high"):
            level = "expressive"
        elif energy in ("low", "very_low") or ac.get("mood") in ("calm", "peaceful"):
            level = "minimal"
        elif vocal == "spoken":
            level = "minimal"
        elif isinstance(dance, (int, float)) and dance > 0.7:
            level = "expressive"
        else:
            level = "natural"

    _level_map = {
        "static": "Only lip-sync and subtle eye movement; the body remains completely still.",
        "minimal": "Subtle head turns, slight shoulder motion, and gentle hand gestures — no large body movement.",
        "natural": "Normal, realistic body movement — head turns, shoulder shifts, arm gestures, gentle swaying.",
        "expressive": "Full-body movement with dynamic gestures, shoulder motion, and rhythmic swaying matching the audio.",
        "dynamic": "Highly energetic full-body motion — dramatic gestures, dancing, athletic movement.",
    }
    hint = _level_map.get(level, _level_map["natural"])
    return f"Movement level: {hint}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_ltx2_mode(
    has_image: bool,
    has_audio: bool,
) -> str:
    """Return 'i2v', 't2v', or 'audio_sync'."""
    if has_audio:
        return "audio_sync"
    if has_image:
        return "i2v"
    return "t2v"


def build_ltx2_meta_prompt(
    *,
    user_prompt: str,
    image_description: str | None = None,
    audio_description: str | None = None,
    audio_characteristics: dict | None = None,
    movement_level: str = "auto",
    ltx2_style: str | None = None,
    audio_generation: str = "enabled",
) -> dict[str, str]:
    """Build a Gemini meta-prompt optimised for LTX-2.3 video generation.

    Returns
    -------
    dict with keys:
        positive   – the full meta-prompt to send to Gemini
        negative   – a negative-prompt string for the runner
        mode       – 'i2v' | 't2v' | 'audio_sync'
    """
    ac: dict = audio_characteristics or {}
    has_image = bool(image_description)
    has_audio = bool(audio_description) or bool(ac)

    mode = detect_ltx2_mode(has_image, has_audio)
    mode_block = {
        "i2v": _MODE_I2V,
        "t2v": _MODE_T2V,
        "audio_sync": _MODE_AUDIO_SYNC,
    }[mode]

    # --- Contextual blocks (natural prose, not ALL-CAPS directives) ---
    blocks: list[str] = [_LTX23_SYSTEM_PROMPT, mode_block]

    # Image context
    if image_description:
        blocks.append(
            f"Reference image (first frame): {image_description}\n"
            "Remember: describe only changes FROM this image — do not repeat what is already visible."
        )

    # Audio context — structured data → prose
    audio_prose = _audio_to_prose(ac)
    if audio_prose:
        blocks.append(audio_prose)
    elif audio_description:
        # Fallback to raw description if no structured data
        blocks.append(f"Audio context: {audio_description}")

    # Audio generation preference
    if audio_generation == "enabled":
        if has_audio:
            blocks.append(
                "Audio is provided — describe the soundscape inline with the action "
                "so the model generates synchronised audio."
            )
        else:
            blocks.append(
                "Generate synchronised audio — weave ambient sounds, effects, or dialogue "
                "naturally into the action description."
            )
    else:
        blocks.append("Video-only generation — do not describe audio.")

    # Style
    style_prose = _style_to_prose(ltx2_style)
    if style_prose:
        blocks.append(style_prose)

    # Movement level
    move_prose = _movement_to_prose(movement_level, ac)
    blocks.append(move_prose)

    # User prompt (last — Gemini treats it as the primary instruction)
    blocks.append(f"User's idea: {user_prompt}")

    positive = "\n\n".join(blocks)

    log_debug(f"LTX2.3 meta-prompt built: mode={mode}, blocks={len(blocks)}, chars={len(positive)}")

    return {
        "positive": positive,
        "negative": LTX23_NEGATIVE_PROMPT,
        "mode": mode,
    }
