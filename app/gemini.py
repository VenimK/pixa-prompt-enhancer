"""
Gemini API client, run_gemini helper, and prompt quality scoring.

Extracted from main.py to reduce monolith size and centralise all Gemini interactions.
"""

import asyncio
import json
import os
import sys
import time
import traceback

import PIL.Image

from app.logger import log_debug

# --- Gemini SDK compatibility layer ---
USE_NEW_GENAI = False
genai_new = None
genai_old = None

try:
    import google.genai as genai_new
    USE_NEW_GENAI = True
except ImportError:
    pass

try:
    import google.generativeai as genai_old
except ImportError:
    pass

if not genai_new and not genai_old:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_version >= "3.14":
        raise ImportError(
            f"Python {python_version} detected. Please install google-genai:\n"
            "pip install google-genai\n"
            "If issues persist, try: pip install --upgrade google-genai"
        )
    else:
        raise ImportError(
            "Google GenAI package not found. Please install:\n"
            "pip install google-genai\n"
            "For older Python versions: pip install google-generativeai"
        )

# --- Client initialisation ---
genai_client = None
genai_model = None
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
log_debug(f"[gemini] Model: {GEMINI_MODEL}")

if "GOOGLE_API_KEY" in os.environ:
    if USE_NEW_GENAI and genai_new:
        genai_client = genai_new.Client(api_key=os.environ["GOOGLE_API_KEY"])
        try:
            genai_model = genai_client.models.get(model=GEMINI_MODEL)
        except Exception:
            genai_model = genai_client
        log_debug("[gemini] Using google.genai (new package)")
    elif genai_old:
        genai_old.configure(api_key=os.environ["GOOGLE_API_KEY"])
        genai_model = genai_old.GenerativeModel(GEMINI_MODEL)
        log_debug(f"[gemini] Using google.generativeai (old package) with model: {GEMINI_MODEL}")

_ALLOWED_GEMINI_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
}


# ---------------------------------------------------------------------------
# Core generate helper
# ---------------------------------------------------------------------------

def run_gemini(
    prompt: str,
    image_path: str | None = None,
    image_paths: list[str] | None = None,
    model_override: str | None = None,
) -> str:
    if "GOOGLE_API_KEY" not in os.environ:
        return "Error: Google API key is not set. Please set the GOOGLE_API_KEY environment variable."

    try:
        model_name = model_override if model_override in _ALLOWED_GEMINI_MODELS else GEMINI_MODEL

        # Use NEW google.genai package
        if USE_NEW_GENAI and genai_client:
            if image_paths and len(image_paths) > 0:
                images = []
                try:
                    for p in image_paths:
                        images.append(PIL.Image.open(p))
                except Exception as img_error:
                    return f"Error loading image(s): {img_error}. Please check the image file format and try again."
                try:
                    start_time = time.time()
                    response = genai_client.models.generate_content(
                        model=model_name,
                        contents=[prompt, *images],
                    )
                    log_debug(f"Gemini API call (multi-image) took {time.time() - start_time:.2f}s")
                    if hasattr(response, "text") and response.text:
                        return response.text
                    return "Error: Gemini API returned empty response. Please try again."
                except Exception as api_error:
                    return f"Error processing image(s) with Gemini API: {api_error}."
            elif image_path:
                try:
                    image = PIL.Image.open(image_path)
                except Exception as img_error:
                    return f"Error loading image: {img_error}. Please check the image format and try again."
                try:
                    start_time = time.time()
                    response = genai_client.models.generate_content(
                        model=model_name,
                        contents=[prompt, image],
                    )
                    log_debug(f"Gemini API call (single image) took {time.time() - start_time:.2f}s")
                    if hasattr(response, "text") and response.text:
                        return response.text
                    return "Error: Gemini API returned empty response. Please try again."
                except Exception as api_error:
                    return f"Error processing image with Gemini API: {api_error}."
            else:
                try:
                    start_time = time.time()
                    response = genai_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    log_debug(f"Gemini API call (text only) took {time.time() - start_time:.2f}s")
                    if hasattr(response, "text") and response.text:
                        return response.text
                    return "Error: Gemini API returned empty response. Please try again."
                except Exception as api_error:
                    return f"Error with Gemini API: {api_error}."

        # Use OLD google.generativeai package
        elif genai_model:
            if image_paths and len(image_paths) > 0:
                images = []
                try:
                    for p in image_paths:
                        images.append(PIL.Image.open(p))
                except Exception as img_error:
                    return f"Error loading image(s): {img_error}. Please check the image file format and try again."
                try:
                    start_time = time.time()
                    response = genai_model.generate_content([prompt, *images])
                    log_debug(f"Gemini API call (multi-image) took {time.time() - start_time:.2f}s")
                    if response.candidates:
                        return response.text
                    return "Error: Gemini API returned empty response. Please try again."
                except Exception as api_error:
                    return f"Error processing image(s) with Gemini API: {api_error}."
            elif image_path:
                try:
                    image = PIL.Image.open(image_path)
                except Exception as img_error:
                    return f"Error loading image: {img_error}. Please check the image file format and try again."
                try:
                    start_time = time.time()
                    response = genai_model.generate_content([prompt, image])
                    log_debug(f"Gemini API call (single image) took {time.time() - start_time:.2f}s")
                    if response.candidates:
                        return response.text
                    return "Error: Gemini API returned empty response. Please try again."
                except Exception as api_error:
                    return f"Error processing image with Gemini API: {api_error}."
            else:
                try:
                    start_time = time.time()
                    response = genai_model.generate_content(prompt)
                    log_debug(f"Gemini API call (text) took {time.time() - start_time:.2f}s")
                    if response.candidates:
                        return response.text
                    return "Error: Gemini API returned empty response. Please try again."
                except Exception as api_error:
                    return f"Error with Gemini API: {api_error}."
        else:
            return "Error: No Gemini API client available. Check your API key and package installation."

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Detailed error: {error_details}")
        return f"An unexpected error occurred: {e}. Please try again later."


async def run_gemini_async(
    prompt: str,
    image_path: str | None = None,
    image_paths: list[str] | None = None,
    model_override: str | None = None,
) -> str:
    """Async wrapper for run_gemini — runs the blocking Gemini call in a thread pool."""
    return await asyncio.to_thread(run_gemini, prompt, image_path, image_paths, model_override)


# ---------------------------------------------------------------------------
# Prompt quality scoring
# ---------------------------------------------------------------------------

_QUALITY_SCORE_KEYS = (
    "clarity",
    "visual_specificity",
    "composition_lighting",
    "consistency",
    "model_compatibility",
    "overall",
)

_DEFAULT_QUALITY_SCORES = {
    "clarity": 7.0,
    "visual_specificity": 7.0,
    "composition_lighting": 7.0,
    "consistency": 7.0,
    "model_compatibility": 7.0,
    "overall": 7.0,
}

_DEFAULT_TOP_IMPROVEMENTS = [
    "Add more concrete subject details.",
    "Specify composition and camera perspective.",
    "Clarify lighting and atmosphere cues.",
]


def _clamp_quality_score(value, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    numeric = max(0.0, min(10.0, numeric))
    return round(numeric, 1)


def _extract_json_object(text: str) -> str | None:
    if not text:
        return None

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        return candidate[start : end + 1]
    return None


def evaluate_enhanced_prompt_quality(
    enhanced_prompt: str,
    prompt_type: str | None = None,
    model: str | None = None,
    gemini_model: str | None = None,
) -> dict | None:
    if not enhanced_prompt:
        return None
    if enhanced_prompt.startswith("Error") or enhanced_prompt.startswith("An unexpected error"):
        return None

    scoring_prompt = (
        "You are a strict quality evaluator for generative AI prompts. "
        "Assess the enhanced prompt and return ONLY valid JSON with this exact schema: "
        '{"quality_scores":{"clarity":0-10,"visual_specificity":0-10,"composition_lighting":0-10,"consistency":0-10,"model_compatibility":0-10,"overall":0-10},'
        '"top_improvements":["string","string","string"]}.\n'
        "Rules: no markdown, no explanations, scores must be numbers, top_improvements must be concise actionable items.\n"
        f"Prompt type context: {prompt_type or 'unknown'}\n"
        f"Model context: {model or 'default'}\n"
        f"Enhanced prompt:\n{enhanced_prompt}"
    )

    try:
        raw_result = run_gemini(scoring_prompt, model_override=gemini_model)
        if raw_result.startswith("Error") or raw_result.startswith("An unexpected error"):
            log_debug(f"Prompt quality scoring skipped due to Gemini error: {raw_result[:120]}")
            return None

        json_payload = _extract_json_object(raw_result)
        if not json_payload:
            log_debug("Prompt quality scoring skipped: no JSON payload found")
            return None

        parsed = json.loads(json_payload)
        if not isinstance(parsed, dict):
            log_debug("Prompt quality scoring skipped: parsed payload is not an object")
            return None

        raw_scores = parsed.get("quality_scores", {})
        quality_scores = {}
        for key in _QUALITY_SCORE_KEYS:
            default = _DEFAULT_QUALITY_SCORES[key]
            value = raw_scores.get(key, default) if isinstance(raw_scores, dict) else default
            quality_scores[key] = _clamp_quality_score(value, default)

        improvements_raw = parsed.get("top_improvements", [])
        improvements: list[str] = []
        if isinstance(improvements_raw, list):
            for item in improvements_raw:
                if not isinstance(item, str):
                    continue
                cleaned = item.strip()
                if cleaned:
                    improvements.append(cleaned[:160])

        if not improvements:
            improvements = _DEFAULT_TOP_IMPROVEMENTS.copy()

        return {
            "quality_scores": quality_scores,
            "top_improvements": improvements[:3],
        }
    except Exception as e:
        log_debug(f"Prompt quality scoring failed: {e}")
        return None
