"""
Ollama provider for local Gemma model inference.

Connects to a local ollama instance (http://localhost:11434) to run
Gemma models for privacy-focused, offline prompt enhancement.
"""

import base64
import json
import os
import time
from typing import Any

import httpx
import PIL.Image

from app.logger import log_debug
from app.providers.base import ModelProvider

# --- Configuration ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma:2b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

log_debug(f"[ollama] Host: {OLLAMA_HOST}, Model: {OLLAMA_MODEL}")


class OllamaProvider(ModelProvider):
    """Ollama provider for local Gemma models."""

    def __init__(self, model: str | None = None):
        self.client = httpx.Client(timeout=OLLAMA_TIMEOUT)
        self.default_model = model or OLLAMA_MODEL
        self._check_connection()

    def _check_connection(self):
        """Verify ollama is running."""
        try:
            resp = self.client.get(f"{OLLAMA_HOST}/api/tags")
            if resp.status_code == 200:
                log_debug("[ollama] Connected successfully")
            else:
                log_debug(f"[ollama] Warning: ollama returned status {resp.status_code}")
        except Exception as e:
            log_debug(f"[ollama] Connection failed: {e}. Is ollama running?")

    def generate_text(
        self,
        prompt: str,
        model_override: str | None = None,
    ) -> str:
        """Generate text from a prompt (no images)."""
        model = model_override or self.default_model
        try:
            start_time = time.time()
            resp = self.client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            log_debug(f"[ollama] Text-only call took {time.time() - start_time:.2f}s")
            resp.raise_for_status()
            result = resp.json()
            return result.get("response", "")
        except Exception as e:
            log_debug(f"[ollama] generate_text error: {e}")
            return f"Error: Ollama generation failed: {e}"

    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        model_override: str | None = None,
    ) -> str:
        """Generate text from a prompt with a single image."""
        model = model_override or self.default_model

        # Check if model supports images (llava, bakllava, etc.)
        multimodal_models = ["llava", "bakllava", "moondream", "minicpm"]
        if not any(m in model.lower() for m in multimodal_models):
            log_debug(f"[ollama] Image input requested but {model} doesn't support images")
            return f"Image analysis is not available with {model}. Use a multimodal model like llava, or switch to Gemini (cloud)."

        try:
            # Convert image to base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            start_time = time.time()
            resp = self.client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                },
            )
            log_debug(f"[ollama] Single-image call took {time.time() - start_time:.2f}s")
            resp.raise_for_status()
            result = resp.json()
            return result.get("response", "")
        except Exception as e:
            log_debug(f"[ollama] generate_with_image error: {e}")
            return f"Error: Ollama image generation failed: {e}"

    def generate_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        model_override: str | None = None,
    ) -> str:
        """Generate text from a prompt with multiple images."""
        model = model_override or self.default_model

        # Check if model supports images
        multimodal_models = ["llava", "bakllava", "moondream", "minicpm"]
        if not any(m in model.lower() for m in multimodal_models):
            log_debug(f"[ollama] Multi-image input requested but {model} doesn't support images")
            return f"Image analysis is not available with {model}. Use a multimodal model like llava, or switch to Gemini (cloud)."

        try:
            images = []
            for path in image_paths:
                with open(path, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode("utf-8"))

            start_time = time.time()
            resp = self.client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": images,
                    "stream": False,
                },
            )
            log_debug(f"[ollama] Multi-image call took {time.time() - start_time:.2f}s")
            resp.raise_for_status()
            result = resp.json()
            return result.get("response", "")
        except Exception as e:
            log_debug(f"[ollama] generate_with_images error: {e}")
            return f"Error: Ollama multi-image generation failed: {e}"

    def evaluate_quality(
        self,
        enhanced_prompt: str,
        prompt_type: str | None = None,
        model: str | None = None,
    ) -> dict | None:
        """Evaluate prompt quality using ollama/Gemma.

        Gemma 2B may struggle with strict JSON output. We use a simplified
        scoring approach with fallback defaults.
        """
        if not enhanced_prompt:
            return None
        if enhanced_prompt.startswith("Error"):
            return None

        # Simplified scoring prompt for smaller models
        scoring_prompt = (
            "Rate this AI prompt from 0-10 on: clarity, visual specificity, "
            "composition/lighting, consistency, model compatibility, and overall quality. "
            "Return ONLY JSON with format: "
            '{"quality_scores":{"clarity":X,"visual_specificity":X,"composition_lighting":X,"consistency":X,"model_compatibility":X,"overall":X},'
            '"top_improvements":["improvement1","improvement2","improvement3"]}\n'
            f"Prompt type: {prompt_type or 'unknown'}\n"
            f"Model: {model or 'default'}\n"
            f"Prompt: {enhanced_prompt}"
        )

        try:
            raw_result = self.generate_text(scoring_prompt)
            if raw_result.startswith("Error"):
                log_debug(f"[ollama] Quality scoring failed: {raw_result[:80]}")
                return _get_default_scores()

            # Try to extract JSON
            json_payload = _extract_json_object(raw_result)
            if not json_payload:
                log_debug("[ollama] No JSON in quality response, using defaults")
                return _get_default_scores()

            parsed = json.loads(json_payload)
            if not isinstance(parsed, dict):
                return _get_default_scores()

            raw_scores = parsed.get("quality_scores", {})
            quality_scores = {}
            for key in ("clarity", "visual_specificity", "composition_lighting", "consistency", "model_compatibility", "overall"):
                try:
                    val = float(raw_scores.get(key, 7.0))
                    quality_scores[key] = max(0.0, min(10.0, round(val, 1)))
                except (ValueError, TypeError):
                    quality_scores[key] = 7.0

            improvements = parsed.get("top_improvements", [])
            if not isinstance(improvements, list) or not improvements:
                improvements = ["Add more specific details.", "Clarify composition.", "Specify lighting."]

            return {
                "quality_scores": quality_scores,
                "top_improvements": improvements[:3],
            }
        except Exception as e:
            log_debug(f"[ollama] Quality scoring exception: {e}")
            return _get_default_scores()


def _extract_json_object(text: str) -> str | None:
    """Extract JSON from text, handling common formatting issues."""
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


def _get_default_scores() -> dict:
    """Return default quality scores when evaluation fails."""
    return {
        "quality_scores": {
            "clarity": 7.0,
            "visual_specificity": 7.0,
            "composition_lighting": 7.0,
            "consistency": 7.0,
            "model_compatibility": 7.0,
            "overall": 7.0,
        },
        "top_improvements": ["Add more specific details.", "Clarify composition.", "Specify lighting."],
    }
