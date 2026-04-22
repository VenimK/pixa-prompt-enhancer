"""
Tests for app/ltx2_prompts.py — LTX-2.3 prompt engineering module.

Covers:
  - Unit tests for build_ltx2_meta_prompt across i2v / t2v / audio_sync modes
  - Mode detection logic
  - Audio-to-prose conversion
  - Negative prompt population
  - Endpoint integration (POST /enhance with prompt_type=LTX2)
"""

import os
import pytest
from unittest.mock import patch

os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-ci")

from app.ltx2_prompts import (
    build_ltx2_meta_prompt,
    detect_ltx2_mode,
    LTX23_NEGATIVE_PROMPT,
    _audio_to_prose,
    _style_to_prose,
    _movement_to_prose,
)
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

class TestModeDetection:
    def test_t2v_no_image_no_audio(self):
        assert detect_ltx2_mode(has_image=False, has_audio=False) == "t2v"

    def test_i2v_with_image(self):
        assert detect_ltx2_mode(has_image=True, has_audio=False) == "i2v"

    def test_audio_sync_with_audio(self):
        assert detect_ltx2_mode(has_image=False, has_audio=True) == "audio_sync"

    def test_audio_sync_takes_priority_over_image(self):
        assert detect_ltx2_mode(has_image=True, has_audio=True) == "audio_sync"


# ---------------------------------------------------------------------------
# build_ltx2_meta_prompt — structure & content
# ---------------------------------------------------------------------------

class TestBuildMetaPrompt:
    def test_t2v_basic(self):
        result = build_ltx2_meta_prompt(user_prompt="a dog running in a park")
        assert "positive" in result
        assert "negative" in result
        assert "mode" in result
        assert result["mode"] == "t2v"
        assert "a dog running in a park" in result["positive"]
        assert result["negative"] == LTX23_NEGATIVE_PROMPT

    def test_i2v_mode_set(self):
        result = build_ltx2_meta_prompt(
            user_prompt="the character starts dancing",
            image_description="A woman in a red dress standing in a studio",
        )
        assert result["mode"] == "i2v"
        # Should include image ref and i2v instruction
        assert "red dress" in result["positive"]
        assert "do not" in result["positive"].lower() or "DO NOT" in result["positive"]

    def test_audio_sync_mode(self):
        result = build_ltx2_meta_prompt(
            user_prompt="singer performs on stage",
            audio_characteristics={"vocal_style": "singing", "tempo": "fast", "tempo_bpm": 140},
        )
        assert result["mode"] == "audio_sync"
        assert "singing" in result["positive"].lower()
        assert "fast" in result["positive"].lower()

    def test_negative_prompt_always_present(self):
        result = build_ltx2_meta_prompt(user_prompt="test")
        assert result["negative"]
        assert "judder" in result["negative"]

    def test_style_included(self):
        result = build_ltx2_meta_prompt(
            user_prompt="a dancer",
            ltx2_style="noir",
        )
        assert "noir" in result["positive"].lower()

    def test_audio_generation_disabled(self):
        result = build_ltx2_meta_prompt(
            user_prompt="silent film",
            audio_generation="disabled",
        )
        assert "do not describe audio" in result["positive"].lower()

    def test_movement_level_static(self):
        result = build_ltx2_meta_prompt(
            user_prompt="portrait shot",
            movement_level="static",
        )
        assert "still" in result["positive"].lower() or "lip-sync" in result["positive"].lower()

    def test_movement_auto_detection_fast_singing(self):
        result = build_ltx2_meta_prompt(
            user_prompt="rock performance",
            movement_level="auto",
            audio_characteristics={"tempo": "fast", "vocal_style": "singing"},
        )
        # Should auto-detect dynamic
        assert "dynamic" in result["positive"].lower() or "energetic" in result["positive"].lower()


# ---------------------------------------------------------------------------
# Audio-to-prose helper
# ---------------------------------------------------------------------------

class TestAudioToProse:
    def test_empty_dict(self):
        assert _audio_to_prose({}) == ""

    def test_none(self):
        assert _audio_to_prose(None) == ""

    def test_singing_vocal(self):
        prose = _audio_to_prose({"vocal_style": "singing", "vocal_range": "high"})
        assert "singing" in prose
        assert "high" in prose

    def test_fast_tempo(self):
        prose = _audio_to_prose({"tempo": "fast", "tempo_bpm": 150})
        assert "fast" in prose
        assert "150" in prose

    def test_genre_rock(self):
        prose = _audio_to_prose({"genre": "rock"})
        assert "rock" in prose.lower()

    def test_high_danceability(self):
        prose = _audio_to_prose({"danceability": 0.85})
        assert "dance" in prose.lower()


# ---------------------------------------------------------------------------
# Style-to-prose helper
# ---------------------------------------------------------------------------

class TestStyleToProse:
    def test_none_style(self):
        assert _style_to_prose(None) == ""

    def test_auto_style(self):
        assert _style_to_prose("auto") == ""

    def test_known_style(self):
        prose = _style_to_prose("cinematic")
        assert "cinematic" in prose.lower()

    def test_unknown_style(self):
        assert _style_to_prose("nonexistent_style_xyz") == ""


# ---------------------------------------------------------------------------
# Movement-to-prose helper
# ---------------------------------------------------------------------------

class TestMovementToProse:
    def test_explicit_static(self):
        prose = _movement_to_prose("static", {})
        assert "still" in prose.lower() or "lip-sync" in prose.lower()

    def test_explicit_dynamic(self):
        prose = _movement_to_prose("dynamic", {})
        assert "energetic" in prose.lower()

    def test_auto_defaults_to_natural(self):
        prose = _movement_to_prose("auto", {})
        assert "natural" in prose.lower() or "realistic" in prose.lower()

    def test_auto_with_fast_singing(self):
        prose = _movement_to_prose("auto", {"tempo": "fast", "vocal_style": "singing"})
        assert "energetic" in prose.lower() or "dynamic" in prose.lower()


# ---------------------------------------------------------------------------
# Endpoint integration (mocked Gemini)
# ---------------------------------------------------------------------------

class TestLTX2Endpoint:
    @patch("app.gemini.run_gemini")
    def test_enhance_ltx2_returns_negative_prompt(self, mock_gemini):
        mock_gemini.return_value = "A young woman is stepping forward in a sunlit park, hair swaying gently."
        resp = client.post("/enhance", json={
            "prompt": "woman walks forward",
            "prompt_type": "LTX2",
            "style": "none",
            "cinematography": "none",
            "lighting": "none",
            "include_quality_scoring": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "enhanced_prompt" in data
        assert data["negative_prompt"] is not None
        assert "judder" in data["negative_prompt"]

    @patch("app.gemini.run_gemini")
    def test_enhance_ltx2_prompt_under_limit(self, mock_gemini):
        mock_gemini.return_value = "A" * 2500  # within 3000 limit
        resp = client.post("/enhance", json={
            "prompt": "test prompt",
            "prompt_type": "LTX2",
            "style": "none",
            "cinematography": "none",
            "lighting": "none",
            "include_quality_scoring": False,
        })
        data = resp.json()
        assert len(data["enhanced_prompt"]) <= 3000

    @patch("app.gemini.run_gemini")
    def test_enhance_image_has_no_negative(self, mock_gemini):
        mock_gemini.return_value = "Enhanced image prompt."
        resp = client.post("/enhance", json={
            "prompt": "a cat",
            "prompt_type": "Image",
            "style": "none",
            "cinematography": "none",
            "lighting": "none",
            "include_quality_scoring": False,
        })
        data = resp.json()
        assert data["negative_prompt"] is None
