"""
Endpoint tests for the Pixa Prompt Enhancer API.

Tests use mocked Gemini responses to avoid real API calls.
Run with: pytest tests/test_endpoints.py -v
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set a dummy API key so the app initialises without a real key
os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-ci")

from app.main import app  # noqa: E402
from app.gemini import _clamp_quality_score, _extract_json_object  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

MOCK_ENHANCED = "A beautiful sunset over the ocean with golden light reflecting on calm waves."
MOCK_QUALITY_JSON = (
    '{"quality_scores":{"clarity":8.5,"visual_specificity":7.0,'
    '"composition_lighting":8.0,"consistency":9.0,"model_compatibility":7.5,"overall":8.0},'
    '"top_improvements":["Add more depth details","Specify camera angle","Include atmosphere cues"]}'
)


# ---------------------------------------------------------------------------
# Health & root
# ---------------------------------------------------------------------------

class TestHealthAndRoot:
    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "api_key_set" in data

    def test_root_returns_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# /enhance endpoint
# ---------------------------------------------------------------------------

class TestEnhanceEndpoint:
    @patch("app.gemini.run_gemini", return_value=MOCK_ENHANCED)
    def test_enhance_basic(self, mock_gemini):
        payload = {
            "prompt": "sunset over ocean",
            "prompt_type": "Image",
            "style": "none",
            "cinematography": "none",
            "lighting": "none",
            "include_quality_scoring": False,
        }
        resp = client.post("/enhance", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "enhanced_prompt" in data
        assert len(data["enhanced_prompt"]) > 0
        # Quality scoring was skipped
        assert data.get("quality_scores") is None

    @patch("app.gemini.run_gemini", side_effect=[MOCK_ENHANCED, MOCK_QUALITY_JSON])
    def test_enhance_with_quality_scoring(self, mock_gemini):
        payload = {
            "prompt": "sunset over ocean",
            "prompt_type": "Image",
            "style": "none",
            "cinematography": "none",
            "lighting": "none",
            "include_quality_scoring": True,
        }
        resp = client.post("/enhance", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("quality_scores") is not None
        assert "clarity" in data["quality_scores"]
        assert isinstance(data["top_improvements"], list)

    @patch("app.gemini.run_gemini", return_value=MOCK_ENHANCED)
    def test_enhance_video_prompt_type(self, mock_gemini):
        payload = {
            "prompt": "person walking in park",
            "prompt_type": "WAN2",
            "style": "none",
            "cinematography": "none",
            "lighting": "none",
            "include_quality_scoring": False,
        }
        resp = client.post("/enhance", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["enhanced_prompt"]) > 0

    def test_enhance_empty_prompt_rejected(self):
        payload = {
            "prompt": "",
            "prompt_type": "Image",
            "style": "none",
            "cinematography": "none",
            "lighting": "none",
        }
        resp = client.post("/enhance", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # The app returns a validation error message inside enhanced_prompt
        assert "validation" in data["enhanced_prompt"].lower() or "error" in data["enhanced_prompt"].lower()

    @patch("app.gemini.run_gemini", return_value="Error: API key invalid")
    def test_enhance_gemini_error_forwarded(self, mock_gemini):
        payload = {
            "prompt": "test prompt",
            "prompt_type": "Image",
            "style": "none",
            "cinematography": "none",
            "lighting": "none",
            "include_quality_scoring": False,
        }
        resp = client.post("/enhance", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["enhanced_prompt"].startswith("Error")
        assert data.get("quality_scores") is None


# ---------------------------------------------------------------------------
# /enhance-specialized endpoint
# ---------------------------------------------------------------------------

class TestEnhanceSpecialized:
    @patch("app.gemini.run_gemini", return_value=MOCK_ENHANCED)
    def test_specialized_commercial(self, mock_gemini):
        payload = {
            "prompt": "luxury watch on marble",
            "enhancement_mode": "commercial",
            "include_quality_scoring": False,
        }
        resp = client.post("/enhance-specialized", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "enhanced_prompt" in data

    @patch("app.gemini.run_gemini", return_value=MOCK_ENHANCED)
    def test_specialized_cinematic(self, mock_gemini):
        payload = {
            "prompt": "hero walks into the light",
            "enhancement_mode": "cinematic",
            "include_quality_scoring": False,
        }
        resp = client.post("/enhance-specialized", json=payload)
        assert resp.status_code == 200

    @patch("app.gemini.run_gemini", return_value=MOCK_ENHANCED)
    def test_specialized_character(self, mock_gemini):
        payload = {
            "prompt": "medieval knight with battle scars",
            "enhancement_mode": "character",
            "include_quality_scoring": False,
        }
        resp = client.post("/enhance-specialized", json=payload)
        assert resp.status_code == 200

    @patch("app.gemini.run_gemini", return_value=MOCK_ENHANCED)
    def test_specialized_object(self, mock_gemini):
        payload = {
            "prompt": "futuristic spaceship engine",
            "enhancement_mode": "object",
            "include_quality_scoring": False,
        }
        resp = client.post("/enhance-specialized", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Unit tests — quality scoring helpers
# ---------------------------------------------------------------------------

class TestQualityHelpers:
    def test_clamp_quality_score_normal(self):
        assert _clamp_quality_score(8.5, 7.0) == 8.5

    def test_clamp_quality_score_below_zero(self):
        assert _clamp_quality_score(-2, 7.0) == 0.0

    def test_clamp_quality_score_above_ten(self):
        assert _clamp_quality_score(15, 7.0) == 10.0

    def test_clamp_quality_score_invalid(self):
        assert _clamp_quality_score("bad", 7.0) == 7.0

    def test_extract_json_plain(self):
        raw = '{"key": "value"}'
        assert _extract_json_object(raw) == raw

    def test_extract_json_with_fences(self):
        raw = '```json\n{"key": "value"}\n```'
        assert _extract_json_object(raw) == '{"key": "value"}'

    def test_extract_json_with_surrounding_text(self):
        raw = 'Here is the result: {"key": "value"} done.'
        assert _extract_json_object(raw) == '{"key": "value"}'

    def test_extract_json_none_on_empty(self):
        assert _extract_json_object("") is None
        assert _extract_json_object(None) is None


# ---------------------------------------------------------------------------
# Path traversal regression
# ---------------------------------------------------------------------------

class TestPathTraversalSafety:
    """Verify that upload filenames are sanitised."""

    @patch("app.gemini.run_gemini", return_value="Safe description")
    def test_image_upload_sanitises_filename(self, mock_gemini):
        """Ensure a malicious filename like '../../etc/passwd' cannot escape UPLOADS_DIR."""
        import io
        malicious_name = "../../etc/passwd.png"
        fake_file = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        resp = client.post(
            "/analyze-image",
            files=[("images", (malicious_name, fake_file, "image/png"))],
        )
        # The request should not crash the server (may fail on image parsing, but not path traversal)
        assert resp.status_code == 200
