"""Unit tests for performance-oriented helpers (audio fallback, image prep, parsers)."""

import os
import sys
import tempfile

os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-ci")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.audio_analysis import analyze_audio_characteristics
from app.image_prep import close_images, prepare_image_for_api
from app.main import _parse_multi_image_analysis, get_provider, _provider_cache


class TestAudioFilenameFallback:
    def test_singing_filename(self):
        result = analyze_audio_characteristics("my_song_vocals.mp3")
        assert result["audio_type"] in {"singing", "music"}
        assert result["has_vocals"] is True
        assert "description" in result
        assert result["description"].endswith(".")

    def test_speech_filename(self):
        result = analyze_audio_characteristics("interview_speech.wav")
        assert result["audio_type"] == "speech"

    def test_fast_tempo_hint(self):
        result = analyze_audio_characteristics("upbeat_dance_track.mp3")
        assert result["tempo"] == "fast"


class TestImagePrep:
    def test_downscales_large_image(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "big.png")
            Image.new("RGB", (4000, 3000), color=(20, 40, 80)).save(path)
            prepared = prepare_image_for_api(path, max_side=1280)
            try:
                assert max(prepared.size) <= 1280
                assert prepared.mode == "RGB"
            finally:
                close_images([prepared])

    def test_converts_rgba_to_rgb(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "alpha.png")
            Image.new("RGBA", (64, 64), color=(10, 20, 30, 128)).save(path)
            prepared = prepare_image_for_api(path)
            try:
                assert prepared.mode == "RGB"
            finally:
                close_images([prepared])


class TestMultiImageParser:
    def test_parses_headed_sections(self):
        raw = """
## Combined
Shared red palette and studio lighting.

## Reference A
A vintage camera on a table.

## Reference B
A sports car in a showroom.

## Style Comparison
• Image A: product photo (warm)
• Image B: automotive (cool)
"""
        parsed = _parse_multi_image_analysis(raw)
        assert "vintage camera" in parsed["a"]
        assert "sports car" in parsed["b"]
        assert "Shared red palette" in parsed["combined"]
        assert "Style Comparison" in parsed["combined"]

    def test_unstructured_text_becomes_combined(self):
        raw = "Just a plain description of both images."
        parsed = _parse_multi_image_analysis(raw)
        assert parsed["combined"] == raw
        assert parsed["a"] is None
        assert parsed["b"] is None


class TestProviderCache:
    def test_gemini_provider_is_reused(self):
        _provider_cache.clear()
        first = get_provider("gemini")
        second = get_provider("gemini")
        assert first is second
