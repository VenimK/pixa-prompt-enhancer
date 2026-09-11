"""Fast audio-feature extraction for LTX-2 prompt enrichment.

Librosa/numpy are imported lazily so the web app can start without paying
the scientific-stack tax on every request. Each expensive transform
(HPSS, beat tracking, STFT-derived features) is computed once.
"""

from __future__ import annotations

import traceback
from typing import Any

from app.logger import log_debug

# Analysis window. Tempo, energy, and vocal presence stabilize well before 15s.
_ANALYSIS_DURATION_S = 15.0
_TARGET_SR = 22050
_HOP_LENGTH = 512


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce numpy scalars / 0-d arrays / None into a Python float."""
    try:
        if value is None:
            return default
        item = getattr(value, "item", None)
        if callable(item):
            return float(item())
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe(obj: Any) -> Any:
    """Convert numpy types so the payload is JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "shape", None) == ():
        return obj.item()
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        try:
            return obj.tolist()
        except Exception:
            return obj
    return obj


def analyze_audio_characteristics(filename: str, file_size: int = 0) -> dict:
    """Fallback filename-based analysis when real audio analysis fails."""
    characteristics = {
        "audio_type": "unknown",
        "tempo": "medium",
        "mood": "neutral",
        "instruments": [],
        "vocals": False,
        "has_vocals": False,
        "description": "",
        "tempo_bpm": None,
        "energy_level": "medium",
        "vocal_confidence": 0.0,
        "danceability": 0.5,
    }

    filename_lower = (filename or "").lower()

    if any(word in filename_lower for word in ["song", "music", "track", "audio", "beat"]):
        characteristics["audio_type"] = "music"
    elif any(word in filename_lower for word in ["speech", "talk", "voice", "dialogue", "speaking"]):
        characteristics["audio_type"] = "speech"
    elif any(word in filename_lower for word in ["sing", "vocal", "lyrics"]):
        characteristics["audio_type"] = "singing"
    elif any(word in filename_lower for word in ["instrumental", "ambient", "background"]):
        characteristics["audio_type"] = "instrumental"

    if any(word in filename_lower for word in ["fast", "quick", "upbeat", "energetic", "dance"]):
        characteristics["tempo"] = "fast"
    elif any(word in filename_lower for word in ["slow", "calm", "relaxing", "ambient", "chill"]):
        characteristics["tempo"] = "slow"

    if any(word in filename_lower for word in ["happy", "joy", "celebration", "upbeat", "party"]):
        characteristics["mood"] = "happy"
    elif any(word in filename_lower for word in ["sad", "emotional", "dramatic", "melancholy"]):
        characteristics["mood"] = "emotional"
    elif any(word in filename_lower for word in ["energetic", "powerful", "intense", "epic"]):
        characteristics["mood"] = "energetic"
    elif any(word in filename_lower for word in ["calm", "peaceful", "relaxing", "meditation"]):
        characteristics["mood"] = "calm"

    if any(word in filename_lower for word in ["vocals", "singing", "voice", "lyrics", "song"]):
        characteristics["vocals"] = True
        characteristics["has_vocals"] = True
        characteristics["vocal_confidence"] = 0.7

    description_parts = []
    if characteristics["audio_type"] == "singing":
        description_parts.append("singing performance with vocals")
    elif characteristics["audio_type"] == "music":
        description_parts.append("musical track")
    elif characteristics["audio_type"] == "speech":
        description_parts.append("spoken dialogue/voice")
    elif characteristics["audio_type"] == "instrumental":
        description_parts.append("instrumental music")
    else:
        description_parts.append("audio track")

    if characteristics["tempo"] == "fast":
        description_parts.append("with fast, energetic rhythm suitable for dancing")
    elif characteristics["tempo"] == "slow":
        description_parts.append("with slow, gentle rhythm")
    else:
        description_parts.append("with moderate tempo")

    if characteristics["mood"] == "happy":
        description_parts.append("creating a joyful, upbeat mood")
    elif characteristics["mood"] == "emotional":
        description_parts.append("with emotional, dramatic atmosphere")
    elif characteristics["mood"] == "energetic":
        description_parts.append("building high energy and excitement")
    elif characteristics["mood"] == "calm":
        description_parts.append("establishing a peaceful, serene mood")

    if characteristics["vocals"]:
        description_parts.append("featuring vocal performance that should be lip-synced")

    characteristics["description"] = " ".join(description_parts) + "."
    return characteristics


def analyze_real_audio_characteristics(file_path: str, filename: str) -> dict:
    """Analyze audio using a single pass of librosa features."""
    try:
        import librosa
        import numpy as np
    except ImportError as e:
        log_debug(f"librosa/numpy not available, falling back to filename analysis: {e}")
        return analyze_audio_characteristics(filename, 0)

    try:
        log_debug(f"Starting audio analysis for: {filename}")
        y, sr = librosa.load(
            file_path,
            sr=_TARGET_SR,
            mono=True,
            duration=_ANALYSIS_DURATION_S,
            res_type="kaiser_fast",
        )
        if y is None or len(y) == 0:
            log_debug("Empty audio buffer, falling back to filename analysis")
            return analyze_audio_characteristics(filename, 0)

        log_debug(f"Audio loaded: {len(y)} samples, {sr} Hz")

        characteristics: dict[str, Any] = {
            "audio_type": "unknown",
            "tempo": "medium",
            "tempo_bpm": None,
            "mood": "neutral",
            "energy_level": "medium",
            "has_vocals": False,
            "vocal_confidence": 0.0,
            "danceability": 0.5,
            "description": "",
            "time_signature": "4/4",
            "beat_strength": "medium",
            "syncopation": "low",
            "vocal_style": "unknown",
            "vocal_range": "medium",
            "performance_type": "studio_recording",
            "genre": "unknown",
            "spectral_characteristics": {},
            "dynamic_range": "medium",
            "emotional_arc": "stable",
            "vocal_count": "unknown",
            "vocal_density": 0.0,
            "vocal_separation": "unknown",
        }

        hop = _HOP_LENGTH

        # --- Core features (each computed once) ---
        tempo_raw, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        tempo = _as_float(tempo_raw, 120.0)
        characteristics["tempo_bpm"] = tempo
        if tempo < 60:
            characteristics["tempo"] = "very_slow"
        elif tempo < 90:
            characteristics["tempo"] = "slow"
        elif tempo < 120:
            characteristics["tempo"] = "medium"
        elif tempo < 140:
            characteristics["tempo"] = "fast"
        else:
            characteristics["tempo"] = "very_fast"

        beat_count = int(getattr(beats, "size", len(beats)))
        beat_diffs = None
        if beat_count > 10:
            beat_diffs = np.diff(beats.astype(float))
            mean_diff = float(np.mean(beat_diffs)) if len(beat_diffs) else 0.0
            if mean_diff > 0:
                beat_consistency = 1.0 - float(np.std(beat_diffs)) / mean_diff
            else:
                beat_consistency = 0.5
            if beat_consistency > 0.8:
                characteristics["beat_strength"] = "strong"
            elif beat_consistency > 0.5:
                characteristics["beat_strength"] = "medium"
            else:
                characteristics["beat_strength"] = "weak"
            characteristics["time_signature"] = "3/4" if mean_diff > 0.8 and beat_count % 3 == 0 else "4/4"
        else:
            beat_consistency = 0.5

        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        energy = _as_float(np.mean(rms), 0.1)
        dynamic_range = _as_float(np.max(rms) - np.min(rms), 0.1) if len(rms) else 0.1
        if energy > 0.15:
            characteristics["energy_level"] = "high"
        elif energy > 0.10:
            characteristics["energy_level"] = "medium"
        else:
            characteristics["energy_level"] = "low"

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        harmonic, percussive = librosa.effects.hpss(y)

        avg_centroid = _as_float(np.mean(spectral_centroids), 0.0)
        spectral_variance = _as_float(np.var(spectral_centroids), 0.0)
        characteristics["spectral_characteristics"] = {
            "brightness": avg_centroid,
            "warmth": bool(avg_centroid < 2000),
            "spectral_variance": spectral_variance,
        }

        # Emotion from already-computed features
        energy_norm = min(energy / 0.2, 1.0)
        tempo_norm = min(tempo / 180.0, 1.0)
        contrast_norm = min(_as_float(np.mean(spectral_contrast), 0.0) / 20.0, 1.0)
        mfcc_norm = min(_as_float(np.std(mfccs, axis=1).mean(), 0.0) / 50.0, 1.0)
        emotion_score = energy_norm * 0.3 + contrast_norm * 0.25 + mfcc_norm * 0.25 + tempo_norm * 0.2
        if emotion_score > 0.7:
            characteristics["mood"] = "energetic"
            characteristics["emotional_intensity"] = "high"
        elif emotion_score > 0.4:
            characteristics["mood"] = "emotional"
            characteristics["emotional_intensity"] = "medium"
        elif emotion_score > 0.2:
            characteristics["mood"] = "contemplative"
            characteristics["emotional_intensity"] = "low"
        else:
            characteristics["mood"] = "futuristic"
            characteristics["emotional_intensity"] = "very_low"

        # Vocal presence
        vocal_score = 0.0
        if avg_centroid > 2000:
            vocal_score += 0.3
        if spectral_variance > 500000:
            vocal_score += 0.2
        mfcc_std = np.std(mfccs, axis=1)
        if _as_float(np.mean(mfcc_std[1:4]), 0.0) > 15:
            vocal_score += 0.3
        if _as_float(np.mean(zcr), 0.0) > 0.05:
            vocal_score += 0.2
        characteristics["vocal_confidence"] = min(float(vocal_score), 1.0)
        characteristics["has_vocals"] = vocal_score > 0.5

        harmonic_energy = float(np.sum(harmonic ** 2))
        total_energy = float(np.sum(y ** 2))
        vocal_density = (harmonic_energy / total_energy) if total_energy > 0 else 0.0
        characteristics["vocal_density"] = float(vocal_density)

        if characteristics["has_vocals"]:
            if vocal_density > 0.8:
                characteristics["vocal_count"] = "choir"
            elif vocal_density > 0.6:
                characteristics["vocal_count"] = "group"
            elif vocal_density > 0.4:
                characteristics["vocal_count"] = "duo"
            elif vocal_density > 0.2:
                characteristics["vocal_count"] = "solo"
            else:
                characteristics["vocal_count"] = "minimal"

            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
            chroma_std = _as_float(np.std(chroma, axis=1).mean(), 0.0)
            if chroma_std > 0.15:
                characteristics["vocal_separation"] = "lead_with_backup"
            elif chroma_std > 0.10:
                characteristics["vocal_separation"] = "harmonized_vocals"
            elif chroma_std > 0.05:
                characteristics["vocal_separation"] = "multiple_voices"
            else:
                characteristics["vocal_separation"] = "unknown"

            # Vocal style without piptrack (piptrack is O(frames*bins) and dominates runtime)
            if tempo > 110 and energy > 0.12:
                characteristics["vocal_style"] = "singing"
            elif tempo < 95 and energy < 0.1:
                characteristics["vocal_style"] = "spoken"
            else:
                characteristics["vocal_style"] = "melodic_speech"
            if avg_centroid > 400:
                characteristics["vocal_range"] = "high"
            elif avg_centroid > 200:
                characteristics["vocal_range"] = "medium"
            else:
                characteristics["vocal_range"] = "low"
        else:
            characteristics["vocal_count"] = "unknown"
            characteristics["vocal_separation"] = "unknown"

        if characteristics["has_vocals"] and characteristics["vocal_count"] == "unknown":
            characteristics["vocal_count"] = "solo"
            characteristics["vocal_separation"] = "single_voice"

        # Performance energy from existing RMS + a cheap onset envelope
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        combined_energy = (energy + _as_float(np.mean(onset_strength), 0.0)) / 2
        if combined_energy > 0.15:
            characteristics["performance_energy"] = "high_energy"
        elif combined_energy > 0.10:
            characteristics["performance_energy"] = "balanced"
        elif combined_energy > 0.05:
            characteristics["performance_energy"] = "subtle"
        else:
            characteristics["performance_energy"] = "minimal"

        centroid_mean = float(np.mean(spectral_centroids)) or 1.0
        rolloff_mean = float(np.mean(spectral_rolloff)) or 1.0
        centroid_variation = float(np.std(spectral_centroids)) / centroid_mean
        rolloff_variation = float(np.std(spectral_rolloff)) / rolloff_mean
        tempo_variation = 0.0
        if beat_diffs is not None and len(beat_diffs) > 1:
            mean_bi = float(np.mean(beat_diffs))
            if mean_bi > 0:
                tempo_variation = float(np.std(beat_diffs)) / mean_bi
        complexity_score = min(centroid_variation * 0.4 + rolloff_variation * 0.4 + tempo_variation * 0.2, 1.0)
        if complexity_score > 0.8:
            characteristics["musical_complexity"] = "complex"
        elif complexity_score > 0.5:
            characteristics["musical_complexity"] = "moderate"
        else:
            characteristics["musical_complexity"] = "simple"

        peak = float(np.max(np.abs(y))) or 1.0
        clipping_ratio = float(np.mean(np.abs(y) > 0.95))
        rms_mean = float(np.mean(rms)) or 1e-9
        rms_std = float(np.std(rms)) / rms_mean
        quality_score = 1.0
        if clipping_ratio > 0.001:
            quality_score -= 0.2
        if peak < 0.05:
            quality_score -= 0.3
        if rms_std > 0.5:
            quality_score -= 0.1
        quality_score = max(quality_score, 0.0)
        if quality_score > 0.8:
            characteristics["audio_quality"] = "excellent"
        elif quality_score > 0.6:
            characteristics["audio_quality"] = "good"
        elif quality_score > 0.4:
            characteristics["audio_quality"] = "fair"
        else:
            characteristics["audio_quality"] = "poor"

        if dynamic_range < 0.05:
            characteristics["dynamic_range"] = "narrow"
        elif dynamic_range < 0.2:
            characteristics["dynamic_range"] = "medium"
        else:
            characteristics["dynamic_range"] = "wide"

        # Syncopation: vectorized onset-vs-beat proximity (was an O(n*m) Python loop)
        if beat_count > 10:
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=sr, hop_length=hop)
            if len(onset_frames) > 0:
                onset_arr = onset_frames.astype(float)[:, None]
                beat_arr = beats.astype(float)[None, :]
                near_beat = np.any(np.abs(onset_arr - beat_arr) < 2.0, axis=1)
                syncopation_ratio = float(np.mean(~near_beat))
                if syncopation_ratio > 0.3:
                    characteristics["syncopation"] = "high"
                elif syncopation_ratio > 0.1:
                    characteristics["syncopation"] = "medium"
                else:
                    characteristics["syncopation"] = "low"

        if beat_count > 10:
            tempo_factor = min(tempo / 120.0, 2.0)
            characteristics["danceability"] = float(max(0.0, min(1.0, beat_consistency * 0.6 + tempo_factor * 0.4)))

        _classify_genre(characteristics, tempo, energy)
        _adjust_for_genre(characteristics)

        perc_mean = float(np.mean(np.abs(percussive))) or 1e-9
        reverb_indicator = float(np.mean(np.abs(harmonic))) / perc_mean
        if reverb_indicator > 2.0:
            characteristics["performance_type"] = "live_performance"
        elif reverb_indicator < 0.5:
            characteristics["performance_type"] = "studio_recording"
        else:
            characteristics["performance_type"] = "home_recording"

        if len(rms) > 10:
            try:
                energy_trend = float(np.polyfit(np.arange(len(rms)), rms, 1)[0])
                if energy_trend > 0.001:
                    characteristics["emotional_arc"] = "building"
                elif energy_trend < -0.001:
                    characteristics["emotional_arc"] = "declining"
                else:
                    characteristics["emotional_arc"] = "stable"
            except Exception:
                characteristics["emotional_arc"] = "stable"

        _finalize_mood_and_type(characteristics)
        characteristics["description"] = _build_description(characteristics)

        log_debug(
            f"Audio analysis done: genre={characteristics['genre']} "
            f"style={characteristics['vocal_style']} bpm={characteristics['tempo_bpm']:.1f}"
        )
        return _json_safe(characteristics)

    except Exception as e:
        log_debug(f"Error in audio analysis: {e}")
        log_debug(traceback.format_exc())
        return analyze_audio_characteristics(filename, 0)


def _classify_genre(characteristics: dict, tempo: float, energy: float) -> None:
    if characteristics["has_vocals"]:
        vocal_style = characteristics.get("vocal_style", "unknown")
        if vocal_style == "singing":
            if tempo > 120 and energy > 0.15:
                brightness = characteristics.get("spectral_characteristics", {}).get("brightness", 0)
                characteristics["genre"] = "metal" if energy > 0.25 and brightness > 3000 else "rock"
            elif tempo > 110 and characteristics["danceability"] > 0.8:
                if characteristics.get("syncopation") == "high" and characteristics.get("beat_strength") == "strong":
                    characteristics["genre"] = "rock" if characteristics.get("energy_level") in ("high", "very_high") else "hip_hop"
                else:
                    characteristics["genre"] = "pop"
            elif tempo < 90 and energy < 0.1:
                characteristics["genre"] = "ballad"
            elif 100 < tempo < 130 and characteristics.get("spectral_characteristics", {}).get("warmth"):
                characteristics["genre"] = "rnb"
            elif tempo > 90 and characteristics.get("spectral_characteristics", {}).get("brightness", 0) < 2000:
                characteristics["genre"] = "indie"
            else:
                characteristics["genre"] = "singer_songwriter"
        elif vocal_style == "spoken":
            if characteristics["beat_strength"] == "strong" and tempo > 90 and characteristics.get("syncopation") == "high":
                characteristics["genre"] = "hip_hop"
            elif characteristics["beat_strength"] == "strong" and tempo > 120:
                characteristics["genre"] = "rock"
            elif tempo < 100 and characteristics["mood"] == "calm":
                characteristics["genre"] = "folk"
            else:
                characteristics["genre"] = "spoken_word"
        elif vocal_style == "melodic_speech":
            if characteristics["beat_strength"] == "strong" and tempo > 100:
                characteristics["genre"] = "rock"
            elif tempo < 100 and characteristics["mood"] in ("calm", "contemplative"):
                characteristics["genre"] = "folk"
            elif characteristics["danceability"] > 0.8:
                characteristics["genre"] = "pop"
            else:
                characteristics["genre"] = "singer_songwriter"
        else:
            if tempo < 100 and characteristics["mood"] == "calm":
                characteristics["genre"] = "folk"
            elif characteristics["beat_strength"] == "strong" and tempo > 120:
                characteristics["genre"] = "rock"
            else:
                characteristics["genre"] = "unknown"
    else:
        brightness = characteristics.get("spectral_characteristics", {}).get("brightness", 0)
        if tempo > 130:
            characteristics["genre"] = "electronic"
        elif tempo < 80:
            characteristics["genre"] = "ambient"
        elif tempo > 100 and brightness > 2500:
            characteristics["genre"] = "classical"
        elif 80 < tempo < 120 and characteristics.get("beat_strength") == "strong":
            characteristics["genre"] = "jazz" if characteristics.get("syncopation") == "high" and characteristics.get("dynamic_range") == "wide" else "instrumental"
        else:
            characteristics["genre"] = "instrumental"


def _adjust_for_genre(characteristics: dict) -> None:
    genre = characteristics["genre"]
    if genre in ("rock", "metal", "pop", "hip_hop", "electronic"):
        characteristics["time_signature"] = "4/4"
    current_energy = characteristics["energy_level"]
    if genre in ("rock", "pop", "hip_hop"):
        if current_energy == "low":
            characteristics["energy_level"] = "medium"
        elif current_energy == "medium" and characteristics.get("beat_strength") == "strong":
            characteristics["energy_level"] = "high"
    elif genre in ("classical", "folk") and current_energy == "high":
        characteristics["energy_level"] = "medium"


def _finalize_mood_and_type(characteristics: dict) -> None:
    if characteristics["energy_level"] in ("high", "very_high") and characteristics["tempo"] in ("fast", "very_fast"):
        characteristics["mood"] = "energetic" if characteristics["has_vocals"] else "energetic_instrumental"
    elif characteristics["energy_level"] in ("low", "very_low") and characteristics["tempo"] in ("slow", "very_slow"):
        characteristics["mood"] = "contemplative" if characteristics.get("vocal_style") == "spoken" else "calm"
    elif characteristics["has_vocals"] and characteristics["energy_level"] == "medium":
        characteristics["mood"] = "emotional"
    elif characteristics["genre"] == "electronic":
        characteristics["mood"] = "futuristic"
    else:
        characteristics["mood"] = "neutral"

    if characteristics["has_vocals"]:
        if characteristics["vocal_style"] == "singing":
            characteristics["audio_type"] = "singing" if characteristics["tempo"] in ("medium", "fast", "very_fast") else "ballad"
        elif characteristics["vocal_style"] == "spoken":
            characteristics["audio_type"] = "speech"
        else:
            characteristics["audio_type"] = "melodic_speech"
    else:
        characteristics["audio_type"] = "instrumental_dance" if characteristics["danceability"] > 0.6 else "instrumental"


def _build_description(characteristics: dict) -> str:
    parts: list[str] = []
    vocal_style = characteristics.get("vocal_style")
    if vocal_style == "singing":
        parts.append(f"{vocal_style} performance with {characteristics['vocal_range']} vocal range")
    elif vocal_style == "spoken":
        parts.append("spoken dialogue with clear diction")
    elif vocal_style == "melodic_speech":
        parts.append("melodic speech with rhythmic delivery")
    else:
        parts.append("instrumental performance")

    bpm = characteristics.get("tempo_bpm") or 0
    if characteristics["beat_strength"] == "strong":
        parts.append(f"with strong {characteristics['tempo']} tempo ({bpm:.1f} BPM)")
    else:
        parts.append(f"with {characteristics['tempo']} tempo ({bpm:.1f} BPM)")

    if characteristics["time_signature"] != "4/4":
        parts.append(f"in {characteristics['time_signature']} time")

    if characteristics["danceability"] > 0.7:
        parts.append("highly syncopated danceable rhythm" if characteristics["syncopation"] == "high" else "strong danceable rhythm")
    elif characteristics["syncopation"] == "high":
        parts.append("complex syncopated rhythm")

    mood_descriptions = {
        "energetic": "creating high energy and excitement",
        "energetic_instrumental": "building instrumental energy",
        "calm": "establishing a peaceful, serene mood",
        "contemplative": "creating an introspective, thoughtful atmosphere",
        "emotional": "with emotional, expressive delivery",
        "futuristic": "with modern, innovative soundscapes",
        "neutral": "with balanced mood",
    }
    parts.append(mood_descriptions.get(characteristics["mood"], "with neutral mood"))

    if characteristics["emotional_arc"] == "building":
        parts.append("with intensity building throughout")
    elif characteristics["emotional_arc"] == "declining":
        parts.append("gradually calming down")

    if characteristics["performance_type"] == "live_performance":
        parts.append("captured in a live setting with natural ambiance")
    elif characteristics["performance_type"] == "studio_recording":
        parts.append("with polished studio production quality")

    genre_details = {
        "pop": "featuring catchy melodic hooks",
        "ballad": "with intimate, emotional delivery",
        "electronic": "with synthesized textures and electronic elements",
        "ambient": "creating atmospheric soundscapes",
        "instrumental": "showcasing musical instrumentation",
        "spoken_word": "with articulate vocal performance",
    }
    if characteristics["genre"] in genre_details:
        parts.append(genre_details[characteristics["genre"]])

    if characteristics["has_vocals"]:
        if characteristics["vocal_confidence"] > 0.8:
            parts.append("featuring prominent vocal performance")
        parts.append("requiring precise lip-sync synchronization")

    return " ".join(parts) + "."
