"""
Pydantic request/response models for the Prompt Enhancer API.

Extracted from main.py to reduce monolith size and improve testability.
"""

from pydantic import BaseModel, field_validator
from app.style_constants import is_valid_style
import logging

logger = logging.getLogger(__name__)


class EnhanceRequest(BaseModel):
    prompt: str
    prompt_type: str  # VEO or WAN2 or Image or 3D or LTX2
    style: str
    cinematography: str
    lighting: str
    image_description: str | None = None
    motion_effect: str | None = None
    text_emphasis: str | None = None
    model: str | None = None  # AI model selection (flux, qwen, nunchaku, etc.)
    model_type: str | None = None  # 3D model type (character, object, vehicle, environment, props)
    wrap_mode: str | None = None  # 'vehicle' | 'people-object' | 'none'
    audio_generation: str | None = None  # LTX-2 audio generation: 'enabled' or 'disabled'
    resolution: str | None = None  # LTX-2 resolution: '4K', '1080p', '720p'
    audio_description: str | None = None  # Description of uploaded audio file
    audio_characteristics: dict | None = None  # Structured audio analysis data from /analyze-audio
    movement_level: str | None = None  # LTX-2 movement level: 'static', 'minimal', 'natural', 'expressive', 'dynamic'
    ltx2_style: str | None = None  # LTX-2 video style: 'music_video', 'cinematic', 'artistic', etc.

    @field_validator('style')
    @classmethod
    def validate_style(cls, v):
        if not isinstance(v, str):
            raise ValueError('Style must be a string')

        # Sanitize input
        v = v.strip()

        # Check for potentially dangerous content
        dangerous_patterns = ['<script', 'javascript:', 'data:', 'vbscript:', 'onload=', 'onerror=']
        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError('Style contains invalid content')

        # Length validation
        if len(v) > 100:
            raise ValueError('Style name too long (max 100 characters)')

        # Validate against known styles (allow empty/auto values)
        if v_lower not in ("", "none", "auto", "automatic"):
            if not is_valid_style(v):
                logger.warning(f"Unknown style '{v}' provided - will use fallback handling")

        return v

    # Audio Integration
    lipsync_intensity: str | None = None  # 'subtle', 'natural', 'exaggerated'
    audio_reactivity: str | None = None  # 'low', 'medium', 'high'
    genre_movement: str | None = None  # 'rock', 'pop', 'classical', 'electronic', 'jazz', 'folk'
    # Timing Control
    movement_speed: str | None = None  # 'slow_motion', 'normal', 'fast'
    pause_points: str | None = None  # 'none', 'occasional', 'frequent'
    transition_smoothness: str | None = None  # 'smooth', 'natural', 'sharp'
    # Character Interaction
    character_coordination: str | None = None  # 'independent', 'synchronized', 'call_response'
    object_interaction: str | None = None  # 'none', 'subtle', 'prominent'
    gemini_model: str | None = None  # override Gemini model for this request
    include_quality_scoring: bool = True  # set False to skip quality scoring (saves a Gemini call)


class EnhanceResponse(BaseModel):
    enhanced_prompt: str
    negative_prompt: str | None = None
    quality_scores: dict[str, float] | None = None
    top_improvements: list[str] | None = None


class AnalyzeResponse(BaseModel):
    description: str


class AnalyzeResponseMulti(BaseModel):
    combined_description: str
    image_a_description: str | None = None
    image_b_description: str | None = None


class SpecializedEnhanceRequest(BaseModel):
    prompt: str
    enhancement_mode: str  # 'commercial', 'cinematic', 'character', 'object', 'ace-step', 'auto'
    image_description: str | None = None
    audio_characteristics: dict | None = None
    prompt_type: str | None = None
    model: str | None = None
    image_analysis: dict | None = None  # For auto-detection
    style: str | None = None
    lighting: str | None = None
    cinematography: str | None = None
    ltx2_style: str | None = None  # LTX-2 video style
    gemini_model: str | None = None  # override Gemini model for this request
    include_quality_scoring: bool = True  # set False to skip quality scoring (saves a Gemini call)
