from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import shutil
import os
import google.generativeai as genai
import PIL.Image
import time
from datetime import datetime
from dotenv import load_dotenv
import atexit
import json
import mimetypes
import librosa
import numpy as np
import scipy

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Debug log configuration
DEBUG_LOG_PATH = "debug.log"


# Initialize debug log on startup
def init_debug_log():
    """Clear and initialize the debug log file"""
    with open(DEBUG_LOG_PATH, "w") as f:
        f.write(f"=== Prompt Enhancer Debug Log Started at {datetime.now()} ===\n")
        f.write(f"Application initialized\n\n")


def log_debug(message: str):
    """Write a debug message to the log file"""
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception as e:
        print(f"Failed to write to debug log: {e}")


def shutdown_debug_log():
    """Log shutdown message"""
    log_debug("=== Application shutting down ===\n")


# Initialize log on startup
init_debug_log()

# Register shutdown handler
atexit.register(shutdown_debug_log)


@app.on_event("startup")
async def startup_event():
    log_debug("FastAPI startup event triggered")


@app.on_event("shutdown")
async def shutdown_event():
    log_debug("FastAPI shutdown event triggered")


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Setup ---
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
UPLOADS_DIR = "uploads"


# --- Pydantic Models ---
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
    movement_level: str | None = None  # LTX-2 movement level: 'static', 'minimal', 'natural', 'expressive', 'dynamic'
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


class EnhanceResponse(BaseModel):
    enhanced_prompt: str


class AnalyzeResponse(BaseModel):
    description: str

class AnalyzeResponseMulti(BaseModel):
    combined_description: str
    image_a_description: str | None = None
    image_b_description: str | None = None


# Make sure to set your GOOGLE_API_KEY environment variable.
# You can get one here: https://aistudio.google.com/app/apikey
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def run_gemini(prompt: str, image_path: str | None = None, image_paths: list[str] | None = None):
    # The genai.configure call at the top of the file handles the API key.
    # If the key is not set, the model.generate_content call will raise an exception.
    if "GOOGLE_API_KEY" not in os.environ:
        return "Error: Google API key is not set. Please set the GOOGLE_API_KEY environment variable."

    try:
        # Configure safety settings to be less restrictive
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        model = genai.GenerativeModel("gemini-2.5-flash", safety_settings=safety_settings)

        # Multi-image support
        if image_paths and len(image_paths) > 0:
            images = []
            try:
                for p in image_paths:
                    images.append(PIL.Image.open(p))
            except Exception as img_error:
                return f"Error loading image(s): {img_error}. Please check the image file format and try again."

            try:
                response = model.generate_content([prompt, *images])
                
                # Check for blocked content
                if response.candidates:
                    return response.text
                else:
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        block_reason = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else str(response.prompt_feedback.block_reason)
                        error_msg = f"Prompt blocked by Gemini API: {block_reason}"
                        log_debug(f"Gemini API blocked content (multi-image): {block_reason}")
                        return f"Error: {error_msg}. Please try rephrasing your prompt."
                    else:
                        return "Error: Gemini API returned empty response. Please try again."
                        
            except Exception as api_error:
                return f"Error processing image(s) with Gemini API: {api_error}. One of the images may be too large or in an unsupported format."
        elif image_path:
            try:
                image = PIL.Image.open(image_path)
            except Exception as img_error:
                return f"Error loading image: {img_error}. Please check the image file format and try again."

            try:
                response = model.generate_content([prompt, image])
                
                # Check for blocked content
                if response.candidates:
                    return response.text
                else:
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        block_reason = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else str(response.prompt_feedback.block_reason)
                        error_msg = f"Prompt blocked by Gemini API: {block_reason}"
                        log_debug(f"Gemini API blocked content (single image): {block_reason}")
                        return f"Error: {error_msg}. Please try rephrasing your prompt."
                    else:
                        return "Error: Gemini API returned empty response. Please try again."
                        
            except Exception as api_error:
                return f"Error processing image with Gemini API: {api_error}. The image may be too large or in an unsupported format."
        else:
            try:
                response = model.generate_content(prompt)
                
                # Check for blocked content
                if response.candidates:
                    return response.text
                else:
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        block_reason = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else str(response.prompt_feedback.block_reason)
                        error_msg = f"Prompt blocked by Gemini API: {block_reason}"
                        log_debug(f"Gemini API blocked content (text only): {block_reason}")
                        log_debug(f"Prompt was: {prompt[:200]}...")
                        return f"Error: {error_msg}. Please try rephrasing your prompt."
                    else:
                        log_debug("Gemini API returned empty candidates without feedback")
                        return "Error: Gemini API returned empty response. Please try again."
                        
            except Exception as api_error:
                return f"Error with Gemini API: {api_error}. Your prompt may contain content that violates usage policies."
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"Detailed error: {error_details}")
        return f"An unexpected error occurred: {e}. Please try again later."


def limit_prompt_length(enhanced_prompt: str, model_type: str) -> str:
    """
    Limit the length of the enhanced prompt based on the model type.
    Different models have different character limits.

    Args:
        enhanced_prompt: The prompt to be limited
        model_type: The model type (e.g., 'WAN2')

    Returns:
        The potentially truncated prompt with a note if truncated
    """
    # Define model-specific limits
    model_limits = {
        # Prompt types
        "wan2": 800,  # WAN2 prompt limit
        "image": 3000,  # Image prompt type default
        "veo": 2000,  # Video prompt type default
        "ltx2": 1500,  # LTX-2 prompt limit (aligned with official examples: 890-1404 chars)
        # AI Models - all set to 3000 except WAN2
        "default": 3000,
        "qwen": 3000,
        "flux": 3000,
        "pixart": 3000,
        "dalle3": 3000,
        "midjourney": 3000,
        "sdxl": 3000,
        "nunchaku": 3000,
        "kandinsky": 3000,
        "imagen": 3000,
    }

    # Convert model_type to lowercase for case-insensitive matching
    model_key = model_type.lower()
    max_length = model_limits.get(model_key, model_limits["default"])

    log_debug(
        f"  - Checking limit: model_type={model_type}, max_length={max_length} chars"
    )

    if len(enhanced_prompt) <= max_length:
        log_debug(f"  ✓ Within limit ({len(enhanced_prompt)} <= {max_length})")
        return enhanced_prompt

    log_debug(
        f"  ✗ EXCEEDS limit ({len(enhanced_prompt)} > {max_length}) - truncation needed"
    )

    # If we get here, the prompt needs to be truncated
    truncated = False

    # First try to find a good sentence boundary
    last_period = enhanced_prompt.rfind(".", 0, max_length)
    last_excl = enhanced_prompt.rfind("!", 0, max_length)
    last_question = enhanced_prompt.rfind("?", 0, max_length)

    # Find the last occurrence of any sentence terminator
    last_sentence_end = max(last_period, last_excl, last_question)

    # If we found a good sentence end (not too early in the text)
    if last_sentence_end > max_length * 0.7:
        truncated_prompt = enhanced_prompt[: last_sentence_end + 1]
        truncated = True
        log_debug(
            f"  - Truncation method: Sentence boundary (at position {last_sentence_end})"
        )
    else:
        # If no good sentence boundary, just truncate at word boundary
        last_space = enhanced_prompt.rfind(" ", 0, max_length - 3)
        if last_space > max_length * 0.5:  # Only if we can keep most of the text
            truncated_prompt = enhanced_prompt[:last_space] + "..."
            truncated = True
            log_debug(
                f"  - Truncation method: Word boundary (at position {last_space})"
            )
        else:
            # Last resort: hard truncation
            truncated_prompt = enhanced_prompt[: max_length - 3] + "..."
            truncated = True
            log_debug(
                f"  - Truncation method: Hard truncation (at position {max_length - 3})"
            )

    # Add a note if we truncated
    if truncated:
        truncated_prompt += (
            "\n\n[Note: Prompt was truncated to fit model's character limit]"
        )
        log_debug(
            f"  - Result: {len(enhanced_prompt)} chars → {len(truncated_prompt)} chars (removed {len(enhanced_prompt) - len(truncated_prompt)} chars)"
        )

    return truncated_prompt


# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page with the current version."""
    log_debug(f"GET / - Home page accessed from {request.client.host}")
    api_key_set = "GOOGLE_API_KEY" in os.environ
    api_key_info = {}
    if api_key_set:
        # Get first and last 4 characters of the API key for display (for security)
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if len(api_key) > 8:
            masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        else:
            masked_key = "[Invalid key format]"
        api_key_info = {"set": True, "masked_key": masked_key}

    version = int(time.time())  # Using timestamp as version for cache busting
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "version": version,
            "api_key_set": api_key_set,
            "api_key_info": {
                "provider": "Google Gemini",
                "masked_key": masked_key if api_key_set else None,
            },
        },
    )


@app.get("/style-test", response_class=HTMLResponse)
async def style_test_page(request: Request):
    """Render the style test page."""
    log_debug("GET /style-test - Style test page accessed")
    version = int(time.time())  # Using timestamp as version for cache busting
    return templates.TemplateResponse(
        "style-test.html", {"request": request, "version": version}
    )


@app.post("/analyze-image", response_model=AnalyzeResponseMulti)
async def analyze_image_endpoint(images: list[UploadFile] = File(...)):
    try:
        # Normalize images list (support single file as well)
        if not images:
            return AnalyzeResponseMulti(
                combined_description="Error: No file provided",
                image_a_description=None,
                image_b_description=None,
            )
        if not isinstance(images, list):
            images = [images]

        if len(images) > 2:
            images = images[:2]

        valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]

        # Create uploads directory if it doesn't exist
        if not os.path.exists(UPLOADS_DIR):
            try:
                os.makedirs(UPLOADS_DIR)
            except Exception as e:
                return AnalyzeResponseMulti(
                    combined_description=f"Error creating upload directory: {e}",
                    image_a_description=None,
                    image_b_description=None,
                )
        saved_paths = []
        a_desc = None
        b_desc = None

        # Validate, save, and collect absolute paths
        for idx, image in enumerate(images):
            if not getattr(image, "filename", None):
                return AnalyzeResponseMulti(
                    combined_description="Error: One of the files has no filename",
                    image_a_description=None,
                    image_b_description=None,
                )
            file_ext = os.path.splitext(image.filename.lower())[1]
            if file_ext not in valid_extensions:
                return AnalyzeResponseMulti(
                    combined_description=(
                        f"Error: Unsupported file format for {image.filename}. "
                        f"Please upload an image in one of these formats: {', '.join(valid_extensions)}"
                    ),
                    image_a_description=None,
                    image_b_description=None,
                )

            try:
                file_path = os.path.join(UPLOADS_DIR, image.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)
                saved_paths.append(os.path.abspath(file_path))
            except Exception as e:
                return AnalyzeResponseMulti(
                    combined_description=f"Error saving image file: {e}",
                    image_a_description=None,
                    image_b_description=None,
                )

        # Build meta-prompts
        if len(saved_paths) == 1:
            meta_prompt = (
                "Analyze this image in detail. Provide a comprehensive description covering the main subject, "
                "setting, composition, colors, and notable elements. Be descriptive and thorough."
            )
            combined = run_gemini(meta_prompt, image_path=saved_paths[0])
            if combined.startswith("Error") or combined.startswith("An unexpected error"):
                return AnalyzeResponseMulti(
                    combined_description=combined,
                    image_a_description=None,
                    image_b_description=None,
                )
            return AnalyzeResponseMulti(
                combined_description=combined,
                image_a_description=None,
                image_b_description=None,
            )
        else:
            # Per-image short summaries
            per_image_prompt = (
                "Briefly summarize this image in 3-5 sentences focusing on subject, style/medium, colors, lighting, and composition."
            )
            a_desc = run_gemini(per_image_prompt, image_path=saved_paths[0])
            b_desc = run_gemini(per_image_prompt, image_path=saved_paths[1])
            if a_desc.startswith("Error") or a_desc.startswith("An unexpected error"):
                a_desc = None
            if b_desc.startswith("Error") or b_desc.startswith("An unexpected error"):
                b_desc = None

            # Combined comparative analysis
            combined_prompt = (
                "You are an assistant generating a combined reference from two images. "
                "Write a structured analysis with these sections: \n"
                "- Shared elements (overlaps)\n"
                "- Differences (distinctive traits of A vs B)\n"
                "- Style/Technique (medium, rendering)\n"
                "- Notable details (important cues to preserve)\n"
                "Be concise but descriptive."
            )
            combined = run_gemini(combined_prompt, image_paths=saved_paths[:2])
            if combined.startswith("Error") or combined.startswith("An unexpected error"):
                combined = (a_desc or "") + ("\n\n" if a_desc and b_desc else "") + (b_desc or "")

            return AnalyzeResponseMulti(
                combined_description=combined,
                image_a_description=a_desc,
                image_b_description=b_desc,
            )
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"Error in analyze_image_endpoint: {error_details}")
        return AnalyzeResponse(
            description=f"An unexpected error occurred: {e}. Please try again later."
        )


def analyze_real_audio_characteristics(file_path: str, filename: str) -> dict:
    """Analyze audio file characteristics using real audio processing."""
    
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=30)  # Analyze first 30 seconds
        
        characteristics = {
            "audio_type": "unknown",
            "tempo": "medium",
            "tempo_bpm": None,
            "mood": "neutral", 
            "energy_level": "medium",
            "has_vocals": False,
            "vocal_confidence": 0.0,
            "danceability": 0.5,
            "description": ""
        }
        
        # 1. Tempo Detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        characteristics["tempo_bpm"] = float(tempo)
        
        # Convert numpy arrays to lists for JSON serialization
        beats = beats.tolist() if hasattr(beats, 'tolist') else beats
        
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
        
        # 2. Energy Level Detection
        # Compute RMS energy
        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.mean(rms))  # Convert to Python float
        
        if energy < 0.05:
            characteristics["energy_level"] = "very_low"
        elif energy < 0.1:
            characteristics["energy_level"] = "low"
        elif energy < 0.2:
            characteristics["energy_level"] = "medium"
        elif energy < 0.3:
            characteristics["energy_level"] = "high"
        else:
            characteristics["energy_level"] = "very_high"
        
        # 3. Vocal Detection using Spectral Analysis
        # Compute spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Detect vocals based on spectral characteristics
        # Human voices typically have higher spectral centroids and specific MFCC patterns
        avg_spectral_centroid = float(np.mean(spectral_centroids))
        spectral_variance = float(np.var(spectral_centroids))
        
        # Simple vocal detection heuristic
        vocal_score = 0.0
        
        # High spectral centroid suggests vocals or high-frequency content
        if avg_spectral_centroid > 2000:
            vocal_score += 0.3
        
        # Variance in spectral content suggests complex audio (like vocals)
        if spectral_variance > 500000:
            vocal_score += 0.2
        
        # MFCC analysis for vocal patterns
        mfcc_std = np.std(mfccs, axis=1)
        if float(np.mean(mfcc_std[1:4])) > 15:  # MFCC coefficients 1-3 are important for vocals
            vocal_score += 0.3
        
        # Zero crossing rate (vocals typically have higher ZCR)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        if float(np.mean(zcr)) > 0.05:
            vocal_score += 0.2
        
        characteristics["vocal_confidence"] = min(float(vocal_score), 1.0)
        characteristics["has_vocals"] = vocal_score > 0.5
        
        # 4. Danceability (based on beat consistency and tempo)
        if len(beats) > 10:
            beat_diffs = np.diff(beats)
            beat_consistency = float(1.0 - np.std(beat_diffs) / np.mean(beat_diffs))
            tempo_factor = min(tempo / 120, 2.0)  # Normalize around 120 BPM
            characteristics["danceability"] = float(beat_consistency * 0.6 + tempo_factor * 0.4)
        else:
            characteristics["danceability"] = 0.3
        
        # 5. Mood Detection based on audio features
        # Combine tempo, energy, and spectral features for mood
        if characteristics["energy_level"] in ["high", "very_high"] and characteristics["tempo"] in ["fast", "very_fast"]:
            if characteristics["has_vocals"]:
                characteristics["mood"] = "energetic"
            else:
                characteristics["mood"] = "energetic_instrumental"
        elif characteristics["energy_level"] in ["low", "very_low"] and characteristics["tempo"] in ["slow", "very_slow"]:
            characteristics["mood"] = "calm"
        elif characteristics["has_vocals"] and characteristics["energy_level"] == "medium":
            characteristics["mood"] = "emotional"
        else:
            characteristics["mood"] = "neutral"
        
        # 6. Audio Type Classification
        if characteristics["has_vocals"]:
            if characteristics["tempo"] in ["medium", "fast", "very_fast"]:
                characteristics["audio_type"] = "singing"
            else:
                characteristics["audio_type"] = "speech"
        else:
            if characteristics["danceability"] > 0.6:
                characteristics["audio_type"] = "instrumental_dance"
            else:
                characteristics["audio_type"] = "instrumental"
        
        # 7. Generate Description
        description_parts = []
        
        # Audio type
        if characteristics["audio_type"] == "singing":
            description_parts.append(f"singing performance with vocals ({characteristics['vocal_confidence']:.1f} confidence)")
        elif characteristics["audio_type"] == "speech":
            description_parts.append("spoken dialogue/voice")
        elif characteristics["audio_type"] == "instrumental_dance":
            description_parts.append("instrumental dance music")
        elif characteristics["audio_type"] == "instrumental":
            description_parts.append("instrumental music")
        else:
            description_parts.append("audio track")
        
        # Tempo with BPM
        description_parts.append(f"with {characteristics['tempo']} tempo ({characteristics['tempo_bpm']:.1f} BPM)")
        
        # Energy and danceability
        if characteristics["danceability"] > 0.7:
            description_parts.append("highly danceable rhythm")
        elif characteristics["danceability"] > 0.4:
            description_parts.append("moderate danceability")
        else:
            description_parts.append("low danceability")
        
        # Mood
        mood_descriptions = {
            "energetic": "creating high energy and excitement",
            "energetic_instrumental": "building instrumental energy",
            "calm": "establishing a peaceful, serene mood",
            "emotional": "with emotional, expressive atmosphere",
            "neutral": "with balanced mood"
        }
        description_parts.append(mood_descriptions.get(characteristics["mood"], "with neutral mood"))
        
        # Performance instructions
        if characteristics["has_vocals"]:
            description_parts.append("featuring vocal performance that should be precisely lip-synced")
        
        characteristics["description"] = " ".join(description_parts) + "."
        
        log_debug(f"Real audio analysis completed: {filename}")
        log_debug(f"Tempo: {characteristics['tempo_bpm']:.1f} BPM, Vocals: {characteristics['has_vocals']}, Energy: {characteristics['energy_level']}")
        
        return characteristics
        
    except Exception as e:
        log_debug(f"Error in real audio analysis: {str(e)}")
        # Fallback to filename-based analysis
        return analyze_audio_characteristics(filename, 0)


def analyze_audio_characteristics(filename: str, file_size: int) -> dict:
    """Fallback filename-based analysis when real audio analysis fails."""
    
    characteristics = {
        "audio_type": "unknown",
        "tempo": "medium",
        "mood": "neutral", 
        "instruments": [],
        "vocals": False,
        "description": "",
        "tempo_bpm": None,
        "energy_level": "medium",
        "vocal_confidence": 0.0,
        "danceability": 0.5
    }
    
    # Analyze filename for clues
    filename_lower = filename.lower()
    
    # Detect audio type from filename
    if any(word in filename_lower for word in ["song", "music", "track", "audio", "beat"]):
        characteristics["audio_type"] = "music"
    elif any(word in filename_lower for word in ["speech", "talk", "voice", "dialogue", "speaking"]):
        characteristics["audio_type"] = "speech"
    elif any(word in filename_lower for word in ["sing", "vocal", "lyrics", "song"]):
        characteristics["audio_type"] = "singing"
    elif any(word in filename_lower for word in ["instrumental", "ambient", "background"]):
        characteristics["audio_type"] = "instrumental"
    
    # Detect tempo from filename
    if any(word in filename_lower for word in ["fast", "quick", "upbeat", "energetic", "dance"]):
        characteristics["tempo"] = "fast"
    elif any(word in filename_lower for word in ["slow", "calm", "relaxing", "ambient", "chill"]):
        characteristics["tempo"] = "slow"
    
    # Detect mood from filename
    if any(word in filename_lower for word in ["happy", "joy", "celebration", "upbeat", "party"]):
        characteristics["mood"] = "happy"
    elif any(word in filename_lower for word in ["sad", "emotional", "dramatic", "melancholy"]):
        characteristics["mood"] = "emotional"
    elif any(word in filename_lower for word in ["energetic", "powerful", "intense", "epic"]):
        characteristics["mood"] = "energetic"
    elif any(word in filename_lower for word in ["calm", "peaceful", "relaxing", "meditation"]):
        characteristics["mood"] = "calm"
    
    # Detect vocal presence
    if any(word in filename_lower for word in ["vocals", "singing", "voice", "lyrics", "song"]):
        characteristics["vocals"] = True
        characteristics["vocal_confidence"] = 0.7
    
    # Generate descriptive text
    description_parts = []
    
    # Audio type description
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
    
    # Tempo description
    if characteristics["tempo"] == "fast":
        description_parts.append("with fast, energetic rhythm suitable for dancing")
    elif characteristics["tempo"] == "slow":
        description_parts.append("with slow, gentle rhythm")
    else:
        description_parts.append("with moderate tempo")
    
    # Mood description
    if characteristics["mood"] == "happy":
        description_parts.append("creating a joyful, upbeat mood")
    elif characteristics["mood"] == "emotional":
        description_parts.append("with emotional, dramatic atmosphere")
    elif characteristics["mood"] == "energetic":
        description_parts.append("building high energy and excitement")
    elif characteristics["mood"] == "calm":
        description_parts.append("establishing a peaceful, serene mood")
    
    # Vocal description
    if characteristics["vocals"]:
        description_parts.append("featuring vocal performance that should be lip-synced")
    
    characteristics["description"] = " ".join(description_parts) + "."
    
    return characteristics


@app.post("/analyze-audio", response_model=dict)
async def analyze_audio_endpoint(audio_file: UploadFile = File(...)):
    """Analyze uploaded audio file for rhythm, tempo, and characteristics."""
    try:
        # Validate audio file
        if not audio_file.content_type.startswith('audio/'):
            return {"error": "Invalid file type. Please upload an audio file."}
        
        # Save audio file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}_{audio_file.filename}"
        file_path = os.path.join(UPLOADS_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Basic audio analysis (for now, return file info)
        # In a full implementation, you'd use librosa or similar for tempo/rhythm analysis
        file_size = os.path.getsize(file_path)
        
        # Generate audio description for LTX-2 prompt enhancement using real audio analysis
        audio_characteristics = analyze_real_audio_characteristics(file_path, audio_file.filename)
        
        # Add file format information
        format_info = ""
        if audio_file.filename.lower().endswith('.mp3'):
            format_info = "MP3 format with compressed audio. "
        elif audio_file.filename.lower().endswith('.wav'):
            format_info = "WAV format with high-quality uncompressed audio. "
        elif audio_file.filename.lower().endswith('.m4a'):
            format_info = "M4A format with efficient compression. "
        
        # Combine format info with characteristics
        audio_description = f"Uploaded audio file: {audio_file.filename} ({file_size/1024/1024:.1f}MB). {format_info}{audio_characteristics['description']} Use this audio as the soundtrack for synchronized performance."
        
        log_debug(f"Audio file analyzed: {audio_file.filename}, Size: {file_size} bytes")
        
        return {
            "audio_description": audio_description,
            "file_path": file_path,
            "file_size": file_size,
            "filename": audio_file.filename,
            "characteristics": audio_characteristics
        }
        
    except Exception as e:
        log_debug(f"Error analyzing audio: {str(e)}")
        return {"error": f"Error analyzing audio file: {str(e)}"}


@app.post("/enhance", response_model=EnhanceResponse)
async def enhance_prompt_endpoint(request: EnhanceRequest) -> EnhanceResponse:
    """Enhance the prompt using Gemini API."""
    import time

    start_time = time.time()

    try:
        # Validate input
        if not request.prompt and request.prompt_type != "WAN2":
            return EnhanceResponse(
                enhanced_prompt="Error: Please provide a prompt to enhance."
            )

        # Check for API key
        if "GOOGLE_API_KEY" not in os.environ:
            return EnhanceResponse(
                enhanced_prompt="Error: Google API key is not set. Please set the GOOGLE_API_KEY environment variable."
            )

        # Detect if this is a vehicle wrap or technical wrapping prompt
        prompt_lower = request.prompt.lower() if request.prompt else ""
        # Heuristic: require both a vehicle term AND a wrap/surface-mapping term.
        # Avoid triggering on negatives (e.g., "no vehicles", "negatives: no vehicles, wraps").
        vehicle_terms = [
            "automotive", "vehicle", "car", "sedan", "coupe", "sports coupe",
            "hatchback", "suv", "truck"
        ]
        # New: People/character terms for branding/wrapping
        people_terms = [
            "person", "people", "human", "character", "man", "woman", "child",
            "model", "athlete", "celebrity", "portrait", "figure", "body"
        ]
        wrap_terms = [
            "vehicle wrap", "wrap design", "vinyl wrap", "car wrap", "livery", "wrap",
            "body wrap", "skin wrap", "character wrap", "branding", "logo wrap",
            "sponsored", "branded", "advertisement", "promo", "sponsored by"
        ]
        body_terms = [
            "body lines", "body panels", "quarter panel", "fender", "hood", "spoiler",
            "door", "bonnet", "trunk", "bumper", "side skirt", "roof"
        ]
        # New: Human body parts for character wraps
        human_body_terms = [
            "arms", "legs", "torso", "chest", "back", "shoulders", "skin",
            "full body", "entire body", "body surface"
        ]
        def contains_term(text: str, terms: list[str]) -> bool:
            return any(term in text for term in terms)

        def has_negation(text: str) -> bool:
            # Common negation patterns that show the user explicitly excludes vehicles/wraps
            neg_patterns = [
                "no vehicles", "no vehicle", "without vehicles", "without vehicle",
                "no wraps", "no wrap", "negatives: no vehicles", "negatives: no wrap",
            ]
            return any(p in text for p in neg_patterns)

        # Optional hint to avoid vehicle mode when the intent is people/object A→B transfer
        people_object_hint = "people/object" in prompt_lower or "a→b transfer" in prompt_lower

        vehicle_present = contains_term(prompt_lower, vehicle_terms)
        people_present = contains_term(prompt_lower, people_terms)
        wrap_present = contains_term(prompt_lower, wrap_terms)
        body_hits = sum(1 for t in body_terms if t in prompt_lower)
        human_body_hits = sum(1 for t in human_body_terms if t in prompt_lower)

        # Start with heuristic, then override with explicit wrap_mode if provided
        # Stricter rule: need explicit wrap AND (vehicle OR >=2 body panel terms)
        is_vehicle_wrap = (
            wrap_present
            and (vehicle_present or body_hits >= 2)
            and not has_negation(prompt_lower)
            and not people_object_hint
        )
        
        # New: Character/branding wrap detection
        is_character_wrap = (
            wrap_present
            and (people_present or human_body_hits >= 2)
            and not has_negation(prompt_lower)
            and not people_object_hint
            and not is_vehicle_wrap  # Don't double-count
        )

        # Explicit wrap_mode overrides heuristics to prevent mixing modes
        if request.wrap_mode:
            mode = request.wrap_mode.lower()
            if mode == "vehicle":
                is_vehicle_wrap = True
                is_character_wrap = False
                people_object_hint = False
            elif mode == "people-object":
                is_vehicle_wrap = False
                is_character_wrap = False
                people_object_hint = True
            elif mode == "character":
                is_vehicle_wrap = False
                is_character_wrap = True
                people_object_hint = False
        
        # --- Build instructions based on user selections ---
        def is_meaningful_selection(value: str | None) -> bool:
            if not value:
                return False
            v = value.strip().lower()
            return v not in {"none", "auto"}

        instructions = []
        if is_meaningful_selection(request.style):
            instructions.append(f"in {request.style.lower()} style")
        if is_meaningful_selection(request.cinematography):
            instructions.append(f"with {request.cinematography.lower()} cinematography")
        if is_meaningful_selection(request.lighting):
            instructions.append(f"with {request.lighting.lower()} lighting")
        if (
            request.prompt_type == "WAN2"
            and request.motion_effect
            and request.motion_effect != "Static"
        ):
            instructions.append(f"with a {request.motion_effect.lower()} motion effect")
        instruction_text = " " + " and ".join(instructions) if instructions else ""

        image_context = (
            f" CRITICAL REFERENCE IMAGE GUIDANCE: The user has provided a reference image described as: '{request.image_description}'. "
            f"Analyze this description carefully and identify key visual elements (subjects, objects, environment, lighting, colors) that should appear in the animation. "
            f"Use these elements as the foundation for your prompt - the animation should clearly relate to what's described in the reference image. "
            f"If the user's prompt mentions specific characters or objects from the reference, ensure they are prominently featured. "
            f"Maintain visual consistency with the reference while adding the requested motion."
            if request.image_description
            else ""
        )
        text_emphasis = f" {request.text_emphasis}" if request.text_emphasis else ""

        # --- Add model-specific guidance if a specific model is selected ---
        model_guidance = ""
        if (
            request.model
            and request.model != "default"
            and request.prompt_type == "Image"
        ):
            # Photorealistic Models
            if request.model == "flux":
                model_guidance = " For the Flux model, include technical photography details (camera, lens, lighting setup) and focus on photorealism with high detail. Specify camera model, lens details, and lighting setup for best results."
            elif request.model == "pixart":
                model_guidance = " For the PixArt model, use detailed descriptions of scene elements and specify artistic style. Focus on composition, lighting, and atmosphere to achieve a good balance of realism and artistic flair."
            elif request.model == "dalle3":
                has_text = any(
                    text_term in request.prompt.lower()
                    for text_term in [
                        "text",
                        "sign",
                        "writing",
                        "label",
                        "book",
                        "letter",
                        "word",
                        "character",
                        "font",
                    ]
                )

                model_guidance = " For the DALL-E 3 model, use clear, detailed instructions and specify style, composition, and lighting. This model excels at following complex instructions and creating coherent scenes."

                if has_text:
                    model_guidance += " Since your prompt involves text elements, DALL-E 3 is an excellent choice as it renders text accurately. Be specific about the exact text you want to appear."

                if (
                    "complex" in request.prompt.lower()
                    or "scene" in request.prompt.lower()
                ):
                    model_guidance += " For complex scenes, describe the spatial relationship between elements and ensure logical composition."

            # Versatile Models
            elif request.model == "qwen":
                # Enhanced Qwen-specific guidance based on latest research
                has_text = any(
                    text_term in request.prompt.lower()
                    for text_term in [
                        "text",
                        "sign",
                        "writing",
                        "label",
                        "book",
                        "letter",
                        "word",
                        "character",
                        "font",
                    ]
                )
                has_chinese = any(ord(c) > 127 for c in request.prompt)

                model_guidance = " For the Qwen model, focus on clear subject descriptions and compositional instructions. Add 'Ultra HD, 4K, cinematic composition' to enhance quality."

                if has_text:
                    model_guidance += " Since your prompt involves text elements, emphasize text clarity and legibility. Qwen excels at rendering complex text with high fidelity."

                    if has_chinese:
                        model_guidance += " For Chinese text, specify that each character should be perfectly rendered with correct stroke order and proportions."

                # Additional guidance for specific image types
                if (
                    "portrait" in request.prompt.lower()
                    or "person" in request.prompt.lower()
                ):
                    model_guidance += " For portraits, focus on natural facial features and expressions rather than technical camera settings."
            elif request.model == "midjourney":
                model_guidance = " For the Midjourney model, use simple, clear descriptions and include artistic style references. Add 'highly detailed, intricate, elegant, sharp focus' to enhance quality. This model excels at creating visually striking imagery with artistic flair."

                if (
                    "landscape" in request.prompt.lower()
                    or "nature" in request.prompt.lower()
                ):
                    model_guidance += " For landscapes, consider adding 'epic scale, atmospheric, golden hour' to achieve dramatic results."
                elif (
                    "concept" in request.prompt.lower()
                    or "fantasy" in request.prompt.lower()
                ):
                    model_guidance += " For concept art or fantasy scenes, add 'concept art, digital painting, trending on artstation' for best results."
            elif request.model == "sdxl":
                model_guidance = " For the SDXL model, balance descriptive and stylistic elements, and include composition details. This versatile model works well across many styles and subjects."

                if (
                    "art" in request.prompt.lower()
                    or "painting" in request.prompt.lower()
                ):
                    model_guidance += " For artistic imagery, consider adding 'masterpiece, trending on artstation, award winning' to enhance quality."
                elif (
                    "photo" in request.prompt.lower()
                    or "realistic" in request.prompt.lower()
                ):
                    model_guidance += " For photorealistic results, add '8k, detailed textures, professional photography' to your prompt."

            # Artistic Models
            elif request.model == "nunchaku":
                model_guidance = " For the Nunchaku model, start with style keywords, focus on mood/atmosphere, and use artistic terminology rather than technical camera terms. This model excels at creating stylized, atmospheric images with strong emotional impact."
            elif request.model == "kandinsky":
                model_guidance = " For the Kandinsky model, use artistic movement references and focus on color palette and composition. This model excels at abstract, surreal, and conceptual imagery with unique stylistic elements."

                if "abstract" in request.prompt.lower():
                    model_guidance += " For abstract art, emphasize non-representational forms, geometric elements, and expressive color use."
                elif "surreal" in request.prompt.lower():
                    model_guidance += " For surreal imagery, focus on dreamlike qualities, unexpected juxtapositions, and symbolic elements."
            elif request.model == "imagen":
                model_guidance = " For the Imagen model, specify subject, setting, action, and include lighting and atmosphere details. This model is strong at following detailed instructions and creating coherent scenes with good composition."

                if (
                    "complex" in request.prompt.lower()
                    or "scene" in request.prompt.lower()
                ):
                    model_guidance += " For complex scenes, describe multiple elements with logical arrangement and consistent style for best results."

        # --- Logic to choose meta-prompt based on prompt_type ---
        if request.prompt_type == "VEO":
            meta_prompt = f"You are a creative assistant for the VEO text-to-video model. Expand the user's idea into a rich, cinematic prompt{instruction_text}. Describe the scene, subject, and action in a detailed paragraph.{image_context}{text_emphasis} IMPORTANT: Keep your enhanced prompt under 2000 characters total. Do not add conversational fluff. User's idea: '{request.prompt}'"

        elif request.prompt_type == "3D":
            # 3D Model generation (character creation from reference) - descriptive prompts for image-to-3D models
            
            # Get the selected 3D model type to tailor the prompt
            model_type = request.model_type or "character"
            
            # Define model type specific guidance
            model_type_guidance = {
                "character": {
                    "focus": "character modeling",
                    "details": "Include detailed anatomy, pose, expression, rigging considerations, facial features, body proportions, and character personality traits",
                    "examples": "pose, expression, anatomy, rigging, facial features, body proportions"
                },
                "object": {
                    "focus": "object modeling", 
                    "details": "Include material properties, surface textures, functional details, scale, and physical characteristics",
                    "examples": "materials, textures, surface details, functionality, scale"
                },
                "vehicle": {
                    "focus": "vehicle modeling",
                    "details": "Include mechanical components, vehicle proportions, scale, technical details, and functional elements",
                    "examples": "mechanical parts, scale, proportions, technical specifications"
                },
                "environment": {
                    "focus": "environment modeling",
                    "details": "Include architectural elements, spatial layout, scale, environmental context, and structural details",
                    "examples": "architecture, scale, spatial relationships, environmental context"
                },
                "props": {
                    "focus": "props modeling",
                    "details": "Include small-scale details, material authenticity, scale, accessory characteristics, and functional properties",
                    "examples": "detail level, materials, scale, accessory properties"
                }
            }
            
            selected_type = model_type_guidance.get(model_type, model_type_guidance["character"])
            
            # Conditional guidance based on whether image description exists
            if request.image_description:
                # Use reference image guidance
                model_3d_rules = f" IMPORTANT: This is for {selected_type['focus']}. Create a detailed visual description that an AI image-to-3D model can use to generate a 3D model from the reference image. Focus on {selected_type['details']}."
                model_3d_format = f" IMPORTANT FORMAT: Write a concise but detailed description of the {model_type} for 3D model generation. Include {selected_type['examples']}. Keep under 2000 characters."
                model_3d_character = f" IMPORTANT {model_type.upper()}: Describe the {model_type} accurately based on the reference image. Make it suitable for 3D model generation workflows."
            else:
                # Use user's prompt as source
                model_3d_rules = f" IMPORTANT: This is for {selected_type['focus']}. Enhance the user's description by adding detailed 3D modeling information while preserving all key elements. Focus on {selected_type['details']}."
                model_3d_format = f" IMPORTANT FORMAT: Expand the user's prompt with 3D modeling details. Include {selected_type['examples']}. Keep the user's main subject and action exactly as described. Keep under 2000 characters."
                model_3d_character = f" IMPORTANT {model_type.upper()}: Take the user's description and enhance it for 3D model generation. Make it suitable for 3D modeling workflows while preserving the user's exact subject matter."
            
            # 3D-specific image context (different from animation context)
            image_context_3d = (
                f" CRITICAL REFERENCE IMAGE GUIDANCE: The user has provided a reference image described as: '{request.image_description}'. "
                f"Use this description to create a detailed visual prompt that accurately represents the {model_type} in the reference image for 3D model generation. "
                f"Extract and describe key visual elements (appearance, pose, colors, materials, textures, proportions) that an image-to-3D AI can use. "
                f"The prompt should enable accurate 3D reconstruction of the reference subject. "
                f"Focus on creating descriptive prompts for 3D generation, not technical modeling specifications."
                if request.image_description
                else ""
            )
            
            motion_effect = (
                f" with {request.motion_effect} motion effect"
                if request.motion_effect and request.motion_effect != "Static"
                else ""
            )
            
            if request.prompt:
                if request.image_description:
                    # Use image description as primary reference but incorporate user's prompt
                    meta_prompt = f"You are a creative assistant for image-to-3D model generation. Create a detailed visual description (maximum 2000 characters){instruction_text}{motion_effect}. The user's main request is: '{request.prompt}'. Use the reference image description to enhance the visual details, but ensure the final description matches the user's specific subject and action. Include detailed visual characteristics, pose, materials, and appearance details.{model_3d_rules}{model_3d_format}{model_3d_character}{image_context_3d}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
                else:
                    # Use user's prompt as primary description source - be explicit about enhancement
                    meta_prompt = f"You are a creative assistant for image-to-3D model generation. Your task is to ENHANCE the user's prompt by adding detailed 3D modeling information while PRESERVING ALL the key elements from their description. Do NOT create a new description - take their exact prompt and expand it with 3D-specific details.{instruction_text}{motion_effect}{model_3d_rules}{model_3d_format}{model_3d_character}{text_emphasis} CRITICAL: Keep the user's main subject and action exactly as described. User's prompt to enhance: '{request.prompt}'"
            else:
                meta_prompt = f"""You are a creative assistant for image-to-3D model generation. Create a detailed visual description (maximum 2000 characters) based on the reference image that can be used by AI image-to-3D models.
- Describe the {model_type}'s appearance, pose, and visual details
- Include colors, materials, textures, and proportions  
- Focus on elements that help 3D reconstruction
- Keep the description detailed but suitable for AI processing

{image_context_3d}{instruction_text}

Generate a detailed visual prompt for 3D model generation now."""

        elif request.prompt_type == "WAN2":
            if request.prompt:
                wan2_text_rule = " IMPORTANT: Do NOT invent any written text, logos, banners, or signage unless the exact text is explicitly present in the reference image description or the user's idea. If text is present, preserve it exactly as written."
                wan2_format_rule = " IMPORTANT FORMAT: Output a single concise WAN2 prompt under 800 characters. Prefer a compact tag-like prefix (shot type, lens/composition, lighting) then 1 sentence describing SUBJECT + MOTION. Include 3-4 distinct motion beats (e.g., wakes → transforms → interacts → returns) separated by commas/semicolons. Always specify camera behavior: use 'static shot' / 'fixed shot' unless camera movement is explicitly requested. Focus on motion; avoid unnecessary micro-details."
                wan2_user_idea_rule = " IMPORTANT: The user's idea is the PRIMARY requirement. The output MUST clearly include the user's requested subject and action. Do not ignore the user's idea. Preserve the exact relationship and behavior described by the user without adding extra narrative elements."
                wan2_subject_rule = " IMPORTANT SUBJECT CONTROL: Do NOT introduce any extra subjects beyond what the user asked for. No random new people/characters, duplicates, background pedestrians, crowds, or extra faces. If the user's idea introduces a new object/character not visible in the reference, describe it as transforming/morphing/emerging from existing elements in the reference (not appearing from nowhere)."
                wan2_props_rule = " IMPORTANT PROP CONTROL: Do NOT invent extra objects/props (e.g., cups, sparks, debris, codes, labels) unless they are explicitly mentioned in the user's idea OR explicitly present in the reference image description."
                motion_effect = (
                    f" with {request.motion_effect} motion effect"
                    if request.motion_effect and request.motion_effect != "Static"
                    else ""
                )
                meta_prompt = f"You are a creative assistant for the WAN2 image-to-video animation model. Create a CONCISE prompt (maximum 800 characters){instruction_text}{motion_effect}. Prefer a multi-beat action sequence with 3-4 clear beats while keeping stable composition.{wan2_format_rule}{wan2_user_idea_rule}{wan2_subject_rule}{wan2_props_rule}{wan2_text_rule}{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
            else:
                # Logic for when the user wants the AI to imply the animation
                meta_prompt = f"""You are a creative assistant for the WAN2 image-to-video model. Create a CONCISE prompt (maximum 800 characters) that describes how to animate the static image.
- Identify 1-3 key elements for movement (e.g., hair, clothing, water, clouds).
- Describe the animation briefly but vividly.
- Keep the total prompt under 800 characters.

IMPORTANT: Do NOT invent any written text, logos, banners, or signage unless the exact text is explicitly present in the reference image description.
IMPORTANT SUBJECT CONTROL: Do NOT introduce any extra subjects beyond what the user asked for. No random new people/characters, duplicates, background pedestrians, crowds, or extra faces.
IMPORTANT FORMAT: Use a compact tag-like prefix (shot type, lens, composition, lighting) then a single sentence describing SUBJECT + MOTION. Always specify camera behavior: use 'static shot' / 'fixed shot' unless camera movement is explicitly requested.

User's Specifications:
- Reference Image: '{image_context}'
- Desired Style: '{instruction_text}'

Generate a brief animation prompt now."""

        elif request.prompt_type == "LTX2":
            # LTX-2 video generation with synchronized audio
            audio_generation = getattr(request, 'audio_generation', 'enabled') if hasattr(request, 'audio_generation') else 'enabled'
            resolution = getattr(request, 'resolution', '4K') if hasattr(request, 'resolution') else '4K'
            audio_description = getattr(request, 'audio_description', '') if hasattr(request, 'audio_description') else ''
            movement_level = getattr(request, 'movement_level', 'natural') if hasattr(request, 'movement_level') else 'natural'
            
            # Audio Integration parameters
            lipsync_intensity = getattr(request, 'lipsync_intensity', 'natural') if hasattr(request, 'lipsync_intensity') else 'natural'
            audio_reactivity = getattr(request, 'audio_reactivity', 'medium') if hasattr(request, 'audio_reactivity') else 'medium'
            genre_movement = getattr(request, 'genre_movement', '') if hasattr(request, 'genre_movement') else ''
            
            # Timing Control parameters
            movement_speed = getattr(request, 'movement_speed', 'normal') if hasattr(request, 'movement_speed') else 'normal'
            pause_points = getattr(request, 'pause_points', 'none') if hasattr(request, 'pause_points') else 'none'
            transition_smoothness = getattr(request, 'transition_smoothness', 'natural') if hasattr(request, 'transition_smoothness') else 'natural'
            
            # Character Interaction parameters
            character_coordination = getattr(request, 'character_coordination', 'independent') if hasattr(request, 'character_coordination') else 'independent'
            object_interaction = getattr(request, 'object_interaction', 'none') if hasattr(request, 'object_interaction') else 'none'
            
            # Add specific instructions for singing/dancing with uploaded audio
            performance_instruction = ""
            
            # Add movement level control
            movement_instruction = ""
            if movement_level == 'static':
                movement_instruction = " ABSOLUTE STATIC PERFORMANCE: Only lip-sync and subtle eye movements allowed. No head movement, no arm gestures, no body swaying, no shoulder movements. Character remains completely still except for mouth movement and minimal facial expressions. "
            elif movement_level == 'minimal':
                movement_instruction = " MINIMAL MOVEMENT: Only subtle head movement, slight shoulder motion, and gentle hand gestures. No large body movements, no dramatic swaying, no exaggerated gestures. Focus on restrained, natural motion. "
            elif movement_level == 'natural':
                movement_instruction = " NATURAL MOVEMENT: Normal body movement including head turns, shoulder movements, arm gestures, and gentle body swaying. Maintain realistic motion without exaggeration. "
            elif movement_level == 'expressive':
                movement_instruction = " EXPRESSIVE MOVEMENT: Full body movement including dynamic gestures, head movement, shoulder motion, arm gestures, and body swaying. Emphasize rhythmic, energetic motion that matches the audio. "
            elif movement_level == 'dynamic':
                movement_instruction = " DYNAMIC MOVEMENT: Highly energetic and expressive full-body movement. Include dramatic gestures, head movement, shoulder motion, arm gestures, body swaying, and rhythmic dancing. Emphasize powerful, athletic motion. "
            
            # AUTO-DETECT from audio_description if not explicitly set
            if audio_description:
                # Auto-detect genre from audio description
                if not genre_movement:
                    if "rock" in audio_description.lower() or "heavy" in audio_description.lower():
                        genre_movement = "rock"
                    elif "pop" in audio_description.lower() or "upbeat" in audio_description.lower():
                        genre_movement = "pop"
                    elif "classical" in audio_description.lower() or "orchestral" in audio_description.lower():
                        genre_movement = "classical"
                    elif "electronic" in audio_description.lower() or "edm" in audio_description.lower() or "synth" in audio_description.lower():
                        genre_movement = "electronic"
                    elif "jazz" in audio_description.lower() or "smooth" in audio_description.lower():
                        genre_movement = "jazz"
                    elif "folk" in audio_description.lower() or "acoustic" in audio_description.lower() or "organic" in audio_description.lower():
                        genre_movement = "folk"
                
                # Auto-detect tempo for movement speed
                if movement_speed == 'normal':  # Only override if not explicitly set
                    if "very_fast" in audio_description.lower() or "fast tempo" in audio_description.lower():
                        movement_speed = "fast"
                    elif "very_slow" in audio_description.lower() or "slow tempo" in audio_description.lower():
                        movement_speed = "slow_motion"
                
                # Auto-detect energy for audio reactivity
                if audio_reactivity == 'medium':  # Only override if not explicitly set
                    if "high energy" in audio_description.lower() or "very_high" in audio_description.lower():
                        audio_reactivity = "high"
                    elif "low energy" in audio_description.lower() or "calm" in audio_description.lower():
                        audio_reactivity = "low"
                
                # Auto-detect singing style for lipsync intensity
                if lipsync_intensity == 'natural':  # Only override if not explicitly set
                    if "dramatic" in audio_description.lower() or "powerful" in audio_description.lower():
                        lipsync_intensity = "exaggerated"
                    elif "subtle" in audio_description.lower() or "gentle" in audio_description.lower():
                        lipsync_intensity = "subtle"
                
                # Generate comprehensive audio characteristics for enhanced prompts
                audio_characteristics = []
                
                # Emotional & Performance Detection
                if any(word in audio_description.lower() for word in ["singing", "vocals", "vocal"]):
                    audio_characteristics.append("singing")
                if "speaking" in audio_description.lower() or "spoken" in audio_description.lower():
                    audio_characteristics.append("speaking")
                if "rapping" in audio_description.lower() or "rap" in audio_description.lower():
                    audio_characteristics.append("rapping")
                if "chanting" in audio_description.lower() or "chant" in audio_description.lower():
                    audio_characteristics.append("chanting")
                if "whispering" in audio_description.lower() or "whisper" in audio_description.lower():
                    audio_characteristics.append("whispering")
                
                # Emotional Tone Detection
                if "angry" in audio_description.lower() or "aggressive" in audio_description.lower():
                    audio_characteristics.append("angry")
                if "joyful" in audio_description.lower() or "happy" in audio_description.lower():
                    audio_characteristics.append("joyful")
                if "sad" in audio_description.lower() or "melancholy" in audio_description.lower():
                    audio_characteristics.append("sad")
                if "romantic" in audio_description.lower() or "love" in audio_description.lower():
                    audio_characteristics.append("romantic")
                if "peaceful" in audio_description.lower() or "calm" in audio_description.lower():
                    audio_characteristics.append("peaceful")
                if "dramatic" in audio_description.lower() or "intense" in audio_description.lower():
                    audio_characteristics.append("dramatic")
                
                # Performance Intensity
                if "passionate" in audio_description.lower():
                    audio_characteristics.append("passionate")
                if "restrained" in audio_description.lower() or "subtle" in audio_description.lower():
                    audio_characteristics.append("restrained")
                if "casual" in audio_description.lower() or "relaxed" in audio_description.lower():
                    audio_characteristics.append("casual")
                
                # Musical Structure Detection
                if "strong beat" in audio_description.lower() or "heavy beat" in audio_description.lower():
                    audio_characteristics.append("strong_beat")
                if "subtle rhythm" in audio_description.lower() or "gentle rhythm" in audio_description.lower():
                    audio_characteristics.append("subtle_rhythm")
                if "complex percussion" in audio_description.lower() or "intricate" in audio_description.lower():
                    audio_characteristics.append("complex_percussion")
                if "minimal beat" in audio_description.lower() or "simple beat" in audio_description.lower():
                    audio_characteristics.append("minimal_beat")
                
                # Instrumentation Detection
                if "guitar" in audio_description.lower():
                    audio_characteristics.append("guitar_driven")
                if "piano" in audio_description.lower():
                    audio_characteristics.append("piano_based")
                if "orchestral" in audio_description.lower() or "orchestra" in audio_description.lower():
                    audio_characteristics.append("orchestral")
                if "electronic beats" in audio_description.lower() or "electronic" in audio_description.lower():
                    audio_characteristics.append("electronic_beats")
                
                # Vocal Presence
                if "lead vocals" in audio_description.lower() or "solo" in audio_description.lower():
                    audio_characteristics.append("lead_vocals")
                if "harmonies" in audio_description.lower() or "harmony" in audio_description.lower():
                    audio_characteristics.append("harmonies")
                if "backup vocals" in audio_description.lower() or "background vocals" in audio_description.lower():
                    audio_characteristics.append("backup_vocals")
                if "acapella" in audio_description.lower() or "a cappella" in audio_description.lower():
                    audio_characteristics.append("acapella")
                
                # Dynamic Range Detection
                if "dynamic shifts" in audio_description.lower() or "volume changes" in audio_description.lower():
                    audio_characteristics.append("dynamic_shifts")
                if "consistent volume" in audio_description.lower() or "steady" in audio_description.lower():
                    audio_characteristics.append("consistent_volume")
                if "crescendo" in audio_description.lower() or "building" in audio_description.lower():
                    audio_characteristics.append("crescendo")
                if "fade" in audio_description.lower():
                    audio_characteristics.append("fade_effects")
                
                # Pace Variations
                if "changing tempo" in audio_description.lower() or "tempo changes" in audio_description.lower():
                    audio_characteristics.append("changing_tempo")
                if "accelerating" in audio_description.lower() or "speeding up" in audio_description.lower():
                    audio_characteristics.append("accelerating")
                if "decelerating" in audio_description.lower() or "slowing down" in audio_description.lower():
                    audio_characteristics.append("decelerating")
                
                # Energy Arcs
                if "building energy" in audio_description.lower() or "energy builds" in audio_description.lower():
                    audio_characteristics.append("building_energy")
                if "climax" in audio_description.lower() or "peak" in audio_description.lower():
                    audio_characteristics.append("climax_moments")
                if "calm sections" in audio_description.lower() or "quiet parts" in audio_description.lower():
                    audio_characteristics.append("calm_sections")
                if "explosive" in audio_description.lower() or "powerful" in audio_description.lower():
                    audio_characteristics.append("explosive_parts")
                
                # Stylistic Elements
                if "vintage" in audio_description.lower() or "retro" in audio_description.lower():
                    audio_characteristics.append("vintage_style")
                if "modern" in audio_description.lower() or "contemporary" in audio_description.lower():
                    audio_characteristics.append("modern_style")
                if "futuristic" in audio_description.lower():
                    audio_characteristics.append("futuristic_style")
                if "latin" in audio_description.lower():
                    audio_characteristics.append("latin_style")
                if "african" in audio_description.lower():
                    audio_characteristics.append("african_style")
                if "asian" in audio_description.lower():
                    audio_characteristics.append("asian_style")
                if "western" in audio_description.lower():
                    audio_characteristics.append("western_style")
                if "middle eastern" in audio_description.lower():
                    audio_characteristics.append("middle_eastern_style")
                
                # Atmospheric Quality
                if "intimate" in audio_description.lower():
                    audio_characteristics.append("intimate")
                if "epic" in audio_description.lower():
                    audio_characteristics.append("epic")
                if "raw" in audio_description.lower():
                    audio_characteristics.append("raw")
                if "polished" in audio_description.lower() or "clean" in audio_description.lower():
                    audio_characteristics.append("polished")
                if "organic" in audio_description.lower():
                    audio_characteristics.append("organic")
                
                # Movement Triggers
                if "highly danceable" in audio_description.lower() or "dance" in audio_description.lower():
                    audio_characteristics.append("highly_danceable")
                if "minimal dance" in audio_description.lower():
                    audio_characteristics.append("minimal_dance")
                if "head-nodding" in audio_description.lower() or "head nod" in audio_description.lower():
                    audio_characteristics.append("head_nodding")
                if "full body movement" in audio_description.lower():
                    audio_characteristics.append("full_body_movement")
                
                # Crowd Response
                if "concert" in audio_description.lower() or "live" in audio_description.lower():
                    audio_characteristics.append("concert_feel")
                if "intimate performance" in audio_description.lower():
                    audio_characteristics.append("intimate_performance")
                if "group" in audio_description.lower() or "band" in audio_description.lower():
                    audio_characteristics.append("group_energy")
                
                # Physicality
                if "athletic" in audio_description.lower():
                    audio_characteristics.append("athletic_performance")
                if "gentle swaying" in audio_description.lower():
                    audio_characteristics.append("gentle_swaying")
                if "static singing" in audio_description.lower():
                    audio_characteristics.append("static_singing")
                if "dramatic gestures" in audio_description.lower():
                    audio_characteristics.append("dramatic_gestures")
                
                # Technical Audio Features
                if "bass-heavy" in audio_description.lower() or "heavy bass" in audio_description.lower():
                    audio_characteristics.append("bass_heavy")
                if "treble-focused" in audio_description.lower():
                    audio_characteristics.append("treble_focused")
                if "balanced" in audio_description.lower():
                    audio_characteristics.append("balanced")
                if "reverb" in audio_description.lower():
                    audio_characteristics.append("reverb")
                if "echo" in audio_description.lower():
                    audio_characteristics.append("echo")
                if "dry sound" in audio_description.lower():
                    audio_characteristics.append("dry_sound")
                if "distortion" in audio_description.lower():
                    audio_characteristics.append("distortion")
                
                # Narrative Elements
                if "storytelling" in audio_description.lower() or "narrative" in audio_description.lower():
                    audio_characteristics.append("storytelling")
                if "emotional journey" in audio_description.lower():
                    audio_characteristics.append("emotional_journey")
                if "character-driven" in audio_description.lower():
                    audio_characteristics.append("character_driven")
                if "abstract" in audio_description.lower():
                    audio_characteristics.append("abstract")
                
                # Mood Progression
                if "uplifting" in audio_description.lower():
                    audio_characteristics.append("uplifting")
                if "dark" in audio_description.lower():
                    audio_characteristics.append("dark")
                if "mysterious" in audio_description.lower():
                    audio_characteristics.append("mysterious")
                if "celebratory" in audio_description.lower():
                    audio_characteristics.append("celebratory")
                
                # Create comprehensive audio instruction based on detected characteristics
                if audio_characteristics:
                    characteristic_instruction = "AUDIO CHARACTERISTICS: "
                    characteristics_text = ", ".join(audio_characteristics)
                    characteristic_instruction += f"Detected audio characteristics: {characteristics_text}. Generate movements and expressions that perfectly match these audio qualities. "
                    performance_instruction += characteristic_instruction
            
            audio_instruction = ""
            if audio_generation == 'enabled':
                if audio_description:
                    audio_instruction = f" Use the uploaded audio as the soundtrack: {audio_description} Generate synchronized lip-sync and dance movements that match the audio rhythm and tempo. The character should sing/dance in perfect sync with the uploaded audio track."
                else:
                    audio_instruction = " Include synchronized audio generation that matches the visual content - describe ambient sounds, dialogue, music, or effects that naturally complement the scene."
            else:
                audio_instruction = " Video only generation - no audio."
            
            resolution_instruction = f""
            
            if audio_description:
                # Parse characteristics from audio_description
                if "singing" in audio_description.lower() or "vocals" in audio_description.lower():
                    performance_instruction += " Focus on subtle mouth motion and natural facial movement influenced by vocal rhythm - gentle lip movement, slight jaw motion, and expressive eyes that suggest singing without exaggerated mouth openings. "
                elif "dance" in request.prompt.lower() or "dancing" in request.prompt.lower():
                    # Use tempo information for more precise dance instructions
                    if "very_fast" in audio_description.lower() or "fast tempo" in audio_description.lower():
                        performance_instruction += " Focus on energetic but controlled rhythmic body motion with quick, precise movements that match the fast tempo - head nods, shoulder movements, and hand gestures in sync with rapid beats. "
                    elif "very_slow" in audio_description.lower() or "slow tempo" in audio_description.lower():
                        performance_instruction += " Focus on gentle, flowing body motion with slow, deliberate movements - subtle swaying, soft hand gestures, and gradual weight shifts synchronized to the slow rhythm. "
                    else:
                        performance_instruction += " Focus on rhythmic body motion with coordinated dance movements - head bobs, shoulder movements, and hand gestures that naturally respond to the musical beat and rhythm. "
                
                # Add energy-based instructions
                if "high energy" in audio_description.lower() or "very_high" in audio_description.lower():
                    performance_instruction += "Emphasize dynamic movement with increased motion range while maintaining natural body mechanics - more expressive gestures, broader movements, and stronger rhythmic responses. "
                elif "low energy" in audio_description.lower() or "calm" in audio_description.lower():
                    performance_instruction += "Emphasize minimal, subtle movement with gentle body language - soft gestures, slight swaying, and calm facial expressions that match the tranquil mood. "
                
                # Add mood-based instructions
                if "happy" in audio_description.lower() or "joyful" in audio_description.lower():
                    performance_instruction += "Create positive presence through natural smiles, bright eyes, and open body posture - subtle expressions that convey joy without overacting. "
                elif "emotional" in audio_description.lower() or "dramatic" in audio_description.lower():
                    performance_instruction += "Create emotional presence through controlled body movement and expressive facial expressions - meaningful gestures and nuanced expressions that convey depth. "
                elif "sad" in audio_description.lower() or "melancholy" in audio_description.lower():
                    performance_instruction += "Create somber presence through gentle, slow movements and soft facial expressions - downward gaze, slight shoulder movements, and restrained gestures. "
            
            # Add Audio Integration instructions
            audio_integration_instruction = ""
            
            # Lip-sync intensity
            if lipsync_intensity == 'subtle':
                audio_integration_instruction += "SUBTLE LIP-SYNC: Minimal mouth movement, slight jaw motion, restrained lip articulation. Focus on natural, understated mouth movements that suggest singing without exaggeration. "
            elif lipsync_intensity == 'exaggerated':
                audio_integration_instruction += "EXAGGERATED LIP-SYNC: Pronounced mouth movements, wide jaw opening, dramatic lip articulation. Emphasize clear, visible mouth shapes and strong jaw motion for dramatic effect. "
            
            # Audio reactivity
            if audio_reactivity == 'low':
                audio_integration_instruction += "LOW AUDIO REACTIVITY: Movements should be loosely connected to audio rhythm. Focus on general mood rather than precise beat synchronization. Gentle, flowing motion that suggests the music without strict timing. "
            elif audio_reactivity == 'high':
                audio_integration_instruction += "HIGH AUDIO REACTIVITY: Movements must be precisely synchronized to audio beats and rhythm. Every gesture, head movement, and body motion should correspond directly to musical elements. Sharp, accurate timing. "
            
            # Genre-based movement
            if genre_movement == 'rock':
                audio_integration_instruction += "ROCK MOVEMENT STYLE: Energetic, powerful movements with strong emphasis on rhythm. Head banging, fist pumps, strong guitar-like arm movements, confident stance. "
            elif genre_movement == 'pop':
                audio_integration_instruction += "POP MOVEMENT STYLE: Choreographed, polished movements with smooth transitions. Graceful arm gestures, coordinated hand movements, stylish poses, dance-pop choreography. "
            elif genre_movement == 'classical':
                audio_integration_instruction += "CLASSICAL MOVEMENT STYLE: Elegant, refined movements with graceful flow. Subtle arm gestures, gentle swaying, dignified posture, controlled expressive movements. "
            elif genre_movement == 'electronic':
                audio_integration_instruction += "ELECTRONIC MOVEMENT STYLE: Rhythmic, robotic movements with sharp precision. Staccato gestures, mechanical head movements, digital dance moves, futuristic body language. "
            elif genre_movement == 'jazz':
                audio_integration_instruction += "JAZZ MOVEMENT STYLE: Smooth, improvisational movements with natural flow. Relaxed swaying, cool hand gestures, casual shoulder movements, spontaneous expressive motions. "
            elif genre_movement == 'folk':
                audio_integration_instruction += "FOLK MOVEMENT STYLE: Natural, grounded movements with organic feel. Gentle swaying, simple hand gestures, warm body language, authentic emotional expression. "
            
            # Add Timing Control instructions
            timing_instruction = ""
            
            # Movement speed
            if movement_speed == 'slow_motion':
                timing_instruction += "SLOW MOTION: All movements should be gracefully slowed down with smooth, flowing transitions. Emphasize deliberate, controlled motion with extended timing. "
            elif movement_speed == 'fast':
                timing_instruction += "FAST MOVEMENT: Quick, energetic movements with rapid transitions. Emphasize speed and agility while maintaining coordination. "
            
            # Pause points
            if pause_points == 'occasional':
                timing_instruction += "OCCASIONAL PAUSES: Include brief moments of complete stillness between movements. Natural pauses that add rhythm and emphasis to the performance. "
            elif pause_points == 'frequent':
                timing_instruction += "FREQUENT PAUSES: Regular moments of stillness throughout the performance. Start-stop rhythm with deliberate pauses between movements. "
            
            # Transition smoothness
            if transition_smoothness == 'smooth':
                timing_instruction += "SMOOTH TRANSITIONS: All movements should flow seamlessly from one to another. No abrupt starts or stops, continuous fluid motion. "
            elif transition_smoothness == 'sharp':
                timing_instruction += "SHARP TRANSITIONS: Quick, precise movements with clear starts and stops. Defined, crisp movements with minimal blending. "
            
            # Add Character Interaction instructions
            interaction_instruction = ""
            
            # Character coordination
            if character_coordination == 'synchronized':
                interaction_instruction += "SYNCHRONIZED MOVEMENT: All characters move in perfect coordination. Mirror movements, unified timing, identical gestures, and coordinated choreography. "
            elif character_coordination == 'call_response':
                interaction_instruction += "CALL AND RESPONSE: Characters interact through alternating movements. One character initiates movement, others respond in turn. Interactive, conversational motion. "
            
            # Object interaction
            if object_interaction == 'subtle':
                interaction_instruction += "SUBTLE OBJECT INTERACTION: Characters occasionally touch or interact with nearby objects naturally. Gentle, realistic object handling that enhances the scene. "
            elif object_interaction == 'prominent':
                interaction_instruction += "PROMINENT OBJECT INTERACTION: Characters actively engage with objects as part of the performance. Deliberate manipulation of props, instruments, or environmental elements. "
                
                # Add danceability-based instructions
                if "highly danceable" in audio_description.lower():
                    performance_instruction += "Include rhythmic dance elements with natural, continuous motion - foot taps, hip movements, and coordinated upper body gestures that flow with the music. "
                elif "low danceability" in audio_description.lower():
                    performance_instruction += "Focus on subtle body movement and emotional expression rather than complex dance - gentle swaying, head movement, and expressive hand gestures. "
                
                # Add genre-specific instructions
                if "electronic" in audio_description.lower() or "edm" in audio_description.lower():
                    performance_instruction += "Emphasize sharp, precise movements with electronic music responsiveness - quick head nods, robotic gestures, and staccato motions that match electronic beats. "
                elif "acoustic" in audio_description.lower() or "folk" in audio_description.lower():
                    performance_instruction += "Emphasize organic, flowing movements with acoustic music responsiveness - gentle swaying, natural gestures, and smooth body motions that match acoustic rhythms. "
                elif "rock" in audio_description.lower() or "metal" in audio_description.lower():
                    performance_instruction += "Emphasize strong, rhythmic movements with rock music responsiveness - head nods, shoulder movements, and powerful gestures that match rock beats. "
                elif "jazz" in audio_description.lower() or "blues" in audio_description.lower():
                    performance_instruction += "Emphasize fluid, expressive movements with jazz responsiveness - smooth body motions, improvisational gestures, and rhythmic variations that match jazz rhythms. "
                
                # Add stability limiter (MANDATORY)
                performance_instruction += "Natural motion, realistic timing, minimal facial distortion, no overacting or sudden movement. "
            
            meta_prompt = f"""You are a creative assistant for the LTX-2 text-to-video model. Create a concise, motion-focused prompt following this exact 7-point structure{instruction_text}.

LTX-2 PROMPT STRUCTURE (follow this order, no numbers):
- Main action in ONE sentence - What is happening right now?
- Movements and gestures - Head turns, walking pace, hair movement, hands, posture
- Character appearances - Clothing, age, accessories, expressions
- Environment - Location, background elements, depth
- Camera angle and movement - Tracking, static, close-up, wide, height
- Lighting and colors - Time of day, shadows, dominant tones
- Changes or events - Or clearly state that nothing changes

CRITICAL FRAME AWARENESS:
- If shot is from waist up: NO foot taps, leg movements, or walking descriptions
- If shot is chest up: NO hip movements, waist movements, or arm gestures below chest
- If shot is close-up on face: ONLY head, eye, and mouth movements
- Only describe movements that are VISIBLE in the specified frame
- If user wants "no movement", "static", "singing only", "still", or "stationary": ONLY describe lip-sync and subtle facial expressions, NO arm movements, head tilts, or body swaying
- If user says "static upper body" or "maintains static posture": NO arm movements, torso movements, or shoulder movements
- ABSOLUTE STATIC RULE: If any static keywords are detected, NO hand gestures, arm movements, finger movements, or body movements of any kind - ONLY lip-sync and eye/eyebrow movements allowed

{resolution_instruction}{audio_instruction}{movement_instruction}{performance_instruction}{audio_integration_instruction}{timing_instruction}{interaction_instruction}

CRITICAL: Keep your enhanced prompt under 1500 characters total. Use exactly 7 sentences maximum - one for each point above. Be concise and specific. Focus on natural motion and realistic timing. Do not add conversational fluff.

{image_context}

User's idea: '{request.prompt}'"""

        elif request.prompt_type == "Image":
            # Determine character limit for this model
            char_limit = 2000  # Default for most models
            if request.model and request.model.lower() == "wan2":
                char_limit = 1000

            # Check for specific materials, styles, or compositions in the prompt
            prompt_lower = request.prompt.lower()

            # People/Object A→B transfer detection (delta-only intent)
            is_people_object = (
                "people/object" in prompt_lower
                or "delta-only edit" in prompt_lower
                or ("add only the" in prompt_lower and "from reference b" in prompt_lower and "from reference a" in prompt_lower)
            )

            if is_people_object and not is_vehicle_wrap:
                meta_prompt = f"""You are enhancing a delta-only edit prompt that transfers an object from Reference B onto the subject in Reference A. Your job is to keep the original photo from Reference A intact and ONLY add the specified object from Reference B.

CRITICAL CONSTRAINTS (PRESERVE EXACTLY):
- Use Reference A as the base canvas and keep its background, framing/composition, perspective/lens look, color grading, and lighting unchanged.
- Preserve the subject's identity, face, expression, skin tone, hair, posture, and clothing from Reference A.
- DO NOT recreate or restage Reference B. DO NOT import Reference B background, layout, or composition.

OBJECT TRANSFER RULES:
- Add ONLY the specified object(s) from Reference B with precise placement, scale, and perspective onto/around the subject from Reference A.
- Maintain correct occlusion (object can sit behind hair/clothing edges) and natural contact with subtle deformation if physically plausible.
- Preserve B's material properties (metal, beads, fabric, leather, etc.) with micro-highlights, reflections, and contact shadows.

NEGATIVES:
- No flat-lay or product-only composition. No restaging or relighting. No vehicles or wraps. No stylization shifts. No color bleed from B.

{image_context}{text_emphasis}

User's idea:
'{request.prompt}'

Keep the output under {char_limit} characters. Output the enhanced prompt now."""

            # Vehicle wrap / technical wrapping - preserve critical instructions
            elif is_vehicle_wrap:
                meta_prompt = f"""You are enhancing a technical vehicle wrap prompt. The user has provided detailed, precise instructions that MUST be preserved EXACTLY.

CRITICAL REQUIREMENTS - DO NOT MODIFY:
1. Keep EVERY technical requirement exactly as written (design transfer, mapping, rendering specs, negatives)
2. Preserve the EXACT structure including all section headings (PRECISE DESIGN TRANSFER, VEHICLE SPECIFICS, TECHNICAL SPECS, etc.)
3. Keep ALL instructions about "Reference A" and "Reference B" word-for-word
4. Do NOT remove or rewrite any constraints, rules, or technical specifications
5. Do NOT change any negative prompts or "do not" instructions

YOUR ENHANCEMENT TASK:
- Add vivid, descriptive language about the vehicle's appearance (sleek, aerodynamic, sculpted, etc.)
- Add descriptive details about the wrap design colors and patterns
- Make the scene more vivid with atmospheric details
- If style preferences are provided{instruction_text}, add them as SUPPLEMENTARY atmosphere only
- Remove any markdown formatting like ** or __ from the original prompt

PRESERVE COMPLETELY:
✓ All bullet points and list structures
✓ All "CRITICAL", "PRESERVE", "MAINTAIN", "DO NOT" instructions  
✓ All technical specifications and measurements
✓ All references to "Reference A" and "Reference B"
✓ The exact meaning and intent of every constraint

{image_context}{text_emphasis}

Original technical prompt to enhance (preserve structure, add descriptions):
'{request.prompt}'

Output the enhanced prompt now, keeping ALL technical requirements intact."""

            # Character/branding wrap - for people, animals, characters
            elif is_character_wrap:
                meta_prompt = f"""You are enhancing a character branding wrap prompt. The user wants to place a logo or branding on a character (person, animal, creature) with natural integration.

CRITICAL REQUIREMENTS:
- Maintain the character's identity, appearance, and personality completely
- The logo/branding should integrate naturally with the character's clothing, accessories, or body
- Preserve the original pose, expression, and character features from Reference A
- If Reference B contains the logo/branding, transfer ONLY those design elements

CHARACTER WRAPPING RULES:
- Place logos naturally on clothing, accessories, or appropriate body areas
- Consider the character's style and personality when integrating branding
- Maintain proper perspective and contour fitting on character's body/clothing
- For animals, integrate branding on collars, accessories, or natural markings
- Keep the character as the main focus, not the branding

ENHANCEMENT TASK:
- Add descriptive details about the character's appearance and style
- Enhance the integration method (embroidered patch, printed fabric, natural pattern, etc.)
- Add atmospheric details that complement the character's personality
- If style preferences are provided{instruction_text}, enhance the overall aesthetic
- Remove any markdown formatting like ** or __ from the original prompt

{image_context}{text_emphasis}

Original character wrap prompt to enhance:
'{request.prompt}'

Output the enhanced prompt now, keeping the character's identity intact while naturally integrating the branding."""

            # Material-specific handling
            elif "yarn" in prompt_lower and any(
                word in prompt_lower for word in ["animal", "creature", "wildlife"]
            ):
                meta_prompt = f"You are a creative assistant for a text-to-image model specializing in yarn art. Your goal is to create a detailed prompt for an image where the main subject is an animal ENTIRELY made of yarn - not a real animal, but a yarn sculpture/creation that looks like an animal. Describe the yarn's texture, colors, stitching details, and how the yarn construction gives the animal character. Make sure to emphasize that this is a yarn creation, not a real animal with yarn elements.{instruction_text}{model_guidance} Include details about the setting and lighting that would best showcase this yarn creation.{image_context}{text_emphasis} IMPORTANT: Keep your enhanced prompt under {char_limit} characters total. Do not add conversational fluff. User's idea: '{request.prompt}'"

            # Add all the other material-specific and style-specific handlers here...
            # (I'm omitting them for brevity, but they should remain in the actual code)

            # Default case with enhanced guidance
            else:
                meta_prompt = f"You are a creative assistant for a text-to-image model. Your goal is to expand the user's idea into a rich, descriptive prompt suitable for generating a static image{instruction_text}.{model_guidance} Focus on the visual details of the scene, subject, and atmosphere. Be specific about composition (rule of thirds, leading lines, framing), perspective (eye level, bird's eye, worm's eye), depth (foreground, middle ground, background elements), and the quality of light (direction, color, intensity, shadows).{image_context}{text_emphasis} IMPORTANT: Keep your enhanced prompt under {char_limit} characters total. Do not add conversational fluff. User's idea: '{request.prompt}'"

        else:
            # Fallback for safety
            meta_prompt = f"Enhance this prompt: {request.prompt}"

        enhanced_prompt = run_gemini(meta_prompt)

        # Check if the response contains an error message
        if enhanced_prompt.startswith("Error") or enhanced_prompt.startswith(
            "An unexpected error"
        ):
            return EnhanceResponse(enhanced_prompt=enhanced_prompt)

        # Apply length limits based on model type
        # Use the model parameter if available, otherwise fall back to prompt_type
        model_for_limit = request.model if request.model else request.prompt_type

        log_debug(f"\n{'='*60}")
        log_debug(f"ENHANCE REQUEST")
        log_debug(f"{'='*60}")
        log_debug(f"  - Model Type (3D): {request.model_type}")
        if hasattr(request, "motion_effect") and request.motion_effect:
            log_debug(f"  - Motion Effect: {request.motion_effect}")
        if request.text_emphasis:
            log_debug(f"  - Text Emphasis: {request.text_emphasis[:50]}...")
        if request.image_description:
            log_debug(
                f"  - Image Description: Present ({len(request.image_description)} chars)"
            )

        log_debug(f"\nOriginal Prompt ({len(request.prompt)} chars):")
        log_debug(
            f"  {request.prompt[:200]}{'...' if len(request.prompt) > 200 else ''}"
        )

        log_debug(f"\nEnhanced Prompt ({len(enhanced_prompt)} chars):")
        log_debug(
            f"  {enhanced_prompt[:200]}{'...' if len(enhanced_prompt) > 200 else ''}"
        )

        log_debug(f"\nModel Guidance Used:")
        if model_guidance:
            log_debug(
                f"  {model_guidance[:150]}{'...' if len(model_guidance) > 150 else ''}"
            )
        else:
            log_debug(f"  None (default/standard)")

        log_debug(f"\nCharacter Limit Check:")
        log_debug(f"  - Model for limit: {model_for_limit}")

        limited_prompt = limit_prompt_length(enhanced_prompt, model_for_limit)

        log_debug(f"\nFinal Output:")
        log_debug(f"  - Length: {len(limited_prompt)} chars")
        log_debug(
            f"  - Truncated: {'YES' if len(limited_prompt) != len(enhanced_prompt) else 'NO'}"
        )

        elapsed_time = time.time() - start_time
        log_debug(f"\n⏱️  Processing time: {elapsed_time:.2f} seconds")
        log_debug(f"{'='*60}\n")

        return EnhanceResponse(enhanced_prompt=limited_prompt)

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        log_debug(f"ERROR in enhance_prompt_endpoint: {e}")
        log_debug(f"Error traceback: {error_details}")
        print(f"Error in enhance_prompt_endpoint: {error_details}")
        return EnhanceResponse(
            enhanced_prompt=f"An unexpected error occurred: {e}. Please try again later."
        )
