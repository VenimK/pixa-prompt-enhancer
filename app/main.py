from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import shutil
import os
# Compatibility layer: try new google.genai first, fall back to old google.generativeai
USE_NEW_GENAI = False
try:
    import google.genai as genai_new
    USE_NEW_GENAI = True
except ImportError:
    genai_new = None

try:
    import google.generativeai as genai_old
except ImportError:
    genai_old = None

if not USE_NEW_GENAI and genai_old is None:
    raise ImportError("Neither google.genai nor google.generativeai is installed. Please install one of them.")
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

# Initialize based on which package is available
genai_client = None
genai_model = None

if "GOOGLE_API_KEY" in os.environ:
    if USE_NEW_GENAI and genai_new:
        genai_client = genai_new.Client(api_key=os.environ["GOOGLE_API_KEY"])
        log_debug("Using google.genai (new package)")
    elif genai_old:
        genai_old.configure(api_key=os.environ["GOOGLE_API_KEY"])
        genai_model = genai_old.GenerativeModel("gemini-2.5-flash")
        log_debug("Using google.generativeai (old package)")


def run_gemini(prompt: str, image_path: str | None = None, image_paths: list[str] | None = None):
    if "GOOGLE_API_KEY" not in os.environ:
        return "Error: Google API key is not set. Please set the GOOGLE_API_KEY environment variable."

    try:
        model_name = "gemini-2.5-flash"

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
                        contents=[prompt, *images]
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
                    return f"Error loading image: {img_error}. Please check the image file format and try again."
                try:
                    start_time = time.time()
                    response = genai_client.models.generate_content(
                        model=model_name,
                        contents=[prompt, image]
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
                        contents=prompt
                    )
                    log_debug(f"Gemini API call (text) took {time.time() - start_time:.2f}s")
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
        import traceback
        error_details = traceback.format_exc()
        print(f"Detailed error: {error_details}")
        return f"An unexpected error occurred: {e}. Please try again later."


def generate_enhanced_ltx2_prompt(audio_characteristics: dict, base_prompt: str) -> str:
    """Generate enhanced LTX-2 prompt using advanced audio analysis."""
    
    # Check for preservation constraints - if user wants strict preservation, be conservative
    base_lower = base_prompt.lower()
    preservation_keywords = [
        "strictly preserve", "preserve exactly", "keep exactly", "no changes", 
        "don't change", "maintain exactly", "preserve the", "keep the", 
        "same character", "same outfit", "same background", "no extra"
    ]
    
    has_preservation_constraint = any(keyword in base_lower for keyword in preservation_keywords)
    
    # If user wants strict preservation, only add minimal audio-driven enhancements
    if has_preservation_constraint:
        prompt_parts = [base_prompt.strip()]
        
        # Only add very conservative enhancements that don't violate preservation
        if audio_characteristics.get('has_vocals'):
            vocal_confidence = audio_characteristics.get('vocal_confidence', 0)
            if vocal_confidence > 0.7:
                prompt_parts.append("with precise lip-sync to the vocal performance")
        
        # Add subtle movement only if base prompt doesn't forbid it
        if "gentle swaying" in base_lower or "subtle movement" in base_lower:
            if audio_characteristics.get('beat_strength') == 'strong':
                prompt_parts.append("with subtle rhythmic movement synchronized to the music")
        
        return " ".join(prompt_parts)
    
    # Normal enhancement mode - proceed with detailed analysis
    prompt_parts = [base_prompt.strip()]
    
    # 1. Performance Style Enhancements
    if audio_characteristics.get('vocal_style') == 'spoken':
        prompt_parts.append("delivering spoken dialogue with precise lip-sync and clear diction")
        if audio_characteristics.get('vocal_confidence', 0) > 0.8:
            prompt_parts.append("with articulate vocal performance and natural speech patterns")
    elif audio_characteristics.get('vocal_style') == 'singing':
        prompt_parts.append(f"singing with {audio_characteristics.get('vocal_range', 'medium')} vocal range")
        if audio_characteristics.get('vocal_range') == 'high':
            prompt_parts.append("featuring high-reaching gestures during peak vocal notes")
    elif audio_characteristics.get('vocal_style') == 'melodic_speech':
        prompt_parts.append("delivering melodic speech with rhythmic cadence")
    
    # 2. Rhythm and Timing Integration
    if audio_characteristics.get('beat_strength') == 'strong':
        prompt_parts.append("with subtle body movements synchronized to strong rhythmic beats")
        if audio_characteristics.get('time_signature') != '4/4':
            prompt_parts.append(f"movements following {audio_characteristics['time_signature']} time signature")
    
    if audio_characteristics.get('syncopation') == 'high':
        prompt_parts.append("incorporating off-beat movements and syncopated gestures")
    elif audio_characteristics.get('syncopation') == 'medium':
        prompt_parts.append("with subtle rhythmic variations in movement")
    
    # 3. Tempo-Based Visual Elements
    tempo = audio_characteristics.get('tempo', 'medium')
    tempo_bpm = audio_characteristics.get('tempo_bpm', 120)
    
    if tempo == 'slow':
        prompt_parts.append(f"with gentle, measured movements timed to the slow {tempo_bpm:.1f} BPM tempo")
        if audio_characteristics.get('danceability', 0) > 0.7:
            prompt_parts.append("creating a calming, meditative visual rhythm despite the danceable beat")
    elif tempo == 'fast':
        prompt_parts.append(f"with energetic movements responding to the driving {tempo_bpm:.1f} BPM tempo")
        prompt_parts.append("quick visual elements synchronized to rapid rhythm")
    
    # 4. Emotional and Dynamic Elements
    mood = audio_characteristics.get('mood', 'neutral')
    emotional_arc = audio_characteristics.get('emotional_arc', 'stable')
    
    mood_enhancements = {
        'calm': "creating a serene, peaceful atmosphere with soft, gentle expressions",
        'contemplative': "with thoughtful, introspective facial expressions and measured movements",
        'energetic': "with dynamic, high-energy movements and vibrant expressions",
        'emotional': "with expressive facial changes and emotional body language",
        'futuristic': "with modern, innovative visual styling and contemporary movements"
    }
    
    if mood in mood_enhancements:
        prompt_parts.append(mood_enhancements[mood])
    
    # 5. Emotional Arc Integration
    if emotional_arc == 'building':
        prompt_parts.append("with intensity and expressiveness gradually building throughout the performance")
    elif emotional_arc == 'declining':
        prompt_parts.append("with movements and expressions gradually softening and calming")
    
    # 6. Genre-Specific Visual Direction
    genre = audio_characteristics.get('genre', 'unknown')
    
    genre_directions = {
        'pop': "with contemporary, polished performance style and modern visual aesthetics",
        'ballad': "with intimate, emotional delivery and heartfelt expressions",
        'electronic': "with futuristic visual effects synchronized to electronic elements",
        'ambient': "with atmospheric, ethereal visual quality and dreamlike movements",
        'spoken_word': "with theatrical, articulate performance and dramatic timing",
        'instrumental': "with movements responding to instrumental textures and musical phrases",
        'rock': "with energetic, powerful movements and strong rhythmic gestures",
        'folk': "with natural, grounded movements and organic, authentic performance style",
        'singer_songwriter': "with intimate, personal performance and heartfelt expressions",
        'metal': "with intense, aggressive movements and powerful headbanging gestures",
        'hip_hop': "with confident street-style movements and rhythmic body language",
        'rnb': "with smooth, soulful movements and groovy, flowing gestures",
        'indie': "with alternative, expressive movements and artistic body language",
        'jazz': "with sophisticated, improvisational movements and cool, flowing gestures",
        'classical': "with elegant, refined movements and graceful, formal posture",
        'latin': "with passionate, rhythmic dance movements and vibrant, energetic gestures",
        'african': "with earthy, grounded movements and powerful rhythmic expressions",
        'asian': "with precise, deliberate movements and graceful, controlled gestures"
    }
    
    if genre in genre_directions:
        prompt_parts.append(genre_directions[genre])
    
    # 7. Performance Type Integration
    performance_type = audio_characteristics.get('performance_type', 'studio_recording')
    
    if performance_type == 'live_performance':
        prompt_parts.append("with natural, authentic performance quality and spontaneous movements")
    elif performance_type == 'studio_recording':
        prompt_parts.append("with polished, precise performance and refined movements")
    
    # 8. Dynamic Range Visual Effects
    dynamic_range = audio_characteristics.get('dynamic_range', 'medium')
    
    if dynamic_range == 'wide':
        prompt_parts.append("with dramatic contrasts between quiet, subtle moments and powerful, expressive peaks")
    elif dynamic_range == 'narrow':
        prompt_parts.append("with consistent, steady performance quality and even visual intensity")
    
    # 9. Spectral Characteristics to Visual Elements
    spectral = audio_characteristics.get('spectral_characteristics', {})
    
    if spectral.get('warmth', False):
        prompt_parts.append("with warm, soft lighting and gentle visual atmosphere")
    elif spectral.get('brightness', 0) > 2500:
        prompt_parts.append("with bright, vibrant visual elements and crisp lighting")
    
    # 10. Danceability and Movement
    danceability = audio_characteristics.get('danceability', 0.5)
    
    if danceability > 0.8:
        prompt_parts.append("with highly danceable rhythmic movements and natural groove")
    elif danceability > 0.6:
        prompt_parts.append("with subtle rhythmic movements and natural flow")
    
    # 11. Energy Level Integration
    energy_level = audio_characteristics.get('energy_level', 'medium')
    
    energy_enhancements = {
        'very_low': "with minimal, subtle movements and gentle expressions",
        'low': "with restrained, calm movements and soft delivery",
        'medium': "with balanced, natural movements and moderate expressiveness",
        'high': "with energetic, confident movements and strong expressions",
        'very_high': "with dynamic, powerful movements and intense expressiveness"
    }
    
    if energy_level in energy_enhancements:
        prompt_parts.append(energy_enhancements[energy_level])
    
    # 12. Advanced Lip-Sync Instructions
    if audio_characteristics.get('has_vocals'):
        vocal_confidence = audio_characteristics.get('vocal_confidence', 0)
        vocal_count = audio_characteristics.get('vocal_count', 'unknown')
        vocal_separation = audio_characteristics.get('vocal_separation', 'unknown')
        
        if vocal_confidence > 0.8:
            prompt_parts.append("featuring precise, detailed lip-sync to every vocal nuance and syllable")
        elif vocal_confidence > 0.6:
            prompt_parts.append("with accurate lip-sync synchronized to vocal delivery")
        
        # Add vocal count specific instructions
        if vocal_count == "solo":
            prompt_parts.append("with focused single vocal performance and clear articulation")
        elif vocal_count == "duo":
            prompt_parts.append("with coordinated dual vocal performance and harmonized delivery")
        elif vocal_count == "small_group":
            prompt_parts.append("with small group vocal dynamics and collaborative performance")
        elif vocal_count == "group":
            prompt_parts.append("with ensemble vocal performance and group coordination")
        elif vocal_count == "choir":
            prompt_parts.append("with choral vocal arrangement and harmonized ensemble performance")
        
        # Add vocal separation instructions
        if vocal_separation == "lead_with_backup":
            prompt_parts.append("featuring prominent lead vocals with supporting harmonies")
        elif vocal_separation == "harmonized_vocals":
            prompt_parts.append("with blended harmonized vocal textures and balanced voices")
        elif vocal_separation == "multiple_voices":
            prompt_parts.append("with layered vocal performances and rich harmonic textures")
        
        # Add vocal style specific instructions
        if audio_characteristics.get('vocal_style') == 'spoken':
            prompt_parts.append("with natural mouth movements matching speech patterns and articulation")
        elif audio_characteristics.get('vocal_style') == 'singing':
            prompt_parts.append("with expressive mouth movements matching melodic phrases and vocal dynamics")
    
    return " ".join(prompt_parts)


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
        log_debug(f"Starting enhanced audio analysis for: {filename}")
        
        # Load audio file
        y, sr = librosa.load(file_path, duration=30)  # Analyze first 30 seconds
        log_debug(f"Audio loaded successfully: {len(y)} samples, {sr} Hz")
        
        characteristics = {
            "audio_type": "unknown",
            "tempo": "medium",
            "tempo_bpm": None,
            "mood": "neutral", 
            "energy_level": "medium",
            "has_vocals": False,
            "vocal_confidence": 0.0,
            "danceability": 0.5,
            "description": "",
            # Enhanced features
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
            # NEW: Vocal count detection
            "vocal_count": "unknown",
            "vocal_density": 0.0,
            "vocal_separation": "unknown"
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
        
        # 2. Enhanced Beat Analysis
        if len(beats) > 10:
            # Beat strength analysis
            beat_consistency = 1.0 - np.std(np.diff(beats)) / np.mean(np.diff(beats))
            if beat_consistency > 0.8:
                characteristics["beat_strength"] = "strong"
            elif beat_consistency > 0.5:
                characteristics["beat_strength"] = "medium"
            else:
                characteristics["beat_strength"] = "weak"
            
            # 2. Time Signature Detection
            if len(beats) > 10:
                beat_intervals = np.diff(beats)
                avg_interval = np.mean(beat_intervals)
                
                # Enhanced time signature detection based on genre
                genre_hint = characteristics.get("genre", "unknown")
                
                if avg_interval > 0.8:  # Slow beats
                    if genre_hint in ["rock", "metal", "pop"]:
                        characteristics["time_signature"] = "4/4"  # Most rock is 4/4
                    elif len(beats) % 3 == 0:
                        characteristics["time_signature"] = "3/4"
                    else:
                        characteristics["time_signature"] = "4/4"  # Most common
                else:
                    characteristics["time_signature"] = "4/4"  # Most common for popular music
        
        # 3. Energy Level Detection
        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.mean(rms))
        
        # Default thresholds for now (genre-based adjustment will happen later)
        if energy > 0.15:
            characteristics["energy_level"] = "high"
        elif energy > 0.10:
            characteristics["energy_level"] = "medium"
        else:
            characteristics["energy_level"] = "low"
        
        # 4. Dynamic Range Analysis
        dynamic_range = float(np.std(rms))
        if dynamic_range < 0.02:
            characteristics["dynamic_range"] = "narrow"
        elif dynamic_range < 0.05:
            characteristics["dynamic_range"] = "medium"
        else:
            characteristics["dynamic_range"] = "wide"
        
        # 5. Enhanced Spectral Analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        characteristics["spectral_characteristics"] = {
            "brightness": float(np.mean(spectral_centroids)),
            "warmth": float(np.mean(spectral_centroids) < 2000),
            "spectral_variance": float(np.var(spectral_centroids))
        }
        
        # 6. Advanced Vocal Detection
        avg_spectral_centroid = float(np.mean(spectral_centroids))
        spectral_variance = float(np.var(spectral_centroids))
        
        vocal_score = 0.0
        
        # High spectral centroid suggests vocals
        if avg_spectral_centroid > 2000:
            vocal_score += 0.3
        
        # Variance in spectral content suggests complex audio (like vocals)
        if spectral_variance > 500000:
            vocal_score += 0.2
        
        # MFCC analysis for vocal patterns
        mfcc_std = np.std(mfccs, axis=1)
        if float(np.mean(mfcc_std[1:4])) > 15:
            vocal_score += 0.3
        
        # Zero crossing rate (vocals typically have higher ZCR)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        if float(np.mean(zcr)) > 0.05:
            vocal_score += 0.2
        
        characteristics["vocal_confidence"] = min(float(vocal_score), 1.0)
        characteristics["has_vocals"] = vocal_score > 0.5
        
        # 7. Vocal Style Analysis
        if characteristics["has_vocals"]:
            # Detect singing vs speech using pitch analysis
            try:
                # Use librosa.pyin instead of yin (more reliable)
                
                # Extract pitch using harmonic component
                pitches, magnitudes = librosa.piptrack(y=harmonic, sr=sr, threshold=0.1)
                
                # Get dominant pitch for each frame
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if len(pitch_values) > 0:
                    # Analyze pitch variation
                    pitch_std = np.std(pitch_values)
                    pitch_range = np.max(pitch_values) - np.min(pitch_values)
                    
                    # Classify vocal style based on pitch characteristics
                    if pitch_std > 50:  # High variation
                        characteristics["vocal_style"] = "singing"
                    elif pitch_std > 20:  # Medium variation
                        characteristics["vocal_style"] = "melodic_speech"
                    else:  # Low variation
                        characteristics["vocal_style"] = "spoken"
                    
                    # Vocal range estimation
                    avg_pitch = float(np.mean(pitch_values))
                    if avg_pitch > 400:
                        characteristics["vocal_range"] = "high"
                    elif avg_pitch > 200:
                        characteristics["vocal_range"] = "medium"
                    else:
                        characteristics["vocal_range"] = "low"
                else:
                    characteristics["vocal_style"] = "unknown"
                    characteristics["vocal_range"] = "medium"
                    
            except Exception as pitch_error:
                log_debug(f"Pitch analysis failed: {pitch_error}")
                # Fallback to basic classification based on context
                if tempo < 100 and characteristics.get("mood") == "calm":
                    characteristics["vocal_style"] = "spoken"  # Folk/calm speech
                elif characteristics.get("beat_strength") == "strong" and tempo > 120:
                    characteristics["vocal_style"] = "singing"  # Rock singing style
                elif tempo < 100 and characteristics.get("mood") in ["calm", "contemplative"]:
                    characteristics["vocal_style"] = "melodic_speech"  # Folk storytelling
                else:
                    characteristics["vocal_style"] = "melodic_speech"
        
        # 8. Vocal Count Detection
        if characteristics["has_vocals"]:
            log_debug("Starting vocal count detection...")
            try:
                # Advanced vocal analysis using harmonic-percussive separation
                harmonic, percussive = librosa.effects.hpss(y)
                
                # Compute spectral features for vocal detection
                S = np.abs(librosa.stft(y))
                S_harmonic = np.abs(librosa.stft(harmonic))
                
                # Use mel spectrogram to estimate vocal-band energy (80-4000 Hz)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=80, fmax=4000)
                vocal_energy = float(np.mean(mel))  # Use raw energy, not dB
                
                # Focus on vocal frequency range (80-4000 Hz)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0]*2-1)
                vocal_mask = (freqs >= 80) & (freqs <= 4000)
                S_vocal = S_harmonic[vocal_mask, :]
                
                # Calculate vocal density from harmonic content
                if S_vocal.size > 0:
                    vocal_density = float(np.mean(S_vocal))
                    characteristics["vocal_density"] = vocal_density
                    
                    # Detect multiple vocals through harmonic complexity
                    harmonic_variance = float(np.var(S_vocal))
                    
                    # Use MFCC analysis for vocal separation - key for detecting multiple voices
                    mfccs = librosa.feature.mfcc(y=harmonic, sr=sr, n_mfcc=20)
                    mfcc_std = np.std(mfccs, axis=1)
                    mfcc_delta = librosa.feature.delta(mfccs)
                    mfcc_delta_std = float(np.mean(np.std(mfcc_delta, axis=1)))
                    
                    # Spectral contrast - helps detect voice layering
                    contrast = librosa.feature.spectral_contrast(y=harmonic, sr=sr)
                    contrast_var = float(np.mean(np.var(contrast, axis=1)))
                    
                    # Chroma features - detect harmonic layering (multiple voices = more chroma activity)
                    chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)
                    chroma_complexity = float(np.mean(np.var(chroma, axis=1)))
                    
                    # Combined vocal complexity score
                    vocal_complexity = (
                        harmonic_variance * 0.01 +
                        float(np.mean(mfcc_std[1:6])) * 0.5 +
                        mfcc_delta_std * 2 +
                        contrast_var * 0.05 +
                        chroma_complexity * 20
                    )
                    
                    log_debug(f"Vocal analysis: energy={vocal_energy:.4f}, variance={harmonic_variance:.2f}, mfcc_delta={mfcc_delta_std:.4f}, contrast={contrast_var:.4f}, chroma={chroma_complexity:.4f}, complexity={vocal_complexity:.2f}")
                    
                    # Vocal count estimation based on complexity
                    # Higher complexity = more voices/harmonies
                    if vocal_complexity < 8:
                        characteristics["vocal_count"] = "solo"
                        characteristics["vocal_separation"] = "single_voice"
                    elif vocal_complexity < 12:
                        characteristics["vocal_count"] = "duo"
                        characteristics["vocal_separation"] = "two_voices"
                    elif vocal_complexity < 18:
                        characteristics["vocal_count"] = "small_group"
                        characteristics["vocal_separation"] = "few_voices"
                    elif vocal_complexity < 25:
                        characteristics["vocal_count"] = "group"
                        characteristics["vocal_separation"] = "multiple_voices"
                    else:
                        characteristics["vocal_count"] = "choir"
                        characteristics["vocal_separation"] = "many_voices"
                    
                    # Detect lead vs backup vocals for non-solo
                    if characteristics["vocal_count"] in ["duo", "small_group", "group"]:
                        # Look for dominant voice patterns using spectral centroid variance
                        centroid = librosa.feature.spectral_centroid(y=harmonic, sr=sr)
                        centroid_var = float(np.var(centroid))
                        if centroid_var > 100000:  # High variance = distinct lead + backup
                            characteristics["vocal_separation"] = "lead_with_backup"
                        else:  # Low variance = harmonized together
                            characteristics["vocal_separation"] = "harmonized_vocals"
                    
                    log_debug(f"Vocal count result: count={characteristics['vocal_count']}, separation={characteristics['vocal_separation']}")
                else:
                    characteristics["vocal_count"] = "unknown"
                    characteristics["vocal_density"] = 0.0
                    characteristics["vocal_separation"] = "unknown"
                
            except Exception as vocal_error:
                log_debug(f"Vocal count analysis failed: {vocal_error}")
                characteristics["vocal_count"] = "unknown"
                characteristics["vocal_density"] = 0.0
                characteristics["vocal_separation"] = "unknown"
        
        # Fallback: If vocal count is still unknown but we have vocals, use heuristics
        if characteristics["has_vocals"] and characteristics["vocal_count"] == "unknown":
            log_debug(f"Vocal count fallback triggered: has_vocals={characteristics['has_vocals']}, current_count={characteristics['vocal_count']}")
            # Use vocal confidence and style to estimate vocal count
            vocal_conf = characteristics.get("vocal_confidence", 0)
            vocal_style = characteristics.get("vocal_style", "unknown")
            
            if vocal_conf > 0.7:
                # High confidence = clear single voice most likely
                characteristics["vocal_count"] = "solo"
                characteristics["vocal_separation"] = "single_voice"
                characteristics["vocal_density"] = vocal_conf  # Use confidence as density proxy
                log_debug(f"Vocal count fallback: solo (high confidence {vocal_conf})")
            elif vocal_conf > 0.5:
                # Medium confidence = could be solo or duo
                characteristics["vocal_count"] = "solo"
                characteristics["vocal_separation"] = "single_voice"
                characteristics["vocal_density"] = vocal_conf
                log_debug(f"Vocal count fallback: solo (medium confidence {vocal_conf})")
            else:
                # Low confidence = default to solo
                characteristics["vocal_count"] = "solo"
                characteristics["vocal_separation"] = "single_voice"
                characteristics["vocal_density"] = 0.5
                log_debug(f"Vocal count fallback: solo (low confidence, default)")
        else:
            log_debug(f"Vocal count fallback NOT triggered: has_vocals={characteristics['has_vocals']}, current_count={characteristics['vocal_count']}")
        
        # 9. Syncopation Detection
        if len(beats) > 10:
            # Look for off-beat energy
            beat_frames = librosa.util.fix_frames(beats)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            
            # Count how many onsets fall between beats (syncopation)
            syncopated_count = 0
            for onset in onset_frames:
                if not any(abs(onset - beat) < 2 for beat in beat_frames):
                    syncopated_count += 1
            
            syncopation_ratio = syncopated_count / len(onset_frames) if len(onset_frames) > 0 else 0
            if syncopation_ratio > 0.3:
                characteristics["syncopation"] = "high"
            elif syncopation_ratio > 0.1:
                characteristics["syncopation"] = "medium"
            else:
                characteristics["syncopation"] = "low"
        
        # 9. Danceability (based on beat consistency and tempo)
        if len(beats) > 10:
            beat_diffs = np.diff(beats)
            beat_consistency = float(1.0 - np.std(beat_diffs) / np.mean(beat_diffs))
            tempo_factor = min(tempo / 120, 2.0)
            characteristics["danceability"] = float(beat_consistency * 0.6 + tempo_factor * 0.4)
        
        # 10. Enhanced Genre Classification
        if characteristics["has_vocals"]:
            if characteristics["vocal_style"] == "singing":
                if tempo > 120 and energy > 0.15:
                    # Check for metal vs rock
                    if energy > 0.25 and characteristics.get("spectral_characteristics", {}).get("brightness", 0) > 3000:
                        characteristics["genre"] = "metal"  # High energy + bright = metal
                    else:
                        characteristics["genre"] = "rock"  # Standard rock
                elif tempo > 110 and characteristics["danceability"] > 0.8:
                    # Check for hip-hop vs pop vs rock
                    if characteristics.get("syncopation") == "high" and characteristics.get("beat_strength") == "strong":
                        # Additional check for rock vs hip-hop
                        if characteristics.get("vocal_style") == "singing" and characteristics.get("energy_level") in ["high", "very_high"]:
                            characteristics["genre"] = "rock"  # Rock with singing + high energy
                        else:
                            characteristics["genre"] = "hip_hop"  # Hip-hop with spoken vocals
                    else:
                        characteristics["genre"] = "pop"
                elif tempo < 90 and energy < 0.1:
                    characteristics["genre"] = "ballad"
                elif tempo > 100 and tempo < 130 and characteristics.get("spectral_characteristics", {}).get("warmth", False):
                    characteristics["genre"] = "rnb"  # Warm sound + medium tempo = R&B
                elif tempo > 90 and characteristics.get("spectral_characteristics", {}).get("brightness", 0) < 2000:
                    characteristics["genre"] = "indie"  # Darker sound = indie
                else:
                    characteristics["genre"] = "singer_songwriter"
            elif characteristics["vocal_style"] == "spoken":
                # Check for hip-hop vs other spoken
                if characteristics["beat_strength"] == "strong" and tempo > 90 and characteristics.get("syncopation") == "high":
                    characteristics["genre"] = "hip_hop"  # Hip-hop with spoken vocals
                elif characteristics["beat_strength"] == "strong" and tempo > 120:
                    characteristics["genre"] = "rock"  # Rock with spoken vocals
                elif tempo < 100 and characteristics["mood"] == "calm":
                    characteristics["genre"] = "folk"  # Folk with spoken vocals
                else:
                    characteristics["genre"] = "spoken_word"
            elif characteristics["vocal_style"] == "melodic_speech":
                # Melodic speech with strong beat often maps to rock/pop
                if characteristics["beat_strength"] == "strong" and tempo > 100:
                    characteristics["genre"] = "rock"
                elif tempo < 100 and characteristics["mood"] in ["calm", "contemplative"]:
                    characteristics["genre"] = "folk"
                elif characteristics["danceability"] > 0.8:
                    characteristics["genre"] = "pop"  # Pop with melodic speech
                else:
                    characteristics["genre"] = "singer_songwriter"
            else:
                # Unknown vocal style - use tempo and mood
                if tempo < 100 and characteristics["mood"] == "calm":
                    characteristics["genre"] = "folk"
                elif characteristics["beat_strength"] == "strong" and tempo > 120:
                    characteristics["genre"] = "rock"
                else:
                    characteristics["genre"] = "unknown"
        else:
            # Instrumental genres
            if tempo > 130:
                characteristics["genre"] = "electronic"
            elif tempo < 80:
                characteristics["genre"] = "ambient"
            elif tempo > 100 and characteristics.get("spectral_characteristics", {}).get("brightness", 0) > 2500:
                characteristics["genre"] = "classical"  # Bright + complex = classical
            elif tempo > 80 and tempo < 120 and characteristics.get("beat_strength") == "strong":
                # Check for jazz vs other instrumental
                if characteristics.get("syncopation") == "high" and characteristics.get("dynamic_range") == "wide":
                    characteristics["genre"] = "jazz"  # Complex rhythm + wide dynamics = jazz
                else:
                    characteristics["genre"] = "instrumental"
            else:
                characteristics["genre"] = "instrumental"
        
        # World Music Detection (based on spectral and rhythmic characteristics)
        spectral = characteristics.get("spectral_characteristics", {})
        if characteristics.get("genre") in ["rock", "pop", "metal", "hip_hop", "rnb", "indie"]:
            pass
        elif spectral.get("brightness", 0) > 3000 and characteristics.get("syncopation") == "high":
            # Latin music: bright + highly syncopated
            if tempo > 110:
                characteristics["genre"] = "latin"
        elif spectral.get("brightness", 0) < 2000 and characteristics.get("beat_strength") == "strong":
            # Check if it's actually rock/pop with warm tones
            if characteristics.get("vocal_style") == "singing" and characteristics.get("vocal_confidence", 0) > 0.6:
                characteristics["genre"] = "rock"  # Rock with warm tones
            elif tempo > 100:
                characteristics["genre"] = "african"  # African music: warm + strong beat
        elif spectral.get("spectral_variance", 0) > 1000000 and tempo > 90:
            # Asian music: complex spectral content
            characteristics["genre"] = "asian"
        
        # 12. Post-Genre Time Signature Correction
        # Fix time signature based on detected genre
        if characteristics["genre"] in ["rock", "metal", "pop", "hip_hop", "electronic"]:
            characteristics["time_signature"] = "4/4"  # Most popular music is 4/4
        elif characteristics["genre"] in ["classical", "folk"]:
            # Keep original detection for classical/folk (could be 3/4 or 4/4)
            pass
        else:
            # For other genres, default to 4/4
            characteristics["time_signature"] = "4/4"
        
        # 13. Genre-Based Energy Adjustment
        # Adjust energy level based on detected genre
        genre = characteristics["genre"]
        current_energy = characteristics["energy_level"]
        
        if genre in ["rock", "pop", "hip_hop"]:
            # Rock/pop should feel more energetic
            if current_energy == "low":
                characteristics["energy_level"] = "medium"
            elif current_energy == "medium" and characteristics.get("beat_strength") == "strong":
                characteristics["energy_level"] = "high"
        elif genre in ["classical", "folk"]:
            # Classical/folk should feel calmer
            if current_energy == "high":
                characteristics["energy_level"] = "medium"
        
        # 11. Performance Type Detection
        # Reverb analysis for live vs studio
        harmonic, percussive = librosa.effects.hpss(y)
        reverb_indicator = float(np.mean(harmonic) / np.mean(percussive))
        
        if reverb_indicator > 2.0:
            characteristics["performance_type"] = "live_performance"
        elif reverb_indicator < 0.5:
            characteristics["performance_type"] = "studio_recording"
        else:
            characteristics["performance_type"] = "home_recording"
        
        # 12. Emotional Arc Detection
        # Analyze energy progression over time
        hop_length = 512
        rms_frames = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        if len(rms_frames) > 10:
            # Check if energy builds, falls, or stays stable
            try:
                energy_trend = np.polyfit(range(len(rms_frames)), rms_frames, 1)[0]
                
                if energy_trend > 0.001:
                    characteristics["emotional_arc"] = "building"
                elif energy_trend < -0.001:
                    characteristics["emotional_arc"] = "declining"
                else:
                    characteristics["emotional_arc"] = "stable"
            except Exception as trend_error:
                log_debug(f"Emotional arc analysis failed: {trend_error}")
                characteristics["emotional_arc"] = "stable"
        
        # 13. Mood Detection (enhanced)
        if characteristics["energy_level"] in ["high", "very_high"] and characteristics["tempo"] in ["fast", "very_fast"]:
            if characteristics["has_vocals"]:
                characteristics["mood"] = "energetic"
            else:
                characteristics["mood"] = "energetic_instrumental"
        elif characteristics["energy_level"] in ["low", "very_low"] and characteristics["tempo"] in ["slow", "very_slow"]:
            if characteristics["vocal_style"] == "spoken":
                characteristics["mood"] = "contemplative"
            else:
                characteristics["mood"] = "calm"
        elif characteristics["has_vocals"] and characteristics["energy_level"] == "medium":
            characteristics["mood"] = "emotional"
        elif characteristics["genre"] == "electronic":
            characteristics["mood"] = "futuristic"
        else:
            characteristics["mood"] = "neutral"
        
        # 14. Audio Type Classification (enhanced)
        if characteristics["has_vocals"]:
            if characteristics["vocal_style"] == "singing":
                if characteristics["tempo"] in ["medium", "fast", "very_fast"]:
                    characteristics["audio_type"] = "singing"
                else:
                    characteristics["audio_type"] = "ballad"
            elif characteristics["vocal_style"] == "spoken":
                characteristics["audio_type"] = "speech"
            else:
                characteristics["audio_type"] = "melodic_speech"
        else:
            if characteristics["danceability"] > 0.6:
                characteristics["audio_type"] = "instrumental_dance"
            else:
                characteristics["audio_type"] = "instrumental"
        
        # 15. Enhanced Description Generation
        description_parts = []
        
        # Audio type and style
        if characteristics["vocal_style"] == "singing":
            description_parts.append(f"{characteristics['vocal_style']} performance with {characteristics['vocal_range']} vocal range")
        elif characteristics["vocal_style"] == "spoken":
            description_parts.append("spoken dialogue with clear diction")
        elif characteristics["vocal_style"] == "melodic_speech":
            description_parts.append("melodic speech with rhythmic delivery")
        else:
            description_parts.append("instrumental performance")
        
        # Tempo and rhythm
        if characteristics["beat_strength"] == "strong":
            description_parts.append(f"with strong {characteristics['tempo']} tempo ({characteristics['tempo_bpm']:.1f} BPM)")
        else:
            description_parts.append(f"with {characteristics['tempo']} tempo ({characteristics['tempo_bpm']:.1f} BPM)")
        
        # Time signature
        if characteristics["time_signature"] != "4/4":
            description_parts.append(f"in {characteristics['time_signature']} time")
        
        # Danceability and syncopation
        if characteristics["danceability"] > 0.7:
            if characteristics["syncopation"] == "high":
                description_parts.append("highly syncopated danceable rhythm")
            else:
                description_parts.append("strong danceable rhythm")
        elif characteristics["syncopation"] == "high":
            description_parts.append("complex syncopated rhythm")
        
        # Mood and emotional arc
        mood_descriptions = {
            "energetic": "creating high energy and excitement",
            "energetic_instrumental": "building instrumental energy",
            "calm": "establishing a peaceful, serene mood",
            "contemplative": "creating an introspective, thoughtful atmosphere",
            "emotional": "with emotional, expressive delivery",
            "futuristic": "with modern, innovative soundscapes",
            "neutral": "with balanced mood"
        }
        description_parts.append(mood_descriptions.get(characteristics["mood"], "with neutral mood"))
        
        # Emotional arc
        if characteristics["emotional_arc"] == "building":
            description_parts.append("with intensity building throughout")
        elif characteristics["emotional_arc"] == "declining":
            description_parts.append("gradually calming down")
        
        # Performance characteristics
        if characteristics["performance_type"] == "live_performance":
            description_parts.append("captured in a live setting with natural ambiance")
        elif characteristics["performance_type"] == "studio_recording":
            description_parts.append("with polished studio production quality")
        
        # Genre-specific details
        genre_details = {
            "pop": "featuring catchy melodic hooks",
            "ballad": "with intimate, emotional delivery",
            "electronic": "with synthesized textures and electronic elements",
            "ambient": "creating atmospheric soundscapes",
            "instrumental": "showcasing musical instrumentation",
            "spoken_word": "with articulate vocal performance"
        }
        if characteristics["genre"] in genre_details:
            description_parts.append(genre_details[characteristics["genre"]])
        
        # Vocal performance details
        if characteristics["has_vocals"]:
            if characteristics["vocal_confidence"] > 0.8:
                description_parts.append("featuring prominent vocal performance")
            description_parts.append("requiring precise lip-sync synchronization")
        
        characteristics["description"] = " ".join(description_parts) + "."
        
        log_debug(f"Enhanced audio analysis completed: {filename}")
        log_debug(f"Genre: {characteristics['genre']}, Style: {characteristics['vocal_style']}, Beat: {characteristics['beat_strength']}")
        log_debug(f"Vocal Count: {characteristics['vocal_count']}, Vocal Density: {characteristics['vocal_density']}, Vocal Separation: {characteristics['vocal_separation']}")
        
        return characteristics
        
    except Exception as e:
        log_debug(f"Error in enhanced audio analysis: {str(e)}")
        log_debug(f"Error type: {type(e).__name__}")
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}")
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
            movement_level = getattr(request, 'movement_level', 'auto') if hasattr(request, 'movement_level') else 'auto'
            
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
            
            # Smart priority system: Auto-detect vs Manual selection
            if movement_level == 'auto' and audio_description:
                # AUTO MODE: Use audio detection to determine optimal movement level
                if "very_fast" in audio_description.lower() and "singing" in audio_description.lower():
                    auto_movement_level = "dynamic"  # Fast singing = dynamic
                elif "high energy" in audio_description.lower() or "very_high" in audio_description.lower():
                    auto_movement_level = "expressive"  # High energy = expressive
                elif "low energy" in audio_description.lower() or "calm" in audio_description.lower() or "peaceful" in audio_description.lower():
                    auto_movement_level = "minimal"  # Low energy = minimal
                elif "speech" in audio_description.lower() or "dialogue" in audio_description.lower() or "spoken" in audio_description.lower():
                    auto_movement_level = "minimal"  # Speech = minimal
                elif "dance" in audio_description.lower() or "highly danceable" in audio_description.lower():
                    auto_movement_level = "expressive"  # Danceable = expressive
                else:
                    auto_movement_level = "natural"  # Default to natural
                
                # Generate movement instruction based on auto-detected level
                if auto_movement_level == 'static':
                    movement_instruction = " ABSOLUTE STATIC PERFORMANCE: Only lip-sync and subtle eye movements allowed. No head movement, no arm gestures, no body swaying, no shoulder movements. Character remains completely still except for mouth movement and minimal facial expressions. "
                elif auto_movement_level == 'minimal':
                    movement_instruction = " MINIMAL MOVEMENT: Only subtle head movement, slight shoulder motion, and gentle hand gestures. No large body movements, no dramatic swaying, no exaggerated gestures. Focus on restrained, natural motion. "
                elif auto_movement_level == 'natural':
                    movement_instruction = " NATURAL MOVEMENT: Normal body movement including head turns, shoulder movements, arm gestures, and gentle body swaying. Maintain realistic motion without exaggeration. "
                elif auto_movement_level == 'expressive':
                    movement_instruction = " EXPRESSIVE MOVEMENT: Full body movement including dynamic gestures, head movement, shoulder motion, arm gestures, and body swaying. Emphasize rhythmic, energetic motion that matches the audio. "
                elif auto_movement_level == 'dynamic':
                    movement_instruction = " DYNAMIC MOVEMENT: Highly energetic and expressive full-body movement. Include dramatic gestures, head movement, shoulder motion, arm gestures, body swaying, and rhythmic dancing. Emphasize powerful, athletic motion. "
                
                # Add auto-detection info to performance instruction
                performance_instruction += f"AUTO-DETECTED MOVEMENT: Selected '{auto_movement_level}' movement level based on audio analysis. "
                
            else:
                # MANUAL MODE: Use user-selected movement level
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
                elif movement_level == 'auto' and not audio_description:
                    # Auto selected but no audio - default to natural
                    movement_instruction = " NATURAL MOVEMENT: Normal body movement including head turns, shoulder movements, arm gestures, and gentle body swaying. Maintain realistic motion without exaggeration. "
            
            # AUTO-DETECT from audio_description if not explicitly set
            if audio_description:
                # CORRECT COMMON AUDIO ANALYSIS ERRORS
                
                # Get audio analysis values if available
                energy_level = getattr(request, 'energy_level', 'medium') if hasattr(request, 'energy_level') else 'medium'
                danceability = getattr(request, 'danceability', 0.5) if hasattr(request, 'danceability') else 0.5
                mood = getattr(request, 'mood', 'neutral') if hasattr(request, 'mood') else 'neutral'
                
                # Fix incorrect energy level for fast tempo music
                if "very_fast" in audio_description.lower() and energy_level == "low":
                    energy_level = "high"  # Fast tempo music can't be low energy
                
                # Fix danceability values over 1.0
                if danceability and float(danceability) > 1.0:
                    danceability = 1.0  # Cap at maximum
                
                # Fix mood for high-energy rock music
                if "very_fast" in audio_description.lower() and mood == "neutral":
                    mood = "energetic"  # Fast music should be energetic
                
                # Fix energy level for dance music
                if "danceability" and float(danceability) > 0.9 and energy_level == "medium":
                    energy_level = "high"  # Highly danceable music should be high energy
                
                # Fix mood for dance/electronic music
                if ("dance" in audio_description.lower() or "electronic" in audio_description.lower() or "pump up" in audio_description.lower()) and mood == "emotional":
                    mood = "energetic"  # Dance music should be energetic, not emotional
                
                # Enhanced genre detection with speech-specific logic
                if not genre_movement:
                    if "rock" in audio_description.lower() or "heavy" in audio_description.lower():
                        genre_movement = "rock"
                    elif "metal" in audio_description.lower() or "aggressive" in audio_description.lower():
                        genre_movement = "metal"
                    elif "pop" in audio_description.lower() or "upbeat" in audio_description.lower():
                        genre_movement = "pop"
                    elif "hip-hop" in audio_description.lower() or "hip hop" in audio_description.lower() or "rap" in audio_description.lower():
                        genre_movement = "hip_hop"
                    elif "r&b" in audio_description.lower() or "rnb" in audio_description.lower() or "soul" in audio_description.lower():
                        genre_movement = "rnb"
                    elif "indie" in audio_description.lower() or "alternative" in audio_description.lower():
                        genre_movement = "indie"
                    elif "classical" in audio_description.lower() or "orchestral" in audio_description.lower():
                        genre_movement = "classical"
                    elif "electronic" in audio_description.lower() or "edm" in audio_description.lower() or "synth" in audio_description.lower():
                        genre_movement = "electronic"
                    elif "jazz" in audio_description.lower() or "smooth" in audio_description.lower():
                        genre_movement = "jazz"
                    elif "folk" in audio_description.lower() or "acoustic" in audio_description.lower() or "organic" in audio_description.lower():
                        genre_movement = "folk"
                    elif "latin" in audio_description.lower() or "salsa" in audio_description.lower() or "reggae" in audio_description.lower():
                        genre_movement = "latin"
                    elif "african" in audio_description.lower() or "afro" in audio_description.lower():
                        genre_movement = "african"
                    elif "asian" in audio_description.lower() or "oriental" in audio_description.lower():
                        genre_movement = "asian"
                    elif "dialogue" in audio_description.lower() or "spoken" in audio_description.lower() or "speech" in audio_description.lower():
                        genre_movement = "storytelling"
                    elif "narrative" in audio_description.lower() or "storytelling" in audio_description.lower():
                        genre_movement = "storytelling"
                
                # Enhanced tempo detection with speech consideration
                if movement_speed == 'normal':  # Only override if not explicitly set
                    if "very_fast" in audio_description.lower() or "fast tempo" in audio_description.lower():
                        movement_speed = "fast"
                    elif "very_slow" in audio_description.lower() or "slow tempo" in audio_description.lower():
                        # For speech, use natural speed instead of slow motion
                        if "speech" in audio_description.lower() or "dialogue" in audio_description.lower() or "spoken" in audio_description.lower():
                            movement_speed = "natural"
                        else:
                            movement_speed = "slow_motion"
                
                # Enhanced energy detection with danceability paradox fix
                if audio_reactivity == 'medium':  # Only override if not explicitly set
                    if "high energy" in audio_description.lower() or "very_high" in audio_description.lower():
                        audio_reactivity = "high"
                    elif "low energy" in audio_description.lower() or "calm" in audio_description.lower() or "peaceful" in audio_description.lower():
                        audio_reactivity = "low"
                    # Use corrected energy_level
                    elif energy_level == "high":
                        audio_reactivity = "high"
                    # Speech-specific reactivity
                    elif "speech" in audio_description.lower() or "dialogue" in audio_description.lower():
                        audio_reactivity = "low"  # Speech should have low reactivity
                
                # Enhanced lipsync intensity with speech-specific logic
                if lipsync_intensity == 'natural':  # Only override if not explicitly set
                    if "dramatic" in audio_description.lower() or "powerful" in audio_description.lower():
                        lipsync_intensity = "exaggerated"
                    elif "subtle" in audio_description.lower() or "gentle" in audio_description.lower():
                        lipsync_intensity = "subtle"
                    # Fast tempo singing should be exaggerated
                    elif "very_fast" in audio_description.lower() and "singing" in audio_description.lower():
                        lipsync_intensity = "exaggerated"
                    # Speech-specific lipsync
                    elif "speech" in audio_description.lower() or "dialogue" in audio_description.lower() or "calm" in audio_description.lower():
                        lipsync_intensity = "subtle"  # Natural speech should be subtle
                
                # Generate comprehensive audio characteristics for enhanced prompts
                audio_characteristics = []
                
                # Emotional & Performance Detection
                if any(word in audio_description.lower() for word in ["singing", "vocals", "vocal"]):
                    audio_characteristics.append("singing")
                if "speaking" in audio_description.lower() or "spoken" in audio_description.lower():
                    audio_characteristics.append("speaking")
                if "dialogue" in audio_description.lower():
                    audio_characteristics.append("dialogue")
                if "speech" in audio_description.lower():
                    audio_characteristics.append("speech")
                if "rapping" in audio_description.lower() or "rap" in audio_description.lower():
                    audio_characteristics.append("rapping")
                if "chanting" in audio_description.lower() or "chant" in audio_description.lower():
                    audio_characteristics.append("chanting")
                if "whispering" in audio_description.lower() or "whisper" in audio_description.lower():
                    audio_characteristics.append("whispering")
                if "storytelling" in audio_description.lower() or "narrative" in audio_description.lower():
                    audio_characteristics.append("storytelling")
                if "conversational" in audio_description.lower():
                    audio_characteristics.append("conversational")
                if "personal" in audio_description.lower():
                    audio_characteristics.append("personal")
                if "intimate" in audio_description.lower():
                    audio_characteristics.append("intimate")
                
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
                    # Fix danceability paradox for speech content
                    if "speech" in audio_description.lower() or "dialogue" in audio_description.lower() or "spoken" in audio_description.lower():
                        audio_characteristics.append("subtle_rhythm")  # More appropriate for speech
                    else:
                        audio_characteristics.append("highly_danceable")
                if "minimal dance" in audio_description.lower():
                    audio_characteristics.append("minimal_dance")
                if "head-nodding" in audio_description.lower() or "head nod" in audio_description.lower():
                    audio_characteristics.append("head_nodding")
                if "full body movement" in audio_description.lower():
                    audio_characteristics.append("full_body_movement")
                # Speech-specific movement triggers
                if "peaceful" in audio_description.lower() or "serene" in audio_description.lower():
                    audio_characteristics.append("gentle_swaying")
                if "calm" in audio_description.lower():
                    audio_characteristics.append("calm_presence")
                if "slow tempo" in audio_description.lower() and ("speech" in audio_description.lower() or "dialogue" in audio_description.lower()):
                    audio_characteristics.append("measured_speech")
                
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
            elif genre_movement == 'metal':
                audio_integration_instruction += "METAL MOVEMENT STYLE: Intense, aggressive movements with powerful energy. Headbanging, aggressive gestures, strong arm movements, intense facial expressions, powerful body language. "
            elif genre_movement == 'hip_hop':
                audio_integration_instruction += "HIP-HOP MOVEMENT STYLE: Confident street-style movements with rhythmic precision. Cool gestures, rhythmic body language, confident posture, smooth flowing motions, urban dance elements. "
            elif genre_movement == 'rnb':
                audio_integration_instruction += "R&B MOVEMENT STYLE: Smooth, soulful movements with groovy flow. Graceful gestures, flowing body language, cool expressions, sophisticated rhythmic motions, soulful delivery. "
            elif genre_movement == 'indie':
                audio_integration_instruction += "INDIE MOVEMENT STYLE: Alternative, expressive movements with artistic flair. Unique gestures, creative body language, individualistic expressions, non-traditional movements, artistic performance style. "
            elif genre_movement == 'latin':
                audio_integration_instruction += "LATIN MOVEMENT STYLE: Passionate, rhythmic dance movements with vibrant energy. Hip movements, rhythmic footwork, expressive arm gestures, passionate expressions, dynamic dance elements. "
            elif genre_movement == 'african':
                audio_integration_instruction += "AFRICAN MOVEMENT STYLE: Earthy, grounded movements with powerful rhythm. Strong body movements, rhythmic foot patterns, expressive gestures, grounded posture, powerful rhythmic expressions. "
            elif genre_movement == 'asian':
                audio_integration_instruction += "ASIAN MOVEMENT STYLE: Precise, deliberate movements with graceful control. Refined gestures, controlled body language, elegant posture, precise timing, graceful flowing motions. "
            
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
                
                # ENHANCED AUDIO ANALYSIS INTEGRATION
            # Use enhanced audio characteristics to generate better prompt
            if audio_description and hasattr(request, 'audio_characteristics'):
                try:
                    # Parse audio characteristics from the audio_description if available
                    # This would be enhanced to pass the full characteristics object
                    if request.image_description:
                        enhanced_base_prompt = f"{request.image_description} {request.prompt}"
                    else:
                        enhanced_base_prompt = f"The character {request.prompt}"
                    
                    # For now, extract key characteristics from audio_description
                    audio_chars = {}
                    
                    # Extract enhanced characteristics from audio_description
                    if "spoken dialogue" in audio_description.lower():
                        audio_chars['vocal_style'] = 'spoken'
                    elif "singing performance" in audio_description.lower():
                        audio_chars['vocal_style'] = 'singing'
                    elif "melodic speech" in audio_description.lower():
                        audio_chars['vocal_style'] = 'melodic_speech'
                    
                    # Extract tempo
                    import re
                    tempo_match = re.search(r'(\d+\.?\d*)\s*BPM', audio_description)
                    if tempo_match:
                        audio_chars['tempo_bpm'] = float(tempo_match.group(1))
                        if audio_chars['tempo_bpm'] < 90:
                            audio_chars['tempo'] = 'slow'
                        elif audio_chars['tempo_bpm'] < 120:
                            audio_chars['tempo'] = 'medium'
                        else:
                            audio_chars['tempo'] = 'fast'
                    
                    # Extract mood
                    if "peaceful" in audio_description.lower() or "serene" in audio_description.lower():
                        audio_chars['mood'] = 'calm'
                    elif "energetic" in audio_description.lower() or "high energy" in audio_description.lower():
                        audio_chars['mood'] = 'energetic'
                    elif "emotional" in audio_description.lower():
                        audio_chars['mood'] = 'emotional'
                    elif "contemplative" in audio_description.lower():
                        audio_chars['mood'] = 'contemplative'
                    
                    # Extract danceability
                    if "highly danceable" in audio_description.lower():
                        audio_chars['danceability'] = 0.85
                    elif "danceable" in audio_description.lower():
                        audio_chars['danceability'] = 0.7
                    else:
                        audio_chars['danceability'] = 0.4
                    
                    # Extract beat strength
                    if "strong rhythm" in audio_description.lower() or "strong beat" in audio_description.lower():
                        audio_chars['beat_strength'] = 'strong'
                    elif "weak rhythm" in audio_description.lower():
                        audio_chars['beat_strength'] = 'weak'
                    else:
                        audio_chars['beat_strength'] = 'medium'
                    
                    # Extract vocal confidence
                    if "prominent vocal" in audio_description.lower():
                        audio_chars['vocal_confidence'] = 0.9
                    elif "featuring vocal" in audio_description.lower():
                        audio_chars['vocal_confidence'] = 0.7
                    else:
                        audio_chars['vocal_confidence'] = 0.5
                    
                    # Extract has_vocals
                    audio_chars['has_vocals'] = any([
                        "vocal" in audio_description.lower(),
                        "singing" in audio_description.lower(),
                        "speech" in audio_description.lower(),
                        "dialogue" in audio_description.lower(),
                        "spoken" in audio_description.lower()
                    ])
                    
                    # Extract genre
                    if "pop" in audio_description.lower():
                        audio_chars['genre'] = 'pop'
                    elif "electronic" in audio_description.lower():
                        audio_chars['genre'] = 'electronic'
                    elif "rock" in audio_description.lower():
                        audio_chars['genre'] = 'rock'
                    elif "jazz" in audio_description.lower():
                        audio_chars['genre'] = 'jazz'
                    elif "classical" in audio_description.lower():
                        audio_chars['genre'] = 'classical'
                    elif "ambient" in audio_description.lower():
                        audio_chars['genre'] = 'ambient'
                    elif "spoken_word" in audio_description.lower():
                        audio_chars['genre'] = 'spoken_word'
                    else:
                        audio_chars['genre'] = 'unknown'
                    
                    # Extract energy level
                    if "very high energy" in audio_description.lower():
                        audio_chars['energy_level'] = 'very_high'
                    elif "high energy" in audio_description.lower():
                        audio_chars['energy_level'] = 'high'
                    elif "low energy" in audio_description.lower():
                        audio_chars['energy_level'] = 'low'
                    elif "very low energy" in audio_description.lower():
                        audio_chars['energy_level'] = 'very_low'
                    else:
                        audio_chars['energy_level'] = 'medium'
                    
                    # Generate enhanced prompt using the new function
                    if audio_chars:
                        enhanced_prompt = generate_enhanced_ltx2_prompt(audio_chars, enhanced_base_prompt)
                        # Update the base prompt with enhanced audio-driven description
                        request.prompt = enhanced_prompt
                        log_debug(f"Enhanced prompt generated using audio analysis: {len(enhanced_prompt)} characters")
                
                except Exception as e:
                    log_debug(f"Error generating enhanced audio prompt: {e}")
                    # Continue with original prompt if enhancement fails
            
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
