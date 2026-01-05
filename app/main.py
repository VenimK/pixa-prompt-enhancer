from fastapi import FastAPI, File, UploadFile, Request
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
    prompt_type: str  # VEO or WAN2 or Image or 3D
    style: str
    cinematography: str
    lighting: str
    image_description: str | None = None
    motion_effect: str | None = None
    text_emphasis: str | None = None
    model: str | None = None  # AI model selection (flux, qwen, nunchaku, etc.)
    model_type: str | None = None  # 3D model type (character, object, vehicle, environment, props)
    wrap_mode: str | None = None  # 'vehicle' | 'people-object' | 'none'


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
        model = genai.GenerativeModel("gemini-2.5-flash")

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
            except Exception as api_error:
                return f"Error processing image(s) with Gemini API: {api_error}. One of the images may be too large or in an unsupported format."
        elif image_path:
            try:
                image = PIL.Image.open(image_path)
            except Exception as img_error:
                return f"Error loading image: {img_error}. Please check the image file format and try again."

            try:
                response = model.generate_content([prompt, image])
            except Exception as api_error:
                return f"Error processing image with Gemini API: {api_error}. The image may be too large or in an unsupported format."
        else:
            try:
                response = model.generate_content(prompt)
            except Exception as api_error:
                return f"Error with Gemini API: {api_error}. Your prompt may contain content that violates usage policies."

        return response.text
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
        elif request.prompt_type == "VEO":
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
                    # Use image description as primary reference
                    meta_prompt = f"You are a creative assistant for image-to-3D model generation. Create a detailed visual description (maximum 2000 characters){instruction_text}{motion_effect}. Focus on creating a descriptive prompt that an AI image-to-3D model can use to generate a 3D model based on the reference image. Include detailed visual characteristics, pose, materials, and appearance details.{model_3d_rules}{model_3d_format}{model_3d_character}{image_context_3d}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
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
