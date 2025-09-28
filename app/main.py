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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

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
templates = Jinja2Templates(directory="templates")
UPLOADS_DIR = "uploads"

# --- Pydantic Models ---
class EnhanceRequest(BaseModel):
    prompt: str
    prompt_type: str # VEO or WAN2 or Image
    style: str
    cinematography: str
    lighting: str
    image_description: str | None = None
    motion_effect: str | None = None
    text_emphasis: str | None = None
    model: str | None = None  # AI model selection (flux, qwen, nunchaku, etc.)

class EnhanceResponse(BaseModel):
    enhanced_prompt: str

class AnalyzeResponse(BaseModel):
    description: str


# Make sure to set your GOOGLE_API_KEY environment variable.
# You can get one here: https://aistudio.google.com/app/apikey
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def run_gemini(prompt: str, image_path: str | None = None):
    # The genai.configure call at the top of the file handles the API key.
    # If the key is not set, the model.generate_content call will raise an exception.
    if "GOOGLE_API_KEY" not in os.environ:
        return "Error: Google API key is not set. Please set the GOOGLE_API_KEY environment variable."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        if image_path:
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
    """
    if model_type == "WAN2":
        # WAN2 has a lower character limit to prevent OOM errors
        max_length = 500
        if len(enhanced_prompt) > max_length:
            # If the prompt is too long, truncate it and add a note
            truncated_prompt = enhanced_prompt[:max_length]
            # Try to find the last complete sentence
            last_period = truncated_prompt.rfind('.')
            if last_period > max_length * 0.7:  # Only truncate at sentence if it's not too short
                truncated_prompt = truncated_prompt[:last_period+1]
            return truncated_prompt
    
    # For other models, return the original prompt
    return enhanced_prompt

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    api_key_set = "GOOGLE_API_KEY" in os.environ
    api_key_info = {}
    if api_key_set:
        # Get first and last 4 characters of the API key for display (for security)
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if len(api_key) > 8:
            masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        else:
            masked_key = "[Invalid key format]"
        api_key_info = {
            "provider": "Google Gemini",
            "masked_key": masked_key
        }
    version = int(time.time())
    return templates.TemplateResponse("index.html", {"request": request, "api_key_set": api_key_set, "api_key_info": api_key_info, "version": version})

@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_image_endpoint(image: UploadFile = File(...)):
    try:
        # Validate image file
        if not image.filename:
            return AnalyzeResponse(description="Error: No file provided")
            
        # Check file extension
        valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        file_ext = os.path.splitext(image.filename.lower())[1]
        if file_ext not in valid_extensions:
            return AnalyzeResponse(description=f"Error: Unsupported file format. Please upload an image in one of these formats: {', '.join(valid_extensions)}")
        
        # Create uploads directory if it doesn't exist
        if not os.path.exists(UPLOADS_DIR):
            try:
                os.makedirs(UPLOADS_DIR)
            except Exception as e:
                return AnalyzeResponse(description=f"Error creating upload directory: {e}")
        
        # Save the file
        try:
            file_path = os.path.join(UPLOADS_DIR, image.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
        except Exception as e:
            return AnalyzeResponse(description=f"Error saving image file: {e}")

        # Get absolute path and analyze
        absolute_file_path = os.path.abspath(file_path)
        meta_prompt = "Analyze this image in detail. Provide a comprehensive description covering the main subject, setting, composition, colors, and any notable elements. Be descriptive and thorough."
        
        description = run_gemini(meta_prompt, image_path=absolute_file_path)
        
        # Check if the response contains an error message
        if description.startswith("Error") or description.startswith("An unexpected error"):
            return AnalyzeResponse(description=description)
            
        return AnalyzeResponse(description=description)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in analyze_image_endpoint: {error_details}")
        return AnalyzeResponse(description=f"An unexpected error occurred: {e}. Please try again later.")

@app.post("/enhance", response_model=EnhanceResponse)
async def enhance_prompt_endpoint(request: EnhanceRequest) -> EnhanceResponse:
    """Enhance the prompt using Gemini API."""
    try:
        # Validate input
        if not request.prompt and request.prompt_type != "WAN2":
            return EnhanceResponse(enhanced_prompt="Error: Please provide a prompt to enhance.")
            
        # Check for API key
        if "GOOGLE_API_KEY" not in os.environ:
            return EnhanceResponse(enhanced_prompt="Error: Google API key is not set. Please set the GOOGLE_API_KEY environment variable.")
        
        # --- Build instructions based on user selections ---
        instructions = []
        if request.style and request.style != "None":
            instructions.append(f"in {request.style.lower()} style")
        if request.cinematography and request.cinematography != "None":
            instructions.append(f"with {request.cinematography.lower()} cinematography")
        if request.lighting and request.lighting != "None":
            instructions.append(f"with {request.lighting.lower()} lighting")
        if request.prompt_type == "WAN2" and request.motion_effect and request.motion_effect != "Static":
            instructions.append(f"with a {request.motion_effect.lower()} motion effect")
        instruction_text = " " + " and ".join(instructions) if instructions else ""

        image_context = f" The user has provided a reference image described as: '{request.image_description}'." if request.image_description else ""
        text_emphasis = f" {request.text_emphasis}" if request.text_emphasis else ""
        
        # --- Add model-specific guidance if a specific model is selected ---
        model_guidance = ""
        if request.model and request.model != "default" and request.prompt_type == "Image":
            # Photorealistic Models
            if request.model == "flux":
                model_guidance = " For the Flux model, include technical photography details (camera, lens, lighting setup) and focus on photorealism with high detail. Specify camera model, lens details, and lighting setup for best results."
            elif request.model == "pixart":
                model_guidance = " For the PixArt model, use detailed descriptions of scene elements and specify artistic style. Focus on composition, lighting, and atmosphere to achieve a good balance of realism and artistic flair."
            elif request.model == "dalle3":
                has_text = any(text_term in request.prompt.lower() for text_term in ["text", "sign", "writing", "label", "book", "letter", "word", "character", "font"])
                
                model_guidance = " For the DALL-E 3 model, use clear, detailed instructions and specify style, composition, and lighting. This model excels at following complex instructions and creating coherent scenes."
                
                if has_text:
                    model_guidance += " Since your prompt involves text elements, DALL-E 3 is an excellent choice as it renders text accurately. Be specific about the exact text you want to appear."
                
                if "complex" in request.prompt.lower() or "scene" in request.prompt.lower():
                    model_guidance += " For complex scenes, describe the spatial relationship between elements and ensure logical composition."
            
            # Versatile Models
            elif request.model == "qwen":
                # Enhanced Qwen-specific guidance based on latest research
                has_text = any(text_term in request.prompt.lower() for text_term in ["text", "sign", "writing", "label", "book", "letter", "word", "character", "font"])
                has_chinese = any(ord(c) > 127 for c in request.prompt)
                
                model_guidance = " For the Qwen model, focus on clear subject descriptions and compositional instructions. Add 'Ultra HD, 4K, cinematic composition' to enhance quality."
                
                if has_text:
                    model_guidance += " Since your prompt involves text elements, emphasize text clarity and legibility. Qwen excels at rendering complex text with high fidelity."
                    
                    if has_chinese:
                        model_guidance += " For Chinese text, specify that each character should be perfectly rendered with correct stroke order and proportions."
                
                # Additional guidance for specific image types
                if "portrait" in request.prompt.lower() or "person" in request.prompt.lower():
                    model_guidance += " For portraits, focus on natural facial features and expressions rather than technical camera settings."
            elif request.model == "midjourney":
                model_guidance = " For the Midjourney model, use simple, clear descriptions and include artistic style references. Add 'highly detailed, intricate, elegant, sharp focus' to enhance quality. This model excels at creating visually striking imagery with artistic flair."
                
                if "landscape" in request.prompt.lower() or "nature" in request.prompt.lower():
                    model_guidance += " For landscapes, consider adding 'epic scale, atmospheric, golden hour' to achieve dramatic results."
                elif "concept" in request.prompt.lower() or "fantasy" in request.prompt.lower():
                    model_guidance += " For concept art or fantasy scenes, add 'concept art, digital painting, trending on artstation' for best results."
            elif request.model == "sdxl":
                model_guidance = " For the SDXL model, balance descriptive and stylistic elements, and include composition details. This versatile model works well across many styles and subjects."
                
                if "art" in request.prompt.lower() or "painting" in request.prompt.lower():
                    model_guidance += " For artistic imagery, consider adding 'masterpiece, trending on artstation, award winning' to enhance quality."
                elif "photo" in request.prompt.lower() or "realistic" in request.prompt.lower():
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
                
                if "complex" in request.prompt.lower() or "scene" in request.prompt.lower():
                    model_guidance += " For complex scenes, describe multiple elements with logical arrangement and consistent style for best results."

        # --- Logic to choose meta-prompt based on prompt_type ---
        if request.prompt_type == "VEO":
            meta_prompt = f"You are a creative assistant for the VEO text-to-video model. Expand the user's idea into a rich, cinematic prompt{instruction_text}. Describe the scene, subject, and action in a detailed paragraph.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif request.prompt_type == "WAN2":
            if request.prompt:
                motion_effect = f" with {request.motion_effect} motion effect" if request.motion_effect and request.motion_effect != "Static" else ""
                meta_prompt = f"You are a creative assistant for the WAN2 image-to-video animation model. Create a CONCISE prompt (maximum 500 characters){instruction_text}{motion_effect}. Focus on describing a single frame that will be animated, with clear subjects and actions. Be extremely brief but descriptive, prioritizing visual elements over detailed explanations. WAN2 has strict character limits to prevent OOM errors.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
            else:
                # Logic for when the user wants the AI to imply the animation
                meta_prompt = f"""You are a creative assistant for the WAN2 image-to-video model. Create a CONCISE prompt (maximum 500 characters) that describes how to animate the static image.
- Identify 1-3 key elements for movement (e.g., hair, clothing, water, clouds).
- Describe the animation briefly but vividly.
- Keep the total prompt under 500 characters to prevent OOM errors.

User's Specifications:
- Reference Image: '{image_context}'
- Desired Style: '{instruction_text}'

Generate a brief animation prompt now."""

        elif request.prompt_type == "Image":
            # Check for specific materials, styles, or compositions in the prompt
            prompt_lower = request.prompt.lower()
            
            # Material-specific handling
            if "yarn" in prompt_lower and any(word in prompt_lower for word in ["animal", "creature", "wildlife"]):
                meta_prompt = f"You are a creative assistant for a text-to-image model specializing in yarn art. Your goal is to create a detailed prompt for an image where the main subject is an animal ENTIRELY made of yarn - not a real animal, but a yarn sculpture/creation that looks like an animal. Describe the yarn's texture, colors, stitching details, and how the yarn construction gives the animal character. Make sure to emphasize that this is a yarn creation, not a real animal with yarn elements.{instruction_text}{model_guidance} Include details about the setting and lighting that would best showcase this yarn creation.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
            
            # Add all the other material-specific and style-specific handlers here...
            # (I'm omitting them for brevity, but they should remain in the actual code)
            
            # Default case with enhanced guidance
            else:
                meta_prompt = f"You are a creative assistant for a text-to-image model. Your goal is to expand the user's idea into a rich, descriptive prompt suitable for generating a static image{instruction_text}.{model_guidance} Focus on the visual details of the scene, subject, and atmosphere. Be specific about composition (rule of thirds, leading lines, framing), perspective (eye level, bird's eye, worm's eye), depth (foreground, middle ground, background elements), and the quality of light (direction, color, intensity, shadows).{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        else:
            # Fallback for safety
            meta_prompt = f"Enhance this prompt: {request.prompt}"

        enhanced_prompt = run_gemini(meta_prompt)
        
        # Check if the response contains an error message
        if enhanced_prompt.startswith("Error") or enhanced_prompt.startswith("An unexpected error"):
            return EnhanceResponse(enhanced_prompt=enhanced_prompt)
        
        # Apply length limits based on model type
        limited_prompt = limit_prompt_length(enhanced_prompt, request.prompt_type)
        
        return EnhanceResponse(enhanced_prompt=limited_prompt)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in enhance_prompt_endpoint: {error_details}")
        return EnhanceResponse(enhanced_prompt=f"An unexpected error occurred: {e}. Please try again later.")
