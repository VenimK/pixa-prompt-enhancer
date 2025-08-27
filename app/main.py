from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import subprocess
import shutil
import os
import google.generativeai as genai
import PIL.Image

app = FastAPI()

# --- Setup ---
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="templates")
UPLOADS_DIR = "uploads"

# --- Pydantic Models ---
class EnhanceRequest(BaseModel):
    prompt: str
    prompt_type: str # VEO or WAN2
    style: str
    cinematography: str
    lighting: str
    image_description: str | None = None

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
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        if image_path:
            image = PIL.Image.open(image_path)
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the Gemini API: {e}"

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    api_key_set = "GOOGLE_API_KEY" in os.environ
    return templates.TemplateResponse("index.html", {"request": request, "api_key_set": api_key_set})

@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_image_endpoint(image: UploadFile = File(...)):
    if not os.path.exists(UPLOADS_DIR):
        os.makedirs(UPLOADS_DIR)
    
    file_path = os.path.join(UPLOADS_DIR, image.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    absolute_file_path = os.path.abspath(file_path)
    meta_prompt = "Analyze this image in detail. Provide a comprehensive description covering the main subject, setting, composition, colors, and any notable elements. Be descriptive and thorough."
    
    description = run_gemini(meta_prompt, image_path=absolute_file_path)
    return AnalyzeResponse(description=description)

@app.post("/enhance", response_model=EnhanceResponse)
async def enhance_prompt_endpoint(request: EnhanceRequest):
    # Build the shared instruction string
    instructions = []
    if request.style != "auto":
        instructions.append(f"in a {request.style.lower()} style")
    if request.cinematography != "auto":
        instructions.append(f"using a {request.cinematography.lower()} shot")
    if request.lighting != "auto":
        instructions.append(f"with {request.lighting.lower()} lighting")
    instruction_text = " " + " and ".join(instructions) if instructions else ""

    image_context = f" The user has provided a reference image described as: '{request.image_description}'." if request.image_description else ""

    # --- Logic to choose meta-prompt based on prompt_type ---
    if request.prompt_type == "VEO":
        meta_prompt = f"You are a creative assistant for the VEO text-to-video model. Expand the user's idea into a rich, cinematic prompt{instruction_text}. Describe the scene, subject, and action in a detailed paragraph.{image_context} Do not add conversational fluff. User's idea: '{request.prompt}'"
    
    elif request.prompt_type == "WAN2":
        if request.prompt:
            # Logic for when the user provides a specific animation idea
            meta_prompt = f"""You are a creative assistant for an Image-to-Video model like WAN2. Your task is to enhance the user's animation idea into a more detailed and vivid prompt.
- The prompt should be action-oriented, focusing on movement and transformation.
- Add specific, imaginative details to the user's core idea. For example, instead of 'the hand crawls', describe *how* it crawls (e.g., 'the hand crawls forward like a spider, its fingers digging into the floor').
- Do not add any conversational fluff.

User's Specifications:
- Reference Image: '{image_context}'
- Animation Idea: '{request.prompt}'
- Desired Style: '{instruction_text}'

Generate the enhanced and detailed WAN2 prompt now."""
        else:
            # Logic for when the user wants the AI to imply the animation
            meta_prompt = f"""You are a creative assistant for an Image-to-Video model like WAN2. Your task is to look at the description of a static image and invent a compelling but subtle animation to make it come alive.
- Identify the most likely elements for movement in the scene (e.g., hair, clothing, water, clouds, fire, trees).
- Create a prompt that describes this subtle animation in a detailed and vivid way.
- Do not add any conversational fluff.

User's Specifications:
- Reference Image: '{image_context}'
- Desired Style: '{instruction_text}'

Generate an implied animation prompt now."""
    
    else:
        # Fallback for safety
        meta_prompt = f"Enhance this prompt: {request.prompt}"

    enhanced_prompt = run_gemini(meta_prompt)
    return EnhanceResponse(enhanced_prompt=enhanced_prompt)
