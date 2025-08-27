from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import subprocess
import shutil
import os

app = FastAPI()

# --- Setup ---
app.mount("/static", StaticFiles(directory="app/static"), name="static")
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

# --- Helper Functions ---
def get_html_response():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

def run_gemini_cli(prompt: str, image_path: str | None = None):
    if image_path:
        # The @ command for file inclusion should precede the text prompt.
        full_prompt = f"@{image_path} {prompt}"
        command = ["gemini", full_prompt]  # No -p flag needed for this format
    else:
        command = ["gemini", "-p", prompt]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except FileNotFoundError:
        return "Error: The 'gemini' command was not found. Please ensure gemini-cli is installed."
    except subprocess.CalledProcessError as e:
        return f"Error executing gemini-cli: {e.stderr}"

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return get_html_response()

@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_image_endpoint(image: UploadFile = File(...)):
    if not os.path.exists(UPLOADS_DIR):
        os.makedirs(UPLOADS_DIR)
    
    file_path = os.path.join(UPLOADS_DIR, image.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    absolute_file_path = os.path.abspath(file_path)
    meta_prompt = "Analyze this image and provide a concise, one-sentence description of its content. Do not add any conversational fluff."
    
    description = run_gemini_cli(meta_prompt, image_path=absolute_file_path)
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

    enhanced_prompt = run_gemini_cli(meta_prompt)
    return EnhanceResponse(enhanced_prompt=enhanced_prompt)
