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
    model = genai.GenerativeModel('gemini-2.5-flash')
    try:
        if image_path:
            image = PIL.Image.open(image_path)
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the Gemini API: {e}"


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
async def enhance_prompt_endpoint(request: EnhanceRequest) -> EnhanceResponse:
    """Enhance the prompt using OpenAI's API."""
    
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
        if request.model == "flux":
            model_guidance = " For the Flux model, include technical photography details (camera, lens, lighting setup) and focus on photorealism with high detail."
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
        elif request.model == "nunchaku":
            model_guidance = " For the Nunchaku model, start with style keywords, focus on mood/atmosphere, and use artistic terminology rather than technical camera terms."

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
        
        elif any(material in prompt_lower for material in ["glass", "crystal", "transparent"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in glass and transparent materials. Your goal is to create a detailed prompt that emphasizes the translucent, reflective, and refractive properties of glass or crystal. Focus on how light interacts with the material - describe highlights, caustics, internal reflections, and the play of light through the transparent elements.{instruction_text}{model_guidance} Include specific details about transparency levels, colors within the glass, and textures.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(material in prompt_lower for material in ["metal", "metallic", "steel", "iron", "bronze", "copper", "gold", "silver"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in metallic materials. Your goal is to create a detailed prompt that emphasizes the distinctive properties of metal - its reflectivity, texture, patina, and interaction with light. Specify the type of metal (steel, iron, bronze, etc.), its finish (polished, brushed, hammered, etc.), and any weathering or aging effects.{instruction_text}{model_guidance} Include details about how light reflects off the metal surfaces and the surrounding environment.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(material in prompt_lower for material in ["wood", "wooden", "timber", "bark"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in wooden materials and textures. Your goal is to create a detailed prompt that emphasizes the organic qualities of wood - its grain patterns, color variations, knots, and textures. Specify the type of wood (oak, pine, mahogany, etc.), its finish (polished, weathered, raw, etc.), and any aging or weathering effects.{instruction_text}{model_guidance} Include details about how light interacts with the wood surfaces, highlighting the warmth and natural character of the material.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(material in prompt_lower for material in ["paper", "origami", "cardboard", "papier-mâché", "papier-mache"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in paper-based materials and art forms. Your goal is to create a detailed prompt that emphasizes the unique qualities of paper - its texture, folds, edges, and interaction with light. Specify the type of paper (origami paper, cardstock, newsprint, etc.), its texture (smooth, rough, crinkled, etc.), and any techniques used (folding, cutting, layering, etc.).{instruction_text}{model_guidance} Include details about the delicacy, precision, and craftsmanship involved in paper art, highlighting shadows, translucency, and dimensional aspects.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(material in prompt_lower for material in ["water", "liquid", "fluid", "splash", "droplet", "ocean", "river", "lake"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in water and fluid dynamics. Your goal is to create a detailed prompt that captures the dynamic, reflective, and transparent qualities of water. Focus on how light interacts with water - describe reflections, refractions, ripples, splashes, and the play of light on and through the water.{instruction_text}{model_guidance} Include specific details about water clarity, movement, surface tension effects, and the surrounding environment's reflection in the water.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(material in prompt_lower for material in ["fire", "flame", "burning", "ember", "smoke"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in fire and flame effects. Your goal is to create a detailed prompt that captures the dynamic, luminous, and transformative qualities of fire. Focus on the interplay of light and shadow created by flames - describe the colors within the fire (yellows, oranges, reds, blues), the intensity and behavior of the flames, and how they illuminate the surrounding environment.{instruction_text}{model_guidance} Include specific details about smoke patterns, ember glow, heat distortion effects, and the contrast between bright flames and darker surroundings.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(material in prompt_lower for material in ["ice", "frost", "frozen", "snow", "crystal"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in ice and frozen elements. Your goal is to create a detailed prompt that captures the crystalline, reflective, and translucent qualities of ice. Focus on how light interacts with frozen surfaces - describe the clarity, internal structures, fracture patterns, and the play of light through and across icy surfaces.{instruction_text}{model_guidance} Include specific details about textures (smooth, rough, crystalline), transparency levels, color tints within the ice, and the surrounding cold environment.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        # Photography style handling
        elif any(style in prompt_lower for style in ["macro", "close-up", "closeup", "microscopic"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in macro photography. Your goal is to create a detailed prompt that captures the intricate details visible only at extreme close-up. Focus on the minute textures, patterns, and structures that become visible at this scale. Specify macro-specific details like shallow depth of field (often f/2.8 or wider), the precise focal point, background bokeh quality, and the scale relationship between subject and frame.{instruction_text}{model_guidance} Include details about lighting that reveals texture (often diffused or ring lights) and the sense of discovery that macro photography provides.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(style in prompt_lower for style in ["aerial", "drone", "bird's eye", "birds eye", "top-down", "overhead"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in aerial photography. Your goal is to create a detailed prompt that captures the unique perspective of viewing subjects from above. Focus on patterns, layouts, and relationships between elements that are only visible from this vantage point. Specify aerial-specific details like altitude perspective (high altitude vs. low drone shot), angle (directly overhead vs. angled), and the sense of scale this perspective provides.{instruction_text}{model_guidance} Include details about lighting conditions, shadows cast (which are prominent in aerial views), atmospheric effects, and how the landscape or subject appears when flattened from above.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(style in prompt_lower for style in ["long exposure", "night photography", "light trail", "star trail", "astrophotography"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in long exposure photography. Your goal is to create a detailed prompt that captures the unique effects achieved through extended exposure times. Focus on the contrast between static elements (which remain sharp) and dynamic elements (which blur, streak, or create light trails). Specify long exposure-specific details like exposure duration effects, light painting possibilities, motion blur characteristics, and star trails or celestial movement if applicable.{instruction_text}{model_guidance} Include details about the necessary stability elements (tripod implied), ambient lighting conditions, and the surreal or ethereal quality that long exposures often create.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(style in prompt_lower for style in ["hdr", "high dynamic range", "bracketed", "tone mapped"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in HDR (High Dynamic Range) photography. Your goal is to create a detailed prompt that captures the expanded range of luminosity and detail possible with HDR techniques. Focus on scenes with extreme brightness variations where both shadow and highlight details are preserved. Specify HDR-specific characteristics like enhanced local contrast, detailed shadows without being muddy, highlights that retain color and detail, and the overall tonal richness.{instruction_text}{model_guidance} Include details about the natural or stylized tone mapping approach, color saturation level, and the balance between realistic and artistic HDR rendering.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        # Style-specific handling
        elif any(style in prompt_lower for style in ["cyberpunk", "cyber", "neon", "futuristic"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in cyberpunk aesthetics. Your goal is to create a detailed prompt that captures the essence of cyberpunk - a high-tech, low-life dystopian future with neon lights, urban decay, advanced technology, and gritty atmosphere. Include specific cyberpunk elements like holographic displays, augmented humans, corporate megastructures, and the contrast between cutting-edge technology and societal decline.{instruction_text}{model_guidance} Focus on the distinctive lighting with neon colors, lens flares, reflective surfaces, and rain-slicked streets.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(style in prompt_lower for style in ["fantasy", "magical", "medieval", "mythical"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in fantasy imagery. Your goal is to create a detailed prompt that brings to life a rich fantasy world with magical elements, mythical creatures, enchanted landscapes, and otherworldly atmosphere. Include specific fantasy tropes and elements appropriate to the scene while avoiding clichés.{instruction_text}{model_guidance} Focus on creating a sense of wonder and the supernatural through lighting, color palette, and atmospheric effects.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        # Artistic reference handling
        elif "like" in prompt_lower and any(artist_term in prompt_lower for artist_term in ["artist", "style of", "artwork", "painting", "illustration"]):
            # Extract potential artist or art style references
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in artistic style emulation. Your goal is to create a detailed prompt that captures the essence of the referenced artistic style or artist's work. Analyze the user's request for specific artists, art movements, or styles mentioned. For each artistic reference, include key characteristics such as: color palette, brushwork/technique, composition tendencies, subject matter treatment, lighting approach, and distinctive stylistic elements.{instruction_text}{model_guidance} Make sure to maintain the integrity of the referenced style while applying it to the user's specific subject matter.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(art_movement in prompt_lower for art_movement in ["impressionism", "cubism", "surrealism", "renaissance", "baroque", "romanticism", "expressionism", "pop art", "minimalism", "abstract", "art nouveau", "art deco"]):
            # Art movement specific guidance
            art_movement = next((movement for movement in ["impressionism", "cubism", "surrealism", "renaissance", "baroque", "romanticism", "expressionism", "pop art", "minimalism", "abstract", "art nouveau", "art deco"] if movement in prompt_lower), "")
            
            art_movement_guidance = {
                "impressionism": "loose brushwork, emphasis on light and its changing qualities, vibrant colors, outdoor scenes, capturing fleeting moments, visible brushstrokes, and a focus on the effects of light and atmosphere over detailed rendering",
                "cubism": "fragmented subjects, geometric forms, multiple viewpoints shown simultaneously, flattened perspective, limited color palette, and analytical deconstruction of forms",
                "surrealism": "dreamlike imagery, unexpected juxtapositions, hyper-realistic rendering of impossible scenes, psychological themes, and elements of the subconscious",
                "renaissance": "perfect proportion, perspective, harmony, balance, classical themes, religious imagery, realistic human anatomy, sfumato technique, and rich symbolism",
                "baroque": "dramatic contrasts between light and shadow (chiaroscuro), dynamic compositions, rich colors, grandeur, emotional intensity, and ornate details",
                "romanticism": "emphasis on emotion, individualism, nature as a powerful force, dramatic landscapes, exotic settings, and heightened drama",
                "expressionism": "distorted forms, exaggerated features, intense colors, emotional impact over physical reality, and subjective perspective",
                "pop art": "bold colors, sharp lines, reproduction of popular imagery, consumer culture references, comic book aesthetics, and mass production techniques",
                "minimalism": "extreme simplicity, geometric abstraction, clean lines, limited color palette, repetition, and absence of decorative elements",
                "abstract": "non-representational forms, emphasis on color, line, and form rather than recognizable subject matter, and visual language that exists independently from visual references",
                "art nouveau": "organic, flowing curvilinear forms, nature-inspired motifs (especially plants and flowers), ornamental quality, asymmetrical designs, and elegant decorative elements",
                "art deco": "bold geometric patterns, symmetry, streamlined forms, rich colors, luxurious materials, stylized representations, and celebration of modern technology"
            }
            
            style_guidance = art_movement_guidance.get(art_movement, "distinctive stylistic elements")
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in the {art_movement} art movement. Your goal is to create a detailed prompt that authentically captures the essence of {art_movement} characterized by {style_guidance}. Apply these stylistic elements to the user's subject matter while maintaining the integrity of the art movement.{instruction_text}{model_guidance} Include specific visual details that would make this image recognizably in the {art_movement} style.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        # Composition guidance
        elif any(comp in prompt_lower for comp in ["portrait", "headshot", "face", "closeup"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in portrait photography. Your goal is to create a detailed prompt for a compelling portrait that captures the subject's essence. Focus on facial features, expression, emotion, and gaze direction. Specify portrait-specific details like framing (tight headshot, half-body, etc.), focal length (85-135mm for flattering perspective), depth of field (often shallow with background bokeh), and lighting setup (Rembrandt, butterfly, split, etc.).{instruction_text}{model_guidance} Include details about the background context and how it relates to the subject.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        elif any(comp in prompt_lower for comp in ["landscape", "vista", "panorama", "scenery"]):
            meta_prompt = f"You are a creative assistant for a text-to-image model specializing in landscape photography. Your goal is to create a detailed prompt for a breathtaking landscape that emphasizes scale, depth, and atmosphere. Focus on the key landscape elements (mountains, forests, water bodies, skies), time of day, weather conditions, and seasonal aspects. Specify landscape-specific details like foreground interest, middle ground, and background layers to create depth.{instruction_text}{model_guidance} Include details about atmospheric perspective, the quality of light across the scene, and any dynamic elements.{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
        
        # Default case with enhanced guidance
        else:
            meta_prompt = f"You are a creative assistant for a text-to-image model. Your goal is to expand the user's idea into a rich, descriptive prompt suitable for generating a static image{instruction_text}.{model_guidance} Focus on the visual details of the scene, subject, and atmosphere. Be specific about composition (rule of thirds, leading lines, framing), perspective (eye level, bird's eye, worm's eye), depth (foreground, middle ground, background elements), and the quality of light (direction, color, intensity, shadows).{image_context}{text_emphasis} Do not add conversational fluff. User's idea: '{request.prompt}'"
    
    else:
        # Fallback for safety
        meta_prompt = f"Enhance this prompt: {request.prompt}"

    enhanced_prompt = run_gemini(meta_prompt)
    
    # Apply length limits based on model type
    limited_prompt = limit_prompt_length(enhanced_prompt, request.prompt_type)
    
    return EnhanceResponse(enhanced_prompt=limited_prompt)
