# PiXa Prompt Enhancer

Pixa Prompt Enhancer is a web-based application that helps you create rich and detailed prompts for text-to-image, text-to-video, and image-to-video models. It uses the Google Gemini API to enhance your ideas with cinematic styles, camera shots, and lighting effects. You can also upload a reference image to guide the prompt generation process.

## Features

- **Multiple Prompting Modes**
  - Image (Text-to-Image)
  - VEO (Text-to-Video)
  - WAN2 (Image-to-Video)
  - LTX-2 (Text-to-Video with Synchronized Audio)

- **Creative Controls**
  - 100+ Artistic Styles
  - Cinematography Techniques
  - Lighting & Composition Options
  - Text Emphasis Control

- **Smart Features**
  - Image-to-Prompt Analysis
  - Two-Image Analysis (A/B) with Combined + Per-Image references
  - Real Audio Analysis (Tempo, Vocals, Energy, Danceability)
  - Audio-Video Synchronization for LTX-2
  - Style Combination
  - Prompt Optimization
  - Character Limit Management

- **User Experience**
  - Clean Web Interface
  - Responsive Design
  - Easy Copy/Paste
  - Dark/Light Mode

## 🎵 LTX-2 Audio Features

The LTX-2 mode includes advanced audio upload and analysis capabilities for perfect audio-video synchronization:

### Audio Upload & Analysis
- **Supported Formats**: MP3, WAV, M4A, OGG, FLAC (up to 50MB)
- **Real Audio Processing**: Uses librosa for professional audio analysis
- **Drag & Drop Interface**: Easy file upload with preview
- **Audio Preview**: Built-in audio player with controls

### Smart Audio Detection
- **Tempo Detection**: Real BPM analysis (e.g., 128.5 BPM)
- **Vocal Detection**: Spectral analysis with confidence scores
- **Energy Level Analysis**: Low/Medium/High/Very High classification
- **Danceability Calculation**: Beat consistency and rhythm analysis
- **Mood Detection**: Energetic/Calm/Emotional classification

### Synchronized Performance Generation
- **Lip-Sync Instructions**: Precise mouth movement matching vocals
- **Dance Synchronization**: Movements matched to audio rhythm and tempo
- **Performance Styles**: Singing, dancing, or speaking based on audio content
- **Mood-Based Instructions**: Energy and emotion matched to audio characteristics

### LTX-2 Controls
- **Audio Generation**: Enable/disable synchronized audio
- **Resolution Options**: 4K, 1080p, 720p output quality
- **Character Limit**: 2500 characters for optimal LTX-2 performance

#### Example Workflow
1. Upload reference image of character
2. Upload audio file (song/music)
3. Select "Video (LTX-2)" mode
4. Write prompt: "Character from reference image sings and dances to uploaded music"
5. Enhanced prompt includes: "singing performance with vocals (0.8 confidence) with fast tempo (128.5 BPM) highly danceable rhythm creating high energy and excitement featuring vocal performance that should be precisely lip-synced"

## 🎨 Available Styles

Pixa Prompt Enhancer offers a wide variety of artistic styles to enhance your prompts. Here's a categorized overview of available styles:

<details>
<summary><strong>🎭 Thematic Styles</strong> (Click to expand)</summary>

- **Animated Series** - Vibrant 2D animation with expressive characters
- **Classic Sitcom** - 90s multi-camera sitcom aesthetic
- **Crime Drama** - Moody, neo-noir atmosphere
- **Cyberpunk Noir** - Neon-drenched futuristic cityscapes
- **Fantasy Series** - Epic high-fantasy with detailed world-building
- **Muppet Show** - Whimsical felt puppets and theatrical lighting
- **Vaporwave** - 80s/90s retro-futurism with pastel gradients

</details>

<details>
<summary><strong>🎨 Artistic Mediums</strong> (Click to expand)</summary>

- **Biomechanical** - Organic-mechanical fusion
- **Claymation** - Stop-motion clay animation
- **Holographic** - Iridescent, light-refracting effects
- **Neon Sign** - Glowing tube lighting and bokeh
- **Paper Cutout** - Layered paper art with depth
- **Stained Glass** - Vibrant cathedral window aesthetic
- **Watercolor** - Visible paper texture and colorwash

</details>

<details>
<summary><strong>📸 Photo & Cinematic</strong> (Click to expand)</summary>

- **Film Noir** - Classic black and white cinematography
- **HDR Photo** - High dynamic range photography
- **Long Exposure** - Light trails and motion blur
- **Macro Photography** - Extreme close-up details
- **Neon Noir** - Cyberpunk-inspired night photography
- **Tilt-Shift** - Miniature effect photography

</details>

<details>
<summary><strong>🎮 Game & Digital Art</strong> (Click to expand)</summary>

- **8-Bit Pixel Art** - Retro video game aesthetic
- **3D Model** - Professional 3D rendering
- **Anime** - Studio-quality anime style
- **Cel Shaded** - Comic book/cartoon rendering
- **Digital Painting** - Painterly digital artwork
- **Low Poly** - Geometric, angular 3D models

</details>

*Note: This is just a selection of available styles. The full list includes over 100 unique styles that can be combined for endless creative possibilities.*

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VenimK/pixa-prompt-enhancer.git
    cd pixa-prompt-enhancer
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate

    # On Windows, use:
    python3 -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    **For LTX-2 Audio Features** (optional but recommended):
    ```bash
    pip install librosa numpy scipy
    ```
    *Note: librosa enables real audio analysis (tempo detection, vocal detection, etc.). Without it, the system will fall back to filename-based analysis.*

4.  **Set up the API Key:**
    This application uses the Google Generative AI SDK and requires a `GOOGLE_API_KEY` to function.

1.  **Get an API Key:**
    -   Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to create your free API key.

2.  **Set the Environment Variable:**
    -   Create a `.env` file in the root directory of the project with the following content:
        ```
        GOOGLE_API_KEY=your_api_key_here
        ```
    -   Replace `your_api_key_here` with the key you obtained.
    -   Alternatively, you can set the environment variable in your terminal:
        ```bash
        export GOOGLE_API_KEY="YOUR_API_KEY_HERE"

        # On Windows Command Prompt, use:
        # set GOOGLE_API_KEY="YOUR_API_KEY_HERE"

        # On Windows PowerShell, use:
        # $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```
    -   **Note:** If using the terminal method, you must set this variable in the same terminal session where you run the application.

### Troubleshooting API Key Issues

If you receive an error like `400 API key not valid`, it usually means the environment variable is not set correctly. Here are a few things to check:

*   **Same Terminal:** Ensure you are running `uvicorn app.main:app --reload` in the *exact same* terminal window where you set the API key.
*   **No Extra Quotes:** When using `set GOOGLE_API_KEY=YOUR_API_KEY_HERE` in the Windows Command Prompt, do not include quotes around your key unless the key itself contains special characters.
*   **Copy-Paste Errors:** Double-check that you have copied the entire API key from Google AI Studio without any extra spaces or missing characters.

## Setup and Usage

1.  **Run the application:**
    ```bash
    uvicorn app.main:app --reload --log-level debug
    ```
    using other port
    uvicorn app.main:app --port 8001 --host 0.0.0.0 --reload --log-level debug

2.  **Open your browser:**
    Navigate to `http://127.0.0.1:8000` in your web browser.

3.  **How to use the application:**
    -   **Optional: Analyze Image(s):**
        -   Drag & drop or click to select up to 2 images.
        -   Use per-slot controls to Replace/Remove A or B. You can also add a second image later.
        -   Click "Analyze Image". The app returns a structured Combined analysis and labeled "Reference A" / "Reference B" summaries.
        -   When you replace or remove an image, analysis auto-refreshes to keep references in sync.
    -   **Optional: Upload Audio (LTX-2 only):**
        -   Select "Video (LTX-2)" as the prompt type.
        -   Audio upload section will appear automatically.
        -   Drag & drop or select an audio file (MP3, WAV, M4A, OGG, FLAC).
        -   System analyzes audio for tempo, vocals, energy, and danceability.
        -   Preview audio with built-in player controls.
    -   **Enhance Your Prompt:**
        -   Select the prompt type (Image, VEO, WAN2, or LTX-2).
        -   Choose an AI model (for Image prompts).
        -   Enter your base prompt in the text area.
        -   Choose your desired style, cinematography, and lighting from the dropdown menus.
        -   Optionally add text emphasis with advanced formatting options.
        -   For WAN2, select motion effects if desired.
        -   For LTX-2, configure audio generation and resolution settings.
        -   Click the "Enhance Prompt" button.
        -   The enhanced prompt will be displayed in the "Enhanced Prompt" section.

### Two-Image Workflow (A/B)

- **Two slots:** Reference A and Reference B with previews and per-slot Replace/Remove.
- **Drag & drop:** Add one or two images; a "+ Add second image" button appears when only A is filled.
- **Analysis output:**
  - Combined section summarizing overlaps, differences, style/technique, and notable details.
  - Separate "Reference A" and "Reference B" summaries for precise control.
- **Auto re-analyze:** Replacing/removing A or B re-runs analysis and updates the references.
- **Prompting tip:** Use the Combined reference for fusion tasks or pick A/B when you need faithful per-image cues. For product label tasks (e.g., wrap A onto B), specify "full-body shrink-sleeve, edge-to-edge, no aluminum visible" and add negatives like "no label gaps, no transparent ink, no warping".

### Wrapping Preset (A → B)

The app includes a **Wrapping Preset** panel in Step 2 with preset flows for:
- **Full wrap / livery** - Complete vehicle/object wrap
- **Partial panels** - Selected surfaces only
- **Decal / sticker** - Die-cut graphics
- **Product wrap (can/bottle/glass)** - Cylindrical object labeling
- **Logo / branding** - Logo placement with size/position/pattern controls

#### Options:
- **Finish:** Glossy vinyl, Matte vinyl, Metallic foil, Paper label, Shrink-wrap film
- **Scope Preset:** All surfaces, Full 360°, Front label only, Neck band, Cap/lid
- **Enforce Reference A palette** - Override target colors with source palette
- **Neutralize B base colors** - Remove original branding/stripes
- **Logo controls** (when Logo/branding selected):
  - **Placement:** Center, Top center, Bottom center, Left, Right, Custom
  - **Size:** Small, Medium, Large, Full coverage, Proportional
  - **Repeating pattern** - Uniform pattern vs single placement

Click **"Insert Wrapping Prompt"** to generate a structured prompt with PRIORITY constraints, palette enforcement, mapping rules, and negatives.

### Prompt Examples (Reference A onto Reference B)

#### Logo/Branding Examples

- **Single centered logo**
```text
PRIORITY: Use ONLY the Reference A palette on wrapped areas. OVERRIDE any original Reference B colors.

Palette (hard): red, white, black. No other hues.

Place the logo/branding from Reference A onto Reference B at centered on the main surface, medium size, clearly visible.

Maintain exact logo aspect ratio and proportions; no distortion, stretching, or warping; crisp edges and sharp details; correct perspective for surface angle; logo sits flat on surface or conforms to curvature naturally.

Photorealistic logo application with glossy vinyl finish; maintain Reference B lighting, shadows, and reflections; logo colors are vibrant and accurate.

Negatives: no color bleed from B, no banding, no artifacts, no partial recolor.
```

- **Repeating logo pattern**
```text
Apply the logo/branding from Reference A as a repeating pattern across Reference B, small and subtle.

Pattern repeats uniformly with consistent spacing; maintain logo aspect ratio and orientation; no distortion, warping, or perspective skew on individual logos; pattern follows surface curvature naturally; crisp edges and sharp details on each instance.

Photorealistic logo application with matte vinyl finish; maintain Reference B lighting, shadows, and reflections; logo colors are vibrant and accurate.
```

#### Product Wrap Examples

- **Full 360° can/bottle wrap**
```text
Transfer the design from Reference A onto the cylindrical surface of Reference B (can/bottle/glass), covering: full 360° circumference.

Wrap seamlessly around the circumference; correct radial perspective and distortion for curved surface; align logo/artwork to front-center; respect top and bottom rims; no visible seam on front view; no stretching or warping at edges.

Photorealistic product packaging with paper label with smooth adhesion; maintain Reference B lighting, reflections, and camera angle; label conforms perfectly to object shape.
```

- **Front label only**
```text
Transfer the design from Reference A onto the cylindrical surface of Reference B (can/bottle/glass), covering: front label/panel only, avoiding back seam.

Wrap seamlessly around the circumference; correct radial perspective and distortion for curved surface; align logo/artwork to front-center; respect top and bottom rims; no visible seam on front view; no stretching or warping at edges.

Photorealistic product packaging with metallic foil label; maintain Reference B lighting, reflections, and camera angle; label conforms perfectly to object shape.
```

### LTX-2 Audio-Video Examples

#### Character Singing with Audio

- **Singing Performance**
```text
Character from reference image performs authentic singing with visible lip-sync, facial expressions showing emotion, and breathing patterns that match the vocal performance. Use the uploaded audio as the soundtrack: singing performance with vocals (0.8 confidence) with fast tempo (128.5 BPM) highly danceable rhythm creating high energy and excitement featuring vocal performance that should be precisely lip-synced. Generate at 4K resolution with cinematic quality.

Focus on authentic singing performance with visible lip-sync, facial expressions showing emotion, and breathing patterns that match the vocal performance. Emphasize joyful, positive expressions and upbeat movements. Create performance with strong dance elements and rhythmic movements.
```

- **Dancing to Music**
```text
Character from reference image dances energetically to the uploaded music with synchronized movements. Use the uploaded audio as the soundtrack: instrumental dance music with fast tempo (140.2 BPM) highly danceable rhythm building high energy and excitement. Generate at 4K resolution with cinematic quality.

Focus on energetic, rapid dance movements synchronized to the fast audio rhythm with dynamic body motions, expressive gestures, and high-energy performance. Emphasize high-energy performance with dynamic movements and intense expressions. Create performance with strong dance elements and rhythmic movements.
```

- **Emotional Performance**
```text
Character from reference image delivers emotional performance with slow, expressive movements. Use the uploaded audio as the soundtrack: singing performance with vocals (0.9 confidence) with slow tempo (75.3 BPM) low danceability with emotional, expressive atmosphere featuring vocal performance that should be precisely lip-synced. Generate at 1080p resolution with cinematic quality.

Focus on authentic singing performance with visible lip-sync, facial expressions showing emotion, and breathing patterns that match the vocal performance. Emphasize gentle, controlled movements with calm expressions. Emphasize emotional facial expressions and dramatic body language. Focus more on emotional expression than complex dance movements.
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
