# PiXa Prompt Enhancer

Pixa Prompt Enhancer is a web-based application that helps you create rich and detailed prompts for text-to-image, text-to-video, and image-to-video models. It uses the Google Gemini API to enhance your ideas with cinematic styles, camera shots, and lighting effects. You can also upload a reference image to guide the prompt generation process.

## Features

-   **Three Prompting Modes:** Supports Image (Text-to-Image), VEO (Text-to-Video), and WAN2 (Image-to-Video) prompt enhancement.
-   **Creative Controls:** Fine-tune your prompts with various styles, cinematography techniques, and lighting options.
-   **Image-to-Prompt:** Upload an image and the application will analyze it to provide a descriptive starting point for your prompt.
-   **Web Interface:** A simple and intuitive web interface for easy prompt creation.

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
    # venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

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
    uvicorn app.main:app --reload
    ```

2.  **Open your browser:**
    Navigate to `http://127.0.0.1:8000` in your web browser.

3.  **How to use the application:**
    -   **Optional: Analyze an Image:**
        -   Drop an image or click to upload a reference image.
        -   Click the "Analyze Image" button to get a description of the image. This description will be used to enhance your prompt.
    -   **Enhance Your Prompt:**
        -   Select the prompt type (Image, VEO, or WAN2).
        -   Choose an AI model (for Image prompts).
        -   Enter your base prompt in the text area.
        -   Choose your desired style, cinematography, and lighting from the dropdown menus.
        -   Optionally add text emphasis with advanced formatting options.
        -   For WAN2, select motion effects if desired.
        -   Click the "Enhance Prompt" button.
        -   The enhanced prompt will be displayed in the "Enhanced Prompt" section.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
