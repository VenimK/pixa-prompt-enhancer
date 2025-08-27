# PiXa Prompt Enhancer

Pixa Prompt Enhancer is a web-based application that helps you create rich and detailed prompts for text-to-video and image-to-video models like VEO and WAN2. It uses the Gemini CLI to enhance your ideas with cinematic styles, camera shots, and lighting effects. You can also upload a reference image to guide the prompt generation process.

## Features

-   **Two Prompting Modes:** Supports both VEO (Text-to-Video) and WAN2 (Image-to-Video) prompt enhancement.
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
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Gemini CLI:**
    This application requires the Gemini CLI to be installed and authenticated. Please follow the official instructions to install and configure the Gemini CLI.

## Setup and Usage

1.  **Run the application:**
    ```bash
    uvicorn app.main:app --reload
    ```

2.  **Open your browser:**
    Navigate to `http://127.0.0.1:8000` in your web browser.

3.  **How to use the application:**
    -   **Optional: Analyze an Image:**
        -   Click on "Choose File" to upload a reference image.
        -   Click the "Analyze Image" button to get a description of the image. This description will be used to enhance your prompt.
    -   **Enhance Your Prompt:**
        -   Select the prompt type (VEO or WAN2).
        -   Enter your base prompt in the text area.
        -   Choose your desired style, cinematography, and lighting from the dropdown menus.
        -   Click the "Enhance Prompt" button.
        -   The enhanced prompt will be displayed in the "Enhanced Prompt" section.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
