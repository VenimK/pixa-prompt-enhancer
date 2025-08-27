
document.addEventListener('DOMContentLoaded', () => {
    // --- Element References ---
    const enhanceButton = document.getElementById('enhance-button');
    const analyzeButton = document.getElementById('analyze-button');
    const imageUpload = document.getElementById('image-upload');
    const promptInput = document.getElementById('prompt-input');
    const promptTypeSelect = document.getElementById('prompt-type-select');
    const styleSelect = document.getElementById('style-select');
    const cinematographySelect = document.getElementById('cinematography-select');
    const lightingSelect = document.getElementById('lighting-select');
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const imageResultContainer = document.getElementById('image-result-container');
    const imagePreview = document.getElementById('image-preview');
    const imageDescriptionText = document.getElementById('image-description-text');
    const scrollToTopBtn = document.getElementById('scroll-to-top-btn');

    // --- Event Listener for Image Analysis ---
    analyzeButton.addEventListener('click', async () => {
        const file = imageUpload.files[0];
        if (!file) {
            alert('Please select an image file first.');
            return;
        }

        imageResultContainer.style.display = 'flex';
        imageDescriptionText.innerText = 'Analyzing image...';
        imagePreview.src = URL.createObjectURL(file); // Show preview immediately

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/analyze-image', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Image analysis failed.');
            }

            const data = await response.json();
            imageDescriptionText.innerText = data.description;
        } catch (error) {
            imageDescriptionText.innerText = 'Error: ' + error.message;
        }
    });

    // --- Event Listener for Prompt Enhancement ---
    enhanceButton.addEventListener('click', async () => {
        const prompt = promptInput.value;
        const promptType = promptTypeSelect.value;
        const style = styleSelect.value;
        const cinematography = cinematographySelect.value;
        const lighting = lightingSelect.value;
        const imageDescription = imageDescriptionText.innerText;

        if (!prompt) {
            alert('Please enter a prompt idea.');
            return;
        }

        resultContainer.style.display = 'block';
        resultText.innerText = 'Enhancing...';

        try {
            const response = await fetch('/enhance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    prompt_type: promptType,
                    style: style,
                    cinematography: cinematography,
                    lighting: lighting,
                    image_description: (imageDescription && !imageDescription.startsWith('Analyzing')) ? imageDescription : ''
                }),
            });

            if (!response.ok) {
                throw new Error('Something went wrong with the request.');
            }

            const data = await response.json();
            resultText.innerText = data.enhanced_prompt;
        } catch (error) {
            resultText.innerText = 'Error: ' + error.message;
        }
    });

    // --- Scroll-to-Top Button Logic ---
    window.onscroll = () => {
        if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
            scrollToTopBtn.style.display = "block";
        } else {
            scrollToTopBtn.style.display = "none";
        }
    };

    scrollToTopBtn.addEventListener('click', () => {
        document.body.scrollTop = 0; // For Safari
        document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
    });
});
