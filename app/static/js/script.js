document.addEventListener('DOMContentLoaded', () => {
    // --- Toast Notification Logic ---
    function showToast(message, type = 'info', duration = 3000) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast-notification ${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('show');
        }, 100); // Delay to allow CSS transition

        setTimeout(() => {
            toast.classList.remove('show');
            // Remove the element after the transition is complete
            toast.addEventListener('transitionend', () => toast.remove());
        }, duration);
    }
    // --- Theme Switcher Logic ---
    const themeToggle = document.getElementById('checkbox');

    function applySavedTheme() {
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        document.body.classList.toggle('dark-mode', isDarkMode);
        if (themeToggle) {
            themeToggle.checked = isDarkMode;
        }
    }

    if (themeToggle) {
        themeToggle.addEventListener('change', function() {
            document.body.classList.toggle('dark-mode');
            localStorage.setItem('darkMode', this.checked);
        });
    }

    applySavedTheme();

    // --- Autosave Logic ---
    const formInputs = {
        prompt: promptInput,
        promptType: promptTypeSelect,
        style: styleSelect,
        cinematography: cinematographySelect,
        lighting: lightingSelect,
        motionEffect: motionEffectSelect
    };

    function saveInputs() {
        const dataToSave = {};
        for (const key in formInputs) {
            if (formInputs[key]) {
                dataToSave[key] = formInputs[key].value;
            }
        }
        localStorage.setItem('userInputs', JSON.stringify(dataToSave));
    }

    function loadInputs() {
        const savedData = localStorage.getItem('userInputs');
        if (savedData) {
            const parsedData = JSON.parse(savedData);
            for (const key in parsedData) {
                if (formInputs[key] && parsedData[key]) {
                    formInputs[key].value = parsedData[key];
                }
            }
            // Manually trigger change event to show/hide motion effect selector
            promptTypeSelect.dispatchEvent(new Event('change'));
        }
    }

    // Add event listeners to all inputs to save on change
    for (const key in formInputs) {
        if (formInputs[key]) {
            formInputs[key].addEventListener('input', saveInputs);
            formInputs[key].addEventListener('change', saveInputs);
        }
    }

    loadInputs(); // Load saved inputs on page load


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
    const copyButton = document.getElementById('copy-button');
    const clearButton = document.getElementById('clear-button');
    const motionEffectSelect = document.getElementById('motion-effect-select');
    const motionEffectContainer = document.getElementById('motion-effect-selector-container');
    const clearImageButton = document.getElementById('clear-image-button');

    // --- Event Listeners ---
    if (analyzeButton) {
        analyzeButton.addEventListener('click', async () => {
            const file = imageUpload.files[0];
            if (!file) {
                showToast('Please select an image file first.', 'error');
                return;
            }

            showToast('Analyzing image...', 'info');
            imageResultContainer.style.display = 'flex';
            imageDescriptionText.innerHTML = '<div class="loader"></div>'; // Show loader in description
            imagePreview.src = URL.createObjectURL(file);
            clearImageButton.style.display = 'inline-block';

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
                showToast('Image analyzed successfully!', 'success');
            } catch (error) {
                imageDescriptionText.innerText = '';
                showToast('Image analysis failed.', 'error');
            }
        });
    }

    if (promptTypeSelect) {
        promptTypeSelect.addEventListener('change', () => {
            if (promptTypeSelect.value === 'WAN2') {
                motionEffectContainer.style.display = 'flex';
            } else {
                motionEffectContainer.style.display = 'none';
            }
        });
    }

    if (enhanceButton) {
        enhanceButton.addEventListener('click', async () => {
            const prompt = promptInput.value;
            const promptType = promptTypeSelect.value;
            const style = styleSelect.value;
            const cinematography = cinematographySelect.value;
            const lighting = lightingSelect.value;
            const imageDescription = imageDescriptionText.innerText;
            const motionEffect = motionEffectSelect.value;

            if (!prompt) {
                showToast('Please enter a prompt idea.', 'error');
                return;
            }

            // --- Start Loading ---
            enhanceButton.disabled = true;
            resultContainer.style.display = 'block';
            resultText.innerHTML = '<div class="loader"></div>';

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
                        image_description: (imageDescription && !imageDescription.startsWith('Analyzing')) ? imageDescription : '',
                        motion_effect: promptType === 'WAN2' ? motionEffect : null
                    }),
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: 'Something went wrong with the request.' }));
                    throw new Error(errorData.detail);
                }

                const data = await response.json();
                resultText.innerText = data.enhanced_prompt;
                showToast('Prompt enhanced successfully!', 'success');
            } catch (error) {
                resultText.innerHTML = ''; // Clear loader on error
                showToast(error.message, 'error');
            } finally {
                // --- End Loading ---
                enhanceButton.disabled = false;
            }
        });
    }

    if (clearImageButton) {
        clearImageButton.addEventListener('click', () => {
            imageUpload.value = ''; // Reset file input
            imageResultContainer.style.display = 'none';
            clearImageButton.style.display = 'none';
            imagePreview.src = '#';
            imageDescriptionText.innerText = '';
        });
    }

    if (clearButton) {
        clearButton.addEventListener('click', () => {
            promptInput.value = '';
            promptInput.focus();
            saveInputs(); // Clear saved data as well
        });
    }

    if (copyButton) {
        copyButton.addEventListener('click', () => {
            navigator.clipboard.writeText(resultText.innerText).then(() => {
                showToast('Copied to clipboard!', 'success');
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                showToast('Failed to copy text.', 'error');
            });
        });
    }

    if (scrollToTopBtn) {
        window.onscroll = () => {
            if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
                scrollToTopBtn.style.display = "block";
            } else {
                scrollToTopBtn.style.display = "none";
            }
        };

        scrollToTopBtn.addEventListener('click', () => {
            document.body.scrollTop = 0;
            document.documentElement.scrollTop = 0;
        });
    }
});
