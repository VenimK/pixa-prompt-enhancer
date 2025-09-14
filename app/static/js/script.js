document.addEventListener('DOMContentLoaded', () => {
    // --- Element References ---
    const els = {
        // Buttons
        enhance: document.getElementById('enhance-button'),
        analyze: document.getElementById('analyze-button'),
        copy: document.getElementById('copy-button'),
        clear: document.getElementById('clear-button'),
        clearImage: document.getElementById('clear-image-button'),
        clearHistory: document.getElementById('clear-history'),
        shortcuts: document.getElementById('shortcuts-btn'),
        closeShortcuts: document.querySelector('.close-shortcuts'),
        scrollToTop: document.getElementById('scroll-to-top-btn'),
        textControlsToggle: document.getElementById('text-controls-toggle'),
        
        // Inputs and Selects
        imageUpload: document.getElementById('image-upload'),
        prompt: document.getElementById('prompt-input'),
        promptType: document.getElementById('prompt-type-select'),
        style: document.getElementById('style-select'),
        cinematography: document.getElementById('cinematography-select'),
        lighting: document.getElementById('lighting-select'),
        motionEffect: document.getElementById('motion-effect-select'),
        template: document.getElementById('template-select'),
        
        // Text Emphasis Controls
        textEmphasis: document.getElementById('text-emphasis-input'),
        textPosition: document.getElementById('text-emphasis-position'),
        textSize: document.getElementById('text-emphasis-size'),
        textStyle: document.getElementById('text-emphasis-style'),
        textColor: document.getElementById('text-emphasis-color'),
        textEffect: document.getElementById('text-emphasis-effect'),
        textMaterial: document.getElementById('text-emphasis-material'),
        textBackground: document.getElementById('text-emphasis-background'),
        textIntegration: document.getElementById('text-emphasis-integration'),
        
        // Containers and Text Elements
        result: document.getElementById('result-container'),
        resultText: document.getElementById('result-text'),
        imageResult: document.getElementById('image-result-container'),
        imagePreview: document.getElementById('image-preview'),
        imageDescription: document.getElementById('image-description-text'),
        motionEffectContainer: document.getElementById('motion-effect-selector-container'),
        historyPanel: document.querySelector('.history-panel'),
        historyList: document.getElementById('history-list'),
        shortcutsOverlay: document.getElementById('shortcuts-overlay'),
        charCounter: document.querySelector('.char-counter'),
        charCount: document.getElementById('char-count'),
        dropZone: document.getElementById('drop-zone'),
        
        // Progress Steps
        steps: document.querySelectorAll('.step'),
        step1: document.querySelector('.step[data-step="1"]'),
        step2: document.querySelector('.step[data-step="2"]'),
        
        // Theme
        themeToggle: document.getElementById('checkbox')
    };

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
        }, 100);

        setTimeout(() => {
            toast.classList.remove('show');
            toast.addEventListener('transitionend', () => toast.remove());
        }, duration);
    }

    // --- Theme Switcher Logic ---
    function applySavedTheme() {
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        document.body.classList.toggle('dark-mode', isDarkMode);
        if (els.themeToggle) {
            els.themeToggle.checked = isDarkMode;
        }
    }

    if (els.themeToggle) {
        els.themeToggle.addEventListener('change', function() {
            document.body.classList.toggle('dark-mode');
            localStorage.setItem('darkMode', this.checked);
        });
    }

    // --- Progress Steps Logic ---
    function updateProgress(stepNumber, status) {
        const step = document.querySelector(`.step[data-step="${stepNumber}"]`);
        if (!step) return;
        
        els.steps.forEach(s => s.classList.remove('active'));
        
        if (status === 'active') {
            step.classList.add('active');
            step.classList.remove('completed');
        } else if (status === 'completed') {
            step.classList.remove('active');
            step.classList.add('completed');
        }
    }

    // --- Event Listeners ---
    if (els.analyze) {
        els.analyze.addEventListener('click', async () => {
            updateProgress(1, 'active');
            const file = els.imageUpload.files[0];
            if (!file) {
                showToast('Please select an image file first.', 'error');
                return;
            }

            showToast('Analyzing image...', 'info');
            els.imageResult.style.display = 'flex';
            els.imageDescription.innerHTML = '<div class="loader"></div>';
            els.imagePreview.src = URL.createObjectURL(file);
            els.clearImage.style.display = 'inline-block';

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
                els.imageDescription.innerText = data.description;
                showToast('Image analyzed successfully!', 'success');
                updateProgress(1, 'completed');
                updateProgress(2, 'active');
            } catch (error) {
                els.imageDescription.innerText = '';
                showToast('Image analysis failed.', 'error');
            }
        });
    }

    if (els.enhance) {
        els.enhance.addEventListener('click', async () => {
            updateProgress(2, 'active');
            const prompt = els.prompt.value;
            const promptType = els.promptType.value;
            const style = els.style.value;
            const cinematography = els.cinematography.value;
            const lighting = els.lighting.value;
            const imageDescription = els.imageDescription.innerText;
            const motionEffect = els.motionEffect.value;
            const textToEmphasis = els.textEmphasis.value;
            const textPosition = els.textPosition.value;

            if (!prompt) {
                showToast('Please enter a prompt idea.', 'error');
                return;
            }

            els.enhance.disabled = true;
            els.result.style.display = 'block';
            els.resultText.innerHTML = '<div class="loader"></div>';

            // Prepare text emphasis details if provided
            let textEmphasisDetails = '';
            if (textToEmphasis) {
                // Get all text options
                const textSize = els.textSize ? els.textSize.value : '';
                const textStyle = els.textStyle ? els.textStyle.value : '';
                const textColor = els.textColor ? els.textColor.value : '';
                const textEffect = els.textEffect ? els.textEffect.value : '';
                const textMaterial = els.textMaterial ? els.textMaterial.value : '';
                const textBackground = els.textBackground ? els.textBackground.value : '';
                const textIntegration = els.textIntegration ? els.textIntegration.value : '';
                
                // Generate comprehensive text description
                textEmphasisDetails = getTextEmphasisDescription(
                    textToEmphasis, 
                    textPosition, 
                    textSize,
                    textStyle, 
                    textColor,
                    textEffect,
                    textMaterial,
                    textBackground,
                    textIntegration
                );
            }

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
                        motion_effect: promptType === 'WAN2' ? motionEffect : null,
                        text_emphasis: textEmphasisDetails
                    }),
                });

                if (!response.ok) {
                    throw new Error('Failed to enhance prompt.');
                }

                const data = await response.json();
                els.resultText.innerText = data.enhanced_prompt;
                showToast('Prompt enhanced successfully!', 'success');
                updateProgress(2, 'completed');
            } catch (error) {
                els.resultText.innerHTML = '';
                showToast(error.message, 'error');
            } finally {
                els.enhance.disabled = false;
            }
        });
    }

    if (els.promptType) {
        els.promptType.addEventListener('change', () => {
            if (els.promptType.value === 'WAN2') {
                els.motionEffectContainer.style.display = 'flex';
            } else {
                els.motionEffectContainer.style.display = 'none';
            }
        });
    }
    
    // --- Templates System ---
    const templates = {
        // Photography Templates
        portrait: {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Close Up',
            lighting: 'Studio',
            basePrompt: 'A professional portrait with perfect lighting and shallow depth of field'
        },
        nature: {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Wide Shot',
            lighting: 'Natural',
            basePrompt: 'A breathtaking landscape with pristine natural beauty and perfect composition'
        },
        aerial: {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Bird\'s Eye',
            lighting: 'Bright',
            basePrompt: 'A stunning aerial view capturing patterns, textures, and scale from high above'
        },
        product: {
            promptType: 'Image',
            style: 'Commercial',
            cinematography: 'Close Up',
            lighting: 'Studio',
            basePrompt: 'A professional product shot with perfect lighting, clean background, and attention to detail'
        },
        food: {
            promptType: 'Image',
            style: 'Commercial',
            cinematography: 'Close Up',
            lighting: 'Soft',
            basePrompt: 'A mouthwatering food photograph with perfect styling, rich colors, and appetizing details'
        },
        underwater: {
            promptType: 'VEO',
            style: 'Photographic',
            cinematography: 'Medium Shot',
            lighting: 'Underwater',
            basePrompt: 'A mesmerizing underwater scene with marine life, light rays filtering through water, and bubbles'
        },
        wildlife: {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Telephoto',
            lighting: 'Natural',
            basePrompt: 'A stunning wildlife photograph capturing an animal in its natural habitat with perfect timing'
        },
        street: {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Medium Shot',
            lighting: 'Natural',
            basePrompt: 'A compelling street photography scene capturing authentic urban life and human interaction'
        },
        architecture: {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Wide Shot',
            lighting: 'Natural',
            basePrompt: 'A striking architectural photograph showcasing impressive building design with perfect perspective'
        },
        
        // Video & Animation Templates
        cinematic: {
            promptType: 'VEO',
            style: 'Cinematic',
            cinematography: 'Wide Shot',
            lighting: 'Golden Hour',
            basePrompt: 'A sweeping cinematic scene with dramatic lighting and atmospheric elements'
        },
        animation: {
            promptType: 'WAN2',
            style: 'Animation',
            cinematography: 'Medium Shot',
            lighting: 'Bright',
            motionEffect: 'Subtle Zoom In',
            basePrompt: 'A vibrant animated scene with smooth motion and expressive characters'
        },
        anime: {
            promptType: 'WAN2',
            style: 'Animation',
            cinematography: 'Medium Shot',
            lighting: 'Bright',
            motionEffect: 'Pan Left',
            basePrompt: 'A detailed anime scene with vibrant colors, expressive characters, and dynamic composition'
        },
        stopmotion: {
            promptType: 'WAN2',
            style: 'Animation',
            cinematography: 'Medium Shot',
            lighting: 'Soft',
            motionEffect: 'Static',
            basePrompt: 'A charming stop-motion animation scene with handcrafted characters and textured materials'
        },
        timelapse: {
            promptType: 'VEO',
            style: 'Cinematic',
            cinematography: 'Wide Shot',
            lighting: 'Dynamic',
            basePrompt: 'A stunning timelapse sequence showing dramatic changes over time with fluid motion'
        },
        musicvideo: {
            promptType: 'VEO',
            style: 'Cinematic',
            cinematography: 'Dynamic',
            lighting: 'Dramatic',
            basePrompt: 'A visually striking music video scene with creative lighting, stylish composition, and energy'
        },
        pixar: {
            promptType: 'WAN2',
            style: 'Animation',
            cinematography: 'Medium Shot',
            lighting: 'Bright',
            motionEffect: 'Subtle Zoom In',
            basePrompt: 'A charming 3D animated scene in Pixar style with expressive characters, detailed textures, and emotional storytelling'
        },
        studioghibli: {
            promptType: 'WAN2',
            style: 'Animation',
            cinematography: 'Wide Shot',
            lighting: 'Natural',
            motionEffect: 'Pan Right',
            basePrompt: 'A beautiful Studio Ghibli inspired scene with lush natural environments, whimsical elements, and dreamlike atmosphere'
        },
        claymation: {
            promptType: 'WAN2',
            style: 'Animation',
            cinematography: 'Medium Shot',
            lighting: 'Soft',
            motionEffect: 'Static',
            basePrompt: 'A whimsical claymation scene with textured clay characters, handcrafted details, and charming imperfections'
        },
        cinemagraph: {
            promptType: 'WAN2',
            style: 'Photographic',
            cinematography: 'Medium Shot',
            lighting: 'Natural',
            motionEffect: 'Subtle Zoom In',
            basePrompt: 'A mesmerizing cinemagraph with subtle isolated motion in an otherwise still photograph'
        },
        slowmo: {
            promptType: 'WAN2',
            style: 'Cinematic',
            cinematography: 'Medium Shot',
            lighting: 'Dramatic',
            motionEffect: 'Slow Motion',
            basePrompt: 'A dramatic slow-motion sequence capturing intricate details of movement with heightened visual impact'
        },
        
        // Creative Styles Templates
        scifi: {
            promptType: 'Image',
            style: 'Sci-Fi',
            cinematography: 'Wide Shot',
            lighting: 'Neon',
            basePrompt: 'A futuristic sci-fi environment with advanced technology and neon lighting'
        },
        cyberpunk: {
            promptType: 'VEO',
            style: 'Sci-Fi',
            cinematography: 'Medium Shot',
            lighting: 'Neon',
            basePrompt: 'A dystopian cyberpunk cityscape with neon lights, rain-slicked streets, and advanced technology'
        },
        fantasy: {
            promptType: 'Image',
            style: 'Fantasy',
            cinematography: 'Medium Shot',
            lighting: 'Magical',
            basePrompt: 'A magical fantasy scene with ethereal lighting and mystical elements'
        },
        horror: {
            promptType: 'Image',
            style: 'Dark',
            cinematography: 'Dutch Angle',
            lighting: 'Low Key',
            basePrompt: 'A chilling horror scene with unsettling atmosphere, shadows, and subtle dread'
        },
        abstract: {
            promptType: 'Image',
            style: 'Abstract',
            cinematography: 'Extreme Close Up',
            lighting: 'High Contrast',
            basePrompt: 'An abstract artistic composition with bold colors, shapes, and textures'
        },
        vintage: {
            promptType: 'Image',
            style: 'Retro',
            cinematography: 'Medium Shot',
            lighting: 'Warm',
            basePrompt: 'A nostalgic vintage scene with period-accurate details, warm tones, and retro aesthetic'
        },
        steampunk: {
            promptType: 'Image',
            style: 'Fantasy',
            cinematography: 'Medium Shot',
            lighting: 'Warm',
            basePrompt: 'An intricate steampunk scene with brass machinery, gears, steam, and Victorian aesthetics'
        },
        vaporwave: {
            promptType: 'Image',
            style: 'Abstract',
            cinematography: 'Wide Shot',
            lighting: 'Neon',
            basePrompt: 'A vaporwave aesthetic scene with retro computing elements, pink and blue gradients, and surreal composition'
        },
        
        // Fine Art Templates
        oilpainting: {
            promptType: 'Image',
            style: 'Fine Art',
            cinematography: 'Medium Shot',
            lighting: 'Dramatic',
            basePrompt: 'A detailed oil painting with rich textures, visible brushstrokes, and masterful use of light and shadow'
        },
        watercolor: {
            promptType: 'Image',
            style: 'Fine Art',
            cinematography: 'Medium Shot',
            lighting: 'Soft',
            basePrompt: 'A delicate watercolor painting with flowing colors, transparent washes, and subtle gradients'
        },
        pencilsketch: {
            promptType: 'Image',
            style: 'Fine Art',
            cinematography: 'Close Up',
            lighting: 'Natural',
            basePrompt: 'A detailed pencil sketch with precise linework, careful shading, and artistic composition'
        },
        digitalart: {
            promptType: 'Image',
            style: 'Digital Art',
            cinematography: 'Medium Shot',
            lighting: 'Dramatic',
            basePrompt: 'A polished digital artwork with vibrant colors, clean lines, and professional rendering'
        },
        
        // Special Effects Templates
        neon: {
            promptType: 'Image',
            style: 'Commercial',
            cinematography: 'Medium Shot',
            lighting: 'Neon',
            basePrompt: 'A scene illuminated by vibrant neon lights creating a colorful glow against a dark background'
        },
        silhouette: {
            promptType: 'Image',
            style: 'Cinematic',
            cinematography: 'Wide Shot',
            lighting: 'Backlit',
            basePrompt: 'A dramatic silhouette scene with strong contrast between the dark subject and bright background'
        },
        longexposure: {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Wide Shot',
            lighting: 'Night',
            basePrompt: 'A long exposure photograph capturing light trails, motion blur, and time passing in a single frame'
        },
        macro: {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Extreme Close Up',
            lighting: 'Soft',
            basePrompt: 'An extreme close-up macro photograph revealing intricate details invisible to the naked eye'
        }
    };
    
    if (els.template) {
        els.template.addEventListener('change', () => {
            const selectedTemplate = templates[els.template.value];
            if (selectedTemplate) {
                els.promptType.value = selectedTemplate.promptType;
                els.style.value = selectedTemplate.style;
                els.cinematography.value = selectedTemplate.cinematography;
                els.lighting.value = selectedTemplate.lighting;
                
                if (selectedTemplate.motionEffect) {
                    els.motionEffect.value = selectedTemplate.motionEffect;
                }
                
                els.prompt.value = selectedTemplate.basePrompt;
                els.promptType.dispatchEvent(new Event('change'));
                
                // Update character counter if it exists
                if (els.charCount) {
                    els.charCount.textContent = els.prompt.value.length;
                }
                
                showToast('Template applied', 'success');
            }
        });
    }

    if (els.clearImage) {
        els.clearImage.addEventListener('click', () => {
            els.imageUpload.value = '';
            els.imageResult.style.display = 'none';
            els.clearImage.style.display = 'none';
            els.imagePreview.src = '#';
            els.imageDescription.innerText = '';
        });
    }

    if (els.clear) {
        els.clear.addEventListener('click', () => {
            els.prompt.value = '';
            els.prompt.focus();
            updateCharCount(); // Update character count when clearing
        });
    }
    
    // --- Text Emphasis Logic ---
    function getTextEmphasisDescription(text, position, size, style, color, effect, material, background, integration) {
        // Start with the base text description
        let textDesc = `The text "${text}"`;
        
        // Add size description if provided
        if (size) {
            switch(size) {
                case 'tiny':
                    textDesc = `The tiny text "${text}"`;
                    break;
                case 'small':
                    textDesc = `The small text "${text}"`;
                    break;
                case 'medium':
                    textDesc = `The medium-sized text "${text}"`;
                    break;
                case 'large':
                    textDesc = `The large text "${text}"`;
                    break;
                case 'huge':
                    textDesc = `The huge text "${text}"`;
                    break;
                case 'giant':
                    textDesc = `The giant, monumental text "${text}"`;
                    break;
            }
        }
        
        // Add style description if provided
        if (style) {
            switch(style) {
                case 'bold':
                    textDesc = textDesc.replace(`The`, `The bold`);
                    break;
                case 'elegant':
                    textDesc = textDesc.replace(`The`, `The elegant serif`);
                    break;
                case 'modern':
                    textDesc = textDesc.replace(`The`, `The clean, modern sans-serif`);
                    break;
                case 'handwritten':
                    textDesc = textDesc.replace(`The`, `The handwritten`);
                    break;
                case 'vintage':
                    textDesc = textDesc.replace(`The`, `The vintage-style`);
                    break;
                case 'futuristic':
                    textDesc = textDesc.replace(`The`, `The futuristic`);
                    break;
                case 'gothic':
                    textDesc = textDesc.replace(`The`, `The gothic-style`);
                    break;
                case 'blocky':
                    textDesc = textDesc.replace(`The`, `The blocky, 3D`);
                    break;
                case 'minimalist':
                    textDesc = textDesc.replace(`The`, `The minimalist`);
                    break;
                case 'ornate':
                    textDesc = textDesc.replace(`The`, `The ornate, decorative`);
                    break;
                case 'pixel':
                    textDesc = textDesc.replace(`The`, `The pixel-style, 8-bit`);
                    break;
                case 'calligraphy':
                    textDesc = textDesc.replace(`The`, `The beautifully calligraphed`);
                    break;
            }
        }
        
        // Add material description if provided
        if (material) {
            switch(material) {
                case 'stone':
                    textDesc += ` made of stone`;
                    break;
                case 'metal':
                    textDesc += ` made of polished metal`;
                    break;
                case 'wood':
                    textDesc += ` carved in wood`;
                    break;
                case 'glass':
                    textDesc += ` made of transparent glass`;
                    break;
                case 'crystal':
                    textDesc += ` made of sparkling crystal`;
                    break;
                case 'ice':
                    textDesc += ` formed from ice`;
                    break;
                case 'paper':
                    textDesc += ` on paper`;
                    break;
                case 'plastic':
                    textDesc += ` made of plastic`;
                    break;
                case 'neon':
                    textDesc += ` formed by neon tubes`;
                    break;
                case 'led':
                    textDesc += ` on an LED display`;
                    break;
                case 'hologram':
                    textDesc += ` as a holographic projection`;
                    break;
                case 'liquid':
                    textDesc += ` formed from flowing liquid`;
                    break;
                case 'fire':
                    textDesc += ` made of flickering flames`;
                    break;
            }
        }
        
        // Add color description if provided
        if (color) {
            switch(color) {
                case 'white':
                    textDesc += ` in white color`;
                    break;
                case 'black':
                    textDesc += ` in black color`;
                    break;
                case 'red':
                    textDesc += ` in bright red color`;
                    break;
                case 'blue':
                    textDesc += ` in vibrant blue color`;
                    break;
                case 'green':
                    textDesc += ` in rich green color`;
                    break;
                case 'purple':
                    textDesc += ` in deep purple color`;
                    break;
                case 'yellow':
                    textDesc += ` in bright yellow color`;
                    break;
                case 'orange':
                    textDesc += ` in warm orange color`;
                    break;
                case 'pink':
                    textDesc += ` in soft pink color`;
                    break;
                case 'teal':
                    textDesc += ` in teal color`;
                    break;
                case 'gold':
                    textDesc += ` in shimmering gold`;
                    break;
                case 'silver':
                    textDesc += ` in metallic silver`;
                    break;
                case 'copper':
                    textDesc += ` in warm copper`;
                    break;
                case 'neon-blue':
                    textDesc += ` in glowing neon blue`;
                    break;
                case 'neon-pink':
                    textDesc += ` in vibrant neon pink`;
                    break;
                case 'neon-green':
                    textDesc += ` in electric neon green`;
                    break;
                case 'glowing':
                    textDesc += ` with a luminous glow`;
                    break;
                case 'rainbow':
                    textDesc += ` in rainbow colors`;
                    break;
                case 'gradient':
                    textDesc += ` in a smooth color gradient`;
                    break;
                case 'contrast':
                    textDesc += ` with high contrast against the background`;
                    break;
            }
        }
        
        // Add effect description if provided
        if (effect) {
            switch(effect) {
                case 'shadow':
                    textDesc += ` with a dramatic drop shadow`;
                    break;
                case 'outline':
                    textDesc += ` with a clear outline`;
                    break;
                case 'embossed':
                    textDesc += ` with an embossed effect`;
                    break;
                case 'beveled':
                    textDesc += ` with beveled edges`;
                    break;
                case 'distressed':
                    textDesc += ` with a worn, distressed appearance`;
                    break;
                case 'glitch':
                    textDesc += ` with a digital glitch effect`;
                    break;
                case 'blur':
                    textDesc += ` with a subtle motion blur`;
                    break;
                case 'sparkle':
                    textDesc += ` with sparkling highlights`;
                    break;
                case 'fire':
                    textDesc += ` with flames emanating from it`;
                    break;
                case 'ice':
                    textDesc += ` with frost and ice crystals forming on it`;
                    break;
                case 'smoke':
                    textDesc += ` with wisps of smoke swirling around it`;
                    break;
            }
        }
        
        // Build the final description with position and background
        let finalDesc = '';
        
        // Add position description
        if (position) {
            switch(position) {
                case 'center':
                    finalDesc = `${textDesc} is prominently displayed in the center of the image`;
                    break;
                case 'top':
                    finalDesc = `${textDesc} is positioned at the top of the image`;
                    break;
                case 'bottom':
                    finalDesc = `${textDesc} is positioned at the bottom of the image`;
                    break;
                case 'left':
                    finalDesc = `${textDesc} is positioned on the left side of the image`;
                    break;
                case 'right':
                    finalDesc = `${textDesc} is positioned on the right side of the image`;
                    break;
                case 'plaque':
                    finalDesc = `A plaque with ${textDesc} is prominently displayed`;
                    break;
                case 'sign':
                    finalDesc = `A sign with ${textDesc} is clearly visible`;
                    break;
                case 'banner':
                    finalDesc = `A banner with ${textDesc} is displayed prominently`;
                    break;
                case 'screen':
                    finalDesc = `A screen showing ${textDesc} is visible`;
                    break;
                case 'etched':
                    finalDesc = `${textDesc} is deeply etched into the surface`;
                    break;
                case 'floating':
                    finalDesc = `${textDesc} appears to be floating in the air`;
                    break;
                case 'graffiti':
                    finalDesc = `${textDesc} appears as stylized graffiti art`;
                    break;
                case 'neon':
                    finalDesc = `${textDesc} is displayed as a glowing sign`;
                    break;
                case 'tattoo':
                    finalDesc = `${textDesc} appears as a detailed tattoo`;
                    break;
                case 'skywriting':
                    finalDesc = `${textDesc} appears as skywriting across the sky`;
                    break;
                default:
                    finalDesc = `${textDesc} is prominently displayed`;
            }
        } else {
            finalDesc = `${textDesc} is prominently displayed`;
        }
        
        // Add background description if provided
        if (background) {
            switch(background) {
                case 'none':
                    // No additional description
                    break;
                case 'contrasting':
                    finalDesc += `, set against a contrasting background that makes it stand out`;
                    break;
                case 'halo':
                    finalDesc += `, surrounded by a soft halo of light`;
                    break;
                case 'spotlight':
                    finalDesc += `, highlighted by a focused spotlight`;
                    break;
                case 'shadow':
                    finalDesc += `, casting dramatic shadows`;
                    break;
                case 'frame':
                    finalDesc += `, enclosed in a decorative frame`;
                    break;
                case 'ribbon':
                    finalDesc += ` on a flowing ribbon`;
                    break;
                case 'clouds':
                    finalDesc += ` against a backdrop of clouds`;
                    break;
                case 'stars':
                    finalDesc += ` against a starry background`;
                    break;
                case 'blur':
                    finalDesc += ` with a blurred background that makes the text pop`;
                    break;
            }
        }
        
        // Add integration description if provided
        if (integration) {
            switch(integration) {
                case 'foreground':
                    finalDesc += `. The text stands out clearly as a foreground element`;
                    break;
                case 'integrated':
                    finalDesc += `. The text is well-integrated with the scene, appearing as a natural part of it`;
                    break;
                case 'part-of':
                    finalDesc += `. The text appears as an organic part of the environment`;
                    break;
                case 'hidden':
                    finalDesc += `. The text is subtly hidden within the scene, visible but not immediately obvious`;
                    break;
                case 'focal':
                    finalDesc += `. The text is the main focal point of the image`;
                    break;
            }
        }
        
        // Ensure the description ends with a period
        if (!finalDesc.endsWith('.')) {
            finalDesc += '.'
        }
        
        // Add a final note about legibility
        finalDesc += ' The text is clearly visible and legible.';
        
        return finalDesc;
    }

    // --- Character Counter Logic ---
    function updateCharCount() {
        if (els.charCount && els.prompt) {
            els.charCount.textContent = els.prompt.value.length;
            
            // Optional: Add visual feedback when approaching limit
            const MAX_CHARS = 1000;
            const count = els.prompt.value.length;
            
            if (els.charCounter) {
                els.charCounter.classList.remove('near-limit', 'at-limit');
                if (count >= MAX_CHARS) {
                    els.charCounter.classList.add('at-limit');
                } else if (count >= MAX_CHARS * 0.9) {
                    els.charCounter.classList.add('near-limit');
                }
            }
        }
    }
    
    // Add input event listener to prompt textarea for character counting
    if (els.prompt && els.charCount) {
        els.prompt.addEventListener('input', updateCharCount);
        
        // Initialize character count on page load
        updateCharCount();
    }
    
    // Initialize text controls toggle functionality
    if (els.textControlsToggle) {
        const textControls = document.querySelector('.text-controls');
        if (textControls) {
            // Start with controls collapsed
            textControls.classList.add('collapsed');
            
            els.textControlsToggle.addEventListener('click', () => {
                textControls.classList.toggle('collapsed');
                els.textControlsToggle.textContent = 
                    textControls.classList.contains('collapsed') ? 'Show more options' : 'Hide options';
            });
        }
    }

    if (els.copy) {
        els.copy.addEventListener('click', () => {
            navigator.clipboard.writeText(els.resultText.innerText).then(() => {
                showToast('Copied to clipboard!', 'success');
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                showToast('Failed to copy text.', 'error');
            });
        });
    }

    if (els.scrollToTop) {
        window.onscroll = () => {
            if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
                els.scrollToTop.style.display = "block";
            } else {
                els.scrollToTop.style.display = "none";
            }
        };

        els.scrollToTop.addEventListener('click', () => {
            document.body.scrollTop = 0;
            document.documentElement.scrollTop = 0;
        });
    }

    // --- Drag and Drop Logic ---
    if (els.dropZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            els.dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            els.dropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            els.dropZone.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            els.dropZone.classList.add('drag-over');
        }

        function unhighlight() {
            els.dropZone.classList.remove('drag-over');
        }

        els.dropZone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const file = dt.files[0];

            if (file && file.type.startsWith('image/')) {
                els.imageUpload.files = dt.files;
                els.analyze.click();
            } else {
                showToast('Please drop an image file', 'error');
            }
        }
    }

    // --- History Panel Logic ---
    if (els.historyPanel) {
        els.historyPanel.addEventListener('mouseenter', () => {
            els.historyPanel.classList.add('show');
        });
        
        els.historyPanel.addEventListener('mouseleave', () => {
            els.historyPanel.classList.remove('show');
        });
    }

    // --- Shortcuts Overlay Logic ---
    if (els.shortcuts && els.shortcutsOverlay) {
        els.shortcuts.addEventListener('click', () => {
            els.shortcutsOverlay.classList.add('show');
        });
        
        els.closeShortcuts.addEventListener('click', () => {
            els.shortcutsOverlay.classList.remove('show');
        });
        
        els.shortcutsOverlay.addEventListener('click', (e) => {
            if (e.target === els.shortcutsOverlay) {
                els.shortcutsOverlay.classList.remove('show');
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && els.shortcutsOverlay.classList.contains('show')) {
                els.shortcutsOverlay.classList.remove('show');
            }
        });
    }

    // Initialize the application
    applySavedTheme();
});
