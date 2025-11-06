document.addEventListener('DOMContentLoaded', () => {
    // --- Initialize Theme Early ---
    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme === 'true') {
        document.documentElement.classList.add('dark-mode');
        document.body.classList.add('dark-mode');
    }
    
    // --- Prompt History Management ---
    const HISTORY_STORAGE_KEY = 'pixa_prompt_history';
    const MAX_HISTORY_ITEMS = 50;
    
    // Load prompt history from local storage
    function loadPromptHistory() {
        try {
            const history = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY)) || [];
            return history;
        } catch (error) {
            console.error('Error loading prompt history:', error);
            return [];
        }
    }
    
    // Save prompt history to local storage
    function savePromptHistory(history) {
        try {
            // Limit history to MAX_HISTORY_ITEMS
            const limitedHistory = history.slice(0, MAX_HISTORY_ITEMS);
            localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(limitedHistory));
        } catch (error) {
            console.error('Error saving prompt history:', error);
        }
    }
    
    // Add a prompt to history
    function addToHistory(promptData) {
        const history = loadPromptHistory();
        
        // Check if this exact prompt already exists
        const existingIndex = history.findIndex(item => 
            item.prompt === promptData.prompt && 
            item.enhancedPrompt === promptData.enhancedPrompt
        );
        
        // If it exists, remove it (we'll add it back at the top)
        if (existingIndex !== -1) {
            history.splice(existingIndex, 1);
        }
        
        // Add the new prompt at the beginning
        history.unshift({
            ...promptData,
            timestamp: new Date().toISOString(),
            id: Date.now().toString()
        });
        
        // Save updated history
        savePromptHistory(history);
        
        // Update the UI
        renderHistoryItems();
    }
    
    // Render history items in the UI
    function renderHistoryItems() {
        if (!els.historyList) return;
        
        const history = loadPromptHistory();
        els.historyList.innerHTML = '';
        
        if (history.length === 0) {
            els.historyList.innerHTML = '<div class="empty-history">No prompt history yet</div>';
            return;
        }
        
        history.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.dataset.id = item.id;
            
            // Format the date
            const date = new Date(item.timestamp);
            const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            // Create the HTML for the history item
            historyItem.innerHTML = `
                <div class="prompt">${truncateText(item.prompt, 50)}</div>
                <div class="metadata">
                    <span>${item.promptType || 'Image'}</span> · 
                    <span>${formattedDate}</span>
                    <button class="star-btn" title="Favorite this prompt">${item.starred ? '★' : '☆'}</button>
                </div>
            `;
            
            // Add click event to load this prompt
            historyItem.addEventListener('click', (e) => {
                if (!e.target.classList.contains('star-btn')) {
                    loadPromptFromHistory(item);
                }
            });
            
            // Add star button functionality
            const starBtn = historyItem.querySelector('.star-btn');
            if (starBtn) {
                if (item.starred) starBtn.classList.add('starred');
                
                starBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    toggleStarredStatus(item.id);
                });
            }
            
            els.historyList.appendChild(historyItem);
        });
    }
    
    // Toggle starred status for a history item
    function toggleStarredStatus(id) {
        const history = loadPromptHistory();
        const itemIndex = history.findIndex(item => item.id === id);
        
        if (itemIndex !== -1) {
            history[itemIndex].starred = !history[itemIndex].starred;
            savePromptHistory(history);
            renderHistoryItems();
        }
    }
    
    // Load a prompt from history
    function loadPromptFromHistory(historyItem) {
        // Set the form values
        els.prompt.value = historyItem.prompt || '';
        if (historyItem.promptType && els.promptType) els.promptType.value = historyItem.promptType;
        if (historyItem.style && els.style) els.style.value = historyItem.style;
        if (historyItem.cinematography && els.cinematography) els.cinematography.value = historyItem.cinematography;
        if (historyItem.lighting && els.lighting) els.lighting.value = historyItem.lighting;
        if (historyItem.motionEffect && els.motionEffect) els.motionEffect.value = historyItem.motionEffect;
        if (historyItem.model && els.modelSelect) els.modelSelect.value = historyItem.model;
        
        // Update character count
        updateCharCount();
        
        // Show the enhanced prompt if available
        if (historyItem.enhancedPrompt) {
            els.result.style.display = 'block';
            els.resultText.innerText = historyItem.enhancedPrompt;
        }
        
        // Update UI based on prompt type
        els.promptType.dispatchEvent(new Event('change'));
        
        showToast('Prompt loaded from history', 'success');
    }
    
    // Clear all history
    function clearHistory() {
        if (confirm('Are you sure you want to clear your prompt history? This cannot be undone.')) {
            localStorage.removeItem(HISTORY_STORAGE_KEY);
            renderHistoryItems();
            showToast('History cleared', 'info');
        }
    }
    
    // Helper function to truncate text
    function truncateText(text, maxLength) {
        if (!text) return '';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }
    
    // Check for shared prompt in URL parameters
    function loadSharedPrompt() {
        const urlParams = new URLSearchParams(window.location.search);
        const sharedPrompt = urlParams.get('shared_prompt');
        
        if (sharedPrompt) {
            // Display the shared prompt
            els.result.style.display = 'block';
            els.resultText.innerText = sharedPrompt;
            
            // Set other parameters if available
            const promptType = urlParams.get('type');
            const style = urlParams.get('style');
            const model = urlParams.get('model');
            
            if (promptType && els.promptType) els.promptType.value = promptType;
            if (style && els.style) els.style.value = style;
            if (model && els.modelSelect) els.modelSelect.value = model;
            
            // Scroll to the result
            setTimeout(() => {
                els.result.scrollIntoView({ behavior: 'smooth' });
                showToast('Shared prompt loaded!', 'success');
            }, 500);
        }
    }
    
    // Initialize history panel toggle
    function initHistoryPanel() {
        const historyToggle = document.createElement('button');
        historyToggle.id = 'history-toggle';
        historyToggle.className = 'history-toggle-btn';
        historyToggle.innerHTML = '📋';
        historyToggle.title = 'Toggle Prompt History';
        document.body.appendChild(historyToggle);
        
        historyToggle.addEventListener('click', () => {
            els.historyPanel.classList.toggle('show');
        });
        
        // Initial render of history items
        renderHistoryItems();
    }

    // --- Element References ---
    const els = {
        // Main form elements
        prompt: document.getElementById('prompt-input'),
        promptType: document.getElementById('prompt-type-select'),
        enhance: document.getElementById('enhance-button'),
        analyze: document.getElementById('analyze-button'),
        clear: document.getElementById('clear-button'),
        clearImage: document.getElementById('clear-image-button'),
        clearHistory: document.getElementById('clear-history'),
        shortcuts: document.getElementById('shortcuts-btn'),
        closeShortcuts: document.querySelector('.close-shortcuts'),
        scrollToTop: document.getElementById('scroll-to-top-btn'),
        share: document.getElementById('share-button'),
        export: document.getElementById('export-button'),
        copy: document.getElementById('copy-button'),
        importFile: document.getElementById('import-file'),
        importLabel: document.getElementById('import-label'),
        saveToCompare: document.getElementById('save-to-compare-button'),
        clearSavedPrompt: document.getElementById('clear-saved-prompt'),
        singleViewBtn: document.getElementById('single-view-btn'),
        compareViewBtn: document.getElementById('compare-view-btn'),
        textControlsToggle: document.getElementById('text-controls-toggle'),
        modelSelect: document.getElementById('model-select'),
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
        
        // Containers and        // Results
        result: document.getElementById('result'),
        resultText: document.getElementById('result-text'),
        singleView: document.getElementById('single-view'),
        compareView: document.getElementById('compare-view'),
        currentPrompt: document.getElementById('current-prompt'),
        savedPrompt: document.getElementById('saved-prompt'),
        imageUpload: document.getElementById('image-upload'),
        imageResult: document.getElementById('image-result-container'),
        imagePreview: document.getElementById('image-preview'),
        imageProgressContainer: document.getElementById('image-progress-container'),
        imageDescription: document.getElementById('image-description-text'),
        motionEffectContainer: document.getElementById('motion-effect-selector-container'),
        historyPanel: document.querySelector('.history-panel'),
        compareResult: document.getElementById('compare-result'),
        compareResultText: document.getElementById('compare-result-text'),
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
        themeToggle: document.getElementById('theme-toggle')
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
        document.documentElement.classList.toggle('dark-mode', isDarkMode);
        document.body.classList.toggle('dark-mode', isDarkMode);
        if (els.themeToggle) {
            els.themeToggle.checked = isDarkMode;
        }
    }

    if (els.themeToggle) {
        els.themeToggle.addEventListener('change', function() {
            const isDark = this.checked;
            document.documentElement.classList.toggle('dark-mode', isDark);
            document.body.classList.toggle('dark-mode', isDark);
            localStorage.setItem('darkMode', isDark);
            
            // Show a toast notification
            showToast(isDark ? '🌙 Dark mode enabled' : '☀️ Light mode enabled', 'info', 2000);
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
            console.log('Analyze button clicked or triggered.');
            updateProgress(1, 'active');
            const file = els.imageUpload.files[0];
            if (!file) {
                console.error('No file selected.');
                showToast('Please select an image file first.', 'error');
                return;
            }
            console.log('File selected:', file.name, 'Size:', file.size, 'Type:', file.type);
            
            // Check file size
            const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
            if (file.size > MAX_FILE_SIZE) {
                showToast(`Image file is too large (${(file.size / (1024 * 1024)).toFixed(2)}MB). Maximum size is 10MB.`, 'error');
                return;
            }
            
            // Check file type
            const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
            if (!validTypes.includes(file.type)) {
                showToast(`Invalid file type: ${file.type}. Please upload a JPEG, PNG, GIF, or WebP image.`, 'error');
                return;
            }

            // Show analyzing state
            showToast('Analyzing image...', 'info');
            els.imageResult.style.display = 'block';
            els.analyze.disabled = true;
            els.analyze.innerHTML = '<div class="loader-small"></div> Analyzing...';
            
            // Create progress container
            const progressContainer = document.createElement('div');
            progressContainer.className = 'upload-progress-container';
            progressContainer.innerHTML = `
                <div class="upload-progress-bar">
                    <div class="upload-progress-fill"></div>
                </div>
                <div class="upload-progress-text">Uploading: 0%</div>
            `;
            
            // Clear previous results and insert progress bar
            els.imageProgressContainer.innerHTML = '';
            els.imageProgressContainer.style.display = 'block';
            els.imageDescription.innerHTML = '';
            els.imageDescription.parentElement.style.display = 'none';
            els.imageProgressContainer.appendChild(progressContainer);
            
            // Show image preview
            els.imagePreview.src = URL.createObjectURL(file);
            els.imagePreview.style.display = 'block';
            els.clearImage.style.display = 'inline-block';

            const formData = new FormData();
            formData.append('image', file);
            
            // Create XMLHttpRequest for upload with progress
            const xhr = new XMLHttpRequest();
            const progressFill = progressContainer.querySelector('.upload-progress-fill');
            const progressText = progressContainer.querySelector('.upload-progress-text');
            
            xhr.upload.addEventListener('progress', (event) => {
                if (event.lengthComputable) {
                    const percentComplete = Math.round((event.loaded / event.total) * 100);
                    progressFill.style.width = percentComplete + '%';
                    progressText.textContent = `Uploading: ${percentComplete}%`;
                }
            });
            
            xhr.addEventListener('load', async () => {
                if (xhr.status === 200) {
                    try {
                        const data = JSON.parse(xhr.responseText);
                        
                        // Clear progress bar
                        els.imageProgressContainer.innerHTML = '';
                        els.imageProgressContainer.style.display = 'none';
                        
                        // Check if the response contains an error message
                        if (data.description && data.description.startsWith('Error:')) {
                            els.imageDescription.innerHTML = `<div class="error-message">${data.description}</div>`;
                            els.imageDescription.parentElement.style.display = 'block';
                            showToast('Image analysis failed: ' + data.description.split(':')[1], 'error');
                        } else {
                            // Success - show the description
                            els.imageDescription.innerHTML = data.description;
                            els.imageDescription.parentElement.style.display = 'block';
                            els.imagePreview.style.display = 'block';
                            showToast('Image analyzed successfully!', 'success');
                            updateProgress(1, 'completed');
                            updateProgress(2, 'active');
                        }
                    } catch (error) {
                        els.imageDescription.innerHTML = '<div class="error-message">Failed to parse server response</div>';
                        els.imageDescription.parentElement.style.display = 'block';
                        showToast('Image analysis failed: Invalid server response', 'error');
                    }
                } else {
                    els.imageProgressContainer.innerHTML = ''; // Clear progress on error
                    els.imageDescription.innerHTML = '<div class="error-message">Server error: ' + xhr.status + '</div>';
                    els.imageDescription.parentElement.style.display = 'block';
                    showToast('Image analysis failed: Server error', 'error');
                }
                
                // Re-enable the analyze button
                els.analyze.disabled = false;
                els.analyze.innerHTML = 'Analyze Image';
            });
            
            xhr.addEventListener('error', () => {
                console.error('XHR network error occurred.');
                els.imageProgressContainer.innerHTML = ''; // Clear progress on error
                els.imageDescription.innerHTML = '<div class="error-message">Network error occurred</div>';
                els.imageDescription.parentElement.style.display = 'block';
                showToast('Image analysis failed: Network error', 'error');
                els.analyze.disabled = false;
                els.analyze.innerHTML = 'Analyze Image';
            });
            
            xhr.addEventListener('abort', () => {
                els.imageDescription.innerHTML = '<div class="error-message">Upload aborted</div>';
                els.imageDescription.parentElement.style.display = 'block';
                showToast('Image analysis aborted', 'warning');
                els.analyze.disabled = false;
                els.analyze.innerHTML = 'Analyze Image';
            });
            
            // Send the request
            xhr.open('POST', '/analyze-image', true);
            console.log('Sending XHR request to /analyze-image...');
            xhr.send(formData);
        });
    }

    if (els.enhance) {
        els.enhance.addEventListener('click', async () => {
            console.log('Enhance button clicked!');
            console.log('els.enhance:', els.enhance);
            console.log('els.prompt:', els.prompt);
            console.log('els.promptType:', els.promptType);
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

            // Validate prompt length
            const charLimit = getCurrentCharLimit();
            if (prompt.length > charLimit) {
                const overLimit = prompt.length - charLimit;
                showToast(`Prompt exceeds character limit by ${overLimit} characters. Please shorten your prompt.`, 'error');
                els.prompt.focus();
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

            // Get selected model and apply model-specific formatting
            const selectedModel = els.modelSelect ? els.modelSelect.value : 'default';
            const formattedPrompt = formatPromptForModel(prompt, selectedModel, promptType);

            try {
                console.log('Making API call to /enhance...');
                console.log('Request data:', {
                    prompt: formattedPrompt,
                    prompt_type: promptType,
                    style: style,
                    cinematography: cinematography,
                    lighting: lighting,
                    motion_effect: motionEffect,
                    image_description: imageDescription,
                    text_emphasis: textEmphasisDetails,
                    model: selectedModel
                });
                const response = await fetch('/enhance', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: formattedPrompt,
                        prompt_type: promptType,
                        style: style,
                        cinematography: cinematography,
                        lighting: lighting,
                        motion_effect: motionEffect,
                        image_description: imageDescription,
                        text_emphasis: textEmphasisDetails,
                        model: selectedModel
                    }),
                });

                if (!response.ok) {
                    console.log('API call failed with status:', response.status);
                    throw new Error('Failed to enhance prompt.');
                }

                console.log('API call successful, parsing response...');
                const data = await response.json();
                console.log('Response data:', data);
                console.log('Enhanced prompt:', data.enhanced_prompt);
                console.log('Result element:', els.result);
                console.log('Result text element:', els.resultText);

                if (data.enhanced_prompt) {
                    console.log('Setting result text...');
                    els.resultText.innerText = data.enhanced_prompt;
                    console.log('Result text set to:', els.resultText.innerText);
                    console.log('Result container elements:');
                    console.log('- result:', els.result);
                    console.log('- resultText:', els.resultText);
                    console.log('- copy button:', els.copy);
                    console.log('- share button:', document.getElementById('share-button'));
                    console.log('- export button:', els.export);
                } else {
                    console.log('No enhanced_prompt in response data');
                }

                if (els.result) {
                    console.log('Showing result container...');
                    els.result.style.display = 'block';
                    console.log('Result container display:', els.result.style.display);

                    // Also check if there's a parent result-container
                    const parentResultContainer = document.getElementById('result-container');
                    if (parentResultContainer) {
                        console.log('Found parent result-container, showing it too...');
                        parentResultContainer.style.display = 'block';
                        console.log('Parent result-container display:', parentResultContainer.style.display);
                    }
                } else {
                    console.log('Result container element not found');
                }

                // Check if the response contains an error message
                if (data.enhanced_prompt.startsWith('Error:') ||
                    data.enhanced_prompt.startsWith('An unexpected error')) {
                    showToast(data.enhanced_prompt.split('.')[0], 'error');
                    updateProgress(2, 'active');
                } else {
                    showToast('Prompt enhanced successfully!', 'success');
                    updateProgress(2, 'completed');

                    // Update current prompt in comparison view if it exists
                    if (els.currentPrompt) {
                        els.currentPrompt.innerText = data.enhanced_prompt;
                    }

                    // Only add to history if it's not an error
                    addToHistory({
                        prompt: prompt,
                        enhancedPrompt: data.enhanced_prompt,
                        promptType: promptType,
                        style: style,
                        cinematography: cinematography,
                        lighting: lighting,
                        motionEffect: motionEffect,
                        model: selectedModel
                    });
                }
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
            
            // Show/hide model selector based on prompt type
            if (els.promptType.value === 'Image' && document.getElementById('model-selector-container')) {
                document.getElementById('model-selector-container').style.display = 'flex';
            } else if (document.getElementById('model-selector-container')) {
                document.getElementById('model-selector-container').style.display = 'none';
            }
            
            // Update character count limit when prompt type changes
            updateCharCount();
        });
    }
    
    // Add event listener for model selection
    if (els.modelSelect) {
        // Function to update model info based on selection
        function updateModelInfo(model) {
            const modelInfoContainer = document.getElementById('model-info-container');
            if (!modelInfoContainer) return;

            // Remove any existing model info
            const existingInfo = document.getElementById('model-info');
            if (existingInfo) existingInfo.remove();

            // Create new model info if a specific model is selected
            if (model && model !== 'default') {
                const modelInfo = document.createElement('div');
                modelInfo.id = 'model-info';
                modelInfo.className = `model-info ${model}-model`;

                let infoContent = '';
                
                // Photorealistic Models
                if (model === 'flux') {
                    infoContent = `<strong>Flux Model Selected</strong>
                    <p>Optimizing for photorealistic imagery with technical details.</p>
                    <em>Best for: Detailed photography-style images with realistic lighting and textures.</em>
                    <em>Tip: Include camera model, lens details, and lighting setup.</em>`;
                } else if (model === 'pixart') {
                    infoContent = `<strong>PixArt Model Selected</strong>
                    <p>Optimizing for detailed compositions with artistic direction.</p>
                    <em>Best for: Images with complex compositions and a good balance of realism and artistic flair.</em>
                    <em>Tip: Use detailed descriptions of scene elements and specify artistic style.</em>`;
                } else if (model === 'dalle3') {
                    infoContent = `<strong>DALL-E 3 Model Selected</strong>
                    <p>Optimizing for coherent scenes with accurate text rendering.</p>
                    <em>Best for: Complex scenes with multiple elements and text that needs to be rendered accurately.</em>
                    <em>Tip: Use clear, detailed instructions and specify style, composition, and lighting.</em>`;
                } 
                // Versatile Models
                else if (model === 'qwen') {
                    infoContent = `<strong>Qwen Model Selected</strong>
                    <p>Optimizing for clear composition and exceptional text rendering. This 20B MMDiT model excels at displaying text in images, especially Chinese characters.</p>
                    <em>Best for: Images containing text, signs, or multilingual content. Also great for portraits with natural expressions.</em>
                    <em>Tip: Add "Ultra HD, 4K, cinematic composition" to enhance quality.</em>`;
                } else if (model === 'midjourney') {
                    infoContent = `<strong>Midjourney Model Selected</strong>
                    <p>Optimizing for artistic compositions with strong aesthetic appeal.</p>
                    <em>Best for: Stylized, visually striking imagery with artistic flair.</em>
                    <em>Tip: Use simple, clear descriptions and include artistic style references.</em>`;
                } else if (model === 'sdxl') {
                    infoContent = `<strong>SDXL Model Selected</strong>
                    <p>Optimizing for detailed compositions with strong artistic direction.</p>
                    <em>Best for: Versatile applications across many styles and subjects.</em>
                    <em>Tip: Balance descriptive and stylistic elements, and include composition details.</em>`;
                } 
                // Artistic Models
                else if (model === 'nunchaku') {
                    infoContent = `<strong>Nunchaku Model Selected</strong>
                    <p>Optimizing for artistic style with mood-focused descriptions.</p>
                    <em>Best for: Stylized, atmospheric images with strong emotional impact.</em>
                    <em>Tip: Include mood/atmosphere descriptions for best results.</em>`;
                    modelInfo.className += ' nunchaku-model';
                } else if (model === 'kandinsky') {
                    infoContent = `<strong>Kandinsky Model Selected</strong>
                    <p>Optimizing for abstract and artistic compositions.</p>
                    <em>Best for: Abstract, surreal, and conceptual imagery with unique stylistic elements.</em>
                    <em>Tip: Use artistic movement references and focus on color palette and composition.</em>`;
                } else if (model === 'imagen') {
                    infoContent = `<strong>Imagen Model Selected</strong>
                    <p>Optimizing for coherent scenes with good composition.</p>
                    <em>Best for: Following detailed instructions and creating coherent scenes.</em>
                    <em>Tip: Specify subject, setting, action, and include lighting and atmosphere details.</em>`;
                }
                
                modelInfo.innerHTML = infoContent;
                
                // Insert after the model selector
                const modelSelectorContainer = document.getElementById('model-selector-container');
                if (modelSelectorContainer) {
                    modelSelectorContainer.insertAdjacentElement('afterend', modelInfo);
                }
            }
        }
        
        // Add event listener to model select dropdown
        els.modelSelect.addEventListener('change', function() {
            updateModelInfo(this.value);
            // Update character count limit when model changes
            updateCharCount();
        });
        
        // Initialize model info with current selection
        if (els.modelSelect.value !== 'default') {
            updateModelInfo(els.modelSelect.value);
        }
        
        // Initialize visibility of model selector based on current prompt type
        if (els.promptType && els.promptType.value !== 'Image' && document.getElementById('model-selector-container')) {
            document.getElementById('model-selector-container').style.display = 'none';
        }
    }
    
    // --- Templates System ---
    // Model-specific template modifiers
    const modelTemplateModifiers = {
        // Photorealistic Models
        'flux': {
            // Flux model prefers technical details and photorealistic elements
            modifyPrompt: (basePrompt, templateType) => {
                let modified = basePrompt;
                
                // Add technical camera details for photography templates
                if (['portrait', 'nature', 'aerial', 'product', 'food', 'wildlife', 'architecture'].includes(templateType)) {
                    modified += ', shot on Sony Alpha A7R IV with 50mm f/1.4 lens, professional lighting, photorealistic, 8K resolution';
                }
                
                // Add specific enhancements based on template type
                if (templateType === 'portrait') {
                    modified += ', perfect skin texture, studio lighting setup with key light and fill light, shallow depth of field';
                } else if (templateType === 'nature' || templateType === 'landscape') {
                    modified += ', golden hour lighting, high dynamic range, crisp details, natural color grading';
                } else if (templateType === 'product') {
                    modified += ', product photography lighting, soft shadows, high detail product texture, commercial quality';
                }
                
                return modified;
            }
        },
        'pixart': {
            // PixArt model balances realism with artistic flair
            modifyPrompt: (basePrompt, templateType) => {
                let modified = basePrompt;
                
                // Add PixArt-specific enhancements
                modified += ', detailed composition, artistic direction, high quality';
                
                // Add specific enhancements based on template type
                if (templateType === 'portrait') {
                    modified += ', detailed facial features, expressive eyes, natural pose';
                } else if (templateType.includes('landscape') || templateType === 'nature') {
                    modified += ', atmospheric perspective, detailed environment, rich textures';
                } else if (templateType.includes('fantasy') || templateType.includes('concept')) {
                    modified += ', imaginative elements, cohesive design, rich details';
                }
                
                return modified;
            }
        },
        'dalle3': {
            // DALL-E 3 model excels at following complex instructions
            modifyPrompt: (basePrompt, templateType) => {
                let modified = basePrompt;
                
                // Add DALL-E 3-specific enhancements
                modified += ', coherent scene, accurate details, high resolution';
                
                // Add specific enhancements based on template type
                if (templateType.includes('text')) {
                    modified += ', perfectly rendered text, legible typography, accurate spelling';
                } else if (templateType === 'portrait') {
                    modified += ', accurate facial proportions, natural expression, detailed features';
                } else if (templateType.includes('complex') || templateType.includes('scene')) {
                    modified += ', multiple elements in harmony, logical composition, consistent perspective';
                }
                
                return modified;
            }
        },
        
        // Versatile Models
        'qwen': {
            // Qwen model excels at text rendering and balanced compositions
            modifyPrompt: (basePrompt, templateType) => {
                let modified = basePrompt;
                
                // Add Qwen-specific enhancements
                modified += ', Ultra HD, 4K, cinematic composition';
                
                // Add specific enhancements based on template type
                if (templateType === 'portrait') {
                    modified += ', natural facial features, realistic skin texture, detailed eyes';
                } else if (templateType.includes('text') || templateType === 'product') {
                    modified += ', clear legible text, high fidelity text rendering';
                }
                
                return modified;
            }
        },
        'midjourney': {
            // Midjourney model excels at artistic compositions
            modifyPrompt: (basePrompt, templateType) => {
                let modified = basePrompt;
                
                // Add Midjourney-specific enhancements
                modified += ', highly detailed, intricate, elegant, sharp focus, dramatic lighting';
                
                // Add specific enhancements based on template type
                if (templateType === 'portrait') {
                    modified += ', volumetric lighting, cinematic, 8k, ultra-realistic';
                } else if (templateType.includes('landscape') || templateType === 'nature') {
                    modified += ', epic scale, atmospheric, golden hour, cinematic';
                } else if (templateType.includes('fantasy') || templateType.includes('concept')) {
                    modified += ', concept art, digital painting, trending on artstation, cinematic';
                }
                
                return modified;
            }
        },
        'sdxl': {
            // SDXL model is versatile across many styles
            modifyPrompt: (basePrompt, templateType) => {
                let modified = basePrompt;
                
                // Add SDXL-specific enhancements
                modified += ', detailed, high resolution, professional quality';
                
                // Add specific enhancements based on template type
                if (templateType === 'portrait') {
                    modified += ', detailed features, professional photography, studio lighting';
                } else if (templateType.includes('art') || templateType.includes('painting')) {
                    modified += ', masterpiece, trending on artstation, award winning';
                } else if (templateType.includes('photo') || templateType === 'realistic') {
                    modified += ', photorealistic, 8k, detailed textures, professional photography';
                }
                
                return modified;
            }
        },
        
        // Artistic Models
        'nunchaku': {
            // Nunchaku model focuses on artistic style and mood
            modifyPrompt: (basePrompt, templateType) => {
                let modified = basePrompt;
                
                // Add artistic style elements
                modified += ', artistic composition, mood-focused, atmospheric';
                
                // Add specific enhancements based on template type
                if (['abstract', 'fantasy', 'cyberpunk', 'vaporwave'].includes(templateType)) {
                    modified += ', stylized, vibrant color palette, dramatic lighting';
                } else if (['oilpainting', 'watercolor', 'pencilsketch'].includes(templateType)) {
                    modified += ', artistic interpretation, expressive style, textured';
                }
                
                return modified;
            }
        },
        'kandinsky': {
            // Kandinsky model excels at abstract and artistic compositions
            modifyPrompt: (basePrompt, templateType) => {
                let modified = basePrompt;
                
                // Add Kandinsky-specific enhancements
                modified += ', abstract elements, artistic composition, unique style';
                
                // Add specific enhancements based on template type
                if (templateType.includes('abstract')) {
                    modified += ', non-representational, geometric forms, expressive color';
                } else if (templateType.includes('surreal')) {
                    modified += ', dreamlike quality, unexpected juxtapositions, symbolic elements';
                } else if (templateType.includes('concept')) {
                    modified += ', conceptual approach, symbolic representation, artistic interpretation';
                }
                
                return modified;
            }
        },
        'imagen': {
            // Imagen model is good at following detailed instructions
            modifyPrompt: (basePrompt, templateType) => {
                let modified = basePrompt;
                
                // Add Imagen-specific enhancements
                modified += ', coherent composition, detailed scene, high quality';
                
                // Add specific enhancements based on template type
                if (templateType === 'portrait') {
                    modified += ', realistic features, natural lighting, detailed textures';
                } else if (templateType.includes('landscape') || templateType === 'nature') {
                    modified += ', natural lighting, atmospheric perspective, detailed environment';
                } else if (templateType.includes('complex') || templateType.includes('scene')) {
                    modified += ', multiple elements, logical arrangement, consistent style';
                }
                
                return modified;
            }
        }
    };
    
    const templates = {
        // Model-Optimized Templates - Photorealistic Models
        'flux-portrait': {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Close Up',
            lighting: 'Studio',
            model: 'flux',
            basePrompt: 'A professional portrait with perfect lighting and shallow depth of field, shot on Sony Alpha A7R IV with 85mm f/1.4 GM lens, studio lighting setup with key light, fill light, and rim light, 8K resolution, photorealistic, perfect skin texture, ultra-detailed eyes'
        },
        'flux-landscape': {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Wide Shot',
            lighting: 'Golden Hour',
            model: 'flux',
            basePrompt: 'A breathtaking landscape with pristine natural beauty and perfect composition, shot on Sony Alpha A7R IV with 16-35mm f/2.8 GM lens, golden hour lighting, high dynamic range, crisp details, natural color grading, photorealistic, 8K resolution'
        },
        'flux-product': {
            promptType: 'Image',
            style: 'Commercial',
            cinematography: 'Close Up',
            lighting: 'Studio',
            model: 'flux',
            basePrompt: 'A professional product shot with perfect lighting, clean background, and attention to detail, shot on Sony Alpha A7R IV with 90mm f/2.8 macro lens, product photography lighting setup, soft shadows, high detail product texture, commercial quality, 8K resolution'
        },
        'pixart-concept': {
            promptType: 'Image',
            style: 'Concept Art',
            cinematography: 'Medium Shot',
            lighting: 'Dramatic',
            model: 'pixart',
            basePrompt: 'A detailed concept art piece with imaginative elements and cohesive design, rich details, atmospheric lighting, detailed composition, artistic direction, high quality'
        },
        'pixart-character': {
            promptType: 'Image',
            style: 'Character Design',
            cinematography: 'Medium Shot',
            lighting: 'Studio',
            model: 'pixart',
            basePrompt: 'A detailed character design with unique features and personality, expressive pose, detailed costume, rich textures, detailed composition, artistic direction, high quality'
        },
        'dalle3-text': {
            promptType: 'Image',
            style: 'Commercial',
            cinematography: 'Close Up',
            lighting: 'Bright',
            model: 'dalle3',
            basePrompt: 'A clean design with perfectly rendered text that reads "SAMPLE TEXT" in clear, legible typography, coherent scene, accurate details, high resolution, perfectly rendered text, legible typography, accurate spelling'
        },
        'dalle3-scene': {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Wide Shot',
            lighting: 'Natural',
            model: 'dalle3',
            basePrompt: 'A detailed scene with multiple elements arranged logically, coherent scene, accurate details, high resolution, multiple elements in harmony, logical composition, consistent perspective'
        },
        
        // Model-Optimized Templates - Versatile Models
        'qwen-text': {
            promptType: 'Image',
            style: 'Commercial',
            cinematography: 'Close Up',
            lighting: 'Bright',
            model: 'qwen',
            basePrompt: 'A clean, professional design with perfectly rendered text that is crisp and legible, Ultra HD, 4K, cinematic composition, high fidelity text rendering, perfect typography'
        },
        'qwen-portrait': {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Close Up',
            lighting: 'Studio',
            model: 'qwen',
            basePrompt: 'A professional portrait with perfect lighting and natural facial features, Ultra HD, 4K, cinematic composition, realistic skin texture, detailed eyes, natural expression'
        },
        'midjourney-landscape': {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Wide Shot',
            lighting: 'Golden Hour',
            model: 'midjourney',
            basePrompt: 'An epic landscape with breathtaking natural beauty, highly detailed, intricate, elegant, sharp focus, dramatic lighting, epic scale, atmospheric, golden hour, cinematic, volumetric lighting, 8k, ultra-realistic'
        },
        'midjourney-fantasy': {
            promptType: 'Image',
            style: 'Fantasy',
            cinematography: 'Medium Shot',
            lighting: 'Magical',
            model: 'midjourney',
            basePrompt: 'A fantasy scene with magical elements and ethereal atmosphere, highly detailed, intricate, elegant, sharp focus, dramatic lighting, concept art, digital painting, trending on artstation, cinematic'
        },
        'sdxl-realistic': {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Medium Shot',
            lighting: 'Natural',
            model: 'sdxl',
            basePrompt: 'A photorealistic scene with perfect lighting and composition, detailed, high resolution, professional quality, photorealistic, 8k, detailed textures, professional photography'
        },
        'sdxl-painting': {
            promptType: 'Image',
            style: 'Digital Art',
            cinematography: 'Medium Shot',
            lighting: 'Dramatic',
            model: 'sdxl',
            basePrompt: 'A digital painting with rich colors and detailed composition, detailed, high resolution, professional quality, masterpiece, trending on artstation, award winning'
        },
        
        // Model-Optimized Templates - Artistic Models
        'nunchaku-fantasy': {
            promptType: 'Image',
            style: 'Fantasy',
            cinematography: 'Medium Shot',
            lighting: 'Magical',
            model: 'nunchaku',
            basePrompt: 'A magical fantasy scene with ethereal lighting and mystical elements, artistic composition, mood-focused, atmospheric, stylized, vibrant color palette, dramatic lighting'
        },
        'nunchaku-abstract': {
            promptType: 'Image',
            style: 'Abstract',
            cinematography: 'Extreme Close Up',
            lighting: 'High Contrast',
            model: 'nunchaku',
            basePrompt: 'An abstract artistic composition with bold colors, shapes, and textures, artistic composition, mood-focused, atmospheric, stylized, vibrant color palette, dramatic lighting, artistic interpretation, expressive style'
        },
        'kandinsky-abstract': {
            promptType: 'Image',
            style: 'Abstract',
            cinematography: 'Medium Shot',
            lighting: 'Vibrant',
            model: 'kandinsky',
            basePrompt: 'An abstract composition with geometric forms and expressive colors, abstract elements, artistic composition, unique style, non-representational, geometric forms, expressive color'
        },
        'kandinsky-surreal': {
            promptType: 'Image',
            style: 'Surreal',
            cinematography: 'Medium Shot',
            lighting: 'Dramatic',
            model: 'kandinsky',
            basePrompt: 'A surreal dreamlike scene with unexpected juxtapositions and symbolic elements, abstract elements, artistic composition, unique style, dreamlike quality, unexpected juxtapositions, symbolic elements'
        },
        'imagen-detailed': {
            promptType: 'Image',
            style: 'Photographic',
            cinematography: 'Wide Shot',
            lighting: 'Natural',
            model: 'imagen',
            basePrompt: 'A detailed scene with multiple elements arranged logically, coherent composition, detailed scene, high quality, multiple elements, logical arrangement, consistent style'
        },
        
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
            const templateValue = els.template.value;
            const selectedTemplate = templates[templateValue];
            
            if (selectedTemplate) {
                // Set form values from template
                els.promptType.value = selectedTemplate.promptType;
                els.style.value = selectedTemplate.style;
                els.cinematography.value = selectedTemplate.cinematography;
                els.lighting.value = selectedTemplate.lighting;
                
                if (selectedTemplate.motionEffect) {
                    els.motionEffect.value = selectedTemplate.motionEffect;
                }
                
                // Get base prompt
                let finalPrompt = selectedTemplate.basePrompt;
                
                // Apply model-specific modifications if a model is selected and it's an Image prompt
                const selectedModel = els.modelSelect ? els.modelSelect.value : 'default';
                if (selectedTemplate.promptType === 'Image' && 
                    selectedModel !== 'default' && 
                    modelTemplateModifiers[selectedModel]) {
                    
                    finalPrompt = modelTemplateModifiers[selectedModel].modifyPrompt(finalPrompt, templateValue);
                    
                    // Show a toast indicating model-specific optimization
                    showToast(`Template optimized for ${selectedModel} model`, 'info');
                }
                
                // Set the final prompt
                els.prompt.value = finalPrompt;
                
                // Set model if template has a model specified
                if (selectedTemplate.model && els.modelSelect) {
                    els.modelSelect.value = selectedTemplate.model;
                    // Update model info
                    updateModelInfo(selectedTemplate.model);
                }
                
                // Update UI based on prompt type
                els.promptType.dispatchEvent(new Event('change'));
                
                // Update character counter
                updateCharCount();
                
                showToast('Template applied', 'success');
            }
        });
        
        // Add event listener to model select to update prompt when model changes
        if (els.modelSelect) {
            els.modelSelect.addEventListener('change', () => {
                // If a template is already selected, reapply it with the new model
                if (els.template.value && templates[els.template.value]) {
                    els.template.dispatchEvent(new Event('change'));
                }
            });
        }
    }

    if (els.clearImage) {
        els.clearImage.addEventListener('click', () => {
            els.imageUpload.value = '';
            els.imageResult.style.display = 'none';
            els.clearImage.style.display = 'none';
            els.imagePreview.src = '#';
            els.imagePreview.style.display = 'none';
            els.imageDescription.innerText = '';
            els.imageDescription.parentElement.style.display = 'none';
            els.imageProgressContainer.innerHTML = '';
            showToast('Image cleared', 'info');
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

    // --- Model-Specific Prompt Formatting ---
    function formatPromptForModel(prompt, model, promptType) {
        // Default formatting if no specific model is selected
        if (!model || model === 'default') {
            return prompt;
        }
        
        // Only apply model-specific formatting for image prompts
        if (promptType !== 'Image') {
            return prompt;
        }
        
        switch(model) {
            case 'flux':
                return formatFluxPrompt(prompt);
            case 'qwen':
                return formatQwenPrompt(prompt);
            case 'nunchaku':
                return formatNunchakuPrompt(prompt);
            default:
                return prompt;
        }
    }
    
    function formatFluxPrompt(prompt) {
        // Flux excels at photorealistic imagery with detailed technical specifications
        let formattedPrompt = prompt.trim();
        
        // Add photography-related technical details if not present
        if (!formattedPrompt.toLowerCase().includes('lens') && 
            !formattedPrompt.toLowerCase().includes('mm') && 
            !formattedPrompt.toLowerCase().includes('camera')) {
            formattedPrompt += ', shot with a professional camera, high-quality lens';
        }
        
        // Add lighting details if not specified
        if (!formattedPrompt.toLowerCase().includes('lighting') && 
            !formattedPrompt.toLowerCase().includes('light')) {
            formattedPrompt += ', perfect lighting';
        }
        
        // Add quality indicators
        if (!formattedPrompt.toLowerCase().includes('detailed') && 
            !formattedPrompt.toLowerCase().includes('detail')) {
            formattedPrompt += ', highly detailed';
        }
        
        // Add photorealism tag
        if (!formattedPrompt.toLowerCase().includes('photorealistic') && 
            !formattedPrompt.toLowerCase().includes('realistic')) {
            formattedPrompt += ', photorealistic';
        }
        
        return formattedPrompt;
    }
    
    function formatQwenPrompt(prompt) {
        // Qwen works well with clear subject descriptions and compositional instructions
        let formattedPrompt = prompt.trim();
        
        // Add clarity indicator if not present
        if (!formattedPrompt.toLowerCase().includes('clear') && 
            !formattedPrompt.toLowerCase().includes('crisp')) {
            formattedPrompt += ', clear and well-composed';
        }
        
        // Add quality indicator if not present
        if (!formattedPrompt.toLowerCase().includes('high quality') && 
            !formattedPrompt.toLowerCase().includes('high-quality')) {
            formattedPrompt += ', high quality';
        }
        
        return formattedPrompt;
    }
    
    function formatNunchakuPrompt(prompt) {
        // Nunchaku focuses on style coherence and artistic interpretations
        let formattedPrompt = prompt.trim();
        
        // Add style indicator at the beginning if not present
        const styleKeywords = ['cinematic', 'artistic', 'stylized', 'dramatic', 'vibrant'];
        let hasStyleKeyword = false;
        
        for (const keyword of styleKeywords) {
            if (formattedPrompt.toLowerCase().includes(keyword)) {
                hasStyleKeyword = true;
                break;
            }
        }
        
        if (!hasStyleKeyword) {
            formattedPrompt = 'Artistic, stylized: ' + formattedPrompt;
        }
        
        // Add mood/atmosphere if not present
        if (!formattedPrompt.toLowerCase().includes('mood') && 
            !formattedPrompt.toLowerCase().includes('atmosphere')) {
            formattedPrompt += ', atmospheric mood';
        }
        
        return formattedPrompt;
    }
    
    // --- Character Counter Logic ---
    // Model-specific character limits
    const MODEL_CHAR_LIMITS = {
        'Image': {
            'default': 3000,
            'flux': 1200,
            'qwen': 2500,  // Increased Qwen limit to 2500 characters
            'nunchaku': 1500
        },
        'VEO': {
            'default': 1000
        },
        'WAN2': {
            'default': 750
        }
    };
    
    // Get current character limit based on selected model and prompt type
    function getCurrentCharLimit() {
        const promptType = els.promptType ? els.promptType.value : 'Image';
        const modelType = els.modelSelect ? els.modelSelect.value : 'default';
        
        // Get the limit for the specific model if available
        if (MODEL_CHAR_LIMITS[promptType] && MODEL_CHAR_LIMITS[promptType][modelType]) {
            return MODEL_CHAR_LIMITS[promptType][modelType];
        }
        
        // Fall back to default limit for the prompt type
        if (MODEL_CHAR_LIMITS[promptType] && MODEL_CHAR_LIMITS[promptType]['default']) {
            return MODEL_CHAR_LIMITS[promptType]['default'];
        }
        
        // Ultimate fallback
        return 1000;
    }
    
    function updateCharCount() {
        if (els.charCount && els.prompt) {
            const count = els.prompt.value.length;
            const charLimit = getCurrentCharLimit();
            
            // Update the character count display with current count and limit
            els.charCount.textContent = `${count} / ${charLimit}`;
            
            // Create or update the model limit info element
            let modelLimitInfo = document.querySelector('.model-limit-info');
            if (!modelLimitInfo) {
                modelLimitInfo = document.createElement('div');
                modelLimitInfo.className = 'model-limit-info';
                modelLimitInfo.innerHTML = `
                    <span class="limit-label">Character limit: <strong>${charLimit}</strong></span>
                    <span class="limit-warning">Approaching limit!</span>
                `;
                els.charCounter.parentNode.appendChild(modelLimitInfo);
            } else {
                modelLimitInfo.querySelector('.limit-label').innerHTML = 
                    `Character limit: <strong>${charLimit}</strong>`;
            }
            
            // Add visual feedback when approaching limit
            if (els.charCounter) {
                els.charCounter.classList.remove('near-limit', 'at-limit');
                modelLimitInfo.classList.remove('warning');
                
                if (count >= charLimit) {
                    els.charCounter.classList.add('at-limit');
                    modelLimitInfo.classList.add('warning');
                    modelLimitInfo.querySelector('.limit-warning').textContent = 'Limit exceeded!';
                } else if (count >= charLimit * 0.9) {
                    els.charCounter.classList.add('near-limit');
                    modelLimitInfo.classList.add('warning');
                    modelLimitInfo.querySelector('.limit-warning').textContent = 'Approaching limit!';
                }
            }
        }
    }
    
    // Add input event listener to prompt textarea for character counting
    if (els.prompt && els.charCount) {
        els.prompt.addEventListener('input', updateCharCount);
        
        // Add event listener for model selection changes
        if (els.modelSelect) {
            els.modelSelect.addEventListener('change', updateCharCount);
        }
        
        // Add event listener for prompt type changes
        if (els.promptType) {
            els.promptType.addEventListener('change', updateCharCount);
        }
        
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
        console.log('Copy button element found:', els.copy);
        els.copy.addEventListener('click', () => {
            console.log('Copy button clicked!');
            const textToCopy = els.resultText.innerText;
            const resultTextLength = textToCopy.length;
            console.log('Text to copy (length:', resultTextLength, '):', textToCopy);
            console.log('Result text element:', els.resultText);
            console.log('Result text HTML:', els.resultText.innerHTML);
            console.log('Result text textContent:', els.resultText.textContent);

            if (!textToCopy || textToCopy.trim() === '') {
                console.log('No text to copy found - text is empty');
                console.log('Checking if resultText element exists and has content...');
                console.log('resultText exists:', !!els.resultText);
                console.log('resultText children:', els.resultText.children.length);
                console.log('resultText parent:', els.resultText.parentElement);
                showToast('No text to copy!', 'error');
                return;
            }

            // If the text looks like a URL, it might be wrong
            if (textToCopy.startsWith('http') && textToCopy.length < 100) {
                console.error('ERROR: Text appears to be a URL instead of prompt text!');
                console.error('This suggests the resultText element contains the wrong content');
                showToast('Error: Wrong content detected. Please refresh and try again.', 'error');
                return;
            }

            // Double-check what we're about to copy
            console.log('About to copy text:', textToCopy.substring(0, 200) + (textToCopy.length > 200 ? '...' : ''));

            // Check if clipboard API is available
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(textToCopy).then(() => {
                    console.log('Text copied successfully to clipboard');
                    console.log('Copied text preview:', textToCopy.substring(0, 100) + '...');
                    showToast('Enhanced prompt copied to clipboard!', 'success');
                }).catch(err => {
                    console.error('Failed to copy text to clipboard: ', err);
                    console.error('Error details:', err.message);
                    // Use fallback method
                    copyTextFallback(textToCopy);
                });
            } else {
                // Clipboard API not available, use fallback immediately
                console.log('Clipboard API not available, using fallback method');
                copyTextFallback(textToCopy);
            }
            
            // Fallback copy function
            function copyTextFallback(text) {
                try {
                    const textArea = document.createElement('textarea');
                    textArea.value = text;
                    textArea.style.position = 'fixed';
                    textArea.style.left = '-9999px';
                    document.body.appendChild(textArea);
                    textArea.select();
                    const successful = document.execCommand('copy');
                    document.body.removeChild(textArea);
                    if (successful) {
                        console.log('Fallback copy successful');
                        showToast('Enhanced prompt copied to clipboard!', 'success');
                    } else {
                        throw new Error('execCommand copy returned false');
                    }
                } catch (fallbackErr) {
                    console.error('Fallback copy also failed:', fallbackErr);
                    showToast('Copy failed. Please select the text manually and copy (Cmd+C or Ctrl+C).', 'error');
                }
            }
        });
    } else {
        console.log('Copy button element NOT found');
    }
    
    // Share button functionality
    const shareButton = document.getElementById('share-button');
    if (shareButton) {
        shareButton.addEventListener('click', () => {
            // Create a shareable URL with the prompt as a parameter
            const enhancedPrompt = els.resultText.innerText;
            if (!enhancedPrompt) {
                showToast('No prompt to share!', 'error');
                return;
            }
            
            // Encode the prompt and other relevant parameters
            const params = new URLSearchParams();
            params.append('shared_prompt', enhancedPrompt);
            
            // Optional: Add other parameters that might be useful for context
            if (els.promptType.value) params.append('type', els.promptType.value);
            if (els.style.value && els.style.value !== 'None') params.append('style', els.style.value);
            if (els.modelSelect.value && els.modelSelect.value !== 'default') params.append('model', els.modelSelect.value);
            
            // Create the shareable URL
            const shareableUrl = `${window.location.origin}${window.location.pathname}?${params.toString()}`;
            
            // Copy the URL to clipboard
            navigator.clipboard.writeText(shareableUrl).then(() => {
                showToast('Shareable URL copied to clipboard!', 'success');
            }).catch(err => {
                console.error('Failed to copy URL: ', err);
            });
        });
    }
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

    // --- Shortcuts Modal Logic ---
    if (els.shortcuts) {
        els.shortcuts.addEventListener('click', () => {
            const modal = document.getElementById('shortcuts-modal');
            if (modal) {
                modal.style.display = 'flex';
                document.body.style.overflow = 'hidden';
            }
        });
    }

    // Close modal when clicking the close button
    const closeModalButtons = document.querySelectorAll('.close-modal');
    closeModalButtons.forEach(button => {
        button.addEventListener('click', () => {
            const modal = button.closest('.modal');
            if (modal) {
                modal.style.display = 'none';
                document.body.style.overflow = '';
            }
        });
    });

    // Close modal when clicking outside the modal content
    window.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) {
            e.target.style.display = 'none';
            document.body.style.overflow = '';
        }
    });

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
    
    // Initialize history panel
    initHistoryPanel();
    
    // Load shared prompt if available in URL
    loadSharedPrompt();
    
    // Add event listener for clear history button
    if (els.clearHistory) {
        els.clearHistory.addEventListener('click', clearHistory);
    }
    
    // --- Prompt Comparison Feature ---
    // Initialize comparison view
    function initComparisonView() {
        // Load saved prompt from localStorage if available
        const savedPromptData = localStorage.getItem('savedPromptForComparison');
        if (savedPromptData) {
            try {
                const data = JSON.parse(savedPromptData);
                els.savedPrompt.innerText = data.prompt;
            } catch (error) {
                console.error('Error loading saved prompt:', error);
            }
        }
        
        // View toggle buttons
        if (els.singleViewBtn) {
            els.singleViewBtn.addEventListener('click', () => {
                els.singleViewBtn.classList.add('active');
                els.compareViewBtn.classList.remove('active');
                els.singleView.classList.add('active');
                els.compareView.classList.remove('active');
            });
        }
        
        if (els.compareViewBtn) {
            els.compareViewBtn.addEventListener('click', () => {
                els.compareViewBtn.classList.add('active');
                els.singleViewBtn.classList.remove('active');
                els.compareView.classList.add('active');
                els.singleView.classList.remove('active');
                
                // Update current prompt in comparison view
                if (els.resultText && els.currentPrompt) {
                    els.currentPrompt.innerText = els.resultText.innerText;
                }
            });
        }
        
        // Save for comparison button
        if (els.saveToCompare) {
            els.saveToCompare.addEventListener('click', () => {
                if (!els.resultText.innerText) {
                    showToast('No prompt to save for comparison', 'error');
                    return;
                }
                
                // Save to localStorage
                const promptData = {
                    prompt: els.resultText.innerText,
                    timestamp: new Date().toISOString()
                };
                
                localStorage.setItem('savedPromptForComparison', JSON.stringify(promptData));
                
                // Update the saved prompt display
                els.savedPrompt.innerText = els.resultText.innerText;
                
                // Switch to comparison view
                els.compareViewBtn.click();
                
                showToast('Prompt saved for comparison', 'success');
            });
        }
        
        // Clear saved prompt button
        if (els.clearSavedPrompt) {
            els.clearSavedPrompt.addEventListener('click', () => {
                localStorage.removeItem('savedPromptForComparison');
                els.savedPrompt.innerText = '';
                showToast('Saved prompt cleared', 'info');
            });
        }
    }
        // Initialize comparison view
        initComparisonView();
        
        // Debug: Check if action buttons are found
        console.log('=== ACTION BUTTONS DEBUG ===');
        console.log('Copy button found:', !!els.copy);
        console.log('Share button found:', !!document.getElementById('share-button'));
        console.log('Export button found:', !!els.export);
        console.log('Save to compare button found:', !!els.saveToCompare);
        console.log('Result container found:', !!els.result);
        console.log('Result text found:', !!els.resultText);
        console.log('Copy button element:', els.copy);
        console.log('===========================');
    
    // --- Event Listeners for Keyboard Shortcuts ---
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter to enhance prompt
        if (e.ctrlKey && e.key === 'Enter') {
            if (els.enhance && !els.enhance.disabled) {
                els.enhance.click();
            }
            e.preventDefault();
        }
        
        // Escape to clear prompt
        if (e.key === 'Escape') {
            if (document.activeElement === els.prompt) {
                els.prompt.value = '';
                updateCharCount();
                e.preventDefault();
            }
        }
        
        // Ctrl+C to copy enhanced prompt
        if (e.ctrlKey && e.key === 'c') {
            if (els.result.style.display === 'block' && document.getSelection().toString() === '') {
                navigator.clipboard.writeText(els.resultText.innerText)
                    .then(() => showToast('Copied to clipboard!', 'success'))
                    .catch(() => showToast('Failed to copy', 'error'));
                e.preventDefault();
            }
        }
        
        // Ctrl+H to toggle history panel
        if (e.ctrlKey && e.key === 'h') {
            els.historyPanel.classList.toggle('show');
            e.preventDefault();
        }
        
        // Ctrl+M to toggle comparison view
        if (e.ctrlKey && e.key === 'm') {
            if (els.result.style.display === 'block') {
                if (els.singleView.classList.contains('active')) {
                    els.compareViewBtn.click();
                } else {
                    els.singleViewBtn.click();
                }
                e.preventDefault();
            }
        }
        
        // Ctrl+S to save current prompt for comparison
        if (e.ctrlKey && e.key === 's') {
            if (els.result.style.display === 'block' && els.saveToCompare) {
                els.saveToCompare.click();
                e.preventDefault();
            }
        }
    });
    
    // --- Export/Import Functionality ---
    if (els.export) {
        els.export.addEventListener('click', () => {
            const enhancedPrompt = els.resultText.innerText;
            if (!enhancedPrompt) {
                showToast('No prompt to export', 'error');
                return;
            }
            
            // Create export data object
            const exportData = {
                version: '1.0',
                timestamp: new Date().toISOString(),
                prompt: els.prompt.value,
                enhancedPrompt: enhancedPrompt,
                settings: {
                    promptType: els.promptType ? els.promptType.value : null,
                    style: els.style ? els.style.value : null,
                    cinematography: els.cinematography ? els.cinematography.value : null,
                    lighting: els.lighting ? els.lighting.value : null,
                    motionEffect: els.motionEffect ? els.motionEffect.value : null,
                    model: els.modelSelect ? els.modelSelect.value : null,
                    imageDescription: els.imageDescription ? els.imageDescription.innerText : null,
                    textEmphasis: els.textEmphasis ? els.textEmphasis.value : null,
                    textPosition: els.textPosition ? els.textPosition.value : null,
                    textSize: els.textSize ? els.textSize.value : null,
                    textStyle: els.textStyle ? els.textStyle.value : null,
                    textColor: els.textColor ? els.textColor.value : null,
                    textEffect: els.textEffect ? els.textEffect.value : null,
                    textMaterial: els.textMaterial ? els.textMaterial.value : null,
                    textBackground: els.textBackground ? els.textBackground.value : null,
                    textIntegration: els.textIntegration ? els.textIntegration.value : null
                }
            };
            
            // Convert to JSON string
            const jsonString = JSON.stringify(exportData, null, 2);
            
            // Create a blob and download link
            const blob = new Blob([jsonString], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            // Create temporary download link
            const downloadLink = document.createElement('a');
            downloadLink.href = url;
            downloadLink.download = `prompt_${new Date().getTime()}.json`;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            
            // Clean up
            document.body.removeChild(downloadLink);
            URL.revokeObjectURL(url);
            
            showToast('Prompt exported successfully!', 'success');
        });
    }
    
    if (els.importFile) {
        els.importFile.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) return;
            
            // Check file type
            if (file.type !== 'application/json') {
                showToast('Please select a JSON file', 'error');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const importData = JSON.parse(e.target.result);
                    
                    // Validate imported data
                    if (!importData.prompt || !importData.enhancedPrompt || !importData.settings) {
                        throw new Error('Invalid import file format');
                    }
                    
                    // Apply imported data
                    els.prompt.value = importData.prompt;
                    els.resultText.innerText = importData.enhancedPrompt;
                    els.result.style.display = 'block';
                    
                    // Apply settings
                    const settings = importData.settings;
                    if (settings.promptType && els.promptType) els.promptType.value = settings.promptType;
                    if (settings.style && els.style) els.style.value = settings.style;
                    if (settings.cinematography && els.cinematography) els.cinematography.value = settings.cinematography;
                    if (settings.lighting && els.lighting) els.lighting.value = settings.lighting;
                    if (settings.motionEffect && els.motionEffect) els.motionEffect.value = settings.motionEffect;
                    if (settings.model && els.modelSelect) els.modelSelect.value = settings.model;
                    
                    // Apply text emphasis settings if they exist
                    if (settings.textEmphasis && els.textEmphasis) {
                        els.textEmphasis.value = settings.textEmphasis;
                        if (settings.textPosition && els.textPosition) els.textPosition.value = settings.textPosition;
                        if (settings.textSize && els.textSize) els.textSize.value = settings.textSize;
                        if (settings.textStyle && els.textStyle) els.textStyle.value = settings.textStyle;
                        if (settings.textColor && els.textColor) els.textColor.value = settings.textColor;
                        if (settings.textEffect && els.textEffect) els.textEffect.value = settings.textEffect;
                        if (settings.textMaterial && els.textMaterial) els.textMaterial.value = settings.textMaterial;
                        if (settings.textBackground && els.textBackground) els.textBackground.value = settings.textBackground;
                        if (settings.textIntegration && els.textIntegration) els.textIntegration.value = settings.textIntegration;
                    }
                    
                    // Update UI based on prompt type
                    if (els.promptType) els.promptType.dispatchEvent(new Event('change'));
                    
                    // Update character count
                    updateCharCount();
                    
                    showToast('Prompt imported successfully!', 'success');
                    
                    // Add to history
                    addToHistory({
                        prompt: importData.prompt,
                        enhancedPrompt: importData.enhancedPrompt,
                        promptType: settings.promptType,
                        style: settings.style,
                        cinematography: settings.cinematography,
                        lighting: settings.lighting,
                        motionEffect: settings.motionEffect,
                        model: settings.model
                    });
                    
                } catch (error) {
                    console.error('Import error:', error);
                    showToast('Failed to import prompt: ' + error.message, 'error');
                }
                
                // Reset the file input
                els.importFile.value = '';
            };
            
            reader.onerror = () => {
                showToast('Error reading file', 'error');
            };
            
            reader.readAsText(file);
        });
    }
});
