    
document.addEventListener('DOMContentLoaded', () => {
    // --- Initialize Theme Early ---
    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme === 'true') {
        document.documentElement.classList.add('dark-mode');
        document.body.classList.add('dark-mode');
    }

    // Wrapping preset logic is defined later, after 'els' is declared

    // --- Two-slot image selection state ---
    let selectedFiles = []; // holds up to 2 File objects
    let autoReAnalyzeOnChange = true;
    let analyzeDebounceTimer = null;
    // Holds last analysis texts for A/B to enable auto object/subject extraction
    let latestAnalysis = { combined: '', a: '', b: '' };

    let analysisInFlight = null;

    function getImageContextForEnhance() {
        const parts = [];
        if (latestAnalysis && latestAnalysis.combined) parts.push(`Combined analysis:\n${latestAnalysis.combined}`);
        if (latestAnalysis && latestAnalysis.a) parts.push(`Reference A:\n${latestAnalysis.a}`);
        if (latestAnalysis && latestAnalysis.b) parts.push(`Reference B:\n${latestAnalysis.b}`);
        const joined = parts.filter(Boolean).join('\n\n');
        if (joined) return joined;
        return (els.imageDescription && els.imageDescription.innerText) ? els.imageDescription.innerText : '';
    }

    async function ensureAnalysisForEnhance() {
        if (analysisInFlight) return analysisInFlight;

        const hasImages = (selectedFiles && selectedFiles.length > 0)
            || (els.imageUpload && els.imageUpload.files && els.imageUpload.files.length > 0);
        const hasAnyAnalysis = !!(latestAnalysis && (latestAnalysis.combined || latestAnalysis.a || latestAnalysis.b));

        if (!hasImages || hasAnyAnalysis) return true;

        analysisInFlight = (async () => {
            try {
                if (els.analyze && !els.analyze.disabled) {
                    els.analyze.click();
                }

                const start = Date.now();
                const timeoutMs = 60000;
                while (Date.now() - start < timeoutMs) {
                    const done = !(els.analyze && els.analyze.disabled);
                    const nowHasAnyAnalysis = !!(latestAnalysis && (latestAnalysis.combined || latestAnalysis.a || latestAnalysis.b));
                    if (done && nowHasAnyAnalysis) return true;
                    await new Promise(r => setTimeout(r, 200));
                }
                return false;
            } finally {
                analysisInFlight = null;
            }
        })();

        return analysisInFlight;
    }

    function reanalyzeIfEnabled() {
        if (!autoReAnalyzeOnChange) return;
        if (!els || !els.analyze) return;
        if (analyzeDebounceTimer) clearTimeout(analyzeDebounceTimer);
        analyzeDebounceTimer = setTimeout(() => {
            // Only trigger if at least one image present
            if (selectedFiles.length > 0 && !els.analyze.disabled) {
                els.analyze.click();
            }
        }, 500);
    }

    function validateFile(f) {
        const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
        const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
        if (f.size > MAX_FILE_SIZE) return `Image too large (${(f.size/(1024*1024)).toFixed(2)}MB). Max 10MB.`;
        if (!validTypes.includes(f.type)) return `Invalid file type: ${f.type}.`;
        return null;
    }

    function renderSelectedPreviews() {
        // Slot A
        if (selectedFiles[0]) {
            const urlA = URL.createObjectURL(selectedFiles[0]);
            if (els.imagePreview) {
                els.imagePreview.src = urlA;
                els.imagePreview.style.display = 'block';
            }
            if (els.imageSlotA) els.imageSlotA.style.display = 'block';
        } else {
            if (els.imagePreview) {
                els.imagePreview.removeAttribute('src');
                els.imagePreview.style.display = 'none';
            }
            if (els.imageSlotA) els.imageSlotA.style.display = 'none';
        }

        // Slot B
        if (selectedFiles[1]) {
            const urlB = URL.createObjectURL(selectedFiles[1]);
            if (els.imagePreviewB) {
                els.imagePreviewB.src = urlB;
                els.imagePreviewB.style.display = 'block';
            }
            if (els.imageSlotB) els.imageSlotB.style.display = 'block';
        } else {
            if (els.imagePreviewB) {
                els.imagePreviewB.removeAttribute('src');
                els.imagePreviewB.style.display = 'none';
            }
            if (els.imageSlotB) els.imageSlotB.style.display = 'none';
        }

        // Add second image prompt
        if (els.addSecondImage) {
            els.addSecondImage.style.display = selectedFiles.length === 1 ? 'block' : 'none';
        }

        // Show/Hide swap button (only when both images are present)
        if (els.swapImages) {
            els.swapImages.style.display = selectedFiles.length === 2 ? 'inline-block' : 'none';
        }

        // Show/Hide clear button and container
        if (els.imageResult) els.imageResult.style.display = selectedFiles.length > 0 ? 'block' : 'none';
        if (els.clearImage) els.clearImage.style.display = selectedFiles.length > 0 ? 'inline-block' : 'none';
    }

    function addFilesToSlots(fileList) {
        const incoming = Array.from(fileList);
        for (const f of incoming) {
            if (selectedFiles.length >= 2) break;
            const err = validateFile(f);
            if (err) { showToast(err, 'error'); continue; }
            selectedFiles.push(f);
        }
        renderSelectedPreviews();
        // If user replaced an image, auto re-analyze
        reanalyzeIfEnabled();
    }

    function replaceSlot(index) {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.addEventListener('change', () => {
            const f = input.files && input.files[0];
            if (!f) return;
            const err = validateFile(f);
            if (err) { showToast(err, 'error'); return; }
            selectedFiles[index] = f;
            renderSelectedPreviews();
            // Auto re-analyze after replacement
            reanalyzeIfEnabled();
        });
        input.click();
    }

    function removeSlot(index) {
        if (index === 0) {
            // Shift B to A if exists
            selectedFiles = selectedFiles[1] ? [selectedFiles[1]] : [];
        } else {
            selectedFiles = selectedFiles[0] ? [selectedFiles[0]] : [];
        }
        renderSelectedPreviews();
        // Auto re-analyze or clear analysis if none left
        if (selectedFiles.length > 0) {
            reanalyzeIfEnabled();
        } else {
            if (els.imageDescription) els.imageDescription.innerHTML = '';
            if (els.imageDescriptionWrapper) els.imageDescriptionWrapper.style.display = 'none';
        }
    }

    // (listener hookups moved below after 'els' is defined)
    
    // --- Image Analysis Collapse State ---
    let isAnalysisExpanded = false; // Start collapsed to save space
    
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
        imagePreviewB: document.getElementById('image-preview-b'),
        imageSlotA: document.getElementById('image-slot-a'),
        imageSlotB: document.getElementById('image-slot-b'),
        swapImages: document.getElementById('swap-images'),
        replaceA: document.getElementById('replace-a'),
        removeA: document.getElementById('remove-a'),
        replaceB: document.getElementById('replace-b'),
        removeB: document.getElementById('remove-b'),
        addSecondImage: document.getElementById('add-second-image'),
        imageProgressContainer: document.getElementById('image-progress-container'),
        imageDescription: document.getElementById('image-description-text'),
        imageDescriptionWrapper: document.getElementById('image-description'),
        motionEffectContainer: document.getElementById('motion-effect-selector-container'),
        historyPanel: document.querySelector('.history-panel'),
        compareResult: document.getElementById('compare-result'),
        compareResultText: document.getElementById('compare-result-text'),
        historyList: document.getElementById('history-list'),
        charCounter: document.querySelector('.char-counter'),
        charCount: document.getElementById('char-count'),
        dropZone: document.getElementById('drop-zone'),
        
        // Progress Steps
        steps: document.querySelectorAll('.step'),
        step1: document.querySelector('.step[data-step="1"]'),
        step2: document.querySelector('.step[data-step="2"]'),
        
        // Theme
        themeToggle: document.getElementById('theme-toggle'),

        // Wrapping Preset controls
        wrapType: document.getElementById('wrap-type'),
        wrapFinish: document.getElementById('wrap-finish'),
        wrapScopePreset: document.getElementById('wrap-scope-preset'),
        wrapScope: document.getElementById('wrap-scope'),
        enforcePalette: document.getElementById('enforce-palette'),
        paletteList: document.getElementById('palette-list'),
        neutralizeB: document.getElementById('neutralize-b'),
        multiStage: document.getElementById('multi-stage'),
        accessoriesOnly: document.getElementById('accessories-only'),
        plainEnhance: document.getElementById('plain-enhance'),
        insertWrapPrompt: document.getElementById('insert-wrap-prompt'),
        objectOverrideContainer: document.getElementById('object-override-container'),
        objectOverride: document.getElementById('object-override'),
        // Vehicle-specific controls
        vehicleControls: document.getElementById('vehicle-controls'),
        transferVehicleColor: document.getElementById('transfer-vehicle-color'),
        // Logo-specific controls
        logoControls: document.getElementById('logo-controls'),
        logoPlacement: document.getElementById('logo-placement'),
        logoSize: document.getElementById('logo-size'),
        logoPattern: document.getElementById('logo-pattern'),
        // Character-specific controls
        characterControls: document.getElementById('character-controls'),
        characterPlacement: document.getElementById('character-placement'),
        characterLogoSize: document.getElementById('character-logo-size'),
        characterIntegration: document.getElementById('character-integration'),
        characterAnimated: document.getElementById('character-animated'),
        // Sticker pack controls
        stickerControls: document.getElementById('sticker-controls'),
        stickerCount: document.getElementById('sticker-count'),
        stickerStyle: document.getElementById('sticker-style'),
        stickerCharacterStyle: document.getElementById('sticker-character-style'),
        stickerSheetOutput: document.getElementById('sticker-sheet-output'),
        stickerCutType: document.getElementById('sticker-cut-type'),
        stickerBorderThickness: document.getElementById('sticker-border-thickness'),
        stickerCamera: document.getElementById('sticker-camera'),
        stickerInnerKeyline: document.getElementById('sticker-inner-keyline'),
        stickerVariation: document.getElementById('sticker-variation'),
        stickerPlanSource: document.getElementById('sticker-plan-source'),
        bubbleTextContainer: document.getElementById('bubble-text-container'),
        bubbleText: document.getElementById('bubble-text')
    };

    // --- Plain Enhance persistence and UI toggle ---
    function togglePlainEnhanceUI() {
        const on = !!(els.plainEnhance && els.plainEnhance.checked);
        
        // Hide elements by finding their parent label or container, but NOT the shared selector
        const hideLabel = (el) => {
            if (!el) return;
            const label = el.closest('label');
            if (label) label.style.display = (on ? 'none' : '');
        };
        
        const hideContainer = (el) => {
            if (!el) return;
            el.style.display = (on ? 'none' : '');
        };
        
        // Hide wrap-related UI when Plain Enhance is ON
        // Hide sibling checkboxes in the same row (use hideLabel for individual labels)
        hideLabel(els.neutralizeB);
        hideLabel(els.multiStage);
        hideLabel(els.accessoriesOnly);
        
        // Hide other wrap control containers
        hideContainer(els.objectOverrideContainer);
        hideContainer(els.vehicleControls);
        hideContainer(els.logoControls);
        hideContainer(els.characterControls);
        hideContainer(els.stickerControls);
        
        // Hide wrap-type selector and related controls (find their parent .selector)
        if (els.wrapType) {
            const wrapSelector = els.wrapType.closest('.selector');
            if (wrapSelector) wrapSelector.style.display = (on ? 'none' : '');
        }
        
        // Hide palette controls
        if (els.enforcePalette) {
            const paletteSelector = els.enforcePalette.closest('.selector');
            if (paletteSelector) paletteSelector.style.display = (on ? 'none' : '');
        }
        
        // Hide insert button
        if (els.insertWrapPrompt) {
            const btnContainer = els.insertWrapPrompt.parentElement;
            if (btnContainer) btnContainer.style.display = (on ? 'none' : '');
        }
    }

    // Initialize Plain Enhance state (default ON) and persist
    if (els.plainEnhance) {
        const savedPlain = localStorage.getItem('plainEnhance');
        // Default to true if not set
        const shouldBeOn = savedPlain === null ? true : (savedPlain === 'true');
        els.plainEnhance.checked = shouldBeOn;
        togglePlainEnhanceUI();
        els.plainEnhance.addEventListener('change', () => {
            localStorage.setItem('plainEnhance', els.plainEnhance.checked ? 'true' : 'false');
            togglePlainEnhanceUI();
        });
    }

    // Populate object override select with detected candidates from Reference B
    function populateObjectOverrideOptions(subjectHint) {
        if (!els.objectOverride) return;
        // Clear previous
        els.objectOverride.innerHTML = '';
        const text = (latestAnalysis.b || latestAnalysis.combined || '').toLowerCase();
        if (!text) return;

        const tiers = [
            // High-priority accessories around head/neck
            ['necklace','pendant','scarf','turtleneck','mask','headband','hairpin','earrings','glasses','sunglasses','hat','cap','brooch'],
            // Hand/wrist/waist small items
            ['bracelet','watch','ring','belt','pin','wallet','cardholder'],
            // Larger outfit/apparel (may restage the look)
            ['overcoat','suit','cape','gauntlets','coat','jacket','blazer','cardigan','sweater','pullover','hoodie','sweatshirt','vest','parka','raincoat','windbreaker','poncho','trench coat',
             'shirt','blouse','polo','t-shirt','tee','tank top','tank','camisole','tube top','crop top','bodysuit',
             'dress','gown','maxi dress','mini dress','midi dress','jumpsuit','romper','playsuit','shawl','hood',
             'handbag','bag','purse','backpack','pants','trousers','jeans','skirt','shorts','leggings','joggers','sweatpants','culottes','capris','overall','dungarees',
             'boots','boot','ankle boots','knee-high boots','thigh-high boots','shoe','loafers','heels','pumps','stilettos','platforms','wedges','oxfords','moccasins','sandal','flip-flop','sneaker','trainer','slippers','clogs','espadrilles','gloves']
        ];
        const hasWord = (w) => {
            const escaped = w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const re = new RegExp(`\\b${escaped}\\b`, 'i');
            return re.test(text);
        };

        const candidates = [];
        const accessoriesOnly = !!(els.accessoriesOnly && els.accessoriesOnly.checked);
        const activeTiers = accessoriesOnly ? tiers.slice(0, 2) : tiers; // exclude large outfit tier if accessories only
        for (const tier of activeTiers) {
            for (const obj of tier) {
                if (hasWord(obj) && !candidates.includes(obj)) candidates.push(obj);
            }
        }

        // Fallback to a few generic choices if nothing detected
        const list = candidates.length ? candidates : (accessoriesOnly
            ? ['necklace','earrings','bracelet','ring','scarf','turtleneck','belt','watch','wallet','cardholder']
            : ['necklace','earrings','bracelet','ring','scarf','turtleneck','handbag','wallet','cardholder','overcoat']
        );
        for (const label of list) {
            const opt = document.createElement('option');
            opt.value = label;
            opt.textContent = label;
            els.objectOverride.appendChild(opt);
        }
    }

    function toggleObjectOverrideVisibility() {
        if (!els.objectOverrideContainer || !els.wrapType) return;
        const isPeopleObject = els.wrapType.value === 'people-object';
        const isCharacter = els.wrapType.value === 'character';
        const isLogo = els.wrapType.value === 'logo';
        const isVehicle = els.wrapType.value === 'vehicle';
        const isStickerPack = els.wrapType.value === 'sticker-pack';
        
        // Hide all control panels first
        if (els.vehicleControls) els.vehicleControls.style.display = 'none';
        if (els.logoControls) els.logoControls.style.display = 'none';
        if (els.characterControls) els.characterControls.style.display = 'none';
        if (els.stickerControls) els.stickerControls.style.display = 'none';
        
        // Show relevant controls
        if (els.vehicleControls) els.vehicleControls.style.display = isVehicle ? 'block' : 'none';
        if (els.logoControls) els.logoControls.style.display = isLogo ? 'block' : 'none';
        if (els.characterControls) els.characterControls.style.display = isCharacter ? 'block' : 'none';
        if (els.stickerControls) els.stickerControls.style.display = isStickerPack ? 'block' : 'none';
        
        // Handle object override container
        els.objectOverrideContainer.style.display = isPeopleObject ? 'flex' : 'none';
        if (isPeopleObject) {
            const subj = detectSubjectFromA(latestAnalysis.a, latestAnalysis.combined) || 'subject';
            populateObjectOverrideOptions(subj);
        }
    }

    // Apply saved theme state and wire up the toggle
    applySavedTheme();
    if (els.themeToggle) {
        els.themeToggle.checked = (localStorage.getItem('darkMode') === 'true');
        els.themeToggle.addEventListener('change', function() {
            const isDark = this.checked;
            document.documentElement.classList.toggle('dark-mode', isDark);
            document.body.classList.toggle('dark-mode', isDark);
            localStorage.setItem('darkMode', isDark);
            showToast(isDark ? ' Dark mode enabled' : ' Light mode enabled', 'info', 2000);
        });
    }

    // Hook wrap-type change for object override UI
    if (els.wrapType) {
        els.wrapType.addEventListener('change', toggleObjectOverrideVisibility);
        // Initialize once on load
        toggleObjectOverrideVisibility();
    }
    if (els.accessoriesOnly) {
        els.accessoriesOnly.addEventListener('change', () => populateObjectOverrideOptions());
    }

    // --- Hookups that require 'els' ---
    // File input -> two-slot state
    if (els.imageUpload) {
        els.imageUpload.addEventListener('change', () => {
            const files = els.imageUpload.files;
            if (!files || files.length === 0) return;
            addFilesToSlots(files);
            // Clear native input to allow re-selecting same file later
            els.imageUpload.value = '';
        });
    }

    // Add second image button (delegated to container to catch inner button)
    if (els.addSecondImage) {
        els.addSecondImage.addEventListener('click', () => {
            if (selectedFiles.length >= 2) return;
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.addEventListener('change', () => {
                if (input.files && input.files[0]) addFilesToSlots([input.files[0]]);
            });
            input.click();
        });
    }

    // Replace/Remove per-slot controls
    if (els.replaceA) els.replaceA.addEventListener('click', () => { if (selectedFiles[0]) replaceSlot(0); else showToast('No image in slot A.', 'warning'); });
    if (els.removeA) els.removeA.addEventListener('click', () => { if (selectedFiles[0]) removeSlot(0); });
    if (els.replaceB) els.replaceB.addEventListener('click', () => { if (selectedFiles.length === 1) { showToast('Add a second image first.', 'info'); } else if (selectedFiles[1]) { replaceSlot(1); } });
    if (els.removeB) els.removeB.addEventListener('click', () => { if (selectedFiles[1]) removeSlot(1); });
    
    // Swap images button
    if (els.swapImages) {
        els.swapImages.addEventListener('click', () => {
            if (selectedFiles.length === 2) {
                [selectedFiles[0], selectedFiles[1]] = [selectedFiles[1], selectedFiles[0]];
                renderSelectedPreviews();
                reanalyzeIfEnabled();
                showToast('Images swapped: A ⇄ B', 'success');
            }
        });
    }

    // Drag & Drop for the drop zone
    if (els.dropZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
            els.dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        ['dragenter', 'dragover'].forEach(evt => {
            els.dropZone.addEventListener(evt, () => els.dropZone.classList.add('is-dragover'));
        });
        ['dragleave', 'drop'].forEach(evt => {
            els.dropZone.addEventListener(evt, () => els.dropZone.classList.remove('is-dragover'));
        });
        els.dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt && dt.files ? dt.files : null;
            if (!files || files.length === 0) return;
            addFilesToSlots(files);
        });
    }

    // --- Drag and Drop to Swap Image Slots ---
    function setupImageSlotSwap() {
        if (!els.imageSlotA || !els.imageSlotB) return;
        
        let draggedSlot = null;
        
        // Drag start
        [els.imageSlotA, els.imageSlotB].forEach(slot => {
            slot.addEventListener('dragstart', (e) => {
                draggedSlot = slot;
                e.dataTransfer.effectAllowed = 'move';
                slot.style.opacity = '0.4';
            });
            
            slot.addEventListener('dragend', (e) => {
                slot.style.opacity = '1';
                draggedSlot = null;
            });
            
            slot.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                if (slot !== draggedSlot) {
                    slot.style.outline = '2px dashed var(--primary-color)';
                }
            });
            
            slot.addEventListener('dragleave', (e) => {
                slot.style.outline = 'none';
            });
            
            slot.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                slot.style.outline = 'none';
                
                // Only swap if dropping on the other slot
                if (draggedSlot && draggedSlot !== slot && selectedFiles.length === 2) {
                    // Swap the files in the array
                    [selectedFiles[0], selectedFiles[1]] = [selectedFiles[1], selectedFiles[0]];
                    renderSelectedPreviews();
                    reanalyzeIfEnabled();
                    showToast('Images swapped', 'success');
                }
            });
        });
    }
    
    // Initialize swap functionality after DOM is ready
    setupImageSlotSwap();

    // --- Wrapping Preset Logic ---
    function extractPaletteFromText(text) {
        if (!text) return '';
        const colors = [
            'black','white','gray','grey','silver','graphite','charcoal',
            'blue','sky blue','light blue','cyan','aqua','teal',
            'green','lime','olive','forest green',
            'red','crimson','burgundy','maroon',
            'orange','coral','peach',
            'yellow','gold','amber',
            'beige','cream','ivory','tan',
            'pink','magenta','fuchsia',
            'purple','violet','lavender','indigo',
            'brown','chocolate','bronze'
        ];
        const found = [];
        const lower = text.toLowerCase();
        colors.forEach(c => { 
            if (lower.includes(c) && !found.includes(c)) found.push(c); 
        });
        return found.slice(0, 6).join(', ');
    }

    function detectObjectType(analysisText, scope, scopePreset) {
        const text = (analysisText + ' ' + scope).toLowerCase();
        
        // Opaque/colored glass detection
        if ((scopePreset && scopePreset === 'glass-opaque') ||
            text.includes('opaque glass') || text.includes('colored glass') ||
            text.includes('frosted glass') || text.includes('milk glass') ||
            text.includes('ceramic glass') || text.includes('painted glass')) {
            return 'glass-opaque';
        }
        
        // Transparent glass detection
        if ((scopePreset && (scopePreset === 'glass-bowl' || scopePreset === 'glass-full')) ||
            text.includes('wine glass') || text.includes('wineglass') ||
            text.includes('champagne glass') || text.includes('goblet') ||
            text.includes('cocktail glass') || text.includes('martini glass') ||
            text.includes('tumbler') || text.includes('drinking glass')) {
            return 'glass';
        }
        
        // Can detection (aluminum/metal cylindrical) - MUST be checked BEFORE bottle
        if (text.includes(' can ') || text.includes(' can,') || text.includes(' can.') || text.includes(' can\'') ||
            text.includes('aluminum can') || text.includes('aluminium can') || 
            text.includes('beer can') || text.includes('soda can') || 
            text.includes('energy drink') || text.includes('beverage can') || 
            text.includes('tin can') || text.includes('metal can') || 
            text.includes('drink can') || text.includes('pop can') ||
            (text.includes('aluminum') && text.includes('beverage')) ||
            (text.includes('aluminium') && text.includes('beverage')) ||
            (text.includes('metal') && text.includes(' can'))) {
            return 'can';
        }
        
        // Bottle detection (glass/plastic bottles with necks)
        if (text.includes('bottle') || text.includes('flask') || 
            text.includes('wine bottle') || text.includes('beer bottle') ||
            text.includes('water bottle') || text.includes('soda bottle')) {
            return 'bottle';
        }
        
        // Cup/mug detection (opaque drinking vessels)
        if (text.includes('coffee cup') || text.includes('mug') ||
            text.includes('teacup') || text.includes('ceramic cup') ||
            text.includes('disposable cup') || text.includes('paper cup')) {
            return 'cup';
        }
        
        // Jar detection (wide-mouth containers)
        if (text.includes('jar') || text.includes('mason jar') ||
            text.includes('preserve jar') || text.includes('canister')) {
            return 'jar';
        }
        
        // Tube/squeeze container
        if (text.includes('tube') || text.includes('squeeze bottle') ||
            text.includes('toothpaste') || text.includes('cosmetic tube')) {
            return 'tube';
        }
        
        // Default to generic container
        return 'container';
    }

    function buildMultiStagePrompt(wrapType, finish, scope, palette, analysisText, scopePreset) {
        // Multi-stage prompting for better consistency
        const extracted = palette || extractPaletteFromText(analysisText);
        const colors = extracted || 'Reference A colors';
        
        // Detect object type
        const objectType = detectObjectType(analysisText, scope, scopePreset);
        
        if (wrapType === 'product') {
            // Object-specific prompts based on type
            switch(objectType) {
                case 'glass-opaque':
                    return `STAGE 1 - BASE COVERAGE ON OPAQUE GLASS:
Create Reference B (opaque/colored/frosted glass) with complete surface coverage using the background/base colors from Reference A (${colors}). For opaque glass: replace the glass color entirely with Reference A colors. For frosted glass: apply solid color over frosted surface. Cover the bowl/cup area completely from rim to base - complete coverage with no original glass color visible. Stem and base can remain in original color if present. Solid color foundation - no artwork yet.

STAGE 2 - ADD MAIN ARTWORK:
Place the primary character/subject from Reference A on the front face of the now-colored opaque glass from Stage 1. Maintain exact proportions and details; correct perspective for glass shape (curved/conical/cylindrical); artwork sits naturally on the colored base. Design wraps around the bowl curvature.

STAGE 3 - ADD DETAILS & FINALIZE:
Add logos, text, or secondary elements from Reference A. Apply ${finish === 'glossy' ? 'smooth glossy' : finish === 'matte' ? 'matte/frosted' : finish} finish; maintain glass shape and any remaining transparency/translucency on stem/base; photorealistic glassware product shot with studio lighting; clean backdrop.

CRITICAL: Opaque glass body is fully colored (no original color showing). Glass shape preserved. Design covers bowl area completely. Stem/base can stay original if applicable.`;

                case 'glass':
                    return `STAGE 1 - APPLY BASE DESIGN TO GLASS:
Take Reference B (glass/wine glass/tumbler) and apply the background/base colors and patterns from Reference A (${colors}) as a printed/etched design ON the glass surface. For transparent glass: design appears as opaque/frosted print. Cover the bowl/cup area completely - no bare/clear glass on the main drinking surface. Maintain glass transparency on stem and base if present.

STAGE 2 - ADD MAIN ARTWORK:
Place the primary character/subject from Reference A on the front-facing side of the glass bowl from Stage 1. The artwork appears as a high-quality print/decal on the glass surface. Correct perspective for curved/conical glass shape; artwork wraps naturally around the bowl curvature. Character details remain sharp and visible.

STAGE 3 - FINALIZE WITH GLASS PROPERTIES:
Add logos, text, or secondary elements from Reference A. Apply ${finish === 'glossy' ? 'smooth printed' : finish === 'matte' ? 'frosted/etched' : finish} surface treatment to the design areas. Maintain glass reflections, highlights, and transparency where appropriate; photorealistic glassware product shot with studio lighting; clean backdrop.

CRITICAL: Design is ON the glass surface (printed/etched), not replacing the glass material. Glass shape and transparency are preserved. Full coverage on bowl area with no bare clear spots.`;

                case 'bottle':
                    return `STAGE 1 - BASE LABEL COVERAGE:
Create label wrap for Reference B (bottle) using background/base colors from Reference A (${colors}). Cover bottle body from neck to base - complete 360° wrap. For glass bottles: opaque label with no bottle color showing through. For plastic bottles: full-coverage wrap. Top neck and cap can remain visible. Solid color foundation - no artwork yet.

STAGE 2 - ADD MAIN ARTWORK:
Place the primary character/subject from Reference A centered on the front face of the bottle label from Stage 1. Maintain exact proportions and details; correct perspective for bottle shape (may be cylindrical, tapered, or curved); artwork sits naturally on the colored label base.

STAGE 3 - ADD DETAILS & FINALIZE:
Add logos, text, or secondary elements from Reference A to appropriate positions (logo near top/shoulder area). Apply ${finish} label finish; photorealistic product bottle shot with studio lighting; clean backdrop. Ensure label edges are clean and wrapped completely around bottle.

CRITICAL: Full label coverage on bottle body with no gaps. Bottle shape (cylindrical/tapered/curved) is preserved. Label wraps seamlessly 360° around bottle.`;

                case 'can':
                    return `STAGE 1 - BASE COVERAGE:
Create Reference B (aluminum/metal can) with complete surface coverage using the background/base colors from Reference A (${colors}). Cover EVERY surface: lid/top, shoulder curve, entire cylindrical body (360°), and bottom rim. Zero bare aluminum, silver, or original metal surface visible anywhere. Solid color foundation - no artwork yet.

STAGE 2 - ADD MAIN ARTWORK:
Place the primary character/subject from Reference A centered on the front face of the now-colored can from Stage 1. Maintain exact proportions and details; correct radial perspective for cylindrical surface; artwork sits naturally on the colored base.

STAGE 3 - ADD DETAILS & FINALIZE:
Add any logos, text, or secondary elements from Reference A to appropriate positions (logo near top/lid area). Ensure seamless integration with Stage 2 artwork. Apply ${finish} finish; photorealistic product mockup with studio lighting; clean backdrop.

CRITICAL: Each stage builds on the previous. Stage 1 MUST cover entire can including lid/top with no bare metal. Stage 2 adds character. Stage 3 adds finishing details.`;

                case 'cup':
                    return `STAGE 1 - BASE CUP COVERAGE:
Create Reference B (cup/mug) with complete surface coverage using the background/base colors from Reference A (${colors}). Cover the cup body 360° from rim to base - complete wrap. Handle remains in original color if present. For ceramic/opaque cups: replace surface color entirely. Solid color foundation - no artwork yet.

STAGE 2 - ADD MAIN ARTWORK:
Place the primary character/subject from Reference A centered on the front face of the cup from Stage 1. Maintain exact proportions and details; correct perspective for cup shape (cylindrical/tapered); artwork wraps naturally around curved surface.

STAGE 3 - ADD DETAILS & FINALIZE:
Add logos, text, or secondary elements from Reference A. Apply ${finish} surface finish; photorealistic product cup shot with studio lighting; clean backdrop. Ensure design wraps completely around cup with no gaps.

CRITICAL: Full coverage on cup body with no bare spots. Cup shape and handle preserved. Design wraps 360° seamlessly.`;

                case 'jar':
                    return `STAGE 1 - JAR LABEL COVERAGE:
Create label for Reference B (jar) using background/base colors from Reference A (${colors}). Cover jar body from shoulder to base - full 360° wrap. Wide-mouth jar: label covers main cylindrical section. Lid and neck can remain visible. Solid color foundation - no artwork yet.

STAGE 2 - ADD MAIN ARTWORK:
Place the primary character/subject from Reference A centered on the front face of the jar label from Stage 1. Maintain exact proportions and details; correct perspective for jar shape (usually straight cylindrical); artwork sits naturally on the colored label base.

STAGE 3 - ADD DETAILS & FINALIZE:
Add logos, text, or secondary elements from Reference A. Apply ${finish} label finish; photorealistic product jar shot with studio lighting; clean backdrop. Ensure label covers jar body completely.

CRITICAL: Full label coverage on jar body with no gaps. Jar shape preserved. Label wraps seamlessly 360°.`;

                case 'tube':
                    return `STAGE 1 - TUBE WRAP COVERAGE:
Create Reference B (tube/squeeze container) with complete surface coverage using the background/base colors from Reference A (${colors}). Cover the tube body from opening/cap to bottom seal - full 360° wrap. Cap can remain in original color. Solid color foundation - no artwork yet.

STAGE 2 - ADD MAIN ARTWORK:
Place the primary character/subject from Reference A on the front face of the tube from Stage 1. Maintain exact proportions and details; account for tube curvature and potential deformation when squeezed; artwork sits naturally on the colored base.

STAGE 3 - ADD DETAILS & FINALIZE:
Add logos, text, or secondary elements from Reference A. Apply ${finish} finish; photorealistic product tube shot with studio lighting; clean backdrop. Ensure wrap covers tube completely.

CRITICAL: Full coverage on tube body with no bare spots. Tube shape preserved. Design wraps seamlessly around tube.`;

                default:
                    if (analysisText.toLowerCase().includes('car') || analysisText.toLowerCase().includes('vehicle') || 
                        analysisText.toLowerCase().includes('auto') || analysisText.toLowerCase().includes('automotive')) {
                        return `STAGE 1 - VEHICLE SURFACE PREP:
Analyze Reference B (target vehicle). Identify and map all body panels, curves, and contours. Create a clean template that matches the exact shape and dimensions of Reference B.

STAGE 2 - DESIGN TRANSFER:
Transfer the complete design, colors, and branding from Reference A (source vehicle) onto the template from Stage 1. Match the following elements:
- Color scheme and gradients
- Racing stripes or graphic patterns
- Brand logos and decals
- Sponsorship graphics
- Any text or numbering

STAGE 3 - INTEGRATION & RENDERING:
Apply the design to Reference B with perfect alignment to body lines and curves. Ensure proper perspective and scaling for all elements. Add realistic material properties (${finish} finish) and ensure all graphics follow the vehicle's contours naturally.

CRITICAL: 
- Maintain Reference B's exact shape, proportions, and perspective
- All graphics must follow the vehicle's curves and contours precisely
- Preserve the lighting and environment from Reference B
- Ensure all text and logos are properly oriented and readable
- Maintain high resolution and crisp edges on all design elements`;
                    } else {
                        // Generic container
                        return `STAGE 1 - BASE COVERAGE:
Create Reference B (container) with complete surface coverage using the background/base colors from Reference A (${colors}). Cover all visible surfaces with solid color foundation - no artwork yet.

STAGE 2 - ADD MAIN ARTWORK:
Place the primary character/subject from Reference A on the main visible face from Stage 1. Maintain exact proportions and details; correct perspective for surface shape.

STAGE 3 - ADD DETAILS & FINALIZE:
Add logos, text, or secondary elements from Reference A. Apply ${finish} finish; photorealistic product shot with studio lighting; clean backdrop.

CRITICAL: Full coverage with no bare spots. Shape preserved. Design applied appropriately for object type`;
                    }
            }
        } else if (wrapType === 'logo') {
            // Read logo-specific controls
            const placement = els.logoPlacement ? els.logoPlacement.value : 'center';
            const size = els.logoSize ? els.logoSize.value : 'medium';
            const isPattern = els.logoPattern ? els.logoPattern.checked : false;
            
            const placementMap = {
                'center': 'centered on the main surface',
                'top-center': 'top-center position',
                'bottom-center': 'bottom-center position',
                'left': 'left side',
                'right': 'right side',
                'custom': scope
            };
            const placementText = placementMap[placement] || 'centered on the main surface';
            
            const sizeMap = {
                'small': 'small and subtle',
                'medium': 'medium size, clearly visible',
                'large': 'large and prominent',
                'full': 'full coverage',
                'proportional': 'proportionally sized to fit the surface'
            };
            const sizeText = sizeMap[size] || 'medium size, clearly visible';
            
            if (isPattern) {
                return `STAGE 1 - PREPARE SURFACE:
Neutralize Reference B surface, replacing any existing colors/patterns with a clean base.

STAGE 2 - APPLY PATTERN:
Apply the logo/branding from Reference A as a repeating pattern across Reference B, ${sizeText}. Pattern repeats uniformly with consistent spacing; maintain logo aspect ratio and orientation on each instance.

STAGE 3 - FINALIZE & RENDER:
${finish} finish with vibrant colors; photorealistic application; maintain Reference B's form and structure; crisp edges and sharp details on each logo instance.`;
            } else {
                return `STAGE 1 - PREPARE SURFACE:
Neutralize Reference B surface, replacing any existing colors/patterns with a clean base that accepts the logo placement.

STAGE 2 - PLACE LOGO:
Apply the logo/branding from Reference A at ${placementText}, ${sizeText}. Maintain exact aspect ratio, crisp edges, and correct positioning.

STAGE 3 - INTEGRATE & RENDER:
${finish} finish with vibrant colors; blend logo naturally into Reference B lighting and shadows; photorealistic application; maintain B's form and structure.`;
            }
        } else {
            // Full/partial/decal wrap
            return `STAGE 1 - COLOR FOUNDATION:
Replace Reference B surface colors with Reference A color palette (${colors}). Maintain B's exact shape and structure while neutralizing original surface colors.

STAGE 2 - APPLY GRAPHICS:
Transfer graphics, patterns, and artwork from Reference A onto the color-prepared surface from Stage 1. Correct perspective and mapping for surface curvature; no warping.

STAGE 3 - FINALIZE DETAILS:
Add fine details, text, logos; apply ${finish} finish; photorealistic rendering with proper lighting and reflections.`;
        }
    }

    // --- Heuristics to extract key object (from B) and subject (from A) ---
    function extractKeyObjectFromB(analysisB, combined) {
        const text = (analysisB || combined || '').toLowerCase();
        if (!text) return null;
        const objects = [
            'necklace','pendant','earrings','ring','bracelet','watch','scarf','turtleneck','tie','bow tie','hat','cap','glasses','sunglasses','mask','helmet','bandana',
            'handbag','bag','purse','backpack','wallet','cardholder','belt','brooch','pin','hairpin','headband','shawl','hood','gloves','tie clip','cufflinks','suspenders','sash',
            'overcoat','coat','jacket','blazer','cardigan','sweater','pullover','hoodie','sweatshirt','vest','parka','raincoat','windbreaker','poncho','trench coat',
            'shirt','blouse','polo','t-shirt','tee','tank top','tank','camisole','tube top','crop top','bodysuit',
            'dress','gown','maxi dress','mini dress','midi dress','jumpsuit','romper','playsuit',
            'pants','trousers','jeans','skirt','shorts','leggings','joggers','sweatpants','culottes','capris','overall','dungarees',
            'boots','boot','ankle boots','knee-high boots','thigh-high boots','shoe','loafers','heels','pumps','stilettos','platforms','wedges','oxfords','moccasins','sandal','flip-flop','sneaker','trainer','slippers','clogs','espadrilles'
        ];
        const materials = ['bead','beaded','metal','gold','silver','steel','leather','fabric','silk','wool','cotton','linen','denim','velvet','crystal','gem','stone','pearl','glass'];
        // Only return pure color words (no preceding tokens)
        const colorsRe = /\b(blue|green|red|yellow|orange|purple|violet|pink|teal|cyan|magenta|gold|silver|black|white|gray|grey|brown|amber|cream|beige)\b/g;

        const hasWord = (w) => {
            // Escape regex special chars in word
            const escaped = w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const re = new RegExp(`\\b${escaped}\\b`, 'i');
            return re.test(text);
        };

        const foundObj = objects.find(o => hasWord(o));
        if (!foundObj) return null;
        const foundMats = materials.filter(m => hasWord(m));
        const colors = Array.from(text.matchAll(colorsRe)).map(m => m[0]).slice(0, 6);
        return {
            label: foundObj,
            materials: foundMats,
            colors
        };
    }

    // Prioritized extraction for human subjects: prefer wearable accessories near face/neck
    function extractKeyObjectFromBWithPriority(subject, analysisB, combined) {
        const text = (analysisB || combined || '').toLowerCase();
        if (!text) return null;
        const hasWord = (w) => {
            const escaped = w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const re = new RegExp(`\\b${escaped}\\b`, 'i');
            return re.test(text);
        };

        // Candidate lists by priority for human portrait edits
        const tiers = [
            // Highest: around-neck/face
            ['necklace','pendant','scarf','turtleneck','headband','hairpin','earrings','glasses','sunglasses','hat','cap','brooch'],
            // Medium: hand/wrist small items
            ['bracelet','watch','ring','belt','pin','wallet','cardholder'],
            // Lower: larger apparel that restages look
            ['overcoat','coat','jacket','blazer','cardigan','sweater','pullover','hoodie','sweatshirt','vest','parka','raincoat','windbreaker','poncho','trench coat',
             'shirt','blouse','polo','t-shirt','tee','tank top','tank','camisole','tube top','crop top','bodysuit',
             'dress','gown','maxi dress','mini dress','midi dress','jumpsuit','romper','playsuit','shawl','costume','hood',
             'handbag','bag','purse','backpack','pants','trousers','jeans','skirt','shorts','leggings','joggers','sweatpants','culottes','capris','overall','dungarees',
             'boots','boot','ankle boots','knee-high boots','thigh-high boots','shoe','loafers','heels','pumps','stilettos','platforms','wedges','oxfords','moccasins','sandal','flip-flop','sneaker','trainer','slippers','clogs','espadrilles','gloves']
        ];

        // If not a clear human subject, fallback to general extractor
        const isHuman = subject && ['woman','man','girl','boy','subject','person'].includes(subject);
        if (!isHuman) return extractKeyObjectFromB(analysisB, combined);

        for (const tier of tiers) {
            const found = tier.find(o => hasWord(o));
            if (found) {
                // Reuse material/color extraction from base
                const base = extractKeyObjectFromB(analysisB, combined) || { materials: [], colors: [] };
                return { label: found, materials: base.materials || [], colors: base.colors || [] };
            }
        }
        return extractKeyObjectFromB(analysisB, combined);
    }

    function detectSubjectFromA(analysisA, combined) {
        const t = (analysisA || combined || '').toLowerCase();
        if (!t) return null;
        if (t.includes('woman') || t.includes('female')) return 'woman';
        if (t.includes('man') || t.includes('male')) return 'man';
        if (t.includes('boy')) return 'boy';
        if (t.includes('girl')) return 'girl';
        if (t.includes('person') || t.includes('portrait') || t.includes('people')) return 'subject';
        return null;
    }

    function buildWrappingPrompt() {
        const wrapType = els.wrapType ? els.wrapType.value : 'full';
        const finish = els.wrapFinish ? els.wrapFinish.value : 'glossy';
        const scopePreset = els.wrapScopePreset ? els.wrapScopePreset.value : '';
        const customScope = (els.wrapScope && els.wrapScope.value.trim()) || '';
        const enforce = els.enforcePalette ? els.enforcePalette.checked : true;
        const palette = (els.paletteList && els.paletteList.value.trim()) || '';
        const neutralize = els.neutralizeB ? els.neutralizeB.checked : true;
        const multiStage = els.multiStage ? els.multiStage.checked : false;

        // Resolve scope from preset or custom
        const scopeMap = {
            'all': 'all visible surfaces',
            '360': 'full 360° circumference',
            'full-can': 'full can from lid/top surface down through shoulder, entire body, and bottom rim - complete coverage with no bare metal visible',
            'body-only': 'cylindrical body only, excluding lid/top and bottom rim',
            'glass-bowl': 'transparent glass bowl/cup area with design printed on surface - keep stem and base transparent/clear',
            'glass-full': 'transparent glass full coverage including bowl, stem, and base with printed design',
            'glass-opaque': 'opaque/colored/frosted glass with complete color replacement on bowl area - replace original glass color entirely',
            'front': 'front label/panel only, avoiding back seam',
            'neck': 'neck band region',
            'cap': 'cap or lid surface only'
        };
        const scope = scopePreset ? (scopeMap[scopePreset] || customScope || 'all visible surfaces') : (customScope || 'all visible surfaces');

        const analysisText = els.imageDescription ? els.imageDescription.innerText.trim() : '';
        
        // Use multi-stage prompting if enabled
        if (multiStage) {
            return buildMultiStagePrompt(wrapType, finish, scope, palette, analysisText, scopePreset);
        }
        let paletteLine = '';
        if (enforce) {
            const extracted = palette || extractPaletteFromText(analysisText);
            paletteLine = extracted
                ? `Palette (hard): ${extracted}. No other hues.`
                : `Palette (hard): Use only colors from Reference A. No other hues.`;
        }

        const priority = enforce 
            ? 'PRIORITY: Use ONLY the Reference A palette on wrapped areas. OVERRIDE any original Reference B colors.' 
            : '';
        const neutralizeLine = neutralize 
            ? 'Replace Reference B surface colors/patterns with Reference A artwork while maintaining Reference B exact shape, form, and structure.' 
            : '';

        let typeSentence = '';
        let mapping = '';
        let rendering = '';

        if (wrapType === 'sticker-pack') {
            // Sticker pack/sheet mode
            const count = els.stickerCount ? els.stickerCount.value : '9';
            const style = els.stickerStyle ? els.stickerStyle.value : 'die-cut';
            const characterStyle = els.stickerCharacterStyle ? els.stickerCharacterStyle.value : 'match-reference';
            const sheetOutput = els.stickerSheetOutput ? els.stickerSheetOutput.value : 'realistic';
            const cutType = els.stickerCutType ? els.stickerCutType.value : 'kiss-cut';
            const borderThickness = els.stickerBorderThickness ? els.stickerBorderThickness.value : 'medium';
            const camera = els.stickerCamera ? els.stickerCamera.value : 'top-down';
            const innerKeyline = !!(els.stickerInnerKeyline && els.stickerInnerKeyline.checked);
            const variation = els.stickerVariation ? els.stickerVariation.value : 'poses';
            const planSource = els.stickerPlanSource ? els.stickerPlanSource.value : 'built-in';

            const styleMap = {
                'die-cut': 'die-cut stickers with white border outline around each character/element',
                'no-border': 'die-cut stickers with no border, following exact character/element outline',
                'rounded': 'rounded square stickers with centered artwork',
                'circle': 'circular stickers with centered artwork'
            };
            const styleText = styleMap[style] || styleMap['die-cut'];

            const variationMap = {
                'poses': 'different poses, expressions, and gestures',
                'elements': 'different props, accessories, and elements',
                'colors': 'different color variants and palettes',
                'scenes': 'different activities, scenes, and situations',
                'emotions': 'different emotions, moods, and feelings',
                'outfits': 'different outfits, costumes, and clothing styles',
                'seasons': 'seasonal variations and holiday themes',
                'interactions': 'interacting with different objects and items',
                'weather': 'different weather conditions and times of day',
                'styles': 'different art styles and rendering techniques',
                'actions': 'different actions and movements (jumping, running, dancing, flying, sitting)',
                'hobbies': 'different hobbies, sports, and recreational activities',
                'gestures': 'different hand gestures, signs, and hand poses',
                'occupations': 'different professions, jobs, and work uniforms',
                'text-bubbles': 'with different speech bubbles, thought bubbles, and text messages'
            };
            
            let variationText = variationMap[variation] || variationMap['poses'];
            
            // If text-bubbles variation is selected and custom text is provided, use it
            if (variation === 'text-bubbles' && els.bubbleText && els.bubbleText.value.trim()) {
                const customTexts = els.bubbleText.value.trim();
                variationText = `with speech bubbles or thought bubbles containing these messages: ${customTexts}`;
            }

            typeSentence = `Create a sticker pack/sheet with ${count} individual ${styleText} featuring the character/subject from Reference A in ${variationText}.`;

            const mappingBackgroundText = sheetOutput === 'transparent'
                ? 'transparent background behind all stickers'
                : 'white matte backing paper behind all stickers (release liner)';

            mapping = `Each sticker is a separate die-cut design; ${mappingBackgroundText}; stickers arranged in an organized grid layout; each sticker shows a complete, self-contained composition; consistent character style and art quality across all ${count} stickers; varied ${variationText} to create a cohesive collection.`;

            const characterStyleMap = {
                'match-reference': 'Match the character/subject style from Reference A (do not force chibi).',
                'cartoon': 'Cartoon style character rendering (not chibi unless explicitly requested).',
                'chibi': 'Chibi style (super deformed): oversized head, large expressive eyes, tiny body, cute proportions.',
                'semi-realistic': 'Semi-realistic character rendering with clean stylized features (not chibi).'
            };
            const characterStyleText = characterStyleMap[characterStyle] || characterStyleMap['match-reference'];

            const borderThicknessMap = {
                thin: 'thin uniform white border (approx 2–3mm)',
                medium: 'medium uniform white border (approx 3–5mm)',
                thick: 'thick uniform white border (approx 6–10mm)'
            };
            const borderThicknessText = borderThicknessMap[borderThickness] || borderThicknessMap.medium;
            const keylineText = innerKeyline ? 'Add a thin dark inner keyline just inside the white border for separation.' : '';

            const cameraText = camera === 'slight-angle'
                ? 'Slight 10–15° angle product photo of the sticker sheet.'
                : 'Top-down flat lay product photo of the sticker sheet.';

            const cutTypeText = cutType === 'die-cut'
                ? 'Die-cut (cut-through) stickers with clean cut edges.'
                : 'Kiss-cut outlines visible around each sticker; stickers remain on backing paper.';

            const outputText = sheetOutput === 'transparent'
                ? 'Transparent background with clean alpha (no checkerboard). Export-ready PNG sticker assets arranged in a neat grid.'
                : 'Printed sticker sheet on white matte backing paper (release liner), with subtle paper texture and visible sheet edges.';

            rendering = `Photorealistic sticker sheet mockup; ${outputText}; ${cameraText} Soft studio lighting with gentle realistic shadows; glossy vinyl sticker surface with subtle specular highlights; ${cutTypeText} ${borderThicknessText}; crisp cut lines; no jagged edges; vibrant colors; sharp details; ${characterStyleText} ${keylineText}`.trim();

            const allowWardrobeChange = (variation === 'outfits' || variation === 'occupations' || variation === 'seasons');

            const stickerNegatives = sheetOutput === 'transparent'
                ? `Negatives: no overlapping stickers, no merged designs, no inconsistent art styles, no blurry details, no incomplete stickers, no cut-off elements, no watermark, no UI, no background scene, no checkerboard pattern; no different character/person; ${allowWardrobeChange ? 'no identity change; keep the same person.' : 'no outfit changes;'} no hat removal; no hairstyle changes.`
                : `Negatives: no overlapping stickers, no merged designs, no inconsistent art styles, no blurry details, no incomplete stickers, no cut-off elements, no watermark, no UI, no busy background scene; do not remove backing paper; do not make stickers float; no extreme perspective distortion; no different character/person; ${allowWardrobeChange ? 'no identity change; keep the same person.' : 'no outfit changes;'} no hat removal; no hairstyle changes.`;

            const identityLock = (characterStyle === 'match-reference')
                ? (allowWardrobeChange
                    ? 'Identity lock: Use the exact same character from Reference A in every sticker. Keep the same person identity, face/head shape, body type, skin tone, and signature accessories (including hat and bag/strap). Wardrobe may change ONLY according to the selected variation plan (outfits/seasonal styling/occupation uniform).'
                    : 'Identity lock: Use the exact same character from Reference A in every sticker. Keep the same person identity, face/head shape, hat, clothing layers, accessories (including bag/strap), and shoes. Do not change age, body type, skin tone, or wardrobe. Only change pose/expression/scene as instructed.')
                : '';

            const fullRedrawInstruction = 'Edit instruction: Replace the entire image with the described sticker sheet mockup composition. Do not preserve the original single-photo framing/background; perform a full scene/layout transformation into a sticker sheet.';

            const styleConsistencyLine = (variation === 'styles')
                ? (sheetOutput === 'realistic'
                    ? `Style variation: All ${count} stickers must remain photorealistic, but each sticker should have a distinct photographic look (e.g., lens choice, depth of field, color grading) while keeping the same character identity and wardrobe.`
                    : `Style variation: Each sticker should use a distinct art style/rendering technique while keeping the same character identity and wardrobe.`)
                : ((sheetOutput === 'realistic' && (characterStyle === 'match-reference' || characterStyle === 'semi-realistic'))
                    ? `Realism consistency: All ${count} stickers must match the same realistic photographic rendering (no illustration/line-art). Consistent fabric textures, skin detail, lighting direction, and lens look across the whole sheet.`
                    : `Style consistency: All ${count} stickers use the same art style, line weight, shading technique, and color saturation.`);

            const n = Math.max(1, parseInt(count, 10) || 9);
            const scenePool = [
                'walking down a city street',
                'sitting on a bench',
                'standing by a window and looking out',
                'holding a coffee cup',
                'standing in an empty room',
                'reading a book',
                'checking a phone',
                'waiting at a crosswalk',
                'carrying a tote bag',
                'opening a door',
                'leaning against a wall',
                'riding an escalator',
                'looking at a storefront',
                'tying shoelaces',
                'holding an umbrella',
                'watering a small plant'
            ];

            const hobbyPool = [
                'jogging (sportswear)',
                'cycling with a bicycle',
                'playing basketball with a basketball',
                'playing soccer with a soccer ball',
                'playing tennis with a tennis racket',
                'skateboarding with a skateboard',
                'playing guitar with a guitar',
                'taking photos with a camera',
                'painting with a small brush/palette',
                'reading a book',
                'working on a laptop',
                'cooking while holding a spatula',
                'doing yoga/stretching on a mat',
                'hiking with a small backpack',
                'gardening with a watering can',
                'playing chess at a table'
            ];

            const actionPool = [
                'jumping',
                'running',
                'walking briskly',
                'sitting casually',
                'standing with hands in pockets',
                'waving',
                'pointing',
                'giving a thumbs-up',
                'looking over shoulder',
                'leaning against a wall',
                'crouching',
                'stretching arms',
                'checking a phone',
                'holding a coffee cup',
                'reading a book',
                'tying shoelaces'
            ];

            const posePool = [
                'standing, hands in pockets, slight head tilt',
                'walking forward, mid-step',
                'sitting casually, relaxed shoulders',
                'leaning against a wall, one foot up',
                'crouching down, looking at the ground',
                'turning back over the shoulder',
                'arms crossed, neutral expression',
                'hands on hips, confident stance',
                'one hand adjusting the cap brim',
                'kneeling on one knee',
                'looking at a watch (no phone)',
                'holding jacket collar slightly',
                'stretching one arm overhead',
                'sitting on steps, elbows on knees',
                'standing with weight on one leg',
                'light jog pose (no props)'
            ];

            const elementPool = [
                'holding a coffee cup',
                'holding a book',
                'holding a phone',
                'carrying a tote bag',
                'holding an umbrella',
                'wearing sunglasses',
                'wearing headphones',
                'holding a camera',
                'holding a skateboard',
                'holding a soccer ball',
                'holding a basketball',
                'holding a tennis racket',
                'holding a small potted plant',
                'holding a takeaway bag',
                'holding keys in hand',
                'holding a water bottle'
            ];

            const emotionPool = [
                'calm and neutral',
                'slight smile',
                'laughing softly',
                'serious and focused',
                'thoughtful/pensive',
                'surprised',
                'confident',
                'curious',
                'tired',
                'excited',
                'bored',
                'determined',
                'relaxed',
                'shy',
                'proud',
                'friendly'
            ];

            const outfitPool = [
                'casual streetwear (same hat, different jacket)',
                'smart casual (same hat, blazer over shirt)',
                'winter outfit (coat + scarf)',
                'rain outfit (rain jacket)',
                'summer outfit (light jacket or no jacket)',
                'sporty outfit (track jacket)',
                'workwear outfit (utility jacket)',
                'night-out outfit (darker refined look)',
                'minimal monochrome outfit',
                'denim-on-denim outfit',
                'hoodie layered under jacket',
                'sweater outfit (knit sweater)',
                'overshirt outfit (flannel/overshirt)',
                'puffer jacket outfit',
                'short-sleeve over long-sleeve layered',
                'light trench coat outfit'
            ];

            const seasonPool = [
                'spring: light jacket, fresh daylight',
                'summer: warmer light, lighter layers',
                'autumn: layered jacket, warm tones',
                'winter: coat + scarf, cold breath visible',
                'rainy day: wet ground reflections, rain jacket',
                'snowy day: soft snow ambience, winter coat',
                'windy day: jacket slightly billowing',
                'golden hour: warm sunset light',
                'overcast day: soft diffused light',
                'night: streetlights and subtle bokeh',
                'foggy morning: light haze',
                'bright midday: crisp shadows',
                'early morning: cool tones',
                'late afternoon: warm highlights',
                'holiday season: subtle festive vibe (no text)',
                'heatwave: sweat sheen, light layers'
            ];

            const interactionPool = [
                'opening a door',
                'pressing an elevator button',
                'reading a sign',
                'looking at a map',
                'picking up a small package',
                'tying shoelaces',
                'sitting and writing in a notebook',
                'holding a coffee cup',
                'checking a phone',
                'paying at a counter (no visible branding)',
                'waiting for a bus',
                'carrying groceries in a bag',
                'opening an umbrella',
                'taking a photo with a camera',
                'waving to someone off-camera',
                'petting a small dog (if possible)'
            ];

            const weatherPool = [
                'sunny day, crisp shadows',
                'overcast day, diffused lighting',
                'rainy day, wet ground reflections',
                'light drizzle with umbrella',
                'foggy atmosphere',
                'snowy day ambience',
                'windy day (jacket moving)',
                'golden hour sunlight',
                'blue hour twilight',
                'night streetlights and bokeh',
                'hot day heat haze (subtle)',
                'cold day with faint breath',
                'cloudy with soft shadows',
                'sunset backlight rim light',
                'indoor warm lighting',
                'indoor cool daylight'
            ];

            const gesturePool = [
                'thumbs up',
                'peace sign',
                'waving',
                'pointing',
                'hands in pockets',
                'arms crossed',
                'hand on chin (thinking)',
                'adjusting cap brim',
                'holding jacket collar',
                'hand raised as if greeting',
                'shrug gesture',
                'clapping once (mid-motion)',
                'fist bump gesture',
                'open palm “stop” gesture',
                'finger-gun gesture (subtle)',
                'hands behind back'
            ];

            const occupationPool = [
                'barista (apron)',
                'photographer (camera)',
                'delivery courier (package)',
                'office worker (laptop/badge)',
                'construction worker (hi-vis vest)',
                'chef (chef coat)',
                'nurse/medical worker (scrubs)',
                'teacher (book/notebook)',
                'mechanic (work coveralls)',
                'artist (paint supplies)',
                'security guard (uniform)',
                'police-like uniform (generic, no logos)',
                'firefighter-like gear (generic)',
                'retail worker (name tag, no branding)',
                'musician (guitar)',
                'gardener (watering can)'
            ];

            const colorPool = [
                'warm palette (subtle warm tones)',
                'cool palette (subtle cool tones)',
                'muted earthy palette',
                'high-contrast black and white',
                'soft pastel accents',
                'vibrant saturated colors',
                'cinematic teal/orange grading',
                'desaturated moody look',
                'bright airy look',
                'night neon accents (subtle)',
                'vintage film fade',
                'clean modern neutral palette',
                'warm sunset grading',
                'cool overcast grading',
                'monochrome with one accent color',
                'duotone-like color grading'
            ];

            const bubblePoolDefault = [
                'Hello!',
                'Thanks!',
                'Yay!',
                'Wow!',
                'LOL',
                'OK!',
                'Bye!',
                'No way!',
                'Nice!',
                'Let\'s go!',
                'BRB',
                'Cool!',
                'Sure!',
                'Oops!',
                'Great!',
                'All good!'
            ];

            const bubblePool = (variation === 'text-bubbles')
                ? ((els.bubbleText && els.bubbleText.value.trim())
                    ? els.bubbleText.value.split(',').map(s => s.trim()).filter(Boolean)
                    : bubblePoolDefault)
                : null;

            const variationPoolMap = {
                scenes: scenePool,
                hobbies: hobbyPool,
                actions: actionPool,
                poses: posePool,
                elements: elementPool,
                emotions: emotionPool,
                outfits: outfitPool,
                seasons: seasonPool,
                interactions: interactionPool,
                weather: weatherPool,
                gestures: gesturePool,
                occupations: occupationPool,
                colors: colorPool,
                'text-bubbles': bubblePool
            };

            const stickerMustIncludeCharacter = (variation !== 'styles')
                ? 'Sticker constraint: Every sticker MUST include the character (not standalone environments/objects). Background elements should be minimal context behind the character, not separate photos. No rectangular photo frames/tiles/polaroids; do not depict any sticker as a framed photograph.'
                : '';

            const propUniqueness = (variation !== 'colors' && variation !== 'styles')
                ? 'Uniqueness constraint: Do NOT repeat primary props across stickers. If a plan line specifies a prop (coffee cup, phone, book, tote bag, umbrella, ball, racket, guitar, camera, etc.), that prop must appear ONLY in that sticker and must NOT appear in any other sticker.'
                : '';

            const pool = variationPoolMap[variation] || null;

            const variationPlanBuiltIn = pool
                ? `${variation === 'scenes' ? 'Scene' : 'Variation'} plan (exactly ${n} unique items; one per sticker; do not reuse):\n${pool.slice(0, Math.min(n, pool.length)).map((s, i) => {
                    let onlyPropText = '';
                    const lower = String(s).toLowerCase();
                    if (lower.includes('coffee cup') || lower.includes('beverage') || lower.includes('cup')) onlyPropText = ' (ONLY sticker with any coffee cup/beverage/cup)';
                    else if (lower.includes('phone')) onlyPropText = ' (ONLY sticker with a phone visible)';
                    else if (lower.includes('book')) onlyPropText = ' (ONLY sticker with a book visible)';
                    else if (lower.includes('tote bag')) onlyPropText = ' (ONLY sticker with a tote bag)';
                    else if (lower.includes('umbrella')) onlyPropText = ' (ONLY sticker with an umbrella)';
                    else if (lower.includes('basketball')) onlyPropText = ' (ONLY sticker with a basketball)';
                    else if (lower.includes('soccer')) onlyPropText = ' (ONLY sticker with a soccer ball)';
                    else if (lower.includes('tennis')) onlyPropText = ' (ONLY sticker with a tennis racket)';
                    else if (lower.includes('skateboard')) onlyPropText = ' (ONLY sticker with a skateboard)';
                    else if (lower.includes('guitar')) onlyPropText = ' (ONLY sticker with a guitar)';
                    else if (lower.includes('camera')) onlyPropText = ' (ONLY sticker with a camera)';
                    else if (lower.includes('laptop')) onlyPropText = ' (ONLY sticker with a laptop)';
                    else if (lower.includes('spatula')) onlyPropText = ' (ONLY sticker with a spatula)';
                    return `${i + 1}. Character: ${s}${onlyPropText}.`;
                }).join('\n')}`
                : '';

            const variationPlanAI = `Variation plan: Generate exactly ${n} unique items (one per sticker) for "${variation}". Output a numbered list 1..${n}. No duplicates. Each item must explicitly describe the character in a distinct ${variationText}.`;

            const variationPlan = planSource === 'ai-enhance'
                ? variationPlanAI
                : variationPlanBuiltIn;

            // Override palette/neutralize lines for sticker mode
            return [
                typeSentence,
                mapping,
                rendering,
                fullRedrawInstruction,
                identityLock,
                `Layout: Arrange ${count} stickers in a grid with even spacing; each sticker maintains consistent scale relative to others.`,
                styleConsistencyLine,
                `Variation: Each sticker shows a unique ${
                    variation === 'poses' ? 'pose/expression' :
                    variation === 'elements' ? 'prop/accessory' :
                    variation === 'colors' ? 'color variant' :
                    variation === 'scenes' ? 'activity/scene' :
                    variation === 'emotions' ? 'emotion/mood' :
                    variation === 'outfits' ? 'outfit/costume' :
                    variation === 'seasons' ? 'seasonal theme' :
                    variation === 'interactions' ? 'object interaction' :
                    variation === 'weather' ? 'weather/time condition' :
                    variation === 'styles' ? 'art style' :
                    variation === 'actions' ? 'action/movement' :
                    variation === 'hobbies' ? 'hobby/sport' :
                    variation === 'gestures' ? 'hand gesture' :
                    variation === 'occupations' ? 'occupation/profession' :
                    variation === 'text-bubbles' ? 'text bubble/message' :
                    'variation'
                } - no duplicates.`,
                stickerMustIncludeCharacter,
                propUniqueness,
                variationPlan,
                stickerNegatives
            ].filter(Boolean).join('\n\n');

        } else if (wrapType === 'logo') {
            // Logo/branding mode
            const placement = els.logoPlacement ? els.logoPlacement.value : 'center';
            const size = els.logoSize ? els.logoSize.value : 'medium';
            const isPattern = els.logoPattern ? els.logoPattern.checked : false;

            const placementMap = {
                'center': 'centered on the main surface',
                'top-center': 'top-center position',
                'bottom-center': 'bottom-center position',
                'left': 'left side',
                'right': 'right side',
                'custom': scope
            };
            const placementText = placementMap[placement] || 'centered on the main surface';

            const sizeMap = {
                'small': 'small and subtle',
                'medium': 'medium size, clearly visible',
                'large': 'large and prominent',
                'full': 'full coverage',
                'proportional': 'proportionally sized to fit the surface'
            };
            const sizeText = sizeMap[size] || 'medium size, clearly visible';

            if (isPattern) {
                typeSentence = `Apply the logo/branding from Reference A as a repeating pattern across Reference B, ${sizeText}.`;
                mapping = 'Pattern repeats uniformly with consistent spacing; maintain logo aspect ratio and orientation; no distortion, warping, or perspective skew on individual logos; pattern follows surface curvature naturally; crisp edges and sharp details on each instance.';
            } else {
                typeSentence = `Place the logo/branding from Reference A onto Reference B at ${placementText}, ${sizeText}.`;
                mapping = 'Maintain exact logo aspect ratio and proportions; no distortion, stretching, or warping; crisp edges and sharp details; correct perspective for surface angle; logo sits flat on surface or conforms to curvature naturally.';
            }

            const finishText = finish === 'metallic' ? 'metallic foil finish' 
                : finish === 'paper' ? 'printed paper finish'
                : finish === 'shrink' ? 'embedded in wrap'
                : finish === 'matte' ? 'matte vinyl finish'
                : 'glossy vinyl finish';
            rendering = `Photorealistic logo application with ${finishText}; maintain Reference B lighting, shadows, and reflections; logo colors are vibrant and accurate.`;
        } else if (wrapType === 'people-object') {
            // People/Object A→B transfer (e.g., subject from A wearing/holding object from B)
            const analysisText = els.imageDescription ? els.imageDescription.innerText.trim() : '';
            const extracted = extractPaletteFromText(analysisText);
            const paletteLine = extracted ? `Color harmony: match object/material colors to Reference A lighting; optionally reflect palette accents (${extracted}) if plausible.` : `Color harmony: match object/material colors to Reference A lighting; optionally reflect subtle accents from Reference A.`;

            // Try to auto-detect specifics from analysis
            const subject = detectSubjectFromA(latestAnalysis.a, latestAnalysis.combined) || 'subject';
            // Read user overrides (allow multiple)
            const selectedObjects = els.objectOverride ? Array.from(els.objectOverride.selectedOptions).map(o => o.value) : [];
            const accessoriesOnly = !!(els.accessoriesOnly && els.accessoriesOnly.checked);
            let objInfo = null;
            if (selectedObjects.length === 0) {
                objInfo = extractKeyObjectFromBWithPriority(subject, latestAnalysis.b, latestAnalysis.combined);
                // If accessories-only is enabled but detected object is a full outfit item, try to fallback to an accessory
                const outfitSet = new Set(['overcoat','suit','cape','gauntlets','coat','jacket','shirt','blouse','bodysuit','dress','pants','trousers','boots','shoe','sandal','flip-flop','boot','sneaker','trainer']);
                if (accessoriesOnly && objInfo && outfitSet.has(objInfo.label)) {
                    // Attempt to pick from top accessory tiers directly
                    const text = (latestAnalysis.b || latestAnalysis.combined || '').toLowerCase();
                    const tier1 = ['necklace','pendant','scarf','turtleneck','mask','headband','hairpin','earrings','glasses','sunglasses','hat','cap','brooch'];
                    const tier2 = ['bracelet','watch','ring','belt','pin'];
                    const hasWord = (w) => {
                        const escaped = w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                        const re = new RegExp(`\\b${escaped}\\b`, 'i');
                        return re.test(text);
                    };
                    const fallback = [...tier1, ...tier2].find(o => hasWord(o));
                    if (fallback) objInfo = { label: fallback, materials: objInfo.materials || [], colors: objInfo.colors || [] };
                }
            } else {
                objInfo = { label: selectedObjects[0], materials: [], colors: [] };
            }
            const objectLabel = selectedObjects.length > 1
                ? `these objects: ${selectedObjects.join(', ')}`
                : (objInfo ? objInfo.label : 'key object/accessory');
            // Build phrase that adds 'the' only for a single object
            const objectPhrase = selectedObjects.length > 1 ? objectLabel : `the ${objectLabel}`;
            // Optional: derive a concise descriptor for apparel from Reference B analysis
            let descriptor = '';
            try {
                const bTxt = (latestAnalysis && latestAnalysis.b ? latestAnalysis.b : '').toLowerCase();
                const addIf = (cond, str) => { if (cond && !descriptor.includes(str)) descriptor += (descriptor ? ', ' : '') + str; };
                if (bTxt) {
                    const isDress = /\bwrap\s+dress\b|\bdress\b/.test(bTxt) && /dress/.test(objectLabel);
                    const isTop = /\b(top|blouse|shirt|tank|tee|t-shirt)\b/.test(objectLabel);
                    const isPants = /\b(pants|trousers|jeans)\b/.test(objectLabel);
                    const isSkirt = /\bskirt\b/.test(objectLabel);

                    // Common garment attributes
                    addIf(/v[-\s]?neck\b/.test(bTxt), 'v-neck');
                    addIf(/crew\s*neck|round\s*neck|scoop\s*neck/.test(bTxt), 'round neckline');
                    addIf(/short[-\s]?sleeve/.test(bTxt), 'short-sleeved');
                    addIf(/long[-\s]?sleeve/.test(bTxt), 'long-sleeved');
                    addIf(/sleeveless/.test(bTxt), 'sleeveless');
                    addIf(/tie[-\s]?waist|belt(ed)?\s+waist/.test(bTxt), 'tie-waist');
                    addIf(/wrap\b/.test(bTxt) && (isDress || isTop || isSkirt), 'wrap style');
                    addIf(/abstract\s+print|pattern/.test(bTxt), 'abstract print');
                    addIf(/stripe(d)?/.test(bTxt), 'striped');
                    addIf(/plaid|check(ed)?/.test(bTxt), 'checked');
                    addIf(/polka\s*dot|dot(ted)?/.test(bTxt), 'polka-dot');
                    // Color cues
                    addIf(/orange/.test(bTxt), 'orange');
                    addIf(/white|off[-\s]?white|cream/.test(bTxt), 'white/cream');
                }
            } catch(_) { /* noop */ }
            const descriptorText = descriptor ? ` (${descriptor})` : '';
            const materialHint = objInfo && objInfo.materials.length ? ` Materials: ${objInfo.materials.join(', ')}.` : '';
            const colorHint = objInfo && objInfo.colors.length ? ` Object colors: ${[...new Set(objInfo.colors)].join(', ')}.` : '';

            typeSentence = `DELTA-ONLY EDIT: Strictly preserve the ${subject} from Reference A exactly (identity, face, expression, skin tone, hair, posture, clothing), the original background, framing/composition, lens characteristics, color grading, and lighting.`;
            mapping = `Add ONLY ${objectPhrase}${descriptorText} from Reference B onto/around the ${subject} from Reference A with precise placement, scale, and perspective. Maintain correct occlusion (object(s) may sit behind hair/clothing edges), natural contact with subtle deformation where physically plausible. Do NOT change any other elements from Reference A.` + (accessoriesOnly ? ' Limit to accessories and small wearables; no full outfit replacement.' : '');
            rendering = [
                `Anchor: Use Reference A as the base canvas. Do NOT recreate or restage Reference B. Do NOT use any background or composition from Reference B.`,
                `Framing: keep the original framing/composition and camera distance from Reference A; maintain the same field of view. Do NOT crop, zoom-in, punch-in, or reframe. No close-ups.`,
                `Lighting & mood: match Reference A's lighting/exposure/DOF exactly; do not restage or relight the scene. Keep the original background and context from Reference A unchanged.`,
                paletteLine,
                `Materials: preserve B's material properties (metal, beads, fabric, leather, etc.) with micro-highlights and reflections; add contact shadows; limit deformation to tiny amounts for realism.` + materialHint + colorHint,
                `Negatives: do NOT alter facial structure, pose, clothing (beyond contact overlap), hair style/length, background, perspective, camera position, or palette from Reference A. Do NOT import any scene elements from Reference B besides ${objectPhrase}. No flat-lay, no product-only composition, no vehicles, wraps, decals, restaging, stylization shifts, or color bleed. No crop, no zoom-in, no close-up, no reframe.` + (accessoriesOnly ? ' No suits, capes, bodysuits, coats, jackets, shirts, pants, shoes.' : '')
            ].join(' ');
        } else if (wrapType === 'product') {
            // Product wrap (cylindrical objects) - detect the actual object type
            const objectType = detectObjectType(analysisText, scope, scopePreset);
            const objectNames = {
                'glass-opaque': 'opaque/frosted glass',
                'glass': 'glass',
                'bottle': 'bottle',
                'can': 'can',
                'cup': 'cup/mug',
                'jar': 'jar',
                'tube': 'tube',
                'container': 'container'
            };
            const objectName = objectNames[objectType] || 'product';
            
            typeSentence = `Transfer the design from Reference A onto the surface of Reference B (${objectName}), covering: ${scope}.`;
            
            // Enhanced mapping for full-can coverage
            if (scopePreset === 'full-can') {
                mapping = 'FULL COVERAGE: Extend artwork from lid/top surface, down through shoulder curve, across entire cylindrical body (360°), to bottom rim. No bare metal, aluminum, or original surface visible anywhere - complete edge-to-edge wrap. Wrap seamlessly around circumference; correct radial perspective and distortion; align primary artwork to front-center; no visible seam on front view; smooth transition over shoulder curve; no stretching, warping, or gaps.';
            } else {
                mapping = 'Wrap seamlessly around the circumference; correct radial perspective and distortion for curved surface; align logo/artwork to front-center; respect top and bottom rims; no visible seam on front view; no stretching or warping at edges.';
            }
            
            const finishText = finish === 'metallic' ? 'metallic foil label' 
                : finish === 'paper' ? 'paper label with smooth adhesion'
                : finish === 'shrink' ? 'tight shrink-wrap film'
                : finish === 'matte' ? 'matte vinyl wrap'
                : 'glossy vinyl wrap';
            rendering = `Photorealistic product packaging with ${finishText}; maintain Reference B lighting, reflections, and camera angle; label conforms perfectly to object shape.`;
        } else if (wrapType === 'full') {
            const finishText = finish === 'metallic' ? 'metallic foil film with reflective properties' 
                : finish === 'matte' ? 'matte vinyl film with subtle texture' 
                : 'glossy vinyl film with deep, vibrant colors';
            
            typeSentence = `Transfer the complete wrap design from Reference A onto the target object in Reference B, maintaining the original shape and features of Reference B.`;
            
            mapping = `CRITICAL INSTRUCTIONS:
- Transfer the complete design from Reference A to Reference B
- Maintain the exact shape, proportions, and features of Reference B
- Preserve the original contours and surface details of Reference B
- Ensure all design elements follow the object's geometry
- No distortion or warping of the design

DESIGN TRANSFER:
- Map all design elements to follow the exact contours of Reference B
- Maintain proper scale and perspective for all elements
- Ensure crisp, high-resolution rendering of all design elements
- Preserve text legibility and logo integrity
- Match the original design's color scheme and style

RENDERING QUALITY:
- Photorealistic materials and lighting
- Accurate reflections that follow the object's curves
- Proper shadows and ambient occlusion
- No distortion or warping of design elements
- Seamless integration with the target object's form`;
            
            rendering = `Photorealistic rendering with a ${finishText} finish; maintain Reference B's camera angle, lighting, and environment; reflections should accurately represent the wrap's material properties.`;
        } else if (wrapType === 'partial') {
            const finishText = finish === 'metallic' ? 'metallic foil film with reflective properties' 
                : finish === 'matte' ? 'matte vinyl film with subtle texture' 
                : 'glossy vinyl film with deep, vibrant colors';
            
            typeSentence = `Apply a partial wrap from Reference A onto the target object in Reference B covering: ${scope}, while maintaining the original shape and features of Reference B.`;
            
            mapping = `CRITICAL INSTRUCTIONS:
- Apply the design from Reference A ONLY to the specified area: ${scope}
- Maintain the exact shape, proportions, and features of Reference B
- Preserve the original contours and surface details of Reference B
- Ensure all design elements follow the object's geometry
- No distortion or warping of the design

DESIGN TRANSFER:
- Map design elements to follow the exact contours of Reference B
- Maintain proper scale and perspective for all elements
- Ensure crisp, high-resolution rendering of all design elements
- Preserve text legibility and logo integrity
- Match the original design's color scheme and style
- Create clean, precise edges at the wrap boundaries

RENDERING QUALITY:
- Photorealistic materials and lighting
- Accurate reflections that follow the object's curves
- Proper shadows and ambient occlusion
- No distortion or warping of design elements
- Seamless integration with the target object's form`;
            
            rendering = `Photorealistic rendering with a ${finishText} finish; maintain Reference B's camera angle, lighting, and environment; reflections should accurately represent the wrap's material properties.`;
        } else if (wrapType === 'vehicle') {
            // Enhanced Vehicle wrap mode - optimized for design transfer while preserving target vehicle's color
            const finishText = finish === 'metallic' ? 'high-gloss metallic vinyl with reflective properties' 
                : finish === 'matte' ? 'premium matte vinyl with subtle texture' 
                : 'high-gloss vinyl with deep, vibrant colors';
            
            // Check if user wants to transfer the base color from Reference A
            const transferColor = els.transferVehicleColor && els.transferVehicleColor.checked;
            console.log('Transfer vehicle color checkbox:', transferColor); // Debug log
            
            if (transferColor) {
                // Transfer everything including base color
                typeSentence = `Transfer the complete vehicle design from Reference A (source vehicle) onto the target vehicle in Reference B, INCLUDING THE BASE COLOR, design elements, and branding.`;
                
                mapping = `CRITICAL INSTRUCTIONS:
- Transfer the COMPLETE design from Reference A, including:
  * Base vehicle color
  * Logos and branding
  * Graphic patterns and artwork
  * Decals and text
  * Design elements and illustrations
- Maintain Reference B's:
  * Body shape and contours
  * Window tint and glass
  * Wheels and tires
  * Lights and trim
  * Interior details
  * Physical proportions and perspective

DESIGN TRANSFER:
- Map all design elements and colors to follow Reference B's exact contours
- Maintain proper scale and perspective for all transferred elements
- Ensure crisp, high-resolution rendering of all design elements
- Preserve text legibility and logo integrity
- Match the original design's complete color scheme and style

RENDERING QUALITY:
- Photorealistic materials and lighting
- Accurate reflections that follow the vehicle's curves
- Proper shadows and ambient occlusion
- No distortion or warping of design elements
- Seamless integration with the target vehicle's form`;
            } else {
                // Transfer only design elements, preserve Reference B's base color
                typeSentence = `Transfer ONLY the design elements from Reference A (source vehicle) onto the target vehicle in Reference B, while MAINTAINING Reference B's original base color, body shape, and physical features.`;
                
                mapping = `CRITICAL INSTRUCTIONS:
- PRESERVE Reference B's ORIGINAL BASE COLOR - DO NOT change the vehicle's factory paint color
- Transfer ONLY the following from Reference A:
  * Logos and branding
  * Graphic patterns and artwork
  * Decals and text
  * Design elements and illustrations
- Maintain Reference B's:
  * Base vehicle color
  * Body shape and contours
  * Window tint and glass
  * Wheels and tires
  * Lights and trim
  * Interior details

DESIGN TRANSFER:
- Map design elements to follow Reference B's exact contours
- Maintain proper scale and perspective for all transferred elements
- Ensure crisp, high-resolution rendering of all design elements
- Preserve text legibility and logo integrity
- Match the original design's color scheme and style

RENDERING QUALITY:
- Photorealistic materials and lighting
- Accurate reflections that follow the vehicle's curves
- Proper shadows and ambient occlusion
- No distortion or warping of design elements
- Seamless integration with the target vehicle's form`;
            }
            
            rendering = `PHOTOREALISTIC RENDERING (STUDIO QUALITY):
- Professional automotive photography with studio lighting
- ${finishText} that accurately reflects the vehicle's environment
- Natural-looking reflections that follow the vehicle's curves
- Soft, diffused lighting that shows off the wrap's details
- Subtle ambient occlusion in panel gaps and recessed areas
- Perfect alignment of all graphics to the vehicle's body lines
- High dynamic range with deep shadows and bright highlights

TECHNICAL REQUIREMENTS:
- Minimum 8K resolution for full detail preservation
- 16-bit color depth for accurate color reproduction
- Photorealistic materials with proper IOR (Index of Refraction)
- Physically accurate lighting and reflections
- No compression artifacts or banding
- Professional commercial photography quality

POST-PROCESSING:
- Apply subtle sharpening to enhance detail visibility
- Ensure consistent white balance across the entire image
- Add minimal contrast for depth and dimension
- Apply slight chromatic aberration for realism
- Include subtle lens distortion if applicable`;
        } else {
            // decal
            typeSentence = `Apply a die-cut decal/sticker from Reference A onto Reference B on: ${scope}.`;
            mapping = 'Correct perspective/UV over surface; align to natural contours; no warping or stretching; crisp edges; follow surface curvature; no background elements.';
            const finishText = finish === 'metallic' ? 'metallic vinyl' : finish === 'matte' ? 'matte vinyl' : 'glossy vinyl';
            rendering = `Photorealistic decal with ${finishText} finish; maintain Reference B lighting and camera angle; cast appropriate shadows; no visible background from Reference A.`;
        }

        let negatives = 'Negatives: no color bleed from B, no banding, no artifacts, no partial recolor.';
        
        // Enhanced negatives for vehicle wraps
        if (wrapType === 'vehicle') {
            negatives = 'Negatives: DO NOT modify Reference B vehicle body shape, proportions, or structure; NO changing the vehicle model or type; NO altering wheel design, lights, or windows; NO removing or changing vehicle features; NO blurry text or logos; NO distorted graphics; NO warped patterns; NO incomplete wrap coverage; NO artifacts or banding; NO unrealistic reflections; NO incorrect perspective; NO color bleed from Reference B onto the wrap design; NO partial application of design elements.';
        }
        
        // Enhanced negatives for full-can coverage
        if (wrapType === 'product' && scopePreset === 'full-can') {
            negatives = 'Negatives: NO bare metal, aluminum, or original surface showing on lid/top, shoulder, body, or rims; no gaps in coverage; no unwrapped areas; no color bleed from B; no banding; no artifacts; no partial recolor; no label stops short of lid/top.';
        }

        return [
            priority,
            paletteLine,
            neutralizeLine,
            typeSentence,
            mapping,
            rendering,
            negatives
        ].filter(Boolean).join('\n\n');
    }

    // Show/hide logo, sticker, and vehicle controls based on wrap type
    if (els.wrapType) {
        els.wrapType.addEventListener('change', () => {
            const wrapType = els.wrapType.value;
            if (els.logoControls) {
                els.logoControls.style.display = wrapType === 'logo' ? 'grid' : 'none';
            }
            if (els.stickerControls) {
                els.stickerControls.style.display = wrapType === 'sticker-pack' ? 'grid' : 'none';
            }
            // Show/hide bubble text input based on variation type
            if (els.bubbleTextContainer && els.stickerVariation) {
                const showBubbleText = wrapType === 'sticker-pack' && els.stickerVariation.value === 'text-bubbles';
                els.bubbleTextContainer.style.display = showBubbleText ? 'block' : 'none';
            }
            if (els.vehicleControls) {
                els.vehicleControls.style.display = wrapType === 'vehicle' ? 'block' : 'none';
            }
        });
    }

    // Show/hide bubble text input when sticker variation changes
    if (els.stickerVariation && els.bubbleTextContainer) {
        els.stickerVariation.addEventListener('change', () => {
            const showBubbleText = els.wrapType && els.wrapType.value === 'sticker-pack' && els.stickerVariation.value === 'text-bubbles';
            els.bubbleTextContainer.style.display = showBubbleText ? 'block' : 'none';
        });
    }

    if (els.insertWrapPrompt) {
        els.insertWrapPrompt.addEventListener('click', () => {
            const wrapText = buildWrappingPrompt();
            if (!els.prompt) return;
            const existing = els.prompt.value.trim();
            els.prompt.value = existing ? `${existing}\n\n${wrapText}` : wrapText;
            if (typeof updateCharCount === 'function') updateCharCount();
            // Move caret to end for convenience
            els.prompt.focus();
            els.prompt.selectionStart = els.prompt.selectionEnd = els.prompt.value.length;
            showToast('Wrapping prompt inserted.', 'success');
        });
    }

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
        const toggleEl = document.getElementById('theme-toggle');
        if (toggleEl) toggleEl.checked = isDarkMode;
    }

    // Theme toggle listener is set after 'els' is created below

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
            const files = selectedFiles.length ? selectedFiles : els.imageUpload.files;
            if (!files || files.length === 0) {
                console.error('No file selected.');
                showToast('Please select 1 or 2 image files first.', 'error');
                return;
            }
            if (files.length > 2) {
                showToast('Only up to 2 images are supported. Using the first two.', 'warning');
            }
            const selected = [ ...files ].slice(0, 2);
            console.log('Files selected:', selected.map(f => `${f.name} (${f.type}, ${(f.size/1024/1024).toFixed(2)}MB)`).join('; '));
            
            // Check file size
            const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
            for (const f of selected) {
                if (f.size > MAX_FILE_SIZE) {
                    showToast(`Image file is too large (${(f.size / (1024 * 1024)).toFixed(2)}MB): ${f.name}. Maximum size is 10MB.`, 'error');
                    return;
                }
            }
            
            // Check file type
            const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
            for (const f of selected) {
                if (!validTypes.includes(f.type)) {
                    showToast(`Invalid file type: ${f.type} for ${f.name}. Please upload a JPEG, PNG, GIF, or WebP image.`, 'error');
                    return;
                }
            }

            // Show analyzing state
            showToast(`Analyzing ${selected.length} image${selected.length > 1 ? 's' : ''}...`, 'info');
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
            els.imageDescriptionWrapper.style.display = 'none';
            els.imageProgressContainer.appendChild(progressContainer);
            
            // Show image preview(s)
            els.imagePreview.src = URL.createObjectURL(selected[0]);
            els.imagePreview.style.display = 'block';
            if (selected[1] && els.imagePreviewB) {
                els.imagePreviewB.src = URL.createObjectURL(selected[1]);
                els.imagePreviewB.style.display = 'block';
            } else if (els.imagePreviewB) {
                els.imagePreviewB.style.display = 'none';
                els.imagePreviewB.removeAttribute('src');
            }
            els.clearImage.style.display = 'inline-block';

            const formData = new FormData();
            // Append files under 'images' field (backend expects list)
            selected.forEach(f => formData.append('images', f));
            
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
                        
                        // Handle new multi-image response
                        const combined = data.combined_description;
                        // Persist raw analysis strings for downstream helpers
                        latestAnalysis = {
                            combined: combined || '',
                            a: (data.image_a_description || ''),
                            b: (data.image_b_description || '')
                        };
                        const isError = typeof combined === 'string' && combined.startsWith('Error');
                        if (isError) {
                            els.imageDescription.innerHTML = `<div class="error-message">${combined}</div>`;
                            els.imageDescriptionWrapper.style.display = 'block';
                            showToast('Image analysis failed.', 'error');
                        } else {
                            // Success - show combined description (and keep per-image for future use)
                            let html = '';
                            if (combined) {
                                html += `<div class="analysis-combined">${combined}</div>`;
                            }
                            if (data.image_a_description || data.image_b_description) {
                                html += '<hr style="opacity:0.2; margin:8px 0;">';
                                if (data.image_a_description) {
                                    html += `<div class="analysis-per-image"><strong>Reference A</strong><br>${data.image_a_description}</div>`;
                                }
                                if (data.image_b_description) {
                                    html += `<div class="analysis-per-image" style="margin-top:6px;"><strong>Reference B</strong><br>${data.image_b_description}</div>`;
                                }
                            }
                            els.imageDescription.innerHTML = html || 'No analysis returned.';
                            els.imageDescriptionWrapper.style.display = 'block';
                            els.imagePreview.style.display = 'block';
                            if (els.imagePreviewB && els.imagePreviewB.src) {
                                els.imagePreviewB.style.display = 'block';
                            }
                            showToast('Image analyzed successfully!', 'success');
                            updateProgress(1, 'completed');
                            updateProgress(2, 'active');
                            // Refresh override candidates if relevant
                            toggleObjectOverrideVisibility();
                            
                            // Recalculate height for collapse/expand functionality
                            // Start collapsed but calculate full height for smooth expand
                            setTimeout(() => {
                                const analysisContent = document.getElementById('analysis-content');
                                const toggleBtn = document.getElementById('toggle-analysis');
                                const toggleTextEl = document.getElementById('toggle-text');
                                const toggleIconEl = document.getElementById('toggle-icon');
                                
                                if (analysisContent && toggleBtn) {
                                    // Store full height for smooth animation later
                                    analysisContent.dataset.fullHeight = analysisContent.scrollHeight;
                                    
                                    // Set to collapsed state initially
                                    analysisContent.style.maxHeight = '100px';
                                    analysisContent.classList.add('collapsed');
                                    isAnalysisExpanded = false;
                                    
                                    if (toggleTextEl) toggleTextEl.textContent = 'Show More';
                                    if (toggleIconEl) toggleIconEl.style.transform = 'rotate(180deg)';
                                    
                                    console.log('Image analysis collapse initialized - starts collapsed at 100px');
                                }
                            }, 100);
                        }
                    } catch (error) {
                        els.imageDescription.innerHTML = '<div class="error-message">Failed to parse server response</div>';
                        els.imageDescriptionWrapper.style.display = 'block';
                        showToast('Image analysis failed: Invalid server response', 'error');
                    }
                } else {
                    els.imageProgressContainer.innerHTML = ''; // Clear progress on error
                    els.imageDescription.innerHTML = '<div class="error-message">Server error: ' + xhr.status + '</div>';
                    els.imageDescriptionWrapper.style.display = 'block';
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
                els.imageDescriptionWrapper.style.display = 'block';
                showToast('Image analysis failed: Network error', 'error');
                els.analyze.disabled = false;
                els.analyze.innerHTML = 'Analyze Image';
            });
            
            xhr.addEventListener('abort', () => {
                els.imageDescription.innerHTML = '<div class="error-message">Upload aborted</div>';
                els.imageDescriptionWrapper.style.display = 'block';
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
            const hasImages = (selectedFiles && selectedFiles.length > 0)
                || (els.imageUpload && els.imageUpload.files && els.imageUpload.files.length > 0);
            if (hasImages) {
                try {
                    await ensureAnalysisForEnhance();
                } catch (e) {
                    console.warn('Auto-analysis before enhance failed:', e);
                }
            }
            const imageDescription = getImageContextForEnhance();
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
                // Determine wrap mode; Plain Enhance overrides to 'none'
                let wrapMode = 'none';
                if (!(els.plainEnhance && els.plainEnhance.checked)) {
                    wrapMode = (els.wrapType && els.wrapType.value === 'vehicle') ? 'vehicle'
                        : (els.wrapType && els.wrapType.value === 'people-object') ? 'people-object'
                        : 'none';
                }

                console.log('Request data:', {
                    prompt: formattedPrompt,
                    prompt_type: promptType,
                    style: style,
                    cinematography: cinematography,
                    lighting: lighting,
                    motion_effect: motionEffect,
                    image_description: imageDescription,
                    text_emphasis: textEmphasisDetails,
                    model: selectedModel,
                    wrap_mode: wrapMode
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
                        model: selectedModel,
                        wrap_mode: wrapMode
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
                } else if (model === 'z-image-turbo') {
                    infoContent = `<strong>Z-Image Turbo Model Selected</strong>
                    <p>Fast photorealistic image model using standard prompt formatting.</p>
                    <em>Best for: Quick photorealistic generations and iteration.</em>
                    <em>Tip: Keep prompts concise and explicit about composition and lighting.</em>`;
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
        },
        
        // Vehicle Wrap Template
        vehicleWrap: {
            promptType: 'Image',
            style: 'Commercial',
            cinematography: 'Medium Shot',
            lighting: 'Studio',
            basePrompt: `A highly detailed, photorealistic static image of a **TARGET_VEHICLE** in a professional setting. The car maintains its original aerodynamic shape with all distinctive features intact. Its **BASE_COLOR** body is extensively branded with **BRANDING_DETAILS** accurately mapped to its curved surfaces without distortion, similar to the reference design.

COMPOSITION:
- **Perspective**: Slightly low-angle perspective
- **Framing**: Dynamic composition with the car as the primary subject
- **Background**: Clean, professional environment that complements the vehicle

LIGHTING & RENDERING:
- **Lighting**: Bright, professional studio lighting
- **Reflections**: Distinct reflections of the environment and branding across the car's **FINISH** finish
- **Shadows**: Soft shadows that highlight the car's contours

TECHNICAL SPECS:
- Ultra-high resolution
- Photorealistic materials and textures
- Accurate color reproduction
- Crisp, detailed rendering of all design elements

STYLE: Commercial product photography with emphasis on the vehicle wrap design and branding.`
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
            els.imageDescriptionWrapper.style.display = 'none';
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
            case 'z-image-turbo':
                return prompt;
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
            'default': 4500,
            'flux': 1200,
            'z-image-turbo': 3000,
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

    // --- History Panel Logic ---
    if (els.historyPanel) {
        els.historyPanel.addEventListener('mouseenter', () => {
            els.historyPanel.classList.add('show');
        });
        
        els.historyPanel.addEventListener('mouseleave', () => {
            els.historyPanel.classList.remove('show');
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
    
    // --- Back to Top Button ---
    const backToTopBtn = document.getElementById('back-to-top');
    
    if (backToTopBtn) {
        // Show/hide button based on scroll position
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                if (window.pageYOffset > 150) {
                    backToTopBtn.classList.add('visible');
                } else {
                    backToTopBtn.classList.remove('visible');
                }
            }, 100);
        });
        
        // Scroll to top when clicked
        backToTopBtn.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
    
    // --- Image Analysis Collapse/Expand ---
    const toggleAnalysisBtn = document.getElementById('toggle-analysis');
    const analysisContent = document.getElementById('analysis-content');
    const toggleText = document.getElementById('toggle-text');
    const toggleIcon = document.getElementById('toggle-icon');
    
    if (toggleAnalysisBtn && analysisContent) {
        // Set initial collapsed state (starts collapsed to save space)
        analysisContent.style.maxHeight = '100px';
        analysisContent.classList.add('collapsed');
        toggleText.textContent = 'Show More';
        toggleIcon.style.transform = 'rotate(180deg)';
        
        toggleAnalysisBtn.addEventListener('click', () => {
            isAnalysisExpanded = !isAnalysisExpanded;
            
            if (isAnalysisExpanded) {
                // Expand
                analysisContent.style.maxHeight = analysisContent.scrollHeight + 'px';
                analysisContent.classList.remove('collapsed');
                toggleText.textContent = 'Show Less';
                toggleIcon.style.transform = 'rotate(0deg)';
                console.log('Analysis expanded to full height');
            } else {
                // Collapse to 100px preview
                analysisContent.style.maxHeight = '100px';
                analysisContent.classList.add('collapsed');
                toggleText.textContent = 'Show More';
                toggleIcon.style.transform = 'rotate(180deg)';
                console.log('Analysis collapsed to 100px preview');
            }
        });
        
        // Update max-height when window resizes
        window.addEventListener('resize', () => {
            if (isAnalysisExpanded) {
                analysisContent.style.maxHeight = analysisContent.scrollHeight + 'px';
            }
        });
    }
});
