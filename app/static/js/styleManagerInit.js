import { styleManager } from './styleManager.js';
import { showToast } from './utils.js';

// Fetch and load styles from CSV file
async function loadStylesFromCSV() {
    try {
        const response = await fetch('/static/data/fooocus_styles.csv');
        if (!response.ok) {
            throw new Error(`Failed to load styles: ${response.status} ${response.statusText}`);
        }
        
        const csvText = await response.text();
        const success = await styleManager.loadStylesFromCSV(csvText);
        
        if (success) {
            console.log('Styles loaded successfully from CSV');
            return true;
        } else {
            console.warn('Failed to parse styles from CSV');
            return false;
        }
    } catch (error) {
        console.error('Error loading styles from CSV:', error);
        return false;
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    
    // Reference to the prompt input
    const promptInput = document.getElementById('prompt-input');
    const negativePromptInput = document.getElementById('negative-prompt') || { value: '' };
    const stylePreview = document.querySelector('.style-preview');
    const styleCategoriesContainer = document.querySelector('.style-categories');
    const applyStyleBtn = document.getElementById('apply-style');
    const clearStyleBtn = document.getElementById('clear-style');
    
    let selectedStyle = null;
    
    // Render style categories and options
    function renderStyleCategories() {
        const categories = styleManager.getCategories();
        
        categories.forEach(category => {
            const styles = styleManager.getStylesByCategory(category);
            
            const categoryElement = document.createElement('div');
            categoryElement.className = 'style-category';
            
            const categoryTitle = document.createElement('h4');
            categoryTitle.textContent = category;
            
            const optionsContainer = document.createElement('div');
            optionsContainer.className = 'style-options';
            
            styles.forEach(style => {
                const optionId = `style-${style.name.toLowerCase().replace(/\s+/g, '-')}`;
                
                const optionElement = document.createElement('div');
                optionElement.className = 'style-option';
                
                const input = document.createElement('input');
                input.type = 'radio';
                input.id = optionId;
                input.name = 'style-option';
                input.value = style.name;
                
                input.addEventListener('change', () => {
                    if (input.checked) {
                        selectedStyle = style;
                        stylePreview.textContent = style.name;
                        // Show a preview of the style
                        if (promptInput.value) {
                            const preview = styleManager.applyStyle(promptInput.value, style.name);
                            stylePreview.textContent = `${style.name}: ${preview.substring(0, 50)}...`;
                        }
                    }
                });
                
                const label = document.createElement('label');
                label.htmlFor = optionId;
                label.title = style.description || style.name;
                label.textContent = style.name;
                
                optionElement.appendChild(input);
                optionElement.appendChild(label);
                optionsContainer.appendChild(optionElement);
            });
            
            categoryElement.appendChild(categoryTitle);
            categoryElement.appendChild(optionsContainer);
            styleCategoriesContainer.appendChild(categoryElement);
        });
    }
    
    // Apply the selected style to the prompt
    function applyStyle() {
        if (!selectedStyle) {
            showToast('Please select a style first', 'warning');
            return;
        }
        
        if (!promptInput.value.trim()) {
            showToast('Please enter a prompt first', 'warning');
            return;
        }
        
        // Apply the style to the prompt
        const enhancedPrompt = styleManager.applyStyle(promptInput.value, selectedStyle.name);
        const negativePrompt = styleManager.getNegativePrompt(selectedStyle.name);
        
        // Update the prompt and negative prompt
        promptInput.value = enhancedPrompt;
        
        if (negativePrompt && negativePromptInput) {
            negativePromptInput.value = negativePromptInput.value 
                ? `${negativePromptInput.value}, ${negativePrompt}`
                : negativePrompt;
        }
        
        // Trigger input events to update character counts
        promptInput.dispatchEvent(new Event('input', { bubbles: true }));
        if (negativePromptInput.dispatchEvent) {
            negativePromptInput.dispatchEvent(new Event('input', { bubbles: true }));
        }
        
        showToast(`Applied ${selectedStyle.name} style`, 'success');
    }
    
    // Clear the selected style
    function clearStyle() {
        // Uncheck all radio buttons
        document.querySelectorAll('input[name="style-option"]').forEach(radio => {
            radio.checked = false;
        });
        
        selectedStyle = null;
        stylePreview.textContent = 'No style selected';
    }
    
    // Event listeners
    applyStyleBtn.addEventListener('click', applyStyle);
    clearStyleBtn.addEventListener('click', clearStyle);
    
    // Initialize the style manager UI
    async function initializeStyleManager() {
        try {
            // First try to load styles from CSV
            const csvLoaded = await loadStylesFromCSV();
            
            // If CSV loading failed, fall back to default styles
            if (!csvLoaded) {
                console.warn('Falling back to default styles');
                const success = await styleManager.loadStyles();
                
                if (!success) {
                    console.error('Failed to load default styles');
                    return;
                }
            }
            
            renderStyleCategories();
        } catch (error) {
            console.error('Error initializing style manager:', error);
        }
    }
    
    initializeStyleManager();
});
