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
    const styleSearchInput = document.getElementById('style-search-input');
    const styleSearchResults = document.querySelector('.style-search-results');
    const applyStyleBtn = document.getElementById('apply-style');
    const clearStyleBtn = document.getElementById('clear-style');
    
    let selectedStyles = [];
    const MAX_STYLES = 3;
    
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
                input.type = 'checkbox';
                input.id = optionId;
                input.name = 'style-option';
                input.value = style.name;
                
                input.addEventListener('change', () => {
                    if (input.checked) {
                        // Check if max styles reached
                        if (selectedStyles.length >= MAX_STYLES) {
                            input.checked = false;
                            showToast(`Maximum ${MAX_STYLES} styles allowed`, 'warning');
                            return;
                        }
                        selectedStyles.push(style);
                    } else {
                        // Remove from selected styles
                        const index = selectedStyles.findIndex(s => s.name === style.name);
                        if (index > -1) {
                            selectedStyles.splice(index, 1);
                        }
                    }
                    
                    // Update preview
                    if (selectedStyles.length > 0) {
                        const styleNames = selectedStyles.map(s => s.name).join(' + ');
                        stylePreview.textContent = `${selectedStyles.length} style${selectedStyles.length > 1 ? 's' : ''}: ${styleNames}`;
                    } else {
                        stylePreview.textContent = 'No style selected';
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
    
    // Apply the selected style(s) to the prompt
    function applyStyle() {
        if (selectedStyles.length === 0) {
            showToast('Please select at least one style', 'warning');
            return;
        }
        
        if (!promptInput.value.trim()) {
            showToast('Please enter a prompt first', 'warning');
            return;
        }
        
        // Tag-based approach: add style keywords to prompt
        const basePrompt = promptInput.value.trim();
        const styleKeywords = selectedStyles.map(s => {
            const styleName = s.name.toLowerCase();
            // Don't add "style" if it already ends with "style"
            return styleName.endsWith('style') ? styleName : styleName + ' style';
        }).join(', ');
        
        // Combine and clean up the prompt (remove duplicate commas and spaces)
        let enhancedPrompt = `${basePrompt}, ${styleKeywords}`;
        enhancedPrompt = enhancedPrompt.replace(/,\s*,+/g, ',').replace(/\s+/g, ' ').trim();
        
        // Combine negative prompts from all selected styles
        const allNegativePrompts = selectedStyles
            .map(s => styleManager.getNegativePrompt(s.name))
            .filter(p => p) // Remove empty prompts
            .join(', ');
        
        // Update the prompt and negative prompt
        promptInput.value = enhancedPrompt;
        
        if (allNegativePrompts && negativePromptInput) {
            negativePromptInput.value = negativePromptInput.value 
                ? `${negativePromptInput.value}, ${allNegativePrompts}`
                : allNegativePrompts;
        }
        
        // Trigger input events to update character counts
        promptInput.dispatchEvent(new Event('input', { bubbles: true }));
        if (negativePromptInput.dispatchEvent) {
            negativePromptInput.dispatchEvent(new Event('input', { bubbles: true }));
        }
        
        const styleCount = selectedStyles.length;
        const styleNames = selectedStyles.map(s => s.name).join(' + ');
        showToast(`Applied ${styleCount} style${styleCount > 1 ? 's' : ''}: ${styleNames}`, 'success');
    }
    
    // Clear the selected styles
    function clearStyle() {
        // Uncheck all checkboxes
        document.querySelectorAll('input[name="style-option"]').forEach(checkbox => {
            checkbox.checked = false;
        });
        
        selectedStyles = [];
        stylePreview.textContent = 'No style selected';
    }
    
    // Filter styles based on search query
    function filterStyles() {
        const query = styleSearchInput.value.toLowerCase().trim();
        
        if (!query) {
            // Show all categories and styles
            document.querySelectorAll('.style-category').forEach(category => {
                category.style.display = 'block';
            });
            document.querySelectorAll('.style-option').forEach(option => {
                option.style.display = 'block';
            });
            styleSearchResults.textContent = '';
            return;
        }
        
        let visibleCount = 0;
        
        // Filter through all categories
        document.querySelectorAll('.style-category').forEach(category => {
            const categoryTitle = category.querySelector('h4').textContent.toLowerCase();
            const options = category.querySelectorAll('.style-option');
            let categoryHasVisible = false;
            
            // Check each style in the category
            options.forEach(option => {
                const label = option.querySelector('label').textContent.toLowerCase();
                const matches = label.includes(query) || categoryTitle.includes(query);
                
                if (matches) {
                    option.style.display = 'block';
                    categoryHasVisible = true;
                    visibleCount++;
                } else {
                    option.style.display = 'none';
                }
            });
            
            // Show/hide category based on whether it has visible styles
            category.style.display = categoryHasVisible ? 'block' : 'none';
        });
        
        // Update search results message
        if (visibleCount === 0) {
            styleSearchResults.textContent = '❌ No styles found';
            styleSearchResults.style.color = '#e74c3c';
        } else {
            styleSearchResults.textContent = `✓ Found ${visibleCount} style${visibleCount !== 1 ? 's' : ''}`;
            styleSearchResults.style.color = '#27ae60';
        }
    }
    
    // Event listeners
    applyStyleBtn.addEventListener('click', applyStyle);
    clearStyleBtn.addEventListener('click', clearStyle);
    
    // Search functionality
    if (styleSearchInput) {
        styleSearchInput.addEventListener('input', filterStyles);
        
        // Clear search on Escape key
        styleSearchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                styleSearchInput.value = '';
                filterStyles();
                styleSearchInput.blur();
            }
        });
    }
    
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
