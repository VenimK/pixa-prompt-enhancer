/**
 * Advanced Theme Editor - Comprehensive CSS theme customization
 * Allows users to edit themes with full CSS features and save custom themes
 */
class ThemeEditor {
    constructor(themeManager) {
        this.themeManager = themeManager;
        this.isOpen = false;
        this.currentTheme = null;
        this.baseTheme = null;
        this.customCSS = '';
        this.customProperties = new Map();
        this.componentStyles = new Map();
        this.history = [];
        this.historyIndex = -1;
        this.isDirty = false;
        
        this.init();
    }

    init() {
        this.createThemeEditor();
        this.bindEvents();
        this.setupCSSProperties();
    }

    createThemeEditor() {
        // Create theme editor modal
        const editor = document.createElement('div');
        editor.id = 'theme-editor';
        editor.className = 'theme-editor';
        editor.innerHTML = `
            <div class="theme-editor-header">
                <h3 class="theme-editor-title">Theme Editor</h3>
                <div class="theme-editor-actions">
                    <button class="btn btn-ghost" id="export-theme-btn" title="Export Theme">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7,10 12,15 17,10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                    </button>
                    <button class="btn btn-ghost" id="preview-btn" title="Toggle Preview">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                            <circle cx="12" cy="12" r="3"/>
                        </svg>
                    </button>
                    <button class="btn btn-ghost" id="reset-btn" title="Reset Changes">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                            <path d="M3 3v5h5"/>
                        </svg>
                    </button>
                    <button class="btn btn-primary" id="save-theme-btn" title="Save Theme">Save Theme</button>
                    <button class="theme-editor-close" aria-label="Close">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M18 6L6 18M6 6l12 12"/>
                        </svg>
                    </button>
                </div>
            </div>
            
            <div class="theme-editor-content">
                <div class="theme-editor-sidebar">
                    <div class="theme-selector-panel">
                        <h4>Base Theme</h4>
                        <select id="base-theme-select" class="form-control">
                            <option value="">Select a base theme...</option>
                        </select>
                    </div>
                    
                    <div class="property-categories">
                        <h4>CSS Properties</h4>
                        <div class="category-list">
                            <button class="category-btn active" data-category="colors">Colors</button>
                            <button class="category-btn" data-category="typography">Typography</button>
                            <button class="category-btn" data-category="layout">Layout</button>
                            <button class="category-btn" data-category="effects">Effects</button>
                            <button class="category-btn" data-category="animations">Animations</button>
                            <button class="category-btn" data-category="advanced">Advanced</button>
                        </div>
                    </div>
                    
                    <div class="component-selector">
                        <h4>Target Component</h4>
                        <select id="component-select" class="form-control">
                            <option value="global">Global Styles</option>
                            <option value="button">Buttons</option>
                            <option value="card">Cards</option>
                            <option value="form">Forms</option>
                            <option value="navbar">Navigation</option>
                            <option value="modal">Modals</option>
                        </select>
                    </div>
                </div>
                
                <div class="theme-editor-main">
                    <div class="property-editor-panel" id="property-editor">
                        <!-- Property editors will be inserted here -->
                    </div>
                    
                    <div class="code-editor-panel" id="code-editor" style="display: none;">
                        <div class="code-editor-header">
                            <h4>CSS Code Editor</h4>
                            <div class="code-editor-actions">
                                <button class="btn btn-sm btn-ghost" id="format-css-btn">Format</button>
                                <button class="btn btn-sm btn-ghost" id="validate-css-btn">Validate</button>
                            </div>
                        </div>
                        <textarea id="css-code" class="code-textarea" placeholder="Enter custom CSS..."></textarea>
                    </div>
                    
                    <div class="preview-panel" id="preview-panel">
                        <div class="preview-header">
                            <h4>Live Preview</h4>
                            <div class="preview-controls">
                                <button class="preview-size-btn active" data-size="desktop">Desktop</button>
                                <button class="preview-size-btn" data-size="tablet">Tablet</button>
                                <button class="preview-size-btn" data-size="mobile">Mobile</button>
                            </div>
                        </div>
                        <div class="preview-content" id="preview-content">
                            <div class="preview-sample">
                                <button class="btn btn-primary">Primary Button</button>
                                <button class="btn btn-secondary">Secondary Button</button>
                                <div class="card">
                                    <h5>Sample Card</h5>
                                    <p>This is a sample card to preview theme changes.</p>
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Sample Input</label>
                                    <input type="text" class="form-control" placeholder="Enter text...">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(editor);
    }

    bindEvents() {
        // Close button
        const closeBtn = document.querySelector('.theme-editor-close');
        closeBtn.addEventListener('click', () => this.close());

        // Base theme selector
        const baseThemeSelect = document.getElementById('base-theme-select');
        baseThemeSelect.addEventListener('change', (e) => this.setBaseTheme(e.target.value));

        // Category buttons
        const categoryBtns = document.querySelectorAll('.category-btn');
        categoryBtns.forEach(btn => {
            btn.addEventListener('click', (e) => this.setCategory(e.target.dataset.category));
        });

        // Component selector
        const componentSelect = document.getElementById('component-select');
        componentSelect.addEventListener('change', (e) => this.setComponent(e.target.value));

        // Export button
        const exportBtn = document.getElementById('export-theme-btn');
        exportBtn.addEventListener('click', () => this.showExportModal());

        // Preview toggle
        const previewBtn = document.getElementById('preview-btn');
        previewBtn.addEventListener('click', () => this.togglePreview());

        // Reset button
        const resetBtn = document.getElementById('reset-btn');
        resetBtn.addEventListener('click', () => this.resetChanges());

        // Save theme button
        const saveBtn = document.getElementById('save-theme-btn');
        saveBtn.addEventListener('click', () => this.saveTheme());

        // Preview size buttons
        const previewSizeBtns = document.querySelectorAll('.preview-size-btn');
        previewSizeBtns.forEach(btn => {
            btn.addEventListener('click', (e) => this.setPreviewSize(e.target.dataset.size));
        });

        // Code editor buttons
        const formatBtn = document.getElementById('format-css-btn');
        formatBtn.addEventListener('click', () => this.formatCSS());

        const validateBtn = document.getElementById('validate-css-btn');
        validateBtn.addEventListener('click', () => this.validateCSS());

        // CSS code textarea
        const cssTextarea = document.getElementById('css-code');
        cssTextarea.addEventListener('input', (e) => this.updateCustomCSS(e.target.value));

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (this.isOpen && !e.target.closest('#theme-editor') && !e.target.closest('#customize-theme-btn')) {
                if (this.isDirty) {
                    this.confirmClose();
                } else {
                    this.close();
                }
            }
        });

        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (this.isOpen && e.key === 'Escape') {
                if (this.isDirty) {
                    this.confirmClose();
                } else {
                    this.close();
                }
            }
        });
    }

    setupCSSProperties() {
        // Define CSS properties for editing
        this.cssProperties = {
            colors: [
                { name: '--primary-color', type: 'color', label: 'Primary Color', default: '#4f46e5' },
                { name: '--primary-hover', type: 'color', label: 'Primary Hover', default: '#4338ca' },
                { name: '--primary-light', type: 'color', label: 'Primary Light', default: '#818cf8' },
                { name: '--success-color', type: 'color', label: 'Success Color', default: '#10b981' },
                { name: '--warning-color', type: 'color', label: 'Warning Color', default: '#f59e0b' },
                { name: '--error-color', type: 'color', label: 'Error Color', default: '#ef4444' },
                { name: '--background-color', type: 'color', label: 'Background Color', default: '#f9fafb' },
                { name: '--surface-color', type: 'color', label: 'Surface Color', default: '#ffffff' },
                { name: '--text-color', type: 'color', label: 'Text Color', default: '#111827' },
                { name: '--text-secondary', type: 'color', label: 'Text Secondary', default: '#374151' },
                { name: '--border-color', type: 'color', label: 'Border Color', default: '#e5e7eb' }
            ],
            typography: [
                { name: '--font-primary', type: 'font', label: 'Primary Font', default: 'Inter, sans-serif' },
                { name: '--font-mono', type: 'font', label: 'Monospace Font', default: 'Fira Code, monospace' },
                { name: '--font-size-xs', type: 'size', label: 'Font Size XS', default: '0.75rem' },
                { name: '--font-size-sm', type: 'size', label: 'Font Size SM', default: '0.875rem' },
                { name: '--font-size-base', type: 'size', label: 'Font Size Base', default: '1rem' },
                { name: '--font-size-lg', type: 'size', label: 'Font Size LG', default: '1.125rem' },
                { name: '--font-size-xl', type: 'size', label: 'Font Size XL', default: '1.25rem' },
                { name: '--font-size-2xl', type: 'size', label: 'Font Size 2XL', default: '1.5rem' },
                { name: '--font-size-3xl', type: 'size', label: 'Font Size 3XL', default: '1.875rem' },
                { name: '--font-size-4xl', type: 'size', label: 'Font Size 4XL', default: '2.25rem' },
                { name: '--font-weight-normal', type: 'weight', label: 'Font Weight Normal', default: '400' },
                { name: '--font-weight-medium', type: 'weight', label: 'Font Weight Medium', default: '500' },
                { name: '--font-weight-semibold', type: 'weight', label: 'Font Weight Semibold', default: '600' },
                { name: '--font-weight-bold', type: 'weight', label: 'Font Weight Bold', default: '700' }
            ],
            layout: [
                { name: '--spacing-xs', type: 'size', label: 'Spacing XS', default: '0.25rem' },
                { name: '--spacing-sm', type: 'size', label: 'Spacing SM', default: '0.5rem' },
                { name: '--spacing-md', type: 'size', label: 'Spacing MD', default: '1rem' },
                { name: '--spacing-lg', type: 'size', label: 'Spacing LG', default: '1.5rem' },
                { name: '--spacing-xl', type: 'size', label: 'Spacing XL', default: '2rem' },
                { name: '--spacing-2xl', type: 'size', label: 'Spacing 2XL', default: '3rem' },
                { name: '--radius-sm', type: 'size', label: 'Border Radius SM', default: '0.25rem' },
                { name: '--radius-md', type: 'size', label: 'Border Radius MD', default: '0.375rem' },
                { name: '--radius-lg', type: 'size', label: 'Border Radius LG', default: '0.5rem' },
                { name: '--radius-xl', type: 'size', label: 'Border Radius XL', default: '0.75rem' },
                { name: '--radius-2xl', type: 'size', label: 'Border Radius 2XL', default: '1rem' },
                { name: '--radius-full', type: 'size', label: 'Border Radius Full', default: '9999px' }
            ],
            effects: [
                { name: '--shadow-xs', type: 'shadow', label: 'Shadow XS', default: '0 1px 2px 0 rgba(0, 0, 0, 0.05)' },
                { name: '--shadow-sm', type: 'shadow', label: 'Shadow SM', default: '0 1px 3px 0 rgba(0, 0, 0, 0.1)' },
                { name: '--shadow-md', type: 'shadow', label: 'Shadow MD', default: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' },
                { name: '--shadow-lg', type: 'shadow', label: 'Shadow LG', default: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' },
                { name: '--shadow-xl', type: 'shadow', label: 'Shadow XL', default: '0 20px 25px -5px rgba(0, 0, 0, 0.1)' },
                { name: '--transition-fast', type: 'transition', label: 'Transition Fast', default: 'all 0.15s ease-in-out' },
                { name: '--transition-normal', type: 'transition', label: 'Transition Normal', default: 'all 0.3s ease-in-out' },
                { name: '--transition-slow', type: 'transition', label: 'Transition Slow', default: 'all 0.5s ease-in-out' }
            ],
            animations: [
                { name: '--animation-bounce', type: 'animation', label: 'Bounce Animation', default: 'bounce 1s infinite' },
                { name: '--animation-fade', type: 'animation', label: 'Fade Animation', default: 'fade 0.3s ease-in-out' },
                { name: '--animation-slide', type: 'animation', label: 'Slide Animation', default: 'slide 0.3s ease-out' },
                { name: '--animation-pulse', type: 'animation', label: 'Pulse Animation', default: 'pulse 2s infinite' }
            ],
            advanced: [
                { name: '--backdrop-blur', type: 'filter', label: 'Backdrop Blur', default: 'blur(10px)' },
                { name: '--transform-scale', type: 'transform', label: 'Transform Scale', default: 'scale(1)' },
                { name: '--transform-rotate', type: 'transform', label: 'Transform Rotate', default: 'rotate(0deg)' },
                { name: '--filter-grayscale', type: 'filter', label: 'Grayscale Filter', default: 'grayscale(0)' },
                { name: '--filter-sepia', type: 'filter', label: 'Sepia Filter', default: 'sepia(0)' }
            ]
        };

        // Populate base theme selector
        this.populateBaseThemes();
    }

    populateBaseThemes() {
        const select = document.getElementById('base-theme-select');
        const themes = this.themeManager.getAvailableThemes();
        
        themes.forEach(theme => {
            const option = document.createElement('option');
            option.value = theme.id;
            option.textContent = theme.name;
            select.appendChild(option);
        });
    }

    open(baseThemeId = null) {
        this.isOpen = true;
        this.baseTheme = baseThemeId || this.themeManager.currentTheme;
        this.currentTheme = this.themeManager.themes.get(this.baseTheme);
        
        // Set base theme in selector
        document.getElementById('base-theme-select').value = this.baseTheme;
        
        // Load current theme properties
        this.loadThemeProperties();
        
        // Show editor
        const editor = document.getElementById('theme-editor');
        editor.classList.add('open');
        editor.style.display = 'block';
        
        // Initialize with colors category
        this.setCategory('colors');
        
        // Apply current theme to preview
        this.updatePreview();
    }

    close() {
        this.isOpen = false;
        const editor = document.getElementById('theme-editor');
        editor.classList.remove('open');
        
        setTimeout(() => {
            editor.style.display = 'none';
        }, 300);
    }

    confirmClose() {
        if (confirm('You have unsaved changes. Are you sure you want to close?')) {
            this.isDirty = false;
            this.close();
        }
    }

    loadThemeProperties() {
        if (!this.currentTheme) return;
        
        // Clear custom properties
        this.customProperties.clear();
        this.componentStyles.clear();
        
        // Load theme colors
        if (this.currentTheme.colors) {
            Object.entries(this.currentTheme.colors).forEach(([key, value]) => {
                const cssVar = `--${key.replace(/([A-Z])/g, '-$1').toLowerCase()}-color`;
                this.customProperties.set(cssVar, value);
            });
        }
        
        // Load theme fonts
        if (this.currentTheme.fonts) {
            Object.entries(this.currentTheme.fonts).forEach(([key, value]) => {
                const cssVar = `--font-${key}`;
                this.customProperties.set(cssVar, value);
            });
        }
        
        // Load theme spacing
        if (this.currentTheme.spacing) {
            Object.entries(this.currentTheme.spacing).forEach(([key, value]) => {
                const cssVar = `--spacing-${key}`;
                this.customProperties.set(cssVar, value.toString());
            });
        }
    }

    setCategory(category) {
        this.currentCategory = category;
        
        // Update category buttons
        const buttons = document.querySelectorAll('.category-btn');
        buttons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.category === category);
        });
        
        // Show property editor
        this.showPropertyEditor(category);
    }

    showPropertyEditor(category) {
        const panel = document.getElementById('property-editor');
        const properties = this.cssProperties[category] || [];
        
        panel.innerHTML = `
            <div class="property-group">
                <h4>${category.charAt(0).toUpperCase() + category.slice(1)} Properties</h4>
                <div class="property-list">
                    ${properties.map(prop => this.createPropertyEditor(prop)).join('')}
                </div>
            </div>
        `;
        
        // Bind property editor events
        this.bindPropertyEditorEvents();
    }

    createPropertyEditor(property) {
        const currentValue = this.customProperties.get(property.name) || property.default;
        
        switch (property.type) {
            case 'color':
                return `
                    <div class="property-item">
                        <label class="property-label">${property.label}</label>
                        <div class="color-input-wrapper">
                            <input type="color" class="color-input" data-property="${property.name}" value="${this.normalizeColor(currentValue)}">
                            <input type="text" class="color-text" data-property="${property.name}" value="${currentValue}" placeholder="${property.default}">
                        </div>
                    </div>
                `;
                
            case 'font':
                return `
                    <div class="property-item">
                        <label class="property-label">${property.label}</label>
                        <select class="font-select" data-property="${property.name}">
                            <option value="Inter, sans-serif" ${currentValue.includes('Inter') ? 'selected' : ''}>Inter</option>
                            <option value="system-ui, sans-serif" ${currentValue.includes('system-ui') ? 'selected' : ''}>System UI</option>
                            <option value="'Times New Roman', serif" ${currentValue.includes('Times') ? 'selected' : ''}>Times New Roman</option>
                            <option value="'Courier New', monospace" ${currentValue.includes('Courier') ? 'selected' : ''}>Courier New</option>
                            <option value="Georgia, serif" ${currentValue.includes('Georgia') ? 'selected' : ''}>Georgia</option>
                            <option value="Verdana, sans-serif" ${currentValue.includes('Verdana') ? 'selected' : ''}>Verdana</option>
                        </select>
                    </div>
                `;
                
            case 'size':
                return `
                    <div class="property-item">
                        <label class="property-label">${property.label}</label>
                        <div class="size-input-wrapper">
                            <input type="range" class="size-range" data-property="${property.name}" min="0" max="100" value="${this.extractSize(currentValue)}">
                            <input type="text" class="size-text" data-property="${property.name}" value="${currentValue}" placeholder="${property.default}">
                        </div>
                    </div>
                `;
                
            case 'weight':
                return `
                    <div class="property-item">
                        <label class="property-label">${property.label}</label>
                        <select class="weight-select" data-property="${property.name}">
                            <option value="100" ${currentValue === '100' ? 'selected' : ''}>Thin (100)</option>
                            <option value="200" ${currentValue === '200' ? 'selected' : ''}>Extra Light (200)</option>
                            <option value="300" ${currentValue === '300' ? 'selected' : ''}>Light (300)</option>
                            <option value="400" ${currentValue === '400' ? 'selected' : ''}>Normal (400)</option>
                            <option value="500" ${currentValue === '500' ? 'selected' : ''}>Medium (500)</option>
                            <option value="600" ${currentValue === '600' ? 'selected' : ''}>Semibold (600)</option>
                            <option value="700" ${currentValue === '700' ? 'selected' : ''}>Bold (700)</option>
                            <option value="800" ${currentValue === '800' ? 'selected' : ''}>Extra Bold (800)</option>
                            <option value="900" ${currentValue === '900' ? 'selected' : ''}>Black (900)</option>
                        </select>
                    </div>
                `;
                
            case 'shadow':
                return `
                    <div class="property-item">
                        <label class="property-label">${property.label}</label>
                        <textarea class="shadow-text" data-property="${property.name}" placeholder="${property.default}">${currentValue}</textarea>
                    </div>
                `;
                
            case 'transition':
                return `
                    <div class="property-item">
                        <label class="property-label">${property.label}</label>
                        <input type="text" class="transition-text" data-property="${property.name}" value="${currentValue}" placeholder="${property.default}">
                    </div>
                `;
                
            case 'animation':
                return `
                    <div class="property-item">
                        <label class="property-label">${property.label}</label>
                        <input type="text" class="animation-text" data-property="${property.name}" value="${currentValue}" placeholder="${property.default}">
                    </div>
                `;
                
            case 'transform':
                return `
                    <div class="property-item">
                        <label class="property-label">${property.label}</label>
                        <input type="text" class="transform-text" data-property="${property.name}" value="${currentValue}" placeholder="${property.default}">
                    </div>
                `;
                
            case 'filter':
                return `
                    <div class="property-item">
                        <label class="property-label">${property.label}</label>
                        <input type="text" class="filter-text" data-property="${property.name}" value="${currentValue}" placeholder="${property.default}">
                    </div>
                `;
                
            default:
                return `
                    <div class="property-item">
                        <label class="property-label">${property.label}</label>
                        <input type="text" class="property-input" data-property="${property.name}" value="${currentValue}" placeholder="${property.default}">
                    </div>
                `;
        }
    }

    bindPropertyEditorEvents() {
        // Color inputs
        const colorInputs = document.querySelectorAll('.color-input');
        colorInputs.forEach(input => {
            input.addEventListener('input', (e) => {
                const property = e.target.dataset.property;
                const value = e.target.value;
                this.updateProperty(property, value);
                
                // Update text input
                const textInput = document.querySelector(`.color-text[data-property="${property}"]`);
                if (textInput) textInput.value = value;
            });
        });

        const colorTexts = document.querySelectorAll('.color-text');
        colorTexts.forEach(input => {
            input.addEventListener('input', (e) => {
                const property = e.target.dataset.property;
                const value = e.target.value;
                this.updateProperty(property, value);
                
                // Update color input
                const colorInput = document.querySelector(`.color-input[data-property="${property}"]`);
                if (colorInput) colorInput.value = this.normalizeColor(value);
            });
        });

        // Size inputs
        const sizeRanges = document.querySelectorAll('.size-range');
        sizeRanges.forEach(input => {
            input.addEventListener('input', (e) => {
                const property = e.target.dataset.property;
                const value = e.target.value + 'rem';
                this.updateProperty(property, value);
                
                // Update text input
                const textInput = document.querySelector(`.size-text[data-property="${property}"]`);
                if (textInput) textInput.value = value;
            });
        });

        const sizeTexts = document.querySelectorAll('.size-text');
        sizeTexts.forEach(input => {
            input.addEventListener('input', (e) => {
                const property = e.target.dataset.property;
                const value = e.target.value;
                this.updateProperty(property, value);
                
                // Update range input
                const rangeInput = document.querySelector(`.size-range[data-property="${property}"]`);
                if (rangeInput) rangeInput.value = this.extractSize(value);
            });
        });

        // Other inputs
        const allInputs = document.querySelectorAll('select, textarea, .property-input, .shadow-text, .transition-text, .animation-text, .transform-text, .filter-text');
        allInputs.forEach(input => {
            input.addEventListener('input', (e) => {
                const property = e.target.dataset.property;
                const value = e.target.value;
                this.updateProperty(property, value);
            });
        });
    }

    updateProperty(property, value) {
        this.customProperties.set(property, value);
        this.isDirty = true;
        this.updatePreview();
        this.addToHistory();
    }

    updatePreview() {
        const preview = document.getElementById('preview-content');
        if (!preview) return;

        // Apply custom properties to preview
        const previewStyles = document.createElement('style');
        previewStyles.id = 'preview-custom-styles';
        
        let css = ':root {';
        this.customProperties.forEach((value, property) => {
            css += `${property}: ${value};`;
        });
        css += '}';
        
        // Add custom CSS
        if (this.customCSS) {
            css += this.customCSS;
        }
        
        // Add component-specific styles
        this.componentStyles.forEach((styles, component) => {
            css += styles;
        });
        
        previewStyles.textContent = css;
        
        // Remove old preview styles
        const oldStyles = document.getElementById('preview-custom-styles');
        if (oldStyles) oldStyles.remove();
        
        // Add new preview styles
        preview.appendChild(previewStyles);
    }

    setBaseTheme(themeId) {
        this.baseTheme = themeId;
        this.currentTheme = this.themeManager.themes.get(themeId);
        this.loadThemeProperties();
        this.updatePreview();
        this.isDirty = true;
    }

    setComponent(component) {
        this.currentComponent = component;
        // Load component-specific styles
        this.loadComponentStyles(component);
    }

    loadComponentStyles(component) {
        // Implementation for loading component-specific styles
        // This would load styles for specific components like buttons, cards, etc.
    }

    togglePreview() {
        const propertyPanel = document.getElementById('property-editor');
        const codePanel = document.getElementById('code-editor');
        const previewPanel = document.getElementById('preview-panel');
        
        if (previewPanel.style.display === 'none') {
            previewPanel.style.display = 'block';
            propertyPanel.style.display = 'block';
            codePanel.style.display = 'none';
        } else {
            previewPanel.style.display = 'none';
            propertyPanel.style.display = 'none';
            codePanel.style.display = 'block';
        }
    }

    setPreviewSize(size) {
        const preview = document.getElementById('preview-content');
        const buttons = document.querySelectorAll('.preview-size-btn');
        
        buttons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.size === size);
        });
        
        // Remove existing size classes
        preview.classList.remove('preview-desktop', 'preview-tablet', 'preview-mobile');
        
        // Add new size class
        preview.classList.add(`preview-${size}`);
    }

    updateCustomCSS(css) {
        this.customCSS = css;
        this.isDirty = true;
        this.updatePreview();
    }

    formatCSS() {
        const textarea = document.getElementById('css-code');
        if (textarea.value) {
            // Simple CSS formatting
            const formatted = textarea.value
                .replace(/\s*{\s*/g, ' {\n  ')
                .replace(/;\s*/g, ';\n  ')
                .replace(/\s*}\s*/g, '\n}\n');
            textarea.value = formatted;
        }
    }

    validateCSS() {
        const css = document.getElementById('css-code').value;
        // Basic CSS validation
        const errors = [];
        
        // Check for basic syntax
        const openBraces = (css.match(/{/g) || []).length;
        const closeBraces = (css.match(/}/g) || []).length;
        
        if (openBraces !== closeBraces) {
            errors.push('Mismatched braces');
        }
        
        if (errors.length > 0) {
            alert('CSS Validation Errors:\n' + errors.join('\n'));
        } else {
            alert('CSS is valid!');
        }
    }

    resetChanges() {
        if (confirm('Are you sure you want to reset all changes?')) {
            this.loadThemeProperties();
            this.customCSS = '';
            this.updatePreview();
            this.isDirty = false;
            
            // Clear code editor
            document.getElementById('css-code').value = '';
        }
    }

    saveTheme() {
        const themeName = prompt('Enter a name for your custom theme:');
        if (!themeName) return;
        
        const customTheme = {
            id: `custom_${Date.now()}`,
            name: themeName,
            category: 'custom',
            description: 'Custom theme created by user',
            baseTheme: this.baseTheme,
            customProperties: Object.fromEntries(this.customProperties),
            customCSS: this.customCSS,
            componentStyles: Object.fromEntries(this.componentStyles),
            createdAt: new Date().toISOString(),
            isCustom: true
        };
        
        // Save theme to theme manager
        this.themeManager.registerTheme(customTheme.id, customTheme);
        
        // Apply the new theme
        this.themeManager.applyTheme(customTheme.id);
        
        this.isDirty = false;
        this.close();
        
        alert(`Theme "${themeName}" saved successfully!`);
    }

    addToHistory() {
        // Add current state to history for undo/redo functionality
        const state = {
            customProperties: new Map(this.customProperties),
            customCSS: this.customCSS,
            componentStyles: new Map(this.componentStyles)
        };
        
        this.history = this.history.slice(0, this.historyIndex + 1);
        this.history.push(state);
        this.historyIndex++;
        
        // Limit history size
        if (this.history.length > 50) {
            this.history.shift();
            this.historyIndex--;
        }
    }

    // Utility methods
    normalizeColor(color) {
        // Convert various color formats to hex
        if (color.startsWith('#')) {
            return color;
        }
        // Add more color format conversions as needed
        return color;
    }

    extractSize(size) {
        // Extract numeric value from size string
        const match = size.match(/(\d+(?:\.\d+)?)/);
        return match ? parseFloat(match[1]) : 0;
    }

    showExportModal() {
        // Create export modal
        const modal = document.createElement('div');
        modal.className = 'export-modal';
        modal.innerHTML = `
            <div class="export-modal-content">
                <div class="export-modal-header">
                    <h4>Export Theme</h4>
                    <button class="export-modal-close" aria-label="Close">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M18 6L6 18M6 6l12 12"/>
                        </svg>
                    </button>
                </div>
                
                <div class="export-modal-body">
                    <div class="export-options">
                        <div class="export-option-group">
                            <label for="export-format">Export Format</label>
                            <select id="export-format" class="form-control">
                                <option value="json">JSON (Complete theme data)</option>
                                <option value="css">CSS (Ready to use)</option>
                                <option value="scss">SCSS (With variables)</option>
                            </select>
                        </div>
                        
                        <div class="export-option-group">
                            <label for="export-filename">Filename</label>
                            <input type="text" id="export-filename" class="form-control" placeholder="my-custom-theme" value="${this.currentTheme?.name || 'custom-theme'}">
                        </div>
                        
                        <div class="export-option-group">
                            <label class="checkbox-label">
                                <input type="checkbox" id="include-metadata" checked>
                                <span>Include theme metadata</span>
                            </label>
                        </div>
                        
                        <div class="export-option-group">
                            <label class="checkbox-label">
                                <input type="checkbox" id="include-preview" checked>
                                <span>Include preview in CSS</span>
                            </label>
                        </div>
                        
                        <div class="export-option-group">
                            <label class="checkbox-label">
                                <input type="checkbox" id="minimize-css">
                                <span>Minimize CSS output</span>
                            </label>
                        </div>
                    </div>
                    
                    <div class="export-preview">
                        <h5>Preview</h5>
                        <div class="export-preview-content" id="export-preview-content">
                            <!-- Preview content will be generated here -->
                        </div>
                    </div>
                </div>
                
                <div class="export-modal-footer">
                    <button class="btn btn-ghost" id="export-cancel-btn">Cancel</button>
                    <button class="btn btn-primary" id="export-confirm-btn">Export Theme</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Show modal
        setTimeout(() => {
            modal.classList.add('show');
        }, 10);
        
        // Bind events
        this.bindExportModalEvents(modal);
        
        // Generate initial preview
        this.updateExportPreview();
    }

    bindExportModalEvents(modal) {
        // Close button
        const closeBtn = modal.querySelector('.export-modal-close');
        closeBtn.addEventListener('click', () => this.closeExportModal(modal));
        
        // Cancel button
        const cancelBtn = modal.querySelector('#export-cancel-btn');
        cancelBtn.addEventListener('click', () => this.closeExportModal(modal));
        
        // Export button
        const exportBtn = modal.querySelector('#export-confirm-btn');
        exportBtn.addEventListener('click', () => this.exportTheme(modal));
        
        // Format change
        const formatSelect = modal.querySelector('#export-format');
        formatSelect.addEventListener('change', () => this.updateExportPreview());
        
        // Filename change
        const filenameInput = modal.querySelector('#export-filename');
        filenameInput.addEventListener('input', () => this.updateExportPreview());
        
        // Checkbox changes
        const checkboxes = modal.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => this.updateExportPreview());
        });
        
        // Close on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeExportModal(modal);
            }
        });
        
        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeExportModal(modal);
            }
        });
    }

    closeExportModal(modal) {
        modal.classList.remove('show');
        setTimeout(() => {
            if (modal.parentNode) {
                modal.parentNode.removeChild(modal);
            }
        }, 300);
    }

    updateExportPreview() {
        const modal = document.querySelector('.export-modal');
        if (!modal) return;
        
        const format = modal.querySelector('#export-format').value;
        const filename = modal.querySelector('#export-filename').value || 'custom-theme';
        const includeMetadata = modal.querySelector('#include-metadata').checked;
        const includePreview = modal.querySelector('#include-preview').checked;
        const minimizeCSS = modal.querySelector('#minimize-css').checked;
        
        try {
            // Create temporary theme data for preview
            const tempTheme = {
                ...this.currentTheme,
                customProperties: Object.fromEntries(this.customProperties),
                customCSS: this.customCSS,
                componentStyles: Object.fromEntries(this.componentStyles)
            };
            
            // Generate preview based on format
            let preview = '';
            const tempId = `temp_${Date.now()}`;
            
            // Register temporary theme for export
            this.themeManager.registerTheme(tempId, tempTheme);
            
            try {
                if (window.themeExporter) {
                    const exported = window.themeExporter.exportTheme(tempId, format);
                    preview = exported.content;
                    
                    // Apply options
                    if (format === 'css' || format === 'scss') {
                        if (!includeMetadata && preview.includes('/*')) {
                            // Remove comments
                            preview = preview.replace(/\/\*[\s\S]*?\*\//g, '');
                        }
                        
                        if (minimizeCSS) {
                            // Minimize CSS
                            preview = preview.replace(/\s+/g, ' ').replace(/;\s*}/g, '}').trim();
                        }
                    }
                    
                    if (!includeMetadata && format === 'json') {
                        const jsonData = JSON.parse(preview);
                        delete jsonData.metadata;
                        preview = JSON.stringify(jsonData, null, 2);
                    }
                } else {
                    preview = 'Theme exporter not available';
                }
            } finally {
                // Clean up temporary theme
                this.themeManager.themes.delete(tempId);
            }
            
            // Update preview content
            const previewContent = modal.querySelector('#export-preview-content');
            if (previewContent) {
                previewContent.textContent = preview;
                previewContent.className = `export-preview-content ${format}`;
            }
            
        } catch (error) {
            const previewContent = modal.querySelector('#export-preview-content');
            if (previewContent) {
                previewContent.textContent = `Error generating preview: ${error.message}`;
                previewContent.className = 'export-preview-content error';
            }
        }
    }

    exportTheme(modal) {
        const format = modal.querySelector('#export-format').value;
        const filename = modal.querySelector('#export-filename').value || 'custom-theme';
        const includeMetadata = modal.querySelector('#include-metadata').checked;
        const includePreview = modal.querySelector('#include-preview').checked;
        const minimizeCSS = modal.querySelector('#minimize-css').checked;
        
        try {
            // Create temporary theme for export
            const tempTheme = {
                ...this.currentTheme,
                customProperties: Object.fromEntries(this.customProperties),
                customCSS: this.customCSS,
                componentStyles: Object.fromEntries(this.componentStyles)
            };
            
            const tempId = `export_${Date.now()}`;
            this.themeManager.registerTheme(tempId, tempTheme);
            
            try {
                if (window.themeExporter) {
                    // Export the theme
                    const exported = window.themeExporter.exportTheme(tempId, format);
                    
                    // Apply options
                    let content = exported.content;
                    
                    if (format === 'css' || format === 'scss') {
                        if (!includeMetadata && content.includes('/*')) {
                            content = content.replace(/\/\*[\s\S]*?\*\//g, '');
                        }
                        
                        if (minimizeCSS) {
                            content = content.replace(/\s+/g, ' ').replace(/;\s*}/g, '}').trim();
                        }
                    }
                    
                    if (!includeMetadata && format === 'json') {
                        const jsonData = JSON.parse(content);
                        delete jsonData.metadata;
                        content = JSON.stringify(jsonData, null, 2);
                    }
                    
                    // Create and trigger download
                    const blob = new Blob([content], { type: exported.mimeType });
                    const url = URL.createObjectURL(blob);
                    
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = `${filename}.${format}`;
                    link.style.display = 'none';
                    
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    URL.revokeObjectURL(url);
                    
                    // Show success message
                    this.showToast(`Theme exported successfully as ${filename}.${format}`, 'success');
                    
                    // Close modal
                    this.closeExportModal(modal);
                    
                } else {
                    this.showToast('Theme exporter not available', 'error');
                }
            } finally {
                // Clean up temporary theme
                this.themeManager.themes.delete(tempId);
            }
            
        } catch (error) {
            this.showToast(`Export failed: ${error.message}`, 'error');
        }
    }

    showToast(message, type = 'info') {
        // Use existing toast system if available
        if (window.showToast) {
            window.showToast(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// Initialize theme editor when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for theme manager to be available
    setTimeout(() => {
        if (window.themeManager) {
            window.themeEditor = new ThemeEditor(window.themeManager);
        }
    }, 100);
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThemeEditor;
}
