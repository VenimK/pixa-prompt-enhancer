/**
 * Theme Manager - Centralized theme management system
 * Handles theme switching, persistence, and CSS variable management
 */
class ThemeManager {
    constructor() {
        this.currentTheme = 'professional-blue';
        this.themes = new Map();
        this.isDarkMode = false;
        this.customSettings = {};
        
        this.init();
    }

    async init() {
        // Load saved preferences
        this.loadSavedPreferences();
        
        // Register default themes
        this.registerDefaultThemes();
        
        // Apply initial theme
        await this.applyTheme(this.currentTheme);
        
        // Set up CSS variable listeners
        this.setupVariableListeners();
    }

    loadSavedPreferences() {
        try {
            const saved = localStorage.getItem('themePreferences');
            if (saved) {
                const prefs = JSON.parse(saved);
                this.currentTheme = prefs.theme || 'professional-blue';
                this.isDarkMode = prefs.darkMode || false;
                this.customSettings = prefs.customSettings || {};
            }
        } catch (error) {
            console.warn('Failed to load theme preferences:', error);
        }
    }

    savePreferences() {
        try {
            const prefs = {
                theme: this.currentTheme,
                darkMode: this.isDarkMode,
                customSettings: this.customSettings,
                lastUpdated: new Date().toISOString()
            };
            localStorage.setItem('themePreferences', JSON.stringify(prefs));
        } catch (error) {
            console.warn('Failed to save theme preferences:', error);
        }
    }

    registerDefaultThemes() {
        // Professional Blue (current default)
        this.registerTheme('professional-blue', {
            name: 'Professional Blue',
            category: 'professional',
            colors: {
                primary: '#4f46e5',
                primaryHover: '#4338ca',
                primaryLight: '#818cf8',
                success: '#10b981',
                warning: '#f59e0b',
                error: '#ef4444',
                background: '#f9fafb',
                surface: '#ffffff',
                text: '#111827',
                textSecondary: '#374151',
                border: '#e5e7eb'
            },
            fonts: {
                primary: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto',
                mono: '"Fira Code", "Monaco", "Cascadia Code", monospace'
            },
            spacing: {
                compact: 0.8,
                normal: 1,
                spacious: 1.2
            }
        });

        // Creative Purple
        this.registerTheme('creative-purple', {
            name: 'Creative Purple',
            category: 'creative',
            colors: {
                primary: '#7c3aed',
                primaryHover: '#6d28d9',
                primaryLight: '#a78bfa',
                success: '#10b981',
                warning: '#f59e0b',
                error: '#ef4444',
                background: '#faf5ff',
                surface: '#ffffff',
                text: '#1f2937',
                textSecondary: '#4b5563',
                border: '#e9d5ff'
            },
            fonts: {
                primary: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto',
                mono: '"Fira Code", "Monaco", "Cascadia Code", monospace'
            },
            spacing: {
                compact: 0.8,
                normal: 1,
                spacious: 1.2
            }
        });

        // Nature Green
        this.registerTheme('nature-green', {
            name: 'Nature Green',
            category: 'nature',
            colors: {
                primary: '#059669',
                primaryHover: '#047857',
                primaryLight: '#34d399',
                success: '#10b981',
                warning: '#f59e0b',
                error: '#ef4444',
                background: '#f0fdf4',
                surface: '#ffffff',
                text: '#064e3b',
                textSecondary: '#047857',
                border: '#bbf7d0'
            },
            fonts: {
                primary: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto',
                mono: '"Fira Code", "Monaco", "Cascadia Code", monospace'
            },
            spacing: {
                compact: 0.8,
                normal: 1,
                spacious: 1.2
            }
        });

        // Sunset Orange
        this.registerTheme('sunset-orange', {
            name: 'Sunset Orange',
            category: 'warm',
            colors: {
                primary: '#ea580c',
                primaryHover: '#c2410c',
                primaryLight: '#fb923c',
                success: '#10b981',
                warning: '#f59e0b',
                error: '#ef4444',
                background: '#fff7ed',
                surface: '#ffffff',
                text: '#1c1917',
                textSecondary: '#78350f',
                border: '#fed7aa'
            },
            fonts: {
                primary: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto',
                mono: '"Fira Code", "Monaco", "Cascadia Code", monospace'
            },
            spacing: {
                compact: 0.8,
                normal: 1,
                spacious: 1.2
            }
        });

        // Ocean Blue
        this.registerTheme('ocean-blue', {
            name: 'Ocean Blue',
            category: 'cool',
            colors: {
                primary: '#0e7490',
                primaryHover: '#0c4a6e',
                primaryLight: '#22d3ee',
                success: '#10b981',
                warning: '#f59e0b',
                error: '#ef4444',
                background: '#f0f9ff',
                surface: '#ffffff',
                text: '#0c4a6e',
                textSecondary: '#075985',
                border: '#bae6fd'
            },
            fonts: {
                primary: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto',
                mono: '"Fira Code", "Monaco", "Cascadia Code", monospace'
            },
            spacing: {
                compact: 0.8,
                normal: 1,
                spacious: 1.2
            }
        });

        // Monochrome
        this.registerTheme('monochrome', {
            name: 'Monochrome',
            category: 'minimal',
            colors: {
                primary: '#374151',
                primaryHover: '#1f2937',
                primaryLight: '#6b7280',
                success: '#059669',
                warning: '#d97706',
                error: '#dc2626',
                background: '#ffffff',
                surface: '#f9fafb',
                text: '#111827',
                textSecondary: '#374151',
                border: '#d1d5db'
            },
            fonts: {
                primary: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto',
                mono: '"Fira Code", "Monaco", "Cascadia Code", monospace'
            },
            spacing: {
                compact: 0.8,
                normal: 1,
                spacious: 1.2
            }
        });

        // Cyberpunk
        this.registerTheme('cyberpunk', {
            name: 'Cyberpunk',
            category: 'futuristic',
            colors: {
                primary: '#ec4899',
                primaryHover: '#db2777',
                primaryLight: '#f472b6',
                accent: '#06b6d4',
                success: '#10b981',
                warning: '#f59e0b',
                error: '#ef4444',
                background: '#0f0f23',
                surface: '#1a1a2e',
                text: '#e0e0e0',
                textSecondary: '#a0a0a0',
                border: '#2a2a4e'
            },
            fonts: {
                primary: '"Orbitron", "Inter", -apple-system, BlinkMacSystemFont',
                mono: '"Fira Code", "Monaco", "Cascadia Code", monospace'
            },
            spacing: {
                compact: 0.8,
                normal: 1,
                spacious: 1.2
            }
        });

        // Retro
        this.registerTheme('retro', {
            name: 'Retro',
            category: 'vintage',
            colors: {
                primary: '#dc2626',
                primaryHover: '#b91c1c',
                primaryLight: '#ef4444',
                accent: '#fbbf24',
                success: '#10b981',
                warning: '#f59e0b',
                error: '#ef4444',
                background: '#fef2f2',
                surface: '#ffffff',
                text: '#1f2937',
                textSecondary: '#7f1d1d',
                border: '#fecaca'
            },
            fonts: {
                primary: '"Press Start 2P", "Courier New", monospace',
                mono: '"Fira Code", "Monaco", "Cascadia Code", monospace'
            },
            spacing: {
                compact: 0.8,
                normal: 1,
                spacious: 1.2
            }
        });
    }

    registerTheme(id, config) {
        this.themes.set(id, {
            id,
            ...config,
            registeredAt: new Date().toISOString()
        });
    }

    async applyTheme(themeId) {
        const theme = this.themes.get(themeId);
        if (!theme) {
            console.warn(`Theme '${themeId}' not found`);
            return false;
        }

        this.currentTheme = themeId;
        
        // Apply CSS custom properties
        this.applyCSSVariables(theme);
        
        // Update body classes
        this.updateBodyClasses(theme);
        
        // Save preferences
        this.savePreferences();
        
        // Dispatch theme change event
        this.dispatchThemeChange(theme);
        
        return true;
    }

    applyCSSVariables(theme) {
        const root = document.documentElement;
        const colors = theme.colors;
        
        // Apply color variables
        Object.entries(colors).forEach(([key, value]) => {
            const cssVar = `--${this.kebabCase(key)}-color`;
            root.style.setProperty(cssVar, value);
        });
        
        // Apply spacing variables
        if (theme.spacing) {
            Object.entries(theme.spacing).forEach(([key, value]) => {
                const cssVar = `--spacing-${key}`;
                root.style.setProperty(cssVar, value.toString());
            });
        }
        
        // Apply font variables
        if (theme.fonts) {
            Object.entries(theme.fonts).forEach(([key, value]) => {
                const cssVar = `--font-${key}`;
                root.style.setProperty(cssVar, value);
            });
        }
    }

    updateBodyClasses(theme) {
        const body = document.body;
        
        // Remove all theme classes
        body.className = body.className.replace(/theme-\S+/g, '').trim();
        
        // Add new theme class
        body.classList.add(`theme-${theme.id}`);
        body.classList.add(`theme-category-${theme.category}`);
        
        // Update dark mode
        if (this.isDarkMode) {
            body.classList.add('dark-mode');
            document.documentElement.classList.add('dark-mode');
        } else {
            body.classList.remove('dark-mode');
            document.documentElement.classList.remove('dark-mode');
        }
    }

    toggleDarkMode() {
        this.isDarkMode = !this.isDarkMode;
        this.updateBodyClasses(this.themes.get(this.currentTheme));
        this.savePreferences();
        this.dispatchThemeChange(this.themes.get(this.currentTheme));
    }

    setDarkMode(enabled) {
        this.isDarkMode = enabled;
        this.updateBodyClasses(this.themes.get(this.currentTheme));
        this.savePreferences();
        this.dispatchThemeChange(this.themes.get(this.currentTheme));
    }

    getAvailableThemes() {
        return Array.from(this.themes.values()).sort((a, b) => a.name.localeCompare(b.name));
    }

    getThemesByCategory(category) {
        return this.getAvailableThemes().filter(theme => theme.category === category);
    }

    getCurrentTheme() {
        return this.themes.get(this.currentTheme);
    }

    createCustomTheme(name, baseThemeId, customColors) {
        const baseTheme = this.themes.get(baseThemeId);
        if (!baseTheme) {
            throw new Error(`Base theme '${baseThemeId}' not found`);
        }

        const customThemeId = `custom-${Date.now()}`;
        const customTheme = {
            ...baseTheme,
            id: customThemeId,
            name: name || `Custom ${baseTheme.name}`,
            category: 'custom',
            colors: { ...baseTheme.colors, ...customColors },
            isCustom: true,
            createdAt: new Date().toISOString()
        };

        this.registerTheme(customThemeId, customTheme);
        return customTheme;
    }

    setupVariableListeners() {
        // Listen for CSS variable changes
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                    this.onCSSVariablesChanged();
                }
            });
        });

        observer.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ['style']
        });
    }

    onCSSVariablesChanged() {
        // Handle CSS variable changes if needed
        this.dispatchThemeChange(this.getCurrentTheme());
    }

    dispatchThemeChange(theme) {
        const event = new CustomEvent('themeChanged', {
            detail: {
                theme,
                isDarkMode: this.isDarkMode,
                timestamp: new Date().toISOString()
            }
        });
        document.dispatchEvent(event);
    }

    kebabCase(str) {
        return str.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase();
    }

    // Utility methods
    exportTheme(themeId) {
        const theme = this.themes.get(themeId);
        if (!theme) return null;

        return {
            ...theme,
            exportedAt: new Date().toISOString(),
            version: '1.0'
        };
    }

    importTheme(themeData) {
        if (!themeData.id || !themeData.name) {
            throw new Error('Invalid theme data: missing id or name');
        }

        this.registerTheme(themeData.id, themeData);
        return themeData;
    }

    resetToDefaults() {
        this.currentTheme = 'professional-blue';
        this.isDarkMode = false;
        this.customSettings = {};
        this.applyTheme(this.currentTheme);
    }
}

// Initialize theme manager when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.themeManager = new ThemeManager();
    
    // Make theme manager globally available
    window.ThemeManager = ThemeManager;
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThemeManager;
}
