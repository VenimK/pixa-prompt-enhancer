/**
 * Theme Selector UI Component
 * Provides user interface for theme selection and customization
 */
class ThemeSelector {
    constructor(themeManager) {
        this.themeManager = themeManager;
        this.isOpen = false;
        this.currentCategory = 'all';
        this.searchQuery = '';
        
        this.init();
    }

    init() {
        this.createThemeSelector();
        this.createThemeToggle();
        this.bindEvents();
        this.loadThemeData();
    }

    createThemeSelector() {
        // Create theme selector modal
        const selector = document.createElement('div');
        selector.id = 'theme-selector';
        selector.className = 'theme-selector';
        selector.style.display = 'none'; // CRITICAL FIX: Hidden by default
        selector.innerHTML = `
            <div class="theme-selector-header">
                <h3 class="theme-selector-title">Select Theme</h3>
                <button class="theme-selector-close" aria-label="Close">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 6L6 18M6 6l12 12"/>
                    </svg>
                </button>
            </div>
            
            <div class="theme-search">
                <input type="text" class="theme-search-input" placeholder="Search themes...">
                <svg class="theme-search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="m21 21-4.35-4.35"/>
                </svg>
            </div>
            
            <div class="recent-themes" id="recent-themes">
                <h4 class="recent-themes-title">Recent</h4>
                <div class="recent-themes-grid" id="recent-themes-grid">
                    <!-- Recent theme cards will be inserted here -->
                </div>
            </div>
            
            <div class="theme-categories">
                <button class="theme-category active" data-category="all">All</button>
                <button class="theme-category" data-category="professional">Professional</button>
                <button class="theme-category" data-category="creative">Creative</button>
                <button class="theme-category" data-category="nature">Nature</button>
                <button class="theme-category" data-category="warm">Warm</button>
                <button class="theme-category" data-category="cool">Cool</button>
                <button class="theme-category" data-category="minimal">Minimal</button>
                <button class="theme-category" data-category="futuristic">Futuristic</button>
                <button class="theme-category" data-category="vintage">Vintage</button>
                <button class="theme-category" data-category="custom">Custom</button>
            </div>
            
            <div class="theme-grid" id="theme-grid">
                <!-- Theme cards will be inserted here -->
            </div>
            
            <div class="theme-actions">
                <button class="btn btn-ghost" id="reset-theme-btn">Reset to Default</button>
                <button class="btn btn-primary" id="customize-theme-btn">Customize Theme</button>
            </div>
        `;

        document.body.appendChild(selector);
    }

    createThemeToggle() {
        // Create theme toggle button
        const toggle = document.createElement('button');
        toggle.id = 'theme-toggle';
        toggle.className = 'theme-toggle';
        toggle.setAttribute('aria-label', 'Toggle theme selector');
        toggle.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="5"/>
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
            </svg>
        `;

        document.body.appendChild(toggle);

        // Create separate light/dark mode toggle button
        const modeToggle = document.createElement('button');
        modeToggle.id = 'mode-toggle';
        modeToggle.className = 'mode-toggle';
        modeToggle.setAttribute('aria-label', 'Toggle light/dark mode');
        modeToggle.innerHTML = `
            <svg id="mode-icon-sun" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="5"/>
                <line x1="12" y1="1" x2="12" y2="3"/>
                <line x1="12" y1="21" x2="12" y2="23"/>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                <line x1="1" y1="12" x2="3" y2="12"/>
                <line x1="21" y1="12" x2="23" y2="12"/>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
            <svg id="mode-icon-moon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display: none;">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
        `;

        document.body.appendChild(modeToggle);

        // Bind light/dark mode toggle event
        modeToggle.addEventListener('click', () => this.toggleLightDarkMode());

        // Initialize mode icon
        this.updateModeIcon();
    }

    toggleLightDarkMode() {
        // Use ThemeManager to toggle dark mode properly (preserves current theme)
        if (window.themeManager) {
            window.themeManager.toggleDarkMode();
            this.updateModeIcon();
            const isDark = window.themeManager.isDarkMode;
            this.showToast(isDark ? 'Switched to dark mode' : 'Switched to light mode', 'info');
        } else {
            // Fallback if ThemeManager not available
            const isDark = document.body.classList.contains('dark-mode');
            if (isDark) {
                document.body.classList.remove('dark-mode');
                document.body.classList.add('light-mode');
                localStorage.setItem('darkMode', 'false');
            } else {
                document.body.classList.remove('light-mode');
                document.body.classList.add('dark-mode');
                localStorage.setItem('darkMode', 'true');
            }
            this.updateModeIcon();
            this.showToast(isDark ? 'Switched to light mode' : 'Switched to dark mode', 'info');
        }
    }

    updateModeIcon() {
        const isDark = document.body.classList.contains('dark-mode');
        const sunIcon = document.getElementById('mode-icon-sun');
        const moonIcon = document.getElementById('mode-icon-moon');
        if (sunIcon && moonIcon) {
            sunIcon.style.display = isDark ? 'none' : 'block';
            moonIcon.style.display = isDark ? 'block' : 'none';
        }
    }

    showToast(message, type = 'info') {
        if (window.showToast) {
            window.showToast(message, type);
        } else {
            console.log(`[${type}] ${message}`);
        }
    }

    bindEvents() {
        // Theme toggle button
        const toggle = document.getElementById('theme-toggle');
        toggle.addEventListener('click', () => this.toggle());

        // Close button
        const closeBtn = document.querySelector('.theme-selector-close');
        closeBtn.addEventListener('click', () => this.close());

        // Category buttons
        const categoryBtns = document.querySelectorAll('.theme-category');
        categoryBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setCategory(e.target.dataset.category);
            });
        });

        // Search input with debouncing
        const searchInput = document.querySelector('.theme-search-input');
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.setSearchQuery(e.target.value);
            }, 150); // 150ms debounce
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (!this.isOpen) return;
            
            const cards = document.querySelectorAll('.theme-card');
            const activeCard = document.querySelector('.theme-card.active');
            let currentIndex = Array.from(cards).indexOf(activeCard);
            
            switch(e.key) {
                case 'ArrowRight':
                    e.preventDefault();
                    if (currentIndex < cards.length - 1) {
                        cards[currentIndex + 1].focus();
                        cards[currentIndex + 1].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    if (currentIndex > 0) {
                        cards[currentIndex - 1].focus();
                        cards[currentIndex - 1].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    if (currentIndex < cards.length - 1) {
                        cards[currentIndex + 1].focus();
                        cards[currentIndex + 1].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    if (currentIndex > 0) {
                        cards[currentIndex - 1].focus();
                        cards[currentIndex - 1].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                    break;
                case 'Enter':
                    if (activeCard) {
                        const themeId = activeCard.dataset.themeId;
                        this.selectTheme(themeId);
                    }
                    break;
            }
        });

        // Reset button
        const resetBtn = document.getElementById('reset-theme-btn');
        resetBtn.addEventListener('click', () => this.resetToDefault());

        // Customize button
        const customizeBtn = document.getElementById('customize-theme-btn');
        customizeBtn.addEventListener('click', () => this.openCustomizer());

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (this.isOpen && !e.target.closest('#theme-selector') && !e.target.closest('#theme-toggle')) {
                this.close();
            }
        });

        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (this.isOpen && e.key === 'Escape') {
                this.close();
            }
        });

        // Listen for theme changes
        document.addEventListener('themeChanged', (e) => {
            this.updateActiveTheme(e.detail.theme.id);
        });
    }

    async loadThemeData() {
        try {
            const response = await fetch('/static/data/themes.json');
            const data = await response.json();
            this.themeData = data;
            this.renderThemes();
        } catch (error) {
            console.error('Failed to load theme data:', error);
            this.renderThemesFromManager();
        }
    }

    renderThemesFromManager() {
        // Fallback to theme manager data
        const themes = this.themeManager.getAvailableThemes();
        this.renderThemes(themes);
    }

    renderThemes(themes = null) {
        const grid = document.getElementById('theme-grid');
        const recentGrid = document.getElementById('recent-themes-grid');
        const recentSection = document.getElementById('recent-themes');
        const themeList = themes || this.themeManager.getAvailableThemes();
        
        // Filter themes
        const filteredThemes = this.filterThemes(themeList);
        
        // Clear grid
        grid.innerHTML = '';
        
        // Render recent themes
        const recentThemes = this.getRecentThemes();
        if (recentThemes.length > 0 && this.currentCategory === 'all' && !this.searchQuery) {
            recentSection.style.display = 'block';
            recentGrid.innerHTML = '';
            recentThemes.slice(0, 3).forEach(theme => {
                const card = this.createThemeCard(theme, true);
                recentGrid.appendChild(card);
            });
        } else {
            recentSection.style.display = 'none';
        }
        
        // Render theme cards
        filteredThemes.forEach(theme => {
            const card = this.createThemeCard(theme);
            grid.appendChild(card);
        });
        
        // Update active theme
        this.updateActiveTheme(this.themeManager.currentTheme);
    }

    filterThemes(themes) {
        let filtered = themes;
        
        // Filter by category
        if (this.currentCategory !== 'all') {
            filtered = filtered.filter(theme => theme.category === this.currentCategory);
        }
        
        // Filter by search query
        if (this.searchQuery) {
            const query = this.searchQuery.toLowerCase();
            filtered = filtered.filter(theme => 
                theme.name.toLowerCase().includes(query) ||
                theme.category.toLowerCase().includes(query) ||
                (theme.description && theme.description.toLowerCase().includes(query))
            );
        }
        
        return filtered;
    }

    createThemeCard(theme) {
        const card = document.createElement('div');
        card.className = 'theme-card';
        card.dataset.themeId = theme.id;
        
        const preview = theme.preview || {
            primary: theme.colors?.primary || '#4f46e5',
            background: theme.colors?.background || '#f9fafb',
            surface: theme.colors?.surface || '#ffffff'
        };
        
        card.innerHTML = `
            <div class="theme-preview" style="background: linear-gradient(135deg, ${preview.primary}, ${preview.surface});">
                <span style="color: ${this.getContrastColor(preview.primary)};">${theme.name}</span>
            </div>
            <div class="theme-info">
                <div class="theme-name">${theme.name}</div>
                <div class="theme-category">${theme.category}</div>
            </div>
        `;
        
        card.addEventListener('click', () => {
            // Clear originalTheme to prevent mouseleave from reverting the selected theme
            originalTheme = null;
            clearTimeout(previewTimeout);
            this.selectTheme(theme.id);
        });
        
        // Add hover preview functionality
        let previewTimeout;
        let originalTheme = null;
        
        card.addEventListener('mouseenter', () => {
            previewTimeout = setTimeout(() => {
                if (!originalTheme) {
                    originalTheme = this.themeManager.currentTheme;
                }
                this.themeManager.applyTheme(theme.id, true); // true = preview mode
            }, 300); // 300ms delay for preview
        });
        
        card.addEventListener('mouseleave', () => {
            clearTimeout(previewTimeout);
            // Only revert if originalTheme exists AND the theme hasn't been selected (clicked)
            if (originalTheme) {
                this.themeManager.applyTheme(originalTheme);
                originalTheme = null;
            }
        });
        
        return card;
    }

    getContrastColor(hexColor) {
        // Simple contrast calculation
        const color = hexColor.replace('#', '');
        const r = parseInt(color.substr(0, 2), 16);
        const g = parseInt(color.substr(2, 2), 16);
        const b = parseInt(color.substr(4, 2), 16);
        const brightness = (r * 299 + g * 587 + b * 114) / 1000;
        return brightness > 128 ? '#000000' : '#ffffff';
    }

    updateActiveTheme(themeId) {
        // Remove active class from all cards
        const cards = document.querySelectorAll('.theme-card');
        cards.forEach(card => {
            card.classList.remove('active');
        });
        
        // Add active class to current theme
        const activeCard = document.querySelector(`[data-theme-id="${themeId}"]`);
        if (activeCard) {
            activeCard.classList.add('active');
        }
    }

    selectTheme(themeId) {
        this.themeManager.applyTheme(themeId);
        this.addToRecentThemes(themeId);
        this.close();
        this.showToast(`Theme changed to ${this.getThemeName(themeId)}`, 'success');
    }

    addToRecentThemes(themeId) {
        const recent = JSON.parse(localStorage.getItem('recentThemes') || '[]');
        // Remove if already exists
        const index = recent.indexOf(themeId);
        if (index > -1) {
            recent.splice(index, 1);
        }
        // Add to beginning
        recent.unshift(themeId);
        // Keep only last 5
        const limited = recent.slice(0, 5);
        localStorage.setItem('recentThemes', JSON.stringify(limited));
    }

    getRecentThemes() {
        const recent = JSON.parse(localStorage.getItem('recentThemes') || '[]');
        return recent.map(id => this.themeManager.themes.get(id)).filter(Boolean);
    }

    getThemeName(themeId) {
        const theme = this.themeManager.themes.get(themeId);
        return theme ? theme.name : themeId;
    }

    setCategory(category) {
        this.currentCategory = category;
        
        // Update category buttons
        const buttons = document.querySelectorAll('.theme-category');
        buttons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.category === category);
        });
        
        this.renderThemes();
    }

    setSearchQuery(query) {
        this.searchQuery = query;
        this.renderThemes();
    }

    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }

    open() {
        this.isOpen = true;
        const selector = document.getElementById('theme-selector');
        selector.classList.add('open');
        selector.style.display = 'block';
        
        // Focus search input
        setTimeout(() => {
            const searchInput = document.querySelector('.theme-search-input');
            if (searchInput) {
                searchInput.focus();
            }
        }, 100);
    }

    close() {
        this.isOpen = false;
        const selector = document.getElementById('theme-selector');
        selector.classList.remove('open');
        
        setTimeout(() => {
            selector.style.display = 'none';
        }, 300);
    }

    resetToDefault() {
        this.themeManager.resetToDefaults();
        this.close();
        this.showToast('Theme reset to default', 'info');
    }

    openCustomizer() {
        // Close theme selector first
        this.close();
        
        // Open theme editor with current theme
        if (window.themeEditor) {
            window.themeEditor.open(this.themeManager.currentTheme);
        } else {
            this.showToast('Theme editor not available', 'error');
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

// Initialize theme selector when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for theme manager to be available
    setTimeout(() => {
        if (window.themeManager) {
            window.themeSelector = new ThemeSelector(window.themeManager);
        }
    }, 100);
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThemeSelector;
}
