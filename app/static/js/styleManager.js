class StyleManager {
    constructor() {
        this.styles = new Map();
        this.categories = new Map();
        this.loaded = false;
        
        // Don't initialize with default styles - we'll load from CSV
        this.loaded = true;
        
        // Bind methods
        this.loadStylesFromCSV = this.loadStylesFromCSV.bind(this);
    }
    
    /**
     * Parse a single line of CSV, handling quoted values
     * @param {string} line - The CSV line to parse
     * @returns {string[]} - Array of column values
     */
    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current);
                current = '';
            } else {
                current += char;
            }
        }
        
        // Add the last field
        result.push(current);
        return result;
    }
    
    /**
     * Add a style to the manager
     * @param {Object} style - The style object to add
     */
    addStyle(style) {
        if (!style || !style.name || !style.prompt) {
            console.warn('Invalid style object:', style);
            return false;
        }
        
        // Add the style to the styles map
        this.styles.set(style.name, style);
        
        // Add the style to its category
        if (!this.categories.has(style.category)) {
            this.categories.set(style.category, []);
        }
        
        // Only add if not already in the category
        const categoryStyles = this.categories.get(style.category);
        if (!categoryStyles.some(s => s.name === style.name)) {
            categoryStyles.push(style);
        }
        
        return true;
    }
    
    /**
     * Load styles from a CSV file
     * @param {string} csvText - The CSV content as a string
     * @returns {boolean} - True if styles were loaded successfully, false otherwise
     */
    async loadStylesFromCSV(csvText) {
        if (!csvText || typeof csvText !== 'string') {
            console.error('Invalid CSV text provided');
            return false;
        }
        
        try {
            // Clear existing styles
            this.styles.clear();
            this.categories.clear();
            
            // Split the CSV into lines and clean them up
            const lines = csvText
                .split('\n')
                .map(line => line.trim())
                .filter(line => line.length > 0);
            
            // Process each line
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                
                try {
                    // Skip empty lines or lines that don't contain a comma (not valid CSV rows)
                    if (!line.includes(',')) continue;
                    
                    // Parse the CSV line (format: name,,,prompt,negative_prompt)
                    const columns = this.parseCSVLine(line);
                    if (columns.length < 5) continue; // Skip invalid lines
                    
                    const name = columns[0].trim();
                    const prompt = columns[3] ? columns[3].trim() : '';
                    const negative_prompt = columns[4] ? columns[4].trim() : '';
                    
                    // Skip if we don't have a name and prompt
                    if (!name || !prompt) continue;
                    
                    // Clean up the style name (capitalize first letter of each word)
                    const cleanStyleName = name
                        .replace(/-/g, ' ')
                        .split(' ')
                        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                        .join(' ');
                    
                    // Extract category from name (format: category-style)
                    let category = 'General';
                    const nameParts = name.split('-');
                    
                    // If the style name has a category prefix (e.g., 'photo-sunset')
                    if (nameParts.length > 1) {
                        const possibleCategory = nameParts[0].toLowerCase();
                        // Map common prefixes to categories
                        const categoryMap = {
                            'photo': 'Photography',
                            'art': 'Art',
                            'artstyle': 'Art Styles',
                            'artist': 'Famous Artists',
                            'digital': 'Digital Art',
                            'paint': 'Painting',
                            'draw': 'Drawing',
                            'sketch': 'Sketch',
                            'anime': 'Anime',
                            'manga': 'Manga',
                            'comic': 'Comic',
                            'cartoon': 'Cartoon',
                            '3d': '3D',
                            'game': 'Gaming',
                            'movie': 'Movie',
                            'director': 'Film Directors',
                            'tv': 'TV',
                            'tvshow': 'TV Shows',
                            'ad': 'Advertisement',
                            'ads': 'Advertisement',
                            'futuristic': 'Futuristic',
                            'cyber': 'Cyberpunk',
                            'steam': 'Steampunk',
                            'misc': 'Miscellaneous',
                            'sai': 'Stable Diffusion',
                            'cinematic': 'Cinematic',
                            'papercraft': 'Papercraft',
                            'superhero': 'Superheroes',
                            'music': 'Music Aesthetics',
                            'era': 'Historical Eras',
                            'cultural': 'Cultural Styles',
                            'weather': 'Weather & Atmosphere',
                            'material': 'Materials & Crafts'
                        };
                        
                        if (categoryMap[possibleCategory]) {
                            category = categoryMap[possibleCategory];
                        }
                    }
                    
                    // Create the style object
                    const style = {
                        name: cleanStyleName,
                        prompt: prompt.replace(/\{prompt\}/g, '{{prompt}}'),
                        negative_prompt: negative_prompt,
                        category: category,
                        description: '' // The CSV doesn't include descriptions
                    };
                    
                    // Add the style to the manager
                    this.addStyle(style);
                    
                } catch (error) {
                    console.warn(`Error parsing style from line ${i + 1}:`, error);
                    continue;
                }
            }
            
            console.log(`Successfully loaded ${this.styles.size} styles from CSV`);
            return true;
            
        } catch (error) {
            console.error('Error loading styles from CSV:', error);
            return false;
        }
    }
    
    /**
     * Get all styles, organized by category
     * @returns {Map<string, Array>} - Map of category names to arrays of style objects
     */
    getAllStyles() {
        return this.categories;
    }
    
    /**
     * Get all categories
     * @returns {string[]} - Array of category names
     */
    getCategories() {
        return Array.from(this.categories.keys()).sort();
    }
    
    /**
     * Get styles for a specific category
     * @param {string} category - The category name
     * @returns {Array} - Array of style objects in the category, or empty array if not found
     */
    getStylesByCategory(category) {
        return this.categories.get(category) || [];
    }
    
    /**
     * Get a style by name
     * @param {string} name - The name of the style to get
     * @returns {Object|undefined} - The style object, or undefined if not found
     */
    getStyle(name) {
        return this.styles.get(name);
    }
    
    /**
     * Apply a style to a prompt
     * @param {string} prompt - The original prompt
     * @param {string} styleName - The name of the style to apply
     * @returns {string} - The styled prompt
     */
    applyStyle(prompt, styleName) {
        const style = this.getStyle(styleName);
        if (!style) return prompt;
        
        // Replace {prompt} or {{prompt}} placeholder with the actual prompt
        return style.prompt.replace(/\{\{?prompt\}?\}/g, prompt);
    }
    
    /**
     * Get the negative prompt for a specific style
     * @param {string} styleName - The name of the style
     * @returns {string} - The negative prompt for the style, or an empty string if not found
     */
    getNegativePrompt(styleName) {
        const style = this.getStyle(styleName);
        return style ? style.negative_prompt || '' : '';
    }
}

// Create and export a singleton instance
const styleManager = new StyleManager();
export { styleManager };