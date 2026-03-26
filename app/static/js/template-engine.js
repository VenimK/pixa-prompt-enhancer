/**
 * Template Engine - Prompt template management system
 * Handles template loading, rendering, and management
 */
class TemplateEngine {
    constructor() {
        this.templates = new Map();
        this.categories = new Map();
        this.userTemplates = new Map();
        this.currentTemplate = null;
        
        this.init();
    }

    async init() {
        await this.loadTemplates();
        this.setupEventListeners();
    }

    async loadTemplates() {
        try {
            const response = await fetch('/static/data/templates.json');
            const data = await response.json();
            
            // Load categories
            this.categories = new Map(Object.entries(data.categories));
            
            // Load templates
            Object.entries(data.templates).forEach(([category, templates]) => {
                Object.entries(templates).forEach(([id, template]) => {
                    this.templates.set(id, {
                        ...template,
                        id,
                        category
                    });
                });
            });
            
            // Load user templates
            this.loadUserTemplates();
            
        } catch (error) {
            console.error('Failed to load templates:', error);
        }
    }

    loadUserTemplates() {
        try {
            const saved = localStorage.getItem('userTemplates');
            if (saved) {
                const userTemplates = JSON.parse(saved);
                Object.entries(userTemplates).forEach(([id, template]) => {
                    this.userTemplates.set(id, template);
                });
            }
        } catch (error) {
            console.warn('Failed to load user templates:', error);
        }
    }

    saveUserTemplates() {
        try {
            const userTemplates = Object.fromEntries(this.userTemplates);
            localStorage.setItem('userTemplates', JSON.stringify(userTemplates));
        } catch (error) {
            console.warn('Failed to save user templates:', error);
        }
    }

    setupEventListeners() {
        // Listen for template requests
        document.addEventListener('templateRequest', (e) => {
            this.handleTemplateRequest(e.detail);
        });

        // Listen for template application
        document.addEventListener('applyTemplate', (e) => {
            this.applyTemplate(e.detail.templateId, e.detail.variables);
        });
    }

    handleTemplateRequest(request) {
        const { action, data } = request;
        
        switch (action) {
            case 'getTemplates':
                this.getTemplates(data.category, data.search);
                break;
            case 'getTemplate':
                this.getTemplate(data.templateId);
                break;
            case 'saveTemplate':
                this.saveTemplate(data.template);
                break;
            case 'deleteTemplate':
                this.deleteTemplate(data.templateId);
                break;
        }
    }

    getTemplates(category = 'all', search = '') {
        let templates = Array.from(this.templates.values());
        
        // Add user templates
        templates = templates.concat(Array.from(this.userTemplates.values()));
        
        // Filter by category
        if (category !== 'all') {
            templates = templates.filter(template => template.category === category);
        }
        
        // Filter by search
        if (search) {
            const query = search.toLowerCase();
            templates = templates.filter(template => 
                template.name.toLowerCase().includes(query) ||
                template.description.toLowerCase().includes(query) ||
                template.tags.some(tag => tag.toLowerCase().includes(query))
            );
        }
        
        // Sort by popularity and name
        templates.sort((a, b) => {
            if (a.popularity && b.popularity) {
                return b.popularity - a.popularity;
            }
            return a.name.localeCompare(b.name);
        });
        
        // Dispatch result
        document.dispatchEvent(new CustomEvent('templatesLoaded', {
            detail: { templates, category, search }
        }));
        
        return templates;
    }

    getTemplate(templateId) {
        const template = this.templates.get(templateId) || this.userTemplates.get(templateId);
        
        if (template) {
            document.dispatchEvent(new CustomEvent('templateLoaded', {
                detail: { template }
            }));
        }
        
        return template;
    }

    renderTemplate(template, variables = {}) {
        if (!template || !template.prompt) {
            return '';
        }
        
        let rendered = template.prompt;
        
        // Replace variables in the prompt
        Object.entries(variables).forEach(([key, value]) => {
            const placeholder = `{${key}}`;
            rendered = rendered.replace(new RegExp(placeholder, 'g'), value || '');
        });
        
        return rendered;
    }

    applyTemplate(templateId, variables = {}) {
        const template = this.getTemplate(templateId);
        
        if (!template) {
            console.error(`Template '${templateId}' not found`);
            return null;
        }
        
        const rendered = this.renderTemplate(template, variables);
        this.currentTemplate = template;
        
        // Apply to prompt textarea
        const promptTextarea = document.getElementById('prompt');
        if (promptTextarea) {
            promptTextarea.value = rendered;
            
            // Trigger change event
            promptTextarea.dispatchEvent(new Event('input', { bubbles: true }));
            promptTextarea.dispatchEvent(new Event('change', { bubbles: true }));
        }
        
        // Dispatch template applied event
        document.dispatchEvent(new CustomEvent('templateApplied', {
            detail: { template, variables, rendered }
        }));
        
        return rendered;
    }

    saveTemplate(template) {
        if (!template.id) {
            template.id = `user_${Date.now()}`;
        }
        
        template.category = template.category || 'user';
        template.createdAt = template.createdAt || new Date().toISOString();
        template.isCustom = true;
        
        this.userTemplates.set(template.id, template);
        this.saveUserTemplates();
        
        // Dispatch template saved event
        document.dispatchEvent(new CustomEvent('templateSaved', {
            detail: { template }
        }));
        
        return template;
    }

    deleteTemplate(templateId) {
        const deleted = this.userTemplates.delete(templateId);
        
        if (deleted) {
            this.saveUserTemplates();
            
            // Dispatch template deleted event
            document.dispatchEvent(new CustomEvent('templateDeleted', {
                detail: { templateId }
            }));
        }
        
        return deleted;
    }

    getCategories() {
        return Array.from(this.categories.entries()).map(([id, category]) => ({
            id,
            ...category
        }));
    }

    getTemplateVariables(templateId) {
        const template = this.getTemplate(templateId);
        return template ? template.variables || [] : [];
    }

    validateTemplate(template) {
        const errors = [];
        
        if (!template.name || template.name.trim() === '') {
            errors.push('Template name is required');
        }
        
        if (!template.prompt || template.prompt.trim() === '') {
            errors.push('Template prompt is required');
        }
        
        if (!template.category) {
            errors.push('Template category is required');
        }
        
        // Check for required variables
        if (template.variables) {
            template.variables.forEach(variable => {
                if (!variable.name || variable.name.trim() === '') {
                    errors.push(`Variable name is required: ${variable.name || 'unnamed'}`);
                }
                
                if (!variable.label || variable.label.trim() === '') {
                    errors.push(`Variable label is required: ${variable.name}`);
                }
                
                if (!variable.type) {
                    errors.push(`Variable type is required: ${variable.name}`);
                }
            });
        }
        
        return {
            isValid: errors.length === 0,
            errors
        };
    }

    exportTemplate(templateId) {
        const template = this.getTemplate(templateId);
        
        if (!template) {
            return null;
        }
        
        return {
            ...template,
            exportedAt: new Date().toISOString(),
            version: '1.0'
        };
    }

    importTemplate(templateData) {
        const validation = this.validateTemplate(templateData);
        
        if (!validation.isValid) {
            throw new Error(`Invalid template: ${validation.errors.join(', ')}`);
        }
        
        // Generate new ID if conflicts
        let templateId = templateData.id;
        if (this.templates.has(templateId) || this.userTemplates.has(templateId)) {
            templateId = `imported_${Date.now()}`;
        }
        
        const template = {
            ...templateData,
            id: templateId,
            isCustom: true,
            importedAt: new Date().toISOString()
        };
        
        return this.saveTemplate(template);
    }

    createTemplateFromPrompt(prompt, name = '') {
        // Extract variables from prompt
        const variablePattern = /\{([^}]+)\}/g;
        const variables = [];
        const matches = prompt.match(variablePattern);
        
        if (matches) {
            const uniqueVars = [...new Set(matches)];
            uniqueVars.forEach(match => {
                const varName = match.slice(1, -1); // Remove { and }
                variables.push({
                    name: varName,
                    label: this.formatVariableLabel(varName),
                    type: 'text',
                    placeholder: `Enter ${varName}`,
                    required: true
                });
            });
        }
        
        return {
            name: name || 'Custom Template',
            category: 'user',
            description: 'Custom template created from current prompt',
            tags: ['custom', 'user'],
            prompt: prompt,
            variables: variables,
            isCustom: true
        };
    }

    formatVariableLabel(varName) {
        // Convert variable name to readable label
        return varName
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }

    getRecentTemplates(limit = 5) {
        // Get recently used templates (this would need tracking)
        return Array.from(this.templates.values())
            .filter(template => template.popularity && template.popularity > 80)
            .sort((a, b) => (b.popularity || 0) - (a.popularity || 0))
            .slice(0, limit);
    }

    searchTemplates(query) {
        const allTemplates = Array.from(this.templates.values())
            .concat(Array.from(this.userTemplates.values()));
        
        const searchQuery = query.toLowerCase();
        
        return allTemplates.filter(template => 
            template.name.toLowerCase().includes(searchQuery) ||
            template.description.toLowerCase().includes(searchQuery) ||
            template.tags.some(tag => tag.toLowerCase().includes(searchQuery)) ||
            template.prompt.toLowerCase().includes(searchQuery)
        );
    }
}

// Initialize template engine when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.templateEngine = new TemplateEngine();
    
    // Make template engine globally available
    window.TemplateEngine = TemplateEngine;
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TemplateEngine;
}
