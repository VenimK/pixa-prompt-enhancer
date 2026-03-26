# Theme Features and Templates System

This document explains the comprehensive theme features and prompt templates system added to the PiXa Prompt Enhancer application.

## 🎨 Advanced Theme System

### Available Themes

The application now includes 8 professionally designed themes:

#### Professional Themes
- **Professional Blue** - Classic blue theme for business environments
- **Monochrome** - Clean black and white minimal theme

#### Creative Themes  
- **Creative Purple** - Vibrant purple for creative professionals
- **Cyberpunk** - Neon-lit futuristic theme with pink/cyan accents
- **Retro** - 80s retro theme with vintage aesthetics

#### Nature-Inspired Themes
- **Nature Green** - Calming green inspired by nature
- **Sunset Orange** - Warm orange reminiscent of sunset colors
- **Ocean Blue** - Deep ocean blue for refreshing experience

### Advanced Theme Editor

#### 🛠️ Full CSS Customization
The advanced theme editor provides comprehensive CSS editing capabilities:

**Color System:**
- **Advanced Color Picker** - Visual color wheel with HSL/RGB/HEX support
- **Gradient Editor** - Linear, radial, and conic gradients
- **Color Presets** - Quick access to common colors
- **Alpha Channel** - Transparency support for all colors

**Typography Controls:**
- **Font Family Selection** - Multiple font options with preview
- **Font Size Controls** - Precise sizing with range sliders
- **Font Weight Options** - From thin (100) to black (900)
- **Line Height & Spacing** - Typography fine-tuning

**Layout & Spacing:**
- **Spacing System** - Consistent spacing values
- **Border Radius** - Rounded corner controls
- **Box Shadows** - Advanced shadow editing
- **Padding & Margins** - Layout spacing controls

**Effects & Animations:**
- **CSS Transitions** - Smooth transition effects
- **Transform Controls** - Scale, rotate, translate
- **Filter Effects** - Blur, grayscale, sepia
- **Custom Animations** - Keyframe animation support

#### 🎯 Component-Specific Styling
Edit individual UI components:
- **Buttons** - Primary, secondary, ghost variants
- **Cards** - Content containers with shadows
- **Forms** - Input fields, labels, groups
- **Navigation** - Navbar and menu styling
- **Modals** - Dialog and overlay styling

#### 📱 Live Preview System
**Real-time Preview:**
- **Instant Updates** - See changes as you edit
- **Component Preview** - Test on specific UI elements
- **Responsive Sizes** - Desktop, tablet, mobile previews
- **Dark/Light Mode** - Preview both theme variants

#### 💾 Theme Management
**Save & Export:**
- **Custom Themes** - Save with unique names and descriptions
- **Export Formats** - JSON, CSS, SCSS export options
- **Theme Backup** - Backup all custom themes
- **Version Control** - Track theme changes

**Import & Clone:**
- **File Import** - Import themes from JSON/CSS files
- **Theme Cloning** - Use existing themes as starting points
- **CSS Import** - Convert CSS to theme format
- **Batch Import** - Import multiple themes

#### 🔧 Advanced Features
**Code Editor:**
- **CSS Syntax Highlighting** - Professional code editing
- **Auto-completion** - CSS property suggestions
- **Error Validation** - Real-time syntax checking
- **Code Formatting** - Auto-format CSS code

**Theme Analytics:**
- **Usage Statistics** - Track theme popularity
- **Performance Metrics** - CSS optimization tips
- **Accessibility Checks** - Contrast and compliance validation

### Theme Features

- **Live Theme Switching** - Instant theme changes without page reload
- **Dark Mode Support** - All themes support both light and dark modes
- **Persistent Settings** - Theme preferences saved automatically
- **Responsive Design** - Themes work perfectly on all screen sizes
- **Accessibility** - High contrast and reduced motion support
- **Professional Interface** - Intuitive editing experience

### How to Use Themes

1. **Access Theme Selector**: Click the theme toggle button (☀️) in the bottom-right corner
2. **Browse Themes**: View all available themes with live previews
3. **Filter by Category**: Use category buttons to filter themes
4. **Search Themes**: Use the search bar to find specific themes
5. **Apply Theme**: Click any theme card to apply it instantly
6. **Customize Theme**: Click "Customize Theme" for advanced editing
7. **Save Custom Themes**: Create and save your own themes

### How to Use Theme Editor

1. **Open Theme Editor**: Click "Customize Theme" from theme selector
2. **Choose Base Theme**: Select an existing theme as starting point
3. **Edit Properties**: 
   - Use visual editors for colors and effects
   - Switch to code editor for advanced CSS
   - Target specific components
4. **Live Preview**: See changes in real-time
5. **Test Responsiveness**: Preview on different screen sizes
6. **Save Theme**: Save with custom name and description

### Theme Categories

- **Professional** - Business-oriented themes
- **Creative** - Artistic and vibrant themes  
- **Nature** - Calming, nature-inspired themes
- **Warm** - Cozy, warm-toned themes
- **Cool** - Refreshing, cool-toned themes
- **Minimal** - Clean, minimalist themes
- **Futuristic** - Modern, sci-fi inspired themes
- **Vintage** - Retro, nostalgic themes
- **Custom** - User-created themes

## 📋 Prompt Templates

### Template Categories

#### Photography Templates
- **Portrait Photography** - Professional portrait setups with lighting and composition options
- **Landscape Photography** - Stunning landscape photography with environmental settings
- **Product Photography** - Clean product photography for marketing and e-commerce

#### Art Style Templates
- **Digital Art** - Digital artwork with various styles and techniques
- **Character Design** - Character design for games, animation, and illustration

#### Professional Templates
- **Marketing Materials** - Professional marketing and advertising visuals
- **Social Media Graphics** - Eye-catching social media graphics for various platforms

#### Technical Templates
- **3D Rendering** - Professional 3D renders for products, architecture, and visualization

### Template Features

- **Variable Substitution** - Templates use `{variable}` syntax for dynamic content
- **Smart Defaults** - Sensible default values for all variables
- **Category Organization** - Templates organized by use case
- **Search Functionality** - Find templates by name, description, or tags
- **Custom Templates** - Create and save your own templates

### How to Use Templates

1. **Open Template Browser**: Access from the main interface (coming soon)
2. **Browse Categories**: Filter templates by category
3. **Search Templates**: Find specific templates using the search bar
4. **Select Template**: Choose a template that matches your needs
5. **Fill Variables**: Enter values for template variables
6. **Apply Template**: One-click application to your prompt

### Template Variables

Templates use variables in the format `{variable_name}`. For example:

```
Professional portrait photography of {subject}, {lighting} lighting, {composition} composition
```

Variables can include:
- **Text Input** - Free-form text entry
- **Select Options** - Predefined choices
- **Required Fields** - Must be filled out
- **Optional Fields** - Can be left empty

## 🛠️ Technical Implementation

### File Structure

```
app/static/
├── css/
│   ├── theme-manager.css          # Theme system styles
│   ├── theme-editor.css          # Advanced theme editor styles
│   ├── color-picker.css          # Color picker component styles
│   └── style-manager.css         # Existing style manager
├── js/
│   ├── theme-manager.js           # Core theme management
│   ├── theme-selector.js          # Theme selector UI
│   ├── theme-editor.js            # Advanced theme editor
│   ├── color-picker.js            # Advanced color picker
│   ├── theme-exporter.js          # Theme export/import
│   └── template-engine.js         # Template system
└── data/
    ├── themes.json                 # Theme configurations
    └── templates.json              # Template definitions
```

### Theme Editor API

```javascript
// Access theme editor
window.themeEditor

// Open theme editor with base theme
window.themeEditor.open('professional-blue');

// Get current theme properties
const properties = window.themeEditor.customProperties;

// Update property
window.themeEditor.updateProperty('--primary-color', '#ff6b6b');

// Save custom theme
window.themeEditor.saveTheme();
```

### Color Picker API

```javascript
// Create color picker
const picker = new ColorPicker({
    enableGradients: true,
    enableAlpha: true,
    presetColors: true
});

// Show color picker
picker.show('#4f46e5');

// Get selected color
const color = picker.getColor();

// Handle color selection
picker.on((color) => {
    console.log('Selected color:', color);
});
```

### Theme Exporter API

```javascript
// Export theme as JSON
const exported = window.themeExporter.exportTheme('custom-theme-123', 'json');

// Download theme
window.themeExporter.downloadTheme('custom-theme-123', 'css');

// Import theme from file
const file = await window.themeExporter.createFileInput();
const theme = await window.themeExporter.uploadTheme(file);

// Create backup
const backup = window.themeExporter.createThemeBackup();
```

## 🎯 Usage Examples

### Example 1: Creating a Custom Theme

```javascript
// Open theme editor with professional blue as base
window.themeEditor.open('professional-blue');

// Edit colors
window.themeEditor.updateProperty('--primary-color', '#ff6b6b');
window.themeEditor.updateProperty('--background-color', '#f8f9fa');

// Edit typography
window.themeEditor.updateProperty('--font-size-base', '1.1rem');
window.themeEditor.updateProperty('--font-weight-medium', '600');

// Add custom CSS
window.themeEditor.updateCustomCSS(`
.btn-primary {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    border: none;
    border-radius: 8px;
}
`);

// Save theme
window.themeEditor.saveTheme('My Custom Theme');
```

### Example 2: Professional Portrait

Template: `portrait photography`
Variables:
- subject: "CEO of tech company"
- lighting: "soft natural"
- composition: "medium shot"
- mood: "professional"
- background: "office environment"

Result:
```
Professional portrait photography of CEO of tech company, soft natural lighting, medium shot composition, professional mood, office environment background, shot on Canon EOS R5 with 85mm f/1.4 lens, corporate headshot style
```

### Example 3: Creative Digital Art

Template: `digital art`
Variables:
- subject: "futuristic cityscape"
- style: "concept art"
- medium: "digital painting"
- color_scheme: "vibrant"
- mood: "energetic"

Result:
```
Digital art of futuristic cityscape in concept art style, digital painting medium, vibrant color scheme, rule of thirds composition, energetic mood, detailed detail level, Studio Ghibli influence
```

## 🔧 Customization Options

### Theme Customization Levels

**Basic Customization:**
- Change primary, secondary, and accent colors
- Adjust font sizes and weights
- Modify spacing and border radius

**Advanced Customization:**
- Full CSS property editing
- Component-specific styling
- Custom animations and transitions
- Gradient and effect editing

**Professional Customization:**
- Direct CSS code editing
- Theme export/import
- Batch theme management
- Performance optimization

### Supported CSS Properties

**Colors:**
- All color properties with alpha channel
- Gradient editing (linear, radial, conic)
- Color scheme generation

**Typography:**
- Font families, sizes, weights
- Line height, letter spacing
- Text decoration and transforms

**Layout:**
- Padding, margins, borders
- Width, height, positioning
- Flexbox and grid properties

**Effects:**
- Box shadows and text shadows
- Transforms and transitions
- Filters and backdrop filters
- Custom animations

## 📱 Mobile Support

The theme system is fully responsive and works seamlessly on:
- Desktop browsers
- Tablet devices  
- Mobile phones
- Touch interfaces

**Mobile Theme Editor:**
- Optimized interface for small screens
- Touch-friendly controls
- Simplified property panels
- Swipe gesture support

## ♿ Accessibility Features

- **High Contrast Mode** - Enhanced contrast for better visibility
- **Reduced Motion** - Respects user's motion preferences
- **Keyboard Navigation** - Full keyboard support for theme editing
- **Screen Reader Support** - Proper ARIA labels and semantic markup
- **Color Validation** - Ensures accessible color combinations

## 🔒 Privacy & Storage

- **Local Storage** - Theme preferences stored locally
- **No Tracking** - No analytics or tracking for theme usage
- **User Data** - Custom themes stored locally only
- **Export/Import** - Users control theme data sharing

## 🚀 Performance

- **Optimized CSS** - Efficient property management
- **Lazy Loading** - Components loaded on demand
- **Caching** - Theme data cached for fast switching
- **Minimal Overhead** - Lightweight implementation

## 🐛 Troubleshooting

### Common Issues

**Theme not applying correctly**
- Check browser console for errors
- Ensure all CSS files are loading
- Try refreshing the page
- Verify theme data integrity

**Theme editor not opening**
- Check if JavaScript is enabled
- Verify theme-editor.js is loaded
- Check for conflicting scripts
- Try clearing browser cache

**Custom theme not saving**
- Check browser local storage permissions
- Ensure theme name is valid
- Verify theme data structure
- Check for duplicate theme IDs

**Color picker not working**
- Ensure color-picker.js is loaded
- Check for CSS conflicts
- Verify canvas support
- Test in different browser

**Export/Import failing**
- Check file format support
- Verify file permissions
- Ensure valid theme data
- Check browser download settings

### Getting Help

For issues or feature requests:
1. Check browser console for error messages
2. Verify all files are properly loaded
3. Test in a different browser
4. Check theme data integrity
5. Contact support with details of the issue

## 🎨 Best Practices

### Theme Design Guidelines

**Color Selection:**
- Use accessible color combinations
- Maintain sufficient contrast ratios
- Consider color blindness
- Test in both light and dark modes

**Typography:**
- Choose readable font combinations
- Maintain consistent sizing
- Ensure proper line height
- Test across devices

**Layout:**
- Use consistent spacing
- Maintain visual hierarchy
- Ensure responsive design
- Test component interactions

**Performance:**
- Minimize CSS complexity
- Use efficient selectors
- Optimize animations
- Test loading performance

---

*This comprehensive theme system provides professional-grade customization capabilities while maintaining excellent performance and accessibility standards. Users can create, customize, and share themes with full CSS control and professional editing tools.*
