/**
 * Advanced Color Picker - Professional color selection with gradients
 * Provides comprehensive color editing capabilities
 */
class ColorPicker {
    constructor(options = {}) {
        this.options = {
            enableGradients: true,
            enableAlpha: true,
            presetColors: true,
            enableEyedropper: false,
            ...options
        };
        
        this.currentColor = '#4f46e5';
        this.currentGradient = null;
        this.history = [];
        this.isPicking = false;
        
        this.init();
    }

    init() {
        this.createColorPicker();
        this.setupPresetColors();
        this.bindEvents();
    }

    createColorPicker() {
        const picker = document.createElement('div');
        picker.className = 'color-picker';
        picker.innerHTML = `
            <div class="color-picker-header">
                <div class="color-display">
                    <div class="color-preview" id="color-preview"></div>
                    <input type="text" class="color-value" id="color-value" value="${this.currentColor}">
                </div>
                <div class="color-tabs">
                    <button class="color-tab active" data-mode="solid">Solid</button>
                    <button class="color-tab" data-mode="gradient">Gradient</button>
                </div>
            </div>
            
            <div class="color-picker-content">
                <!-- Solid Color Picker -->
                <div class="color-panel" id="solid-panel">
                    <div class="color-wheel-container">
                        <canvas class="color-wheel" id="color-wheel" width="200" height="200"></canvas>
                        <div class="color-slider">
                            <input type="range" class="hue-slider" id="hue-slider" min="0" max="360" value="0">
                        </div>
                    </div>
                    <div class="color-inputs">
                        <div class="color-input-group">
                            <label>HEX</label>
                            <input type="text" class="hex-input" id="hex-input" value="${this.currentColor}">
                        </div>
                        <div class="color-input-group">
                            <label>RGB</label>
                            <div class="rgb-inputs">
                                <input type="number" class="rgb-input" id="rgb-r" min="0" max="255" value="79">
                                <input type="number" class="rgb-input" id="rgb-g" min="0" max="255" value="70">
                                <input type="number" class="rgb-input" id="rgb-b" min="0" max="255" value="229">
                            </div>
                        </div>
                        <div class="color-input-group">
                            <label>HSL</label>
                            <div class="hsl-inputs">
                                <input type="number" class="hsl-input" id="hsl-h" min="0" max="360" value="239">
                                <input type="number" class="hsl-input" id="hsl-s" min="0" max="100" value="79">
                                <input type="number" class="hsl-input" id="hsl-l" min="0" max="100" value="59">
                            </div>
                        </div>
                        ${this.options.enableAlpha ? `
                        <div class="color-input-group">
                            <label>Alpha</label>
                            <div class="alpha-inputs">
                                <input type="range" class="alpha-slider" id="alpha-slider" min="0" max="100" value="100">
                                <input type="number" class="alpha-input" id="alpha-input" min="0" max="100" value="100">
                            </div>
                        </div>
                        ` : ''}
                    </div>
                </div>
                
                <!-- Gradient Picker -->
                <div class="color-panel" id="gradient-panel" style="display: none;">
                    <div class="gradient-preview" id="gradient-preview"></div>
                    <div class="gradient-type">
                        <label>Gradient Type</label>
                        <select class="gradient-type-select" id="gradient-type">
                            <option value="linear">Linear</option>
                            <option value="radial">Radial</option>
                            <option value="conic">Conic</option>
                        </select>
                    </div>
                    <div class="gradient-angle">
                        <label>Angle</label>
                        <input type="range" class="angle-slider" id="angle-slider" min="0" max="360" value="90">
                        <input type="number" class="angle-input" id="angle-input" min="0" max="360" value="90">
                    </div>
                    <div class="gradient-stops">
                        <label>Color Stops</label>
                        <div class="gradient-stops-list" id="gradient-stops">
                            <!-- Gradient stops will be added here -->
                        </div>
                        <button class="btn btn-sm btn-ghost" id="add-gradient-stop">Add Stop</button>
                    </div>
                </div>
            </div>
            
            ${this.options.presetColors ? `
            <div class="preset-colors">
                <label>Preset Colors</label>
                <div class="preset-colors-grid" id="preset-colors">
                    <!-- Preset colors will be added here -->
                </div>
            </div>
            ` : ''}
            
            <div class="color-picker-actions">
                <button class="btn btn-ghost" id="color-picker-cancel">Cancel</button>
                <button class="btn btn-primary" id="color-picker-apply">Apply</button>
            </div>
        `;

        this.element = picker;
        this.setupColorWheel();
        this.initializeGradient();
    }

    setupColorWheel() {
        const canvas = document.getElementById('color-wheel');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = Math.min(centerX, centerY) - 10;
        
        // Draw color wheel
        for (let angle = 0; angle < 360; angle += 1) {
            const startAngle = (angle - 1) * Math.PI / 180;
            const endAngle = angle * Math.PI / 180;
            
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.arc(centerX, centerY, radius, startAngle, endAngle);
            ctx.closePath();
            
            const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
            gradient.addColorStop(0, `hsl(${angle}, 0%, 100%)`);
            gradient.addColorStop(0.7, `hsl(${angle}, 100%, 50%)`);
            gradient.addColorStop(1, `hsl(${angle}, 100%, 40%)`);
            
            ctx.fillStyle = gradient;
            ctx.fill();
        }
        
        // Add center circle for lightness
        const lightnessGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius * 0.7);
        lightnessGradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
        lightnessGradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.5)');
        lightnessGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * 0.7, 0, 2 * Math.PI);
        ctx.fillStyle = lightnessGradient;
        ctx.fill();
    }

    setupPresetColors() {
        if (!this.options.presetColors) return;
        
        const presetColors = [
            '#000000', '#ffffff', '#ff0000', '#00ff00', '#0000ff',
            '#ffff00', '#ff00ff', '#00ffff', '#ff8800', '#8800ff',
            '#00ff88', '#ff0088', '#88ff00', '#0088ff', '#888888',
            '#444444', '#cccccc', '#ffcccc', '#ccffcc', '#ccccff',
            '#ffffcc', '#ffccff', '#ccffff', '#4f46e5', '#7c3aed',
            '#059669', '#ea580c', '#0e7490', '#374151', '#ec4899',
            '#dc2626'
        ];
        
        const container = document.getElementById('preset-colors');
        if (!container) return;
        
        presetColors.forEach(color => {
            const colorSwatch = document.createElement('div');
            colorSwatch.className = 'preset-color-swatch';
            colorSwatch.style.backgroundColor = color;
            colorSwatch.addEventListener('click', () => this.setColor(color));
            container.appendChild(colorSwatch);
        });
    }

    initializeGradient() {
        this.currentGradient = {
            type: 'linear',
            angle: 90,
            stops: [
                { color: '#4f46e5', position: 0 },
                { color: '#7c3aed', position: 100 }
            ]
        };
        
        this.updateGradientStops();
        this.updateGradientPreview();
    }

    bindEvents() {
        // Tab switching
        const tabs = document.querySelectorAll('.color-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchMode(e.target.dataset.mode);
            });
        });
        
        // Color wheel interaction
        const canvas = document.getElementById('color-wheel');
        if (canvas) {
            canvas.addEventListener('click', (e) => this.handleColorWheelClick(e));
            canvas.addEventListener('mousemove', (e) => this.handleColorWheelMove(e));
        }
        
        // Hue slider
        const hueSlider = document.getElementById('hue-slider');
        if (hueSlider) {
            hueSlider.addEventListener('input', (e) => this.updateFromHue(e.target.value));
        }
        
        // Color inputs
        const hexInput = document.getElementById('hex-input');
        if (hexInput) {
            hexInput.addEventListener('input', (e) => this.updateFromHex(e.target.value));
        }
        
        // RGB inputs
        ['rgb-r', 'rgb-g', 'rgb-b'].forEach(id => {
            const input = document.getElementById(id);
            if (input) {
                input.addEventListener('input', () => this.updateFromRGB());
            }
        });
        
        // HSL inputs
        ['hsl-h', 'hsl-s', 'hsl-l'].forEach(id => {
            const input = document.getElementById(id);
            if (input) {
                input.addEventListener('input', () => this.updateFromHSL());
            }
        });
        
        // Alpha inputs
        if (this.options.enableAlpha) {
            const alphaSlider = document.getElementById('alpha-slider');
            const alphaInput = document.getElementById('alpha-input');
            
            if (alphaSlider) {
                alphaSlider.addEventListener('input', (e) => {
                    alphaInput.value = e.target.value;
                    this.updateAlpha(e.target.value / 100);
                });
            }
            
            if (alphaInput) {
                alphaInput.addEventListener('input', (e) => {
                    alphaSlider.value = e.target.value;
                    this.updateAlpha(e.target.value / 100);
                });
            }
        }
        
        // Gradient controls
        const gradientType = document.getElementById('gradient-type');
        if (gradientType) {
            gradientType.addEventListener('change', (e) => {
                this.currentGradient.type = e.target.value;
                this.updateGradientPreview();
            });
        }
        
        const angleSlider = document.getElementById('angle-slider');
        const angleInput = document.getElementById('angle-input');
        
        if (angleSlider) {
            angleSlider.addEventListener('input', (e) => {
                angleInput.value = e.target.value;
                this.currentGradient.angle = parseInt(e.target.value);
                this.updateGradientPreview();
            });
        }
        
        if (angleInput) {
            angleInput.addEventListener('input', (e) => {
                angleSlider.value = e.target.value;
                this.currentGradient.angle = parseInt(e.target.value);
                this.updateGradientPreview();
            });
        }
        
        // Add gradient stop button
        const addStopBtn = document.getElementById('add-gradient-stop');
        if (addStopBtn) {
            addStopBtn.addEventListener('click', () => this.addGradientStop());
        }
        
        // Action buttons
        const cancelBtn = document.getElementById('color-picker-cancel');
        const applyBtn = document.getElementById('color-picker-apply');
        
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.cancel());
        }
        
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this.apply());
        }
    }

    switchMode(mode) {
        this.currentMode = mode;
        
        // Update tabs
        const tabs = document.querySelectorAll('.color-tab');
        tabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.mode === mode);
        });
        
        // Update panels
        const solidPanel = document.getElementById('solid-panel');
        const gradientPanel = document.getElementById('gradient-panel');
        
        if (solidPanel && gradientPanel) {
            if (mode === 'solid') {
                solidPanel.style.display = 'block';
                gradientPanel.style.display = 'none';
            } else {
                solidPanel.style.display = 'none';
                gradientPanel.style.display = 'block';
            }
        }
    }

    handleColorWheelClick(e) {
        const canvas = e.target;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        
        const angle = Math.atan2(y - centerY, x - centerX) * 180 / Math.PI + 90;
        const distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
        const maxDistance = Math.min(centerX, centerY) - 10;
        
        if (distance <= maxDistance) {
            const saturation = Math.min(100, (distance / maxDistance) * 100);
            const lightness = 50 + (1 - distance / maxDistance) * 30;
            
            this.setColorFromHSL(angle, saturation, lightness);
        }
    }

    handleColorWheelMove(e) {
        if (!this.isPicking) return;
        this.handleColorWheelClick(e);
    }

    updateFromHue(hue) {
        const hsl = this.hexToHSL(this.currentColor);
        this.setColorFromHSL(hue, hsl.s, hsl.l);
    }

    updateFromHex(hex) {
        if (this.isValidHex(hex)) {
            this.setColor(hex);
        }
    }

    updateFromRGB() {
        const r = parseInt(document.getElementById('rgb-r')?.value || 0);
        const g = parseInt(document.getElementById('rgb-g')?.value || 0);
        const b = parseInt(document.getElementById('rgb-b')?.value || 0);
        
        if (this.isValidRGB(r, g, b)) {
            const hex = this.rgbToHex(r, g, b);
            this.setColor(hex);
        }
    }

    updateFromHSL() {
        const h = parseInt(document.getElementById('hsl-h')?.value || 0);
        const s = parseInt(document.getElementById('hsl-s')?.value || 0);
        const l = parseInt(document.getElementById('hsl-l')?.value || 0);
        
        if (this.isValidHSL(h, s, l)) {
            this.setColorFromHSL(h, s, l);
        }
    }

    updateAlpha(alpha) {
        this.currentAlpha = alpha;
        this.updateColorDisplay();
    }

    setColor(color) {
        this.currentColor = color;
        this.updateColorDisplay();
        this.updateColorInputs();
    }

    setColorFromHSL(h, s, l) {
        this.currentColor = this.hslToHex(h, s, l);
        this.updateColorDisplay();
        this.updateColorInputs();
    }

    updateColorDisplay() {
        const preview = document.getElementById('color-preview');
        const valueDisplay = document.getElementById('color-value');
        
        if (preview) {
            preview.style.backgroundColor = this.currentColor;
        }
        
        if (valueDisplay) {
            valueDisplay.value = this.currentColor;
        }
    }

    updateColorInputs() {
        const hex = this.currentColor;
        const rgb = this.hexToRGB(hex);
        const hsl = this.hexToHSL(hex);
        
        // Update HEX input
        const hexInput = document.getElementById('hex-input');
        if (hexInput) hexInput.value = hex;
        
        // Update RGB inputs
        document.getElementById('rgb-r').value = rgb.r;
        document.getElementById('rgb-g').value = rgb.g;
        document.getElementById('rgb-b').value = rgb.b;
        
        // Update HSL inputs
        document.getElementById('hsl-h').value = Math.round(hsl.h);
        document.getElementById('hsl-s').value = Math.round(hsl.s);
        document.getElementById('hsl-l').value = Math.round(hsl.l);
        
        // Update hue slider
        const hueSlider = document.getElementById('hue-slider');
        if (hueSlider) hueSlider.value = Math.round(hsl.h);
    }

    updateGradientStops() {
        const container = document.getElementById('gradient-stops');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.currentGradient.stops.forEach((stop, index) => {
            const stopElement = document.createElement('div');
            stopElement.className = 'gradient-stop';
            stopElement.innerHTML = `
                <input type="color" class="stop-color" value="${stop.color}">
                <input type="number" class="stop-position" value="${stop.position}" min="0" max="100">
                <button class="btn btn-sm btn-ghost remove-stop" data-index="${index}">×</button>
            `;
            
            container.appendChild(stopElement);
        });
        
        // Bind events for gradient stops
        container.querySelectorAll('.stop-color').forEach(input => {
            input.addEventListener('input', (e) => {
                const index = parseInt(e.target.closest('.gradient-stop').querySelector('.remove-stop').dataset.index);
                this.currentGradient.stops[index].color = e.target.value;
                this.updateGradientPreview();
            });
        });
        
        container.querySelectorAll('.stop-position').forEach(input => {
            input.addEventListener('input', (e) => {
                const index = parseInt(e.target.closest('.gradient-stop').querySelector('.remove-stop').dataset.index);
                this.currentGradient.stops[index].position = parseInt(e.target.value);
                this.updateGradientPreview();
            });
        });
        
        container.querySelectorAll('.remove-stop').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const index = parseInt(e.target.dataset.index);
                this.removeGradientStop(index);
            });
        });
    }

    updateGradientPreview() {
        const preview = document.getElementById('gradient-preview');
        if (!preview) return;
        
        const gradient = this.buildGradientString();
        preview.style.background = gradient;
    }

    buildGradientString() {
        const { type, angle, stops } = this.currentGradient;
        
        const sortedStops = stops.sort((a, b) => a.position - b.position);
        const stopString = sortedStops.map(stop => `${stop.color} ${stop.position}%`).join(', ');
        
        switch (type) {
            case 'linear':
                return `linear-gradient(${angle}deg, ${stopString})`;
            case 'radial':
                return `radial-gradient(circle, ${stopString})`;
            case 'conic':
                return `conic-gradient(from ${angle}deg, ${stopString})`;
            default:
                return `linear-gradient(${angle}deg, ${stopString})`;
        }
    }

    addGradientStop() {
        const newPosition = this.currentGradient.stops.length > 0 
            ? Math.min(100, this.currentGradient.stops[this.currentGradient.stops.length - 1].position + 10)
            : 50;
        
        this.currentGradient.stops.push({
            color: '#ffffff',
            position: newPosition
        });
        
        this.updateGradientStops();
        this.updateGradientPreview();
    }

    removeGradientStop(index) {
        if (this.currentGradient.stops.length > 2) {
            this.currentGradient.stops.splice(index, 1);
            this.updateGradientStops();
            this.updateGradientPreview();
        }
    }

    // Utility methods
    hexToRGB(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    }

    rgbToHex(r, g, b) {
        return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
    }

    hexToHSL(hex) {
        const rgb = this.hexToRGB(hex);
        return this.rgbToHSL(rgb.r, rgb.g, rgb.b);
    }

    rgbToHSL(r, g, b) {
        r /= 255;
        g /= 255;
        b /= 255;
        
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;
        
        if (max === min) {
            h = s = 0;
        } else {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            
            switch (max) {
                case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
                case g: h = ((b - r) / d + 2) / 6; break;
                case b: h = ((r - g) / d + 4) / 6; break;
            }
        }
        
        return {
            h: h * 360,
            s: s * 100,
            l: l * 100
        };
    }

    hslToHex(h, s, l) {
        h /= 360;
        s /= 100;
        l /= 100;
        
        let r, g, b;
        
        if (s === 0) {
            r = g = b = l;
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1/6) return p + (q - p) * 6 * t;
                if (t < 1/2) return q;
                if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                return p;
            };
            
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
        }
        
        return this.rgbToHex(Math.round(r * 255), Math.round(g * 255), Math.round(b * 255));
    }

    isValidHex(hex) {
        return /^#?[0-9A-F]{6}$/i.test(hex);
    }

    isValidRGB(r, g, b) {
        return r >= 0 && r <= 255 && g >= 0 && g <= 255 && b >= 0 && b <= 255;
    }

    isValidHSL(h, s, l) {
        return h >= 0 && h <= 360 && s >= 0 && s <= 100 && l >= 0 && l <= 100;
    }

    normalizeColor(color) {
        if (color.startsWith('#')) {
            return color;
        }
        return '#' + color;
    }

    // Public API
    show(color = null) {
        if (color) {
            this.setColor(color);
        }
        
        document.body.appendChild(this.element);
        this.element.style.display = 'block';
        
        // Position the picker
        this.positionPicker();
    }

    hide() {
        this.element.style.display = 'none';
        if (this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
    }

    positionPicker() {
        // Position picker relative to trigger element
        // This would be implemented based on the trigger element
    }

    getColor() {
        return this.currentMode === 'gradient' 
            ? this.buildGradientString()
            : this.currentColor;
    }

    cancel() {
        this.hide();
        this.onCancel?.();
    }

    apply() {
        this.hide();
        this.onApply?.(this.getColor());
    }

    on(callback) {
        this.onApply = callback;
        return this;
    }

    onCancel(callback) {
        this.onCancel = callback;
        return this;
    }
}

// Initialize color picker when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.ColorPicker = ColorPicker;
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ColorPicker;
}
