const fs = require('fs');
const path = require('path');

const inputFile = path.join(__dirname, '../data/fooocus_styles.csv');
const outputFile = path.join(__dirname, '../app/static/data/fooocus_styles.json');

// Read the CSV file
const csvContent = fs.readFileSync(inputFile, 'utf-8');
const lines = csvContent.split('\n').filter(line => line.trim() !== '');

const styles = [];
let headers = [];
let inQuotes = false;
let currentLine = [];
let currentField = '';

// Simple CSV parser that handles quoted fields with commas
for (const line of lines) {
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            currentLine.push(currentField.trim());
            currentField = '';
        } else {
            currentField += char;
        }
    }
    
    // If we're still in quotes, continue to the next line
    if (inQuotes) {
        currentField += '\n';
        continue;
    }
    
    // Add the last field
    if (currentField) {
        currentLine.push(currentField.trim());
        currentField = '';
    }
    
    // Skip empty lines or header rows
    if (currentLine.length === 0 || currentLine[0].toLowerCase().includes('style name') || currentLine[0] === '') {
        currentLine = [];
        continue;
    }
    
    // If this is the first data row, it's the header
    if (headers.length === 0) {
        headers = currentLine.map(h => h.trim());
    } else {
        // Process a data row
        const style = {};
        for (let i = 0; i < Math.min(headers.length, currentLine.length); i++) {
            if (headers[i] && currentLine[i]) {
                style[headers[i].toLowerCase().replace(/\s+/g, '_')] = currentLine[i].trim();
            }
        }
        
        // Only add if we have a style name and prompt
        if (style.style_name && style.prompt) {
            styles.push({
                name: style.style_name,
                prompt: style.prompt,
                negative_prompt: style.negative_prompt || '',
                category: style.category || 'Uncategorized',
                description: style.description || ''
            });
        }
    }
    
    currentLine = [];
}

// Ensure output directory exists
const outputDir = path.dirname(outputFile);
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Save the styles to a JSON file
fs.writeFileSync(outputFile, JSON.stringify(styles, null, 2));
console.log(`Parsed ${styles.length} styles to ${outputFile}`);
