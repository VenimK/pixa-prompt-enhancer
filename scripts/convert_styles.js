const fs = require('fs');
const csv = require('csv-parser');
const path = require('path');

const inputFile = path.join(__dirname, '../data/fooocus_styles.csv');
const outputFile = path.join(__dirname, '../app/static/data/fooocus_styles.json');

const styles = [];

// Ensure output directory exists
const outputDir = path.dirname(outputFile);
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

fs.createReadStream(inputFile)
    .pipe(csv({
        mapHeaders: ({ header }) => header.trim(),
        mapValues: ({ value }) => value.trim()
    }))
    .on('data', (row) => {
        if (row['Style Name'] && row.Prompt) {
            const style = {
                name: row['Style Name'],
                prompt: row.Prompt,
                negative_prompt: row['Negative Prompt'] || '',
                category: row.Category || 'Uncategorized',
                description: row.Description || ''
            };
            styles.push(style);
        }
    })
    .on('end', () => {
        fs.writeFileSync(outputFile, JSON.stringify(styles, null, 2));
        console.log(`Converted ${styles.length} styles to ${outputFile}`);
    });
