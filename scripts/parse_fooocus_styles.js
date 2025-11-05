const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');

const inputFile = path.join(__dirname, '../data/fooocus_styles.csv');
const outputFile = path.join(__dirname, '../app/static/data/fooocus_styles.json');

// Read the CSV file
const fileContent = fs.readFileSync(inputFile, 'utf-8');

// Parse the CSV content
const records = parse(fileContent, {
  columns: true,
  skip_empty_lines: true,
  trim: true,
  relax_quotes: true,
  skip_records_with_error: true
});

const styles = [];

// Process each record
records.forEach(record => {
  // Skip rows without a style name or prompt
  if (!record['Style Name'] || !record.Prompt) {
    return;
  }

  const style = {
    name: record['Style Name'].trim(),
    prompt: record.Prompt.trim(),
    negative_prompt: (record['Negative Prompt'] || '').trim(),
    category: (record.Category || 'Uncategorized').trim(),
    description: (record.Description || '').trim()
  };

  styles.push(style);
});

// Ensure output directory exists
const outputDir = path.dirname(outputFile);
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Save the styles to a JSON file
fs.writeFileSync(outputFile, JSON.stringify(styles, null, 2));
console.log(`Parsed ${styles.length} styles to ${outputFile}`);
