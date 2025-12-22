#!/usr/bin/env node

/**
 * Script to translate all English documentation files to Urdu
 * Uses MyMemory Translation API
 */

const fs = require('fs');
const path = require('path');

// Configuration
const DOCS_DIR = path.join(__dirname, '../docs');
const URDU_I18N_DIR = path.join(__dirname, '../i18n/ur/docusaurus-plugin-content-docs/current');
const TRANSLATION_API = 'https://api.mymemory.translated.net/get';
const BATCH_DELAY = 300; // ms delay between API calls to avoid rate limiting

// Ensure Urdu directory exists
if (!fs.existsSync(URDU_I18N_DIR)) {
  fs.mkdirSync(URDU_I18N_DIR, { recursive: true });
  console.log('✓ Created Urdu i18n directory');
}

/**
 * Translate text using MyMemory API
 */
async function translateText(text) {
  try {
    if (!text || text.trim().length === 0) {
      return text;
    }

    // Skip very long texts - split them
    if (text.length > 500) {
      const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
      const translated = await Promise.all(
        sentences.map(s => translateText(s.trim()))
      );
      return translated.join(' ');
    }

    const url = `${TRANSLATION_API}?q=${encodeURIComponent(text)}&langpair=en|ur`;
    const response = await fetch(url);
    const data = await response.json();

    if (data && data.responseData && data.responseData.translatedText) {
      return data.responseData.translatedText;
    }

    return text;
  } catch (error) {
    console.error('Translation error:', error.message);
    return text;
  }
}

/**
 * Translate markdown frontmatter
 */
async function translateFrontmatter(lines) {
  const result = [];
  let inFrontmatter = false;
  let fmStart = 0;

  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === '---') {
      if (!inFrontmatter) {
        inFrontmatter = true;
        fmStart = i;
        result.push(lines[i]);
      } else {
        // End of frontmatter
        inFrontmatter = false;
        result.push(lines[i]);
        return { result, contentStart: i + 1 };
      }
    } else if (inFrontmatter) {
      result.push(lines[i]);
    } else {
      return { result, contentStart: i };
    }
  }

  return { result, contentStart: lines.length };
}

/**
 * Translate markdown content while preserving code blocks
 */
async function translateMarkdown(content) {
  const lines = content.split('\n');
  const { result: fmLines, contentStart } = await translateFrontmatter(lines);

  const contentLines = lines.slice(contentStart);
  const translatedContent = [];
  let inCodeBlock = false;
  let codeBlockFence = '';

  for (let line of contentLines) {
    // Check for code block markers
    if (line.startsWith('```')) {
      inCodeBlock = !inCodeBlock;
      if (inCodeBlock) {
        codeBlockFence = line;
      }
      translatedContent.push(line);
      continue;
    }

    // Don't translate inside code blocks
    if (inCodeBlock) {
      translatedContent.push(line);
      continue;
    }

    // Skip empty lines
    if (!line.trim()) {
      translatedContent.push(line);
      continue;
    }

    // Translate the line
    const translated = await translateText(line);
    translatedContent.push(translated);

    // Add delay to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, BATCH_DELAY));
  }

  return fmLines.join('\n') + '\n' + translatedContent.join('\n');
}

/**
 * Get all markdown files recursively
 */
function getAllMarkdownFiles(dir, fileList = []) {
  const files = fs.readdirSync(dir);

  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      getAllMarkdownFiles(filePath, fileList);
    } else if (file.endsWith('.md')) {
      fileList.push(filePath);
    }
  });

  return fileList;
}

/**
 * Main translation process
 */
async function main() {
  console.log('🚀 Starting documentation translation to Urdu...\n');

  const markdownFiles = getAllMarkdownFiles(DOCS_DIR);
  console.log(`📄 Found ${markdownFiles.length} markdown files to translate\n`);

  let successCount = 0;
  let errorCount = 0;

  for (const filePath of markdownFiles) {
    try {
      const relativePath = path.relative(DOCS_DIR, filePath);
      const urduFilePath = path.join(URDU_I18N_DIR, relativePath);
      const urduDir = path.dirname(urduFilePath);

      // Ensure directory exists
      if (!fs.existsSync(urduDir)) {
        fs.mkdirSync(urduDir, { recursive: true });
      }

      // Read English content
      const englishContent = fs.readFileSync(filePath, 'utf-8');
      console.log(`📝 Translating: ${relativePath}`);

      // Translate content
      const urduContent = await translateMarkdown(englishContent);

      // Write Urdu content
      fs.writeFileSync(urduFilePath, urduContent, 'utf-8');
      console.log(`✓ Saved: ${relativePath}\n`);

      successCount++;
    } catch (error) {
      console.error(`✗ Error translating ${filePath}:`, error.message);
      errorCount++;
    }
  }

  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log(`✅ Translation complete!`);
  console.log(`✓ Successfully translated: ${successCount} files`);
  if (errorCount > 0) {
    console.log(`✗ Errors: ${errorCount} files`);
  }
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
  console.log('Users can now access Urdu content at: /ur/docs/...');
}

main().catch(console.error);
