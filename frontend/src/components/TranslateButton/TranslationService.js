import axios from 'axios';

// Use MyMemory Translation API (free, no API key required)
const TRANSLATION_API = 'https://api.mymemory.translated.net/get';

/**
 * Extracts text from HTML while preserving structure
 */
const extractTextSegments = (htmlContent) => {
  const parser = new DOMParser();
  const doc = parser.parseFromString(htmlContent, 'text/html');
  const segments = [];
  
  // Walk through all text nodes
  const walker = document.createTreeWalker(
    doc.body,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode: (node) => {
        // Skip empty text and code blocks
        const parent = node.parentElement;
        if (!node.textContent.trim()) return NodeFilter.FILTER_REJECT;
        if (parent.tagName === 'CODE' || parent.tagName === 'PRE') return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    }
  );
  
  let node;
  while (node = walker.nextNode()) {
    const text = node.textContent.trim();
    if (text.length > 0) {
      segments.push({
        original: text,
        node: node
      });
    }
  }
  
  return { segments, doc };
};

/**
 * Translates a single text segment to Urdu
 */
const translateSegment = async (text) => {
  try {
    // MyMemory API has a 500 character limit per request
    if (text.length > 500) {
      // Split into sentences and translate separately
      const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
      const translations = await Promise.all(
        sentences.map(sentence => translateSegment(sentence.trim()))
      );
      return translations.join(' ');
    }
    
    const response = await axios.get(TRANSLATION_API, {
      params: {
        q: text,
        langpair: 'en|ur'
      }
    });
    
    if (response.data && response.data.responseData) {
      return response.data.responseData.translatedText;
    }
    
    return text; // Return original if translation fails
  } catch (error) {
    console.warn('Translation segment failed:', error);
    return text; // Return original on error
  }
};

/**
 * Translates HTML content to Urdu while preserving code blocks and structure
 * @param {string} htmlContent - The HTML content to translate
 * @returns {Promise<string>} - The translated HTML content
 */
export const translateToUrdu = async (htmlContent) => {
  try {
    // Extract text segments
    const { segments, doc } = extractTextSegments(htmlContent);
    
    if (segments.length === 0) {
      throw new Error('No translatable content found');
    }
    
    // Translate segments in batches to avoid rate limiting
    const batchSize = 5;
    for (let i = 0; i < segments.length; i += batchSize) {
      const batch = segments.slice(i, i + batchSize);
      
      await Promise.all(
        batch.map(async (segment) => {
          try {
            const translated = await translateSegment(segment.original);
            segment.node.textContent = translated;
            
            // Small delay to avoid rate limiting
            await new Promise(resolve => setTimeout(resolve, 200));
          } catch (err) {
            console.warn('Failed to translate segment:', err);
            // Keep original text on error
          }
        })
      );
    }
    
    return doc.body.innerHTML;

  } catch (error) {
    console.error("Error translating to Urdu:", error);
    throw new Error('Translation failed. Please try again later.');
  }
};

/**
 * Translates plain text to Urdu (for simpler use cases)
 * @param {string} text - The text to translate
 * @returns {Promise<string>} - The translated text
 */
export const translateTextToUrdu = async (text) => {
  try {
    return await translateSegment(text);
  } catch (error) {
    console.error("Error translating text to Urdu:", error);
    throw new Error('Translation failed. Please try again.');
  }
};
