import React, { useEffect, useState } from 'react';
import { useTranslation } from '../../hooks/useTranslation';

// Component to translate larger content blocks (like blog posts, book chapters)
const TranslateContent = ({ content, contentType = 'text' }) => {
  const { currentLanguage, translate, isLoading } = useTranslation();
  const [translatedContent, setTranslatedContent] = useState(content);

  useEffect(() => {
    const translateContentBlock = async () => {
      if (currentLanguage === 'en' || !content) {
        setTranslatedContent(content);
        return;
      }

      try {
        if (contentType === 'html') {
          // For HTML content, we need to parse and translate text nodes
          const translated = await translateHtmlContent(content, translate);
          setTranslatedContent(translated);
        } else {
          // For plain text content
          const result = await translate(content, currentLanguage);
          setTranslatedContent(result);
        }
      } catch (error) {
        console.error('Content translation error:', error);
        setTranslatedContent(content); // Fallback to original content
      }
    };

    translateContentBlock();
  }, [content, currentLanguage, translate, contentType]);

  if (isLoading && content) {
    return <div className="translation-loading">Translating content...</div>;
  }

  if (contentType === 'html') {
    return <div dangerouslySetInnerHTML={{ __html: translatedContent }} />;
  }

  return <>{translatedContent}</>;
};

// Helper function to translate HTML content while preserving structure
const translateHtmlContent = async (htmlString, translateFn) => {
  // This is a simplified approach - in a real implementation, you'd want to use a proper HTML parser
  // For now, we'll extract text content, translate it, and return the modified HTML
  const parser = new DOMParser();
  const doc = parser.parseFromString(htmlString, 'text/html');

  // Find all text nodes and translate them
  const walker = doc.createTreeWalker(
    doc.body,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode: function(node) {
        // Only translate text nodes that have actual content
        return node.nodeValue.trim().length > 0 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    }
  );

  const textNodes = [];
  let node;
  while (node = walker.nextNode()) {
    textNodes.push(node);
  }

  // Translate each text node
  for (const textNode of textNodes) {
    if (textNode.nodeValue.trim()) {
      try {
        const translated = await translateFn(textNode.nodeValue);
        textNode.nodeValue = translated;
      } catch (error) {
        console.error('Text node translation error:', error);
        // Keep original text if translation fails
      }
    }
  }

  return doc.body.innerHTML;
};

export default TranslateContent;