// Test the urduToRoman function logic
const fs = require('fs');

// Read the script content
const scriptContent = fs.readFileSync('frontend/static/js/translate-button.js', 'utf8');

// Extract the urduToRoman function body
const functionMatch = scriptContent.match(/function urduToRoman\(urduText\) \{([\s\S]*?)return romanText;/);
if (functionMatch) {
  console.log('urduToRoman function found');

  // The function body
  const functionBody = functionMatch[1];

  // Create a test function based on the logic
  const urduToRoman = new Function('urduText', functionBody + 'return romanText;');

  // Test with some sample Urdu text
  const testCases = [
    'السلام علیکم',
    'میں ایک کتاب پڑھ رہا ہوں',
    'آج کیسے ہو',
    ' physical ai اور humanoid robotics '
  ];

  console.log('\nTesting urduToRoman function:');
  testCases.forEach(test => {
    try {
      const result = urduToRoman(test);
      console.log(`Input: "${test}" -> Output: "${result}"`);
    } catch (e) {
      console.error('Error in test:', e);
    }
  });
} else {
  console.log('urduToRoman function not found');
}