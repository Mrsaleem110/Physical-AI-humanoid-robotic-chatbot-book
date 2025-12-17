---
id: 5
title: "Enhance Urdu Translations"
stage: "general"
date_iso: "2025-12-12"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "master"
user: "user"
command: "Enhance Urdu translation files to cover all UI elements"
labels: ["i18n", "translation", "localization", "urdu"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml: |
  - C:\Users\Chohan Laptop's\A.I Hackathon 1\humanoid-chatbot-book\docs\i18n\ur\docusaurus-theme-classic\navbar.json
  - C:\Users\Chohan Laptop's\A.I Hackathon 1\humanoid-chatbot-book\docs\i18n\ur\docusaurus-theme-classic\footer.json
  - C:\Users\Chohan Laptop's\A.I Hackathon 1\humanoid-chatbot-book\docs\i18n\ur\docusaurus-theme-classic\404.json
  - C:\Users\Chohan Laptop's\A.I Hackathon 1\humanoid-chatbot-book\docs\i18n\ur\docusaurus-plugin-content-docs\current.json
tests_yaml: |
  - "Manual testing of all UI elements in Urdu language"
  - "Verification that navbar elements are properly translated"
  - "Verification that footer elements are properly translated"
  - "Verification that 404 page is properly translated"
---

# Enhance Urdu Translations

## PROMPT_TEXT
not working in urdu Humanoid Robotics Logo
Physical AI & Humanoid Robotics
Book
English
Sign In
GitHub

Page Not Found
We could not find what you were looking for.

Please contact the owner of the site that linked you to the original URL and let them know their link is broken.

Docs
Book Home
Community
Stack Overflow
Discord
More
GitHub
Copyright © 2025 Physical AI & Humanoid Robotics Book. Built with Docusaurus.

## RESPONSE_TEXT
I've enhanced the Urdu translation files to cover all the UI elements you mentioned:

1. Updated navbar.json to include translations for logo alt text, locale dropdown label, and other navbar elements
2. Updated footer.json to include translations for footer section titles like 'Docs', 'Community', and 'More'
3. Created a specific 404.json file to translate the "Page Not Found" message and related text
4. Created docs-related translation file for sidebar and version-related elements

The translations now cover:
- Navbar elements: "Humanoid Robotics Logo", "Physical AI & Humanoid Robotics", "Book", "English", "Sign In", "GitHub"
- 404 page: "Page Not Found", "We could not find what you were looking for", etc.
- Footer elements: "Docs", "Book Home", "Community", "Stack Overflow", "Discord", "More", "GitHub", and copyright text

## OUTCOME
- Comprehensive Urdu translations for all major UI elements
- Proper translation of navbar, footer, 404 page, and docs elements
- All elements mentioned in the issue are now properly translated
- Improved localization coverage for Urdu language