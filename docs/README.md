# GitHub Pages Deployment Documentation

This directory contains the built Docusaurus site for the Physical AI & Humanoid Robotics Book, deployed via GitHub Pages.

## Deployment Process

The site is automatically deployed using GitHub Actions when changes are pushed to the `main` branch. Here's how the process works:

1. The GitHub Actions workflow (`.github/workflows/deploy.yml`) triggers on pushes to the `main` branch
2. It sets up Node.js and installs dependencies in the `frontend` directory
3. It builds the Docusaurus site using `npm run build`
4. It copies the built files to this `docs` directory
5. It deploys the content to the `gh-pages` branch for GitHub Pages

## Site Structure

- **Source**: The Docusaurus site is located in the `frontend` directory
- **Configuration**: `frontend/docusaurus.config.js` contains the site configuration
- **Content**: Documentation content is in `frontend/docs/`
- **Deployment**: Built site is deployed to this `docs` directory

## Base URL Configuration

The site is configured with the base URL `/Physical-AI-humanoid-robotic-chatbot-book/` to work correctly with GitHub Pages at:
https://Mrsaleem110.github.io/Physical-AI-humanoid-robotic-chatbot-book/

## Manual Deployment

If you need to build and deploy manually:

```bash
cd frontend
npm install
npm run build
# Copy contents of frontend/build/ to docs/ directory
```

## Troubleshooting

If the site doesn't load properly:
1. Check that GitHub Pages is enabled in your repository settings
2. Verify that the source is set to "Deploy from a branch" with branch "gh-pages" and folder "/ (root)"
3. Ensure the `docusaurus.config.js` file has the correct `baseUrl`

## Internationalization

The site supports multiple languages:
- English (default)
- Urdu (RTL)
- Korean

The language files are located in the `frontend/i18n` directory.