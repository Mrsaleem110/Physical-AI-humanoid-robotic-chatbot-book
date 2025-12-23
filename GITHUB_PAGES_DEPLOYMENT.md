# GitHub Pages Deployment Guide

## Overview
This project is configured for deployment to GitHub Pages using GitHub Actions. The site is built using Docusaurus and automatically deployed when changes are pushed to the main branch.

Last updated: December 23, 2025

## How It Works
1. Source code is in the `frontend/` directory
2. GitHub Actions workflow (`.github/workflows/deploy.yml`) builds the site on every push to main
3. Built site is deployed to the `docs/` directory
4. GitHub Pages serves content from the `docs/` directory on the `gh-pages` branch

## Configuration
- Base URL: `/Physical-AI-humanoid-robotic-chatbot-book/`
- Repository: `Mrsaleem110/Physical-AI-humanoid-robotic-chatbot-book`
- Deployment branch: `gh-pages`
- Source directory: `frontend/`

## To Enable GitHub Pages
1. Go to your repository Settings
2. Navigate to Pages section
3. Set Source to "Deploy from a branch"
4. Select branch "gh-pages" and folder "/ (root)"
5. Click Save

## Site URL
Your site will be available at: https://Mrsaleem110.github.io/Physical-AI-humanoid-robotic-chatbot-book/

## Required Files
The following files are essential for GitHub Pages deployment:
- `.github/workflows/deploy.yml` - GitHub Actions workflow
- `frontend/docusaurus.config.js` - Site configuration with correct base URL
- `docs/` directory - Deployment target (auto-generated)

## Manual Build (if needed)
```bash
cd frontend
npm install
npm run build
# Built files will be in frontend/build/
```

Then copy contents of `frontend/build/` to `docs/` directory.