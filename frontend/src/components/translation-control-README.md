# Translation Control Feature

## Overview
This feature provides users with more control over language switching behavior on the website. Instead of automatically redirecting to different language versions, users now have options to control when and how the language switching occurs.

## Features Added

1. **User Preference Control**: Users can choose to disable automatic language switching
2. **Confirmation Dialogs**: When automatic switching is disabled, users are asked to confirm before changing languages
3. **Persistent Settings**: User preferences are saved in browser's localStorage
4. **Keyboard Shortcut**: Ctrl+Shift+T to toggle translation preference on/off

## How It Works

1. The system checks for a user preference stored in localStorage
2. If auto-translation is disabled, a confirmation dialog appears before language switching
3. Users can toggle the preference using the control that appears on the page or with the keyboard shortcut

## Files Modified

- `frontend/static/js/multilingual-dropdown.js` - Added user control logic
- `frontend/static/js/controlled-translation.js` - New file with preference controls
- `frontend/docusaurus.config.js` - Added new script and removed duplicate translation UI elements

## Configuration

The duplicate translation UI elements were removed from the navbar to prevent confusion:
- Removed custom HTML dropdown container (was causing duplicate controls)
- Kept the standard Docusaurus locale dropdown for cleaner UX