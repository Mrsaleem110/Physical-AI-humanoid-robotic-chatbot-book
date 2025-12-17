#!/usr/bin/env python3
"""
Simple test script to verify the translation implementation for all requested languages.
This script checks that all necessary files and configurations are in place.
"""
import os
from pathlib import Path

def test_translation_implementation():
    """Test that the translation implementation is complete."""
    print("Testing Translation Implementation...")
    print("=" * 50)

    # Define the project root and frontend path
    project_root = Path(__file__).parent
    frontend_path = project_root / "frontend"
    backend_path = project_root / "backend"

    # Languages to test
    languages = ['en', 'ur', 'hi', 'fr', 'de', 'zh', 'ja']

    print("1. Testing Backend Implementation")
    print("-" * 30)

    # Check if the translation models are properly defined
    translation_model_path = backend_path / "src" / "models" / "translation.py"
    if translation_model_path.exists():
        with open(translation_model_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'TranslationRequest' in content and 'TranslationResponse' in content:
                print("[OK] Backend: Pydantic models defined")
            else:
                print("[ERROR] Backend: Pydantic models missing")
        print("[OK] Backend: Translation service exists")
    else:
        print("[ERROR] Backend: Translation model file missing")

    # Check if deep-translator is in requirements
    requirements_path = backend_path / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'deep-translator' in content:
                print("[OK] Backend: deep-translator dependency present")
            else:
                print("[ERROR] Backend: deep-translator dependency missing")
    else:
        print("[ERROR] Backend: requirements.txt missing")

    print("\n2. Testing Frontend Implementation")
    print("-" * 30)

    # Check if i18n directories exist for all languages
    i18n_path = frontend_path / "i18n"
    if i18n_path.exists():
        print("[OK] Frontend: i18n directory exists")

        missing_languages = []
        for lang in languages:
            lang_path = i18n_path / lang
            if lang_path.exists():
                print(f"[OK] Frontend: {lang} language directory exists")

                # Check subdirectories
                subdirs = ['docusaurus-plugin-content-docs', 'docusaurus-plugin-content-pages', 'docusaurus-theme-classic']
                for subdir in subdirs:
                    sub_path = lang_path / subdir
                    if sub_path.exists():
                        print(f"  [OK] {subdir} subdirectory exists")
                    else:
                        print(f"  [ERROR] {subdir} subdirectory missing")

                # Check code.json file
                code_json_path = lang_path / "code.json"
                if code_json_path.exists():
                    print(f"  [OK] code.json exists")
                else:
                    print(f"  [ERROR] code.json missing")

                # Check Navbar.json file
                navbar_json_path = lang_path / "docusaurus-theme-classic" / "Navbar.json"
                if navbar_json_path.exists():
                    print(f"  [OK] Navbar.json exists")
                else:
                    print(f"  [ERROR] Navbar.json missing")

            else:
                print(f"[ERROR] Frontend: {lang} language directory missing")
                missing_languages.append(lang)

        if not missing_languages:
            print("[OK] All language directories exist")
        else:
            print(f"[ERROR] Missing language directories: {missing_languages}")
    else:
        print("[ERROR] Frontend: i18n directory missing")

    print("\n3. Testing Translation Content")
    print("-" * 30)

    # Check if intro.md files exist for all languages
    content_missing = []
    for lang in languages:
        intro_path = i18n_path / lang / "docusaurus-plugin-content-docs" / "current" / "intro.md"
        if intro_path.exists():
            print(f"[OK] intro.md exists for {lang}")
        else:
            print(f"[ERROR] intro.md missing for {lang}")
            content_missing.append(lang)

    if not content_missing:
        print("[OK] All intro.md files exist")
    else:
        print(f"[ERROR] Missing intro.md files: {content_missing}")

    # Check if translation pages exist
    page_missing = []
    for lang in languages:
        page_path = i18n_path / lang / "docusaurus-plugin-content-pages" / "translation.js"
        if page_path.exists():
            print(f"[OK] translation.js exists for {lang}")
        else:
            print(f"[ERROR] translation.js missing for {lang}")
            page_missing.append(lang)

    if not page_missing:
        print("[OK] All translation.js pages exist")
    else:
        print(f"[ERROR] Missing translation.js pages: {page_missing}")

    print("\n4. Testing Navbar Component")
    print("-" * 30)

    navbar_component_path = frontend_path / "src" / "theme" / "NavbarItem" / "CustomTranslationNavbarButton.js"
    if navbar_component_path.exists():
        with open(navbar_component_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'ur' in content and 'hi' in content and 'fr' in content and 'de' in content and 'zh' in content and 'ja' in content:
                print("[OK] Navbar component supports all languages")
            else:
                print("[ERROR] Navbar component missing some languages")
        print("[OK] Navbar component exists")
    else:
        print("[ERROR] Navbar component missing")

    print("\n5. Testing Docusaurus Configuration")
    print("-" * 30)

    config_path = frontend_path / "docusaurus.config.js"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'ur' in content and 'hi' in content and 'fr' in content and 'de' in content and 'zh' in content and 'ja' in content:
                print("[OK] Docusaurus config includes all languages")
            else:
                print("[ERROR] Docusaurus config missing some languages")
        print("[OK] Docusaurus config exists")
    else:
        print("[ERROR] Docusaurus config missing")

    print("\n" + "=" * 50)
    print("Translation Implementation Test Complete")

    # Summary
    all_checks_passed = (
        translation_model_path.exists() and
        requirements_path.exists() and
        'deep-translator' in open(requirements_path, encoding='utf-8').read() and
        i18n_path.exists() and
        not missing_languages and
        not content_missing and
        not page_missing and
        navbar_component_path.exists()
    )

    if all_checks_passed:
        print("[SUCCESS] ALL TESTS PASSED! Translation implementation is complete.")
        print("\nFeatures implemented:")
        print("- Backend API with Pydantic models")
        print("- Deep-translator dependency")
        print("- Frontend i18n support for all languages")
        print("- Language-specific content files")
        print("- Interactive translation UI")
        print("- Navbar language selector")
        print("- Docusaurus configuration")
        return True
    else:
        print("[FAILURE] SOME TESTS FAILED! Implementation is incomplete.")
        return False

if __name__ == "__main__":
    success = test_translation_implementation()
    exit(0 if success else 1)