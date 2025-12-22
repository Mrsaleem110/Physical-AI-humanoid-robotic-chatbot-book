import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import LanguageSelector from '../LanguageSelector';
import { TranslationProvider } from '../TranslationProvider';

// Mock the useTranslation hook
jest.mock('../../../hooks/useTranslation', () => ({
  useTranslation: jest.fn(),
}));

const MockProvider = ({ children, mockValue }) => (
  <TranslationProvider defaultLanguage="en">
    {children}
  </TranslationProvider>
);

describe('LanguageSelector Component', () => {
  const mockChangeLanguage = jest.fn();

  const renderWithProvider = (mockValue) => {
    const defaultMockValue = {
      currentLanguage: 'en',
      availableLanguages: [
        { code: 'en', name: 'English' },
        { code: 'ur', name: 'Urdu' },
        { code: 'fr', name: 'French' },
      ],
      changeLanguage: mockChangeLanguage,
      isLoading: false,
      getLanguageName: (code) => code,
    };

    const combinedMockValue = { ...defaultMockValue, ...mockValue };

    return render(
      <TranslationProvider defaultLanguage={combinedMockValue.currentLanguage}>
        <LanguageSelector />
      </TranslationProvider>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders language selector dropdown with available languages', () => {
    renderWithProvider();

    const selectElement = screen.getByRole('combobox');
    expect(selectElement).toBeInTheDocument();

    // Check that all available languages are present as options
    expect(screen.getByRole('option', { name: /English/ })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /Urdu/ })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /French/ })).toBeInTheDocument();
  });

  test('displays current language as selected', () => {
    renderWithProvider({ currentLanguage: 'fr' });

    const selectElement = screen.getByRole('combobox');
    expect(selectElement.value).toBe('fr');
  });

  test('calls changeLanguage when a new language is selected', async () => {
    renderWithProvider();

    const selectElement = screen.getByRole('combobox');
    fireEvent.change(selectElement, { target: { value: 'ur' } });

    await waitFor(() => {
      expect(mockChangeLanguage).toHaveBeenCalledWith('ur');
    });
  });

  test('disables dropdown when loading', () => {
    renderWithProvider({ isLoading: true });

    const selectElement = screen.getByRole('combobox');
    expect(selectElement).toBeDisabled();
  });

  test('shows loading indicator when loading', () => {
    renderWithProvider({ isLoading: true });

    expect(screen.getByText('Translating...')).toBeInTheDocument();
  });

  test('displays language flags and names correctly', () => {
    renderWithProvider();

    const selectElement = screen.getByRole('combobox');
    const options = screen.getAllByRole('option');

    // Check that some options include flags and names
    expect(screen.getByRole('option', { name: /🇺🇸 English/ })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /🇵🇰 Urdu/ })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /🇫🇷 French/ })).toBeInTheDocument();
  });
});