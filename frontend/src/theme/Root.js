import React from 'react';
import { UserProvider } from '../contexts/UserContext';
import { TranslationProvider } from '../contexts/TranslationContext';

export default function Root({ children }) {
  return (
    <TranslationProvider>
      <UserProvider>
        {children}
      </UserProvider>
    </TranslationProvider>
  );
}