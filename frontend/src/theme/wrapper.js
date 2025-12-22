import React, { useEffect } from 'react';
import { UserProvider } from '../contexts/UserContext';

export default function Wrapper({ children }) {
  useEffect(() => {
    // TranslateButton will be mounted via navbar config HTML item
    // and rendered through the navbar system
  }, []);

  return (
    <UserProvider>
      {children}
    </UserProvider>
  );
}