import React from 'react';
import { useUser } from '../../../contexts/UserContext';
import { Redirect } from '@docusaurus/router';
// Translator removed from doc wrapper to avoid UI clutter; use navbar dropdown instead

export default function DocItemLayoutWrapper({ children }) {
  const { user, isLoading } = useUser();

  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '50vh',
        fontSize: '18px'
      }}>
        Loading...
      </div>
    );
  }

  if (!user) {
    return <Redirect to="/auth" />;
  }

  return (
    <>
      {/* Translator removed from this area; use navbar dropdown */}
      {children}
    </>
  );
}