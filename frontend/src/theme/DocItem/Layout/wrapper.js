import React from 'react';
import { useUser } from '../../../contexts/UserContext';
import { Redirect } from '@docusaurus/router';
import TranslateButton from '../../../components/TranslateButton';

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
      <div style={{
        position: 'sticky',
        top: '10px',
        zIndex: 1000,
        marginBottom: '20px',
        textAlign: 'right',
        paddingRight: '20px'
      }}>
        <TranslateButton />
      </div>
      {children}
    </>
  );
}