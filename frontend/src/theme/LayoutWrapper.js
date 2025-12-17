import React, { useEffect } from 'react';
import { useUser } from '../contexts/UserContext';
import { Redirect } from '@docusaurus/router';
import Layout from '@theme/Layout';

const LayoutWrapper = (props) => {
  const { user, isLoading } = useUser();
  const isDocsPage = typeof window !== 'undefined' && window.location.pathname.startsWith('/docs/');

  // Only apply auth check for docs pages
  if (isDocsPage) {
    if (isLoading) {
      return (
        <Layout title="Loading..." description="Please wait while we check your authentication status">
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '50vh',
            fontSize: '18px'
          }}>
            Loading...
          </div>
        </Layout>
      );
    }

    if (!user) {
      return <Redirect to="/auth" />;
    }
  }

  return props.children;
};

export default LayoutWrapper;