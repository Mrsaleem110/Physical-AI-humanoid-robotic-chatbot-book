import React from 'react';
import { useUser } from '../contexts/UserContext';
import { Redirect } from '@docusaurus/router';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

const LayoutWrapper = (props) => {
  const { user, isLoading } = useUser();
  const { siteConfig } = useDocusaurusContext();
  const baseUrl = (siteConfig && siteConfig.baseUrl) ? siteConfig.baseUrl : '/';

  let relativePath = '/';
  if (typeof window !== 'undefined') {
    const pathname = window.location.pathname || '/';
    const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    if (normalizedBase !== '/' && pathname.startsWith(normalizedBase)) {
      relativePath = pathname.slice(normalizedBase.length - 1);
    } else {
      relativePath = pathname;
    }
  }

  const isAuthPage = relativePath === '/auth' || relativePath === `${baseUrl}auth`;

  // Consider some paths as static/assets so we don't redirect them
  const isStaticAsset = relativePath.startsWith('/static') || relativePath.startsWith('/img') || relativePath.endsWith('.ico') || relativePath.includes('.') || relativePath.startsWith('/assets');

  // If checking auth status, show a loading page
  if (isLoading) {
    return (
      <Layout title="Loading..." description="Checking authentication status">
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

  // If user is not authenticated and visiting a non-auth, non-static page, redirect to /auth
  if (!user && !isAuthPage && !isStaticAsset) {
    return <Redirect to={`${baseUrl}auth`} />;
  }

  return props.children;
};

export default LayoutWrapper;