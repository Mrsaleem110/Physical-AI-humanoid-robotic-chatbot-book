import React, { useEffect } from 'react';
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

  const isDocsPage = relativePath.startsWith('/docs/');
  const isAuthPage = relativePath === '/auth';
  const isHomePage = relativePath === '/';

  // Apply auth check only for docs pages (not the home page, as home page handles auth itself)
  if (isDocsPage && !isAuthPage) {
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
      return <Redirect to={`${baseUrl}auth`} />;
    }
  }

  return props.children;
};

export default LayoutWrapper;