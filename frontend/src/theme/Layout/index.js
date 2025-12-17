import React, { useEffect, useState } from 'react';
import { useUser } from '../../contexts/UserContext';
import { Redirect, useLocation } from '@docusaurus/router';
import OriginalLayout from '@theme-original/Layout';

export default function LayoutWrapper(props) {
  const { user, isLoading } = useUser();
  const location = useLocation();
  const [shouldRedirect, setShouldRedirect] = useState(false);
  const [checked, setChecked] = useState(false);

  // Check if the path starts with /docs/ or any locale followed by /docs/
  // But exclude the intro page to allow access without authentication for language switching
  const isDocsPage = (location.pathname.startsWith('/docs/') ||
                     /^\/[a-z]{2}\/docs\//.test(location.pathname))
                     && !location.pathname.includes('/docs/intro');

  useEffect(() => {
    if (!isLoading) {
      if (isDocsPage && !user) {
        setShouldRedirect(true);
      }
      setChecked(true);
    }
  }, [user, isLoading, isDocsPage]);

  if (shouldRedirect) {
    return <Redirect to="/auth" />;
  }

  if (isLoading && isDocsPage && !user) {
    return (
      <OriginalLayout {...props}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '50vh',
          fontSize: '18px'
        }}>
          Checking authentication...
        </div>
      </OriginalLayout>
    );
  }

  return (
    <OriginalLayout {...props} />
  );
}