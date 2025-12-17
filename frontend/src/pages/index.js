import React, { useEffect } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import FloatingChatbot from '@site/src/components/FloatingChatbot';
import TranslateButton from '@site/src/components/TranslateButton';
import { useUser } from '@site/src/contexts/UserContext';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Agentic Sphere
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  const { user, isLoading } = useUser();

  // If user is not logged in, redirect to auth page
  useEffect(() => {
    if (!isLoading && !user) {
      window.location.href = '/auth';
    }
  }, [user, isLoading]);

  // Show loading state while checking authentication
  if (!isLoading && !user) {
    return (
      <Layout
        title={`Redirecting...`}
        description="Redirecting to authentication page">
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '60vh',
          fontSize: '18px'
        }}>
          Redirecting to signup page...
        </div>
      </Layout>
    );
  }

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <Layout
        title={`Loading...`}
        description="Loading your experience">
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '60vh',
          fontSize: '18px'
        }}>
          Loading...
        </div>
      </Layout>
    );
  }

  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Comprehensive Guide to Building Intelligent Humanoid Robots by">
      <div style={{ position: 'relative' }}>
        <TranslateButton />
      </div>
      <HomepageHeader />
      <main>

        <HomepageFeatures />

      </main>
      <FloatingChatbot />
    </Layout>
  );
}