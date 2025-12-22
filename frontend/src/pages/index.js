import React, { useEffect } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { Redirect } from '@docusaurus/router';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import FloatingChatbot from '@site/src/components/FloatingChatbot';
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
  const {user, isLoading} = useUser();

  // Show loading state while checking authentication status
  if (isLoading) {
    return (
      <Layout
        title={`Loading - ${siteConfig.title}`}
        description="Loading your personalized learning experience">
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '60vh',
          fontSize: '18px'
        }}>
          Loading your personalized experience...
        </div>
      </Layout>
    );
  }

  // Redirect to auth page if user is not authenticated using Docusaurus Redirect
  const baseUrl = siteConfig.baseUrl || '/';

  if (!user) {
    return <Redirect to={`${baseUrl}auth`} />;
  }

  // If user is authenticated but hasn't completed personalization, send them to the background questions
  const needsPersonalization = !user.preferredLearningStyle || !user.softwareExperience;
  if (needsPersonalization) {
    return <Redirect to={`${baseUrl}auth?step=2`} />;
  }

  // Show loading state while checking authentication status
  if (isLoading) {
    return (
      <Layout
        title={`Loading - ${siteConfig.title}`}
        description="Loading your personalized learning experience">
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '60vh',
          fontSize: '18px'
        }}>
          Loading your personalized experience...
        </div>
      </Layout>
    );
  }

  // Show content only if user is authenticated
  if (user) {
    return (
      <Layout
        title={`Welcome to ${siteConfig.title}`}
        description="Discover Physical AI and Humanoid Robotics with our interactive learning platform">
        <HomepageHeader />
        <main>
          {/* Show welcome message for new users */}
          <section className="margin-top--lg margin-bottom--lg">
            <div className="container">
              <div className="row">
                <div className="col col--8 col--offset-2">
                  <div className="text--center padding-horiz--md">
                    <h2>Welcome, {user.name || 'User'}!</h2>
                    <p>
                      Congratulations on joining the Agentic Sphere learning platform.
                      You're now ready to explore Physical AI and Humanoid Robotics with our
                      personalized learning experience.
                    </p>
                    <p>
                      Start your journey by exploring our comprehensive modules,
                      use the chatbot for instant help, and customize your learning
                      experience based on your background.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>
          <HomepageFeatures />
        </main>
        <FloatingChatbot />
      </Layout>
    );
  }

  // If not authenticated, show redirecting message
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
        Redirecting to authentication page...
      </div>
    </Layout>
  );
}