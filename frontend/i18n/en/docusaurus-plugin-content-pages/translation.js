import React from 'react';
import Layout from '@theme/Layout';
import PageTranslation from '@site/src/components/PageTranslation';

export default function TranslationPage() {
  return (
    <Layout title="Translation" description="Translation functionality page">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Translation Functionality</h1>
            <p>This page demonstrates the translation capabilities of our platform.</p>
            <PageTranslation />
            <div className="margin-vert--lg">
              <h2>How Translation Works</h2>
              <p>Our platform supports real-time translation of content into multiple languages including Urdu, Hindi, French, German, Chinese, and Japanese.</p>
              <p>Select a language from the dropdown above to see the page content translated in real-time.</p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}