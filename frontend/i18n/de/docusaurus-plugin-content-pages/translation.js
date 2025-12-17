import React from 'react';
import Layout from '@theme/Layout';
import PageTranslation from '@site/src/components/PageTranslation';

export default function TranslationPage() {
  return (
    <Layout title="Übersetzung" description="Übersetzungsfunktionalität Seite">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Übersetzungsfunktionalität</h1>
            <p>Diese Seite zeigt die Übersetzungsfähigkeiten unserer Plattform.</p>
            <PageTranslation />
            <div className="margin-vert--lg">
              <h2>Wie die Übersetzung funktioniert</h2>
              <p>Unsere Plattform unterstützt die Echtzeit-Übersetzung von Inhalten in mehrere Sprachen, darunter Urdu, Hindi, Französisch, Deutsch, Chinesisch und Japanisch.</p>
              <p>Wählen Sie eine Sprache aus dem Dropdown-Menü oben, um den Seiteninhalt in Echtzeit übersetzt zu sehen.</p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}