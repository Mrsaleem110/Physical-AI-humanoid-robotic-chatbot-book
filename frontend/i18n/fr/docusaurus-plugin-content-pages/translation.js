import React from 'react';
import Layout from '@theme/Layout';
import PageTranslation from '@site/src/components/PageTranslation';

export default function TranslationPage() {
  return (
    <Layout title="Traduction" description="Page de fonctionnalité de traduction">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Fonctionnalité de traduction</h1>
            <p>Cette page démontre les capacités de traduction de notre plateforme.</p>
            <PageTranslation />
            <div className="margin-vert--lg">
              <h2>Comment fonctionne la traduction</h2>
              <p>Notre plateforme prend en charge la traduction en temps réel de contenu dans plusieurs langues, notamment l'ourdou, l'hindi, le français, l'allemand, le chinois et le japonais.</p>
              <p>Sélectionnez une langue dans le menu déroulant ci-dessus pour voir le contenu de la page traduit en temps réel.</p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}