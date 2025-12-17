import React from 'react';
import Layout from '@theme/Layout';
import PageTranslation from '@site/src/components/PageTranslation';

export default function TranslationPage() {
  return (
    <Layout title="अनुवाद" description="अनुवाद क्षमता पृष्ठ">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>अनुवाद क्षमता</h1>
            <p>यह पृष्ठ हमारे प्लेटफॉर्म की अनुवाद क्षमताओं को दर्शाता है।</p>
            <PageTranslation />
            <div className="margin-vert--lg">
              <h2>अनुवाद कैसे काम करता है</h2>
              <p>हमारा प्लेटफॉर्म उर्दू, हिंदी, फ्रेंच, जर्मन, चीनी और जापानी सहित कई भाषाओं में सामग्री के वास्तविक समय अनुवाद का समर्थन करता है।</p>
              <p>ड्रॉपडाउन से एक भाषा चुनें ताकि वास्तविक समय में अनुवादित सामग्री देखी जा सके।</p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}