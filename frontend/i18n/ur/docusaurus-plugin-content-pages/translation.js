import React from 'react';
import Layout from '@theme/Layout';
import PageTranslation from '@site/src/components/PageTranslation';

export default function TranslationPage() {
  return (
    <Layout title="ترجمہ" description="ترجمہ کی صلاحیت کا صفحہ">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>ترجمہ کی صلاحیت</h1>
            <p>یہ صفحہ ہمارے پلیٹ فارم کی ترجمہ کی صلاحیات کو ظاہر کرتا ہے۔</p>
            <PageTranslation />
            <div className="margin-vert--lg">
              <h2>ترجمہ کیسے کام کرتا ہے</h2>
              <p>ہمارا پلیٹ فارم اردو، ہندی، فرانسیسی، جرمن، چینی اور جاپانی سمیت متعدد زبانوں میں مواد کے حقیقی وقت کے ترجمے کی حمایت کرتا ہے۔</p>
              <p>اوپر ڈراپ ڈاؤن سے زبان منتخب کریں تاکہ حقیقی وقت میں ترجمہ شدہ مواد دیکھ سکیں۔</p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}