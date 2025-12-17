import React from 'react';
import Layout from '@theme/Layout';
import PageTranslation from '@site/src/components/PageTranslation';

export default function TranslationPage() {
  return (
    <Layout title="翻訳" description="翻訳機能ページ">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>翻訳機能</h1>
            <p>このページでは、当プラットフォームの翻訳機能を紹介しています。</p>
            <PageTranslation />
            <div className="margin-vert--lg">
              <h2>翻訳の仕組み</h2>
              <p>当プラットフォームは、ウルドゥー語、ヒンディー語、フランス語、ドイツ語、中国語、日本語など、複数の言語へのコンテンツのリアルタイム翻訳をサポートしています。</p>
              <p>上記のドロップダウンから言語を選択すると、ページコンテンツがリアルタイムで翻訳されます。</p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}