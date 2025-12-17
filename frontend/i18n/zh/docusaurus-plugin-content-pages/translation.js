import React from 'react';
import Layout from '@theme/Layout';
import PageTranslation from '@site/src/components/PageTranslation';

export default function TranslationPage() {
  return (
    <Layout title="翻译" description="翻译功能页面">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>翻译功能</h1>
            <p>此页面展示了我们平台的翻译功能。</p>
            <PageTranslation />
            <div className="margin-vert--lg">
              <h2>翻译如何工作</h2>
              <p>我们的平台支持将内容实时翻译成多种语言，包括乌尔都语、印地语、法语、德语、中文和日语。</p>
              <p>从上方下拉菜单中选择一种语言，以查看页面内容的实时翻译。</p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}