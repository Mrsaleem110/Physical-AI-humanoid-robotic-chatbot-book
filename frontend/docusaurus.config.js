// @ts-check
// `@type` JSDoc annotations allow IDEs and type-checking tools to autocomplete
// @ts-check

/** @type {import('@docusaurus/types').Config} */
// Keep a single base URL constant so script paths work in dev and production
const BASE_URL = '/Physical-AI-humanoid-robotic-chatbot-book/';

const config = {
  title: 'Agentic Sphere - Physical AI & Humanoid Robotics',
  tagline: 'Discover a next generation web book dedicated to Physical AI and Humanoid Robotics Access expertly curated content in multiple languages and interact with an intelligent chatbot trained on the book itself. A modern learning experience designed for professionals, researchers and future innovators.               (made by )',

  favicon: 'img/logo.svg',
  
  // Set the production url of your site here
  url: 'https://Mrsaleem110.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<org-name>/<repo-name>/'
  baseUrl: BASE_URL,
  trailingSlash: true,

  // GitHub pages deployment config.
  organizationName: 'Mrsaleem110', // Usually your GitHub org/user name.
  projectName: 'Physical-AI-humanoid-robotic-chatbot-book', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur', 'ko'],
    localeConfigs: {
      en: {
        label: 'English',
        direction: 'ltr',
      },
      ur: {
        label: 'Urdu',
        direction: 'rtl', // Right to left for Urdu
      },
      ko: {
        label: 'Korean',
        direction: 'ltr',
      },
    },
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/Mrsaleem110/Physical-AI-humanoid-robotic-chatbot-book/tree/main/docs/',
          // Enable JSX in markdown
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: false, // Disable blog for this book
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
  scripts: [
    {
      src: `${BASE_URL}js/translate-button.js`,
      async: true,
    },
    {
      src: `${BASE_URL}js/controlled-translation.js`,
      async: true,
    },
  ],

  customFields: {
    libreTranslateUrl: process.env.REACT_APP_LIBRETRANSLATE_URL || 'http://localhost:5000/translate',
  },

  clientModules: [
    require.resolve('./src/theme/wrapper.js'),
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/logo.svg',
      navbar: {
        title: 'Agentic Sphere',
        logo: {
          alt: 'Agentic Sphere Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book',
          },
          {
            type: 'html',
            value: '<div id="translate-button-mount" style="display: flex; align-items: center;"></div>',
            position: 'right',
          },
          {
            to: '/auth',
            label: 'Sign In',
            position: 'right',
          },
          {
            href: 'https://github.com/Mrsaleem110/Physical-AI-humanoid-robotic-chatbot-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Book Home',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/humanoid-robotics',
              },
              {
                label: 'Discord',
                href: 'https://discord.com/channels/@me',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/Mrsaleem110/Physical-AI-humanoid-robotic-chatbot-book',
              },
            ],
          },
        ],
        copyright: `Copyright @ 2025 Physical AI & Humanoid Robotics Book. Build by Agentic Sphere.`,
      },
      prism: {
        theme: require('prism-react-renderer').themes.github,
        darkTheme: require('prism-react-renderer').themes.dracula,
      },
    }),
};

module.exports = config;