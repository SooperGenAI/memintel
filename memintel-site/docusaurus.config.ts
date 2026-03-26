import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import type * as OpenApiPlugin from 'docusaurus-plugin-openapi-docs';

const config: Config = {
  title: 'Memintel',
  tagline: 'Deterministic semantic compiler and runtime for agentic AI systems.',
  favicon: 'img/favicon.ico',

  future: { v4: true },

  url: 'https://www.memintel.io',
  baseUrl: '/',
  organizationName: 'SooperGenAI',
  projectName: 'memintel',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'warn',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: { defaultLocale: 'en', locales: ['en'] },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          showLastUpdateTime: false,
          showLastUpdateAuthor: false,
          docItemComponent: '@theme/ApiItem',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      'docusaurus-plugin-openapi-docs',
      {
        id: 'openapi',
        docsPluginId: 'classic',
        config: {
          memintel: {
            specPath: 'static/api/openapi.yaml',
            outputDir: 'docs/api-reference/generated',
            sidebarOptions: {
              groupPathsBy: 'tag',
              categoryLinkSource: 'tag',
            },
            showSchemas: true,
          } satisfies OpenApiPlugin.Options,
        },
      },
    ],
    // Fix webpack polyfill errors from postman-code-generators
    function webpackPolyfillPlugin() {
      return {
        name: 'webpack-polyfill-plugin',
        configureWebpack() {
          return {
            resolve: {
              fallback: {
                path: false,
                fs: false,
                os: false,
              },
            },
          };
        },
      };
    },
  ],

  themes: ['docusaurus-theme-openapi-docs'],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    // languageTabs belongs here in themeConfig, not in the plugin config
    languageTabs: [
      { highlight: 'bash',       language: 'curl',   logoClass: 'bash'   },
      { highlight: 'python',     language: 'python', logoClass: 'python' },
      { highlight: 'typescript', language: 'nodejs', logoClass: 'nodejs' },
    ],
    announcementBar: {
      id: 'early_access',
      content: '⚡ Memintel is in early access — <a href="https://github.com/SooperGenAI/memintel" target="_blank">star us on GitHub</a> to follow the build.',
      backgroundColor: '#0f172a',
      textColor: '#94a3b8',
      isCloseable: true,
    },
    navbar: {
      title: 'Memintel',
      hideOnScroll: false,
      logo: {
        alt: 'Memintel',
        src: 'img/logo.svg',
        srcDark: 'img/logo-dark.svg',
        width: 28,
        height: 28,
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'introSidebar',
          position: 'left',
          label: 'Introduction',
        },
        {
          type: 'docSidebar',
          sidebarId: 'adminSidebar',
          position: 'left',
          label: 'Admin Guide',
        },
        {
          type: 'docSidebar',
          sidebarId: 'deploySidebar',
          position: 'left',
          label: 'Deploy',
        },
        {
          type: 'docSidebar',
          sidebarId: 'apiSidebar',
          position: 'left',
          label: 'API Reference',
        },
        {
          type: 'docSidebar',
          sidebarId: 'pythonSidebar',
          position: 'left',
          label: 'Python SDK',
        },
        {
          type: 'docSidebar',
          sidebarId: 'tutorialsSidebar',
          position: 'left',
          label: 'Tutorials',
        },
        {
          href: 'https://github.com/SooperGenAI/memintel',
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
            { label: 'Introduction', to: '/docs/intro/overview' },
            { label: 'API Reference', to: '/docs/api-reference/overview' },
            { label: 'Python SDK', to: '/docs/python-sdk/python-overview' },
          ],
        },
        {
          title: 'Project',
          items: [
            { label: 'GitHub', href: 'https://github.com/SooperGenAI/memintel' },
            { label: 'Issues', href: 'https://github.com/SooperGenAI/memintel/issues' },
            { label: 'Discussions', href: 'https://github.com/SooperGenAI/memintel/discussions' },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Memintel.`,
    },
    prism: {
      theme: prismThemes.oneDark,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: ['python', 'typescript', 'yaml', 'bash'],
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
