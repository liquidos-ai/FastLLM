import type {Config} from '@docusaurus/types';
import {themes as prismThemes} from 'prism-react-renderer';

const baseUrl = process.env.DOCUSAURUS_BASE_URL ?? '/';
const config: Config = {
  title: 'FastLLM',
  tagline: 'Rust LLM gateway SDK with cloud and local model orchestration.',
  favicon: 'img/logo.png',
  url: 'https://liquidos-ai.github.io',
  baseUrl,
  organizationName: 'liquidos-ai',
  projectName: 'FastLLM',
  onBrokenLinks: 'throw',
  trailingSlash: true,
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  themes: [],
  presets: [
    [
      'classic',
      {
        docs: {
          path: 'content',
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/liquidos-ai/FastLLM/tree/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],
  themeConfig: {
    image: 'img/logo.png',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'FastLLM',
      hideOnScroll: true,
      logo: {
        alt: 'FastLLM',
        src: 'img/logo.svg',
        width: 24,
        height: 24,
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/guides/getting-started/',
          label: 'Getting Started',
          position: 'left',
        },
        {
          href: 'https://docs.rs/releases/search?query=fastllm',
          label: 'docs.rs',
          position: 'left',
        },
        {
          href: 'https://github.com/liquidos-ai/FastLLM',
          position: 'right',
          className: 'navbar-github-link',
          'aria-label': 'GitHub Repository',
        },
      ],
    },
    footer: {
      style: 'light',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Overview',
              to: '/',
            },
            {
              label: 'Getting Started',
              to: '/guides/getting-started/',
            },
            {
              label: 'Runtime Architecture',
              to: '/runtime/architecture/',
            },
          ],
        },
        {
          title: 'Reference',
          items: [
            {
              label: 'docs.rs',
              href: 'https://docs.rs/releases/search?query=fastllm',
            },
            {
              label: 'Typed Configuration',
              to: '/reference/typed-configuration/',
            },
          ],
        },
        {
          title: 'Project',
          items: [
            {
              label: 'Repository',
              href: 'https://github.com/liquidos-ai/FastLLM',
            },
            {
              label: 'Contributing',
              href: 'https://github.com/liquidos-ai/FastLLM',
            },
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} <a href="https://liquidos.ai" target="_blank" rel="noopener noreferrer">LiquidOS AI</a>`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'json', 'toml', 'rust'],
    },
  },
};

export default config;
