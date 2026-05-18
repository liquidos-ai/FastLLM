import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'index',
    {
      type: 'category',
      label: 'Guides',
      items: ['guides/getting-started'],
    },
    {
      type: 'category',
      label: 'Runtime',
      items: ['runtime/architecture', 'runtime/local-models'],
    },
    {
      type: 'category',
      label: 'Reference',
      items: ['reference/typed-configuration'],
    },
  ],
};

export default sidebars;
