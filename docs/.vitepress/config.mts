import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Parallel Image Reader',
  description: 'KiraAI 并行识图插件 — 并发调用 VLM 描述图片',
  lang: 'zh-CN',
  base: '/kira-plugin-parallel-image-reader/',

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '指南', link: '/guide/install' },
      { text: 'GitHub', link: 'https://github.com/CelestNya/kira-plugin-parallel-image-reader' },
    ],

    sidebar: [
      {
        text: '指南',
        items: [
          { text: '安装', link: '/guide/install' },
          { text: '工作流程', link: '/guide/workflow' },
        ],
      },
      {
        text: '配置',
        items: [
          { text: '选项说明', link: '/config/options' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/CelestNya/kira-plugin-parallel-image-reader' },
    ],

    footer: {
      message: 'MIT License',
      copyright: 'Copyright 2026 CelestNya',
    },
  },
})
