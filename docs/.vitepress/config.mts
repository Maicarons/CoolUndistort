import { defineConfig } from "vitepress";

export default defineConfig({
  lang: "zh-CN",
  title: "CoolUndistort",
  description:
    "统一图像去畸变与矩形整形：Rust 推理 + Python 训练 + Tauri GUI。AutoLambda 盲估计 λ，开箱即用。",
  base: "/CoolUndistort/",
  head: [["link", { rel: "icon", type: "image/svg+xml", href: "/favicon.svg" }]],
  themeConfig: {
    nav: [
      { text: "指南", link: "/DEVELOPMENT", activeMatch: "/DEVELOPMENT" },
      { text: "训练", link: "/TRAINING" },
      { text: "GUI", link: "/GUI" },
      { text: "研究", link: "/research" },
      { text: "GitHub", link: "https://github.com/Maicarons/CoolUndistort" }
    ],
    sidebar: [
      {
        text: "项目",
        items: [
          { text: "开发文档", link: "/DEVELOPMENT" },
          { text: "训练", link: "/TRAINING" },
          { text: "GUI（Tauri）", link: "/GUI" }
        ]
      },
      {
        text: "研究",
        items: [{ text: "UniRect 论文深读报告", link: "/research" }]
      }
    ],
    socialLinks: [{ icon: "github", link: "https://github.com/Maicarons/CoolUndistort" }],
    outline: { level: [2, 3], label: "本页目录" },
    docFooter: { prev: "上一页", next: "下一页" },
    lastUpdated: { text: "最后更新" },
    returnToTopLabel: "回到顶部",
    sidebarMenuLabel: "目录"
  }
});
