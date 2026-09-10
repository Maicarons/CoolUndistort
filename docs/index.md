---
layout: home

hero:
  name: CoolUndistort
  text: 统一图像去畸变与矩形整形
  tagline: 推理全部 Rust · 训练 Python · Tauri 桌面 GUI —— 内置 AutoLambda 盲估计权重，无需标定，开箱即用
  actions:
    - theme: brand
      text: 快速开始
      link: /DEVELOPMENT
    - theme: alt
      text: 训练指南
      link: /TRAINING
    - theme: alt
      text: 论文研究
      link: /research
  image:
    src: /lens.svg
    alt: CoolUndistort

features:
  - icon: 🔍
    title: 自动盲去畸变
    details: 单图盲估计除法模型 λ（tract 纯 Rust 加载内置 ONNX），CLI 侧 6 点 sweep MAE 0.0147，无需标定板。
  - icon: ⚙️
    title: 完整几何工具箱
    details: identity / division / Brown-Conrady 标定模型、TPS 闭式解 warp、T4 旋转校正，单张与批量一视同仁。
  - icon: 🖥️
    title: Tauri 桌面 GUI
    details: Vue 3 + Element Plus「光学实验台」界面：拖拽批量、前后对比滑块、实时 λ 读数、一键保存。
  - icon: 🧪
    title: 可复现训练
    details: AutoLambda 多尺度合成训练 + UniRect-lite（RP-TPS + Mamba）论文复现线，断点续训，ONNX 一键导出。
---
