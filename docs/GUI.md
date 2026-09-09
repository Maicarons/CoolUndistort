# GUI 文档（Tauri）

> 技术选型：Tauri v2（Rust 后端 + Vite+TS 前端）。**推理全部在 Rust 后端**，
> 经 `crates/infer` 的 `undistort` 入口；前端只做上传/展示，经 `invoke("undistort_image")` 调用。
> 无 Python 服务、无 HTTP 后端。

## 1. 目录

```
gui/
├── package.json            # 前端工程（vite + @tauri-apps/cli）
├── vite.config.ts
├── index.html              # 任务下拉 T1-T4 / mode / λ 输入 / 前后对比
├── src/
│   ├── main.ts             # 上传 / invoke("undistort_image") / base64 回显
│   └── styles.css
└── src-tauri/
    ├── Cargo.toml          # 依赖 coolundistort-infer（path 引用 workspace 外，保持与 CLI 同一核）
    ├── tauri.conf.json
    └── src/main.rs         # undistort_image command：解码 → undistort → 编码 PNG → base64
```

## 2. 运行（无需 Python）

```powershell
cd gui
npm install
npm run tauri dev
```

调用契约：`invoke("undistort_image", { imageBytes: number[], task: "t1"|"t2"|"t3"|"t4", mode: "fisheye"|"tps"|"checkpoint", lambda: number })`
→ 返回 base64 PNG。`tps`/`checkpoint` 在 P1 权重就绪前返回 `NeedsWeights` 错误，前端直接展示。

## 3. 功能清单

- P0：图片上传 + 预览、任务下拉（T1–T4）、mode 选择、λ 输入、Rust 本地推理、前后对比展示、关于页（AGPL 文本 + 源码链接）。
- P1：prompt 可视化叠加、批量处理（复用同一 `undistort` 入口循环调）、K/D 标定面板（参数进 `CameraParams`）。
- P2：ONNX 轻量模型内置（仍在 Rust 内加载推理，不引入 Python 运行时）。

## 4. 打包

```powershell
cd gui
npm run tauri build   # 产物在 src-tauri/target/release/bundle/
```

注意：AGPL 下分发安装包必须同步提供源码（放 GUI 关于页 + GitHub Release 源码包）。
MOWA 权重（NTU S-Lab License）**不得打进安装包**。
