# GUI 文档（Tauri）

> 技术选型：Tauri v2（Rust 后端 + Vite+TS 前端）。**推理全部在 Rust 后端**，
> 经 `crates/infer` 的 `undistort_full` 入口；前端只做上传/展示，经 `invoke("undistort_image")` 调用。
> 无 Python 服务、无 HTTP 后端。

## 1. 目录

```
gui/
├── package.json / vite.config.ts / index.html
├── src/main.ts    # 多选批量 / invoke / 对比滑块 / 保存
├── src/styles.css # 对比叠加样式
└── src-tauri/src/main.rs  # undistort_image：标定/TPS/ONNX/倾角全参数透传
```

## 2. 运行（无需 Python）

```powershell
cd gui
npm install
npm run tauri dev
```

调用契约：`invoke("undistort_image", { imageBytes, task, mode, lambda, calibJson, angleDeg, onnxPath, tpsDeltasJson, tpsGrid })`
→ 返回 base64 PNG。`tps` 无 deltas / `checkpoint` 无 onnx 文件时返回明确错误，前端直接展示。

## 3. 功能

- 多选批量队列（点选切换首张，逐张推理，状态栏报进度）。
- 前后对比滑块（after 图层 `clip-path` 跟随 slider）。
- 标定面板：λ 快调 / calib.json 文件载入（identity/division/brown-conrady）/ T4 倾角。
- TPS / 权重：deltas.json + grid 输入、model.onnx 路径（权重就绪即用）。
- 保存结果 PNG；footer 关于（AGPL + GitHub 源码链接）。

## 4. 打包

```powershell
cd gui
npm run tauri build   # 产物在 src-tauri/target/release/bundle/
```

注意：AGPL 下分发安装包必须同步提供源码（放 GUI 关于页 + GitHub Release 源码包）。
MOWA 权重（NTU S-Lab License）**不得打进安装包**。
