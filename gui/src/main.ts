// CoolUndistort GUI: upload -> invoke undistort_image (Rust) -> before/after view.
import { invoke } from "@tauri-apps/api/core";

const el = (id: string) => document.getElementById(id) as HTMLElement;
const before = el("before") as HTMLImageElement;
const after = el("after") as HTMLImageElement;
const statusEl = el("status");

let file: File | null = null;
(el("file") as HTMLInputElement).addEventListener("change", (e) => {
  file = (e.target as HTMLInputElement).files?.[0] ?? null;
  if (file) before.src = URL.createObjectURL(file);
});

document.getElementById("run")?.addEventListener("click", async () => {
  if (!file) {
    statusEl.textContent = "请先选择图片";
    return;
  }
  statusEl.textContent = "推理中...";
  try {
    const buf = new Uint8Array(await file.arrayBuffer());
    const png: string = await invoke("undistort_image", {
      imageBytes: Array.from(buf),
      task: (el("task") as HTMLSelectElement).value,
      mode: (el("mode") as HTMLSelectElement).value,
      lambda: parseFloat((el("lambda") as HTMLInputElement).value) || 0.35,
    });
    after.src = `data:image/png;base64,${png}`;
    statusEl.textContent = "完成（Rust 本地推理）";
  } catch (err) {
    statusEl.textContent = `失败: ${String(err)}`;
  }
});
