// CoolUndistort GUI: upload(+batch) -> invoke undistort_image (Rust) -> compare slider.
import { invoke } from "@tauri-apps/api/core";

const el = (id: string) => document.getElementById(id) as HTMLElement;
const before = el("before") as HTMLImageElement;
const after = el("after") as HTMLImageElement;
const statusEl = el("status");
const queueEl = el("queue") as HTMLUListElement;
const slider = el("slider") as HTMLInputElement;

let files: File[] = [];
let calibText = "";
let deltasText = "";
let lastResultUrl: string | null = null;
let lastLambda: number | null = null;

slider.addEventListener("input", () => {
  after.style.clipPath = `inset(0 0 0 ${slider.value}%)`;
});
after.style.clipPath = "inset(0 0 0 50%)";

async function readText(f: File): Promise<string> {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result ?? ""));
    r.onerror = () => rej(r.error);
    r.readAsText(f);
  });
}

function renderQueue() {
  queueEl.innerHTML = "";
  files.forEach((f, i) => {
    const li = document.createElement("li");
    li.textContent = `${i + 1}. ${f.name}`;
    li.onclick = () => {
      before.src = URL.createObjectURL(f);
      files = [f, ...files.filter((_, j) => j !== i)];
      renderQueue();
    };
    queueEl.appendChild(li);
  });
}

(el("file") as HTMLInputElement).addEventListener("change", (e) => {
  const list = Array.from((e.target as HTMLInputElement).files ?? []);
  if (!list.length) return;
  files = [...files, ...list];
  before.src = URL.createObjectURL(list[0]);
  renderQueue();
});
(el("clear") as HTMLButtonElement).addEventListener("click", () => {
  files = [];
  renderQueue();
});
(el("calib") as HTMLInputElement).addEventListener("change", async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  calibText = f ? await readText(f) : "";
  statusEl.textContent = f ? `已载入标定 ${f.name}` : "已清除标定";
});
(el("deltas") as HTMLInputElement).addEventListener("change", async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  deltasText = f ? await readText(f) : "";
  statusEl.textContent = f ? `已载入 TPS deltas ${f.name}` : "已清除 deltas";
});

async function runOne(f: File): Promise<void> {
  const buf = new Uint8Array(await f.arrayBuffer());
  const mode = (el("mode") as HTMLSelectElement).value;
  if (mode === "auto") {
    const res = (await invoke("undistort_auto", {
      imageBytes: Array.from(buf),
      onnxPath: (el("onnx") as HTMLInputElement).value,
    })) as { image: string; lambda: number };
    lastResultUrl = `data:image/png;base64,${res.image}`;
    after.src = lastResultUrl;
    lastLambda = res.lambda;
    return;
  }
  const png: string = await invoke("undistort_image", {
    imageBytes: Array.from(buf),
    task: (el("task") as HTMLSelectElement).value,
    mode,
    lambda: parseFloat((el("lambda") as HTMLInputElement).value) || 0.35,
    calibJson: calibText,
    angleDeg: parseFloat((el("angle") as HTMLInputElement).value) || 0,
    onnxPath: (el("onnx") as HTMLInputElement).value,
    tpsDeltasJson: deltasText,
    tpsGrid: (el("grid") as HTMLInputElement).value,
  });
  lastResultUrl = `data:image/png;base64,${png}`;
  after.src = lastResultUrl;
  lastLambda = null;
}

(el("run") as HTMLButtonElement).addEventListener("click", async () => {
  if (!files.length) {
    statusEl.textContent = "请先选择图片（可多选批量）";
    return;
  }
  for (let i = 0; i < files.length; i++) {
    statusEl.textContent = `推理中 ${i + 1}/${files.length} ...`;
    try {
      before.src = URL.createObjectURL(files[i]);
      await runOne(files[i]);
    } catch (err) {
      statusEl.textContent = `失败 [${files[i].name}]: ${String(err)}`;
      return;
    }
  }
  const suffix = lastLambda !== null ? `，λ≈${lastLambda.toFixed(4)}` : "";
  statusEl.textContent = `完成 ${files.length} 张（Rust 本地推理${suffix}）`;
});

(el("save") as HTMLButtonElement).addEventListener("click", () => {
  if (!lastResultUrl) {
    statusEl.textContent = "暂无结果可保存";
    return;
  }
  const a = document.createElement("a");
  a.href = lastResultUrl;
  a.download = "undistorted.png";
  a.click();
});
