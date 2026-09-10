<script setup lang="ts">
// CoolUndistort main app: state + layout (AGPL-3.0-or-later).
import { computed, reactive, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { ElMessage } from "element-plus";
import ControlRail from "./components/ControlRail.vue";
import CompareCanvas from "./components/CompareCanvas.vue";
import BatchStrip from "./components/BatchStrip.vue";

export interface QueueItem {
  id: number;
  file: File;
  url: string;
  resultUrl: string | null;
  lambda: number | null;
  done: boolean;
  failed: boolean;
}

const mode = ref("auto");
const task = ref("t2");
const lambda = ref(0.35);
const angleDeg = ref(0);
const grid = ref("10x12");
const onnxPath = ref("");
let calibText = "";
let deltasText = "";

const items = reactive<QueueItem[]>([]);
const activeId = ref<number | null>(null);
const activeIndex = computed(() => items.findIndex((i) => i.id === activeId.value));
const active = computed(() => (activeIndex.value >= 0 ? items[activeIndex.value] : null));

const running = ref(false);
const progress = ref(0);

function readText(f: File): Promise<string> {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result ?? ""));
    r.onerror = () => rej(r.error);
    r.readAsText(f);
  });
}

function addFiles(list: File[]) {
  for (const f of list) {
    items.push({ id: Date.now() + Math.random(), file: f, url: URL.createObjectURL(f), resultUrl: null, lambda: null, done: false, failed: false });
  }
  if (activeId.value === null && items.length) activeId.value = items[0].id;
}

function onPick(files: File[]) {
  if (!files.length) return;
  addFiles(files);
  ElMessage.success({ message: `已加入 ${files.length} 张`, grouping: true });
}

function selectItem(id: number) {
  activeId.value = id;
}

function removeItem(id: number) {
  const idx = items.findIndex((i) => i.id === id);
  if (idx >= 0) items.splice(idx, 1);
  if (activeId.value === id) activeId.value = items[0]?.id ?? null;
}

function clearAll() {
  items.splice(0);
  activeId.value = null;
}

async function loadCalib(f: File) {
  calibText = await readText(f);
  ElMessage.success(`已载入标定 ${f.name}`);
}

function clearCalib() {
  calibText = "";
}

async function loadDeltas(f: File) {
  deltasText = await readText(f);
  ElMessage.success(`已载入 TPS deltas ${f.name}`);
}

function clearDeltas() {
  deltasText = "";
}

async function runOne(item: QueueItem) {
  const buf = new Uint8Array(await item.file.arrayBuffer());
  if (mode.value === "auto") {
    const res = (await invoke("undistort_auto", {
      imageBytes: Array.from(buf),
      onnxPath: onnxPath.value,
    })) as { image: string; lambda: number };
    item.resultUrl = `data:image/png;base64,${res.image}`;
    item.lambda = res.lambda;
  } else {
    const png: string = await invoke("undistort_image", {
      imageBytes: Array.from(buf),
      task: task.value,
      mode: mode.value,
      lambda: lambda.value,
      calibJson: calibText,
      angleDeg: angleDeg.value,
      onnxPath: onnxPath.value,
      tpsDeltasJson: deltasText,
      tpsGrid: grid.value,
    });
    item.resultUrl = `data:image/png;base64,${png}`;
    item.lambda = null;
  }
  item.done = true;
  item.failed = false;
}

async function runAll() {
  if (!items.length) {
    ElMessage.warning("请先添加图片（支持多选批量）");
    return;
  }
  running.value = true;
  progress.value = 0;
  for (let i = 0; i < items.length; i++) {
    activeId.value = items[i].id;
    try {
      await runOne(items[i]);
    } catch (err) {
      items[i].failed = true;
      ElMessage.error(`[${items[i].file.name}] ${String(err)}`);
      running.value = false;
      return;
    }
    progress.value = Math.round(((i + 1) / items.length) * 100);
  }
  running.value = false;
  const auto = mode.value === "auto" && active.value?.lambda != null;
  ElMessage.success(`完成 ${items.length} 张 · Rust 本地推理${auto ? ` · λ≈${active.value!.lambda!.toFixed(4)}` : ""}`);
}

function saveResult() {
  if (!active.value?.resultUrl) {
    ElMessage.info("当前图片还没有结果");
    return;
  }
  const a = document.createElement("a");
  a.href = active.value.resultUrl;
  a.download = `${active.value.file.name.replace(/\.[^.]+$/, "")}_undistorted.png`;
  a.click();
}
</script>

<template>
  <div class="bench">
    <header class="topbar rise rise-1">
      <div class="brand">
        <svg class="mark" viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">
          <circle cx="16" cy="16" r="13" fill="none" stroke="var(--amber)" stroke-width="2" />
          <circle cx="16" cy="16" r="6.5" fill="none" stroke="var(--cyan)" stroke-width="1.4" />
          <path d="M16 3v6M16 23v6M3 16h6M23 16h6" stroke="var(--ink-faint)" stroke-width="1.2" />
        </svg>
        <div class="brandtext">
          <strong>COOLUNDISTORT</strong>
          <span class="mono sub">光学实验台 · v1.1</span>
        </div>
      </div>
      <div class="topmeta mono">
        <span class="dot" :class="running ? 'busy' : 'idle'" />
        {{ running ? "RUNNING" : "READY" }} · RUST LOCAL
      </div>
    </header>

    <div class="body">
      <ControlRail
        class="rise rise-2"
        v-model:mode="mode"
        v-model:task="task"
        v-model:lambda="lambda"
        v-model:angle-deg="angleDeg"
        v-model:grid="grid"
        v-model:onnx-path="onnxPath"
        :has-selection="items.length > 0"
        :running="running"
        :has-result="!!active?.resultUrl"
        @pick="onPick"
        @calib="loadCalib"
        @calib-clear="clearCalib"
        @deltas="loadDeltas"
        @deltas-clear="clearDeltas"
        @run="runAll"
        @save="saveResult"
      />

      <main class="stage rise rise-3">
        <CompareCanvas :item="active" :mode="mode" />
      </main>
    </div>

    <BatchStrip
      class="rise rise-4"
      :items="items"
      :active-id="activeId"
      :progress="progress"
      :running="running"
      @select="selectItem"
      @remove="removeItem"
      @clear="clearAll"
    />
  </div>
</template>

<style scoped>
.bench { height: 100%; display: flex; flex-direction: column; padding: 14px 18px 12px; box-sizing: border-box; gap: 12px; }

.topbar { display: flex; justify-content: space-between; align-items: center; }
.brand { display: flex; align-items: center; gap: 12px; }
.mark { filter: drop-shadow(0 0 8px rgba(255, 180, 84, 0.35)); }
.brandtext { display: flex; flex-direction: column; line-height: 1.15; }
.brandtext strong { font-weight: 800; letter-spacing: 0.14em; font-size: 15px; }
.brandtext .sub { color: var(--ink-faint); font-size: 11px; letter-spacing: 0.08em; }
.topmeta { color: var(--ink-dim); font-size: 12px; display: flex; align-items: center; gap: 8px; letter-spacing: 0.1em; }
.dot { width: 8px; height: 8px; border-radius: 50%; }
.dot.idle { background: var(--good); box-shadow: 0 0 8px rgba(130, 217, 146, 0.6); }
.dot.busy { background: var(--amber); box-shadow: 0 0 8px rgba(255, 180, 84, 0.7); animation: glowpulse 1.2s infinite; }

.body { flex: 1; display: flex; gap: 14px; min-height: 0; }
.stage { flex: 1; min-width: 0; display: flex; }
</style>
