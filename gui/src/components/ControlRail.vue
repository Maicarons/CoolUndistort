<script setup lang="ts">
// Left control rail: upload / mode / task / calibration / actions (AGPL-3.0-or-later).
import { computed, ref } from "vue";

const props = defineProps<{
  mode: string;
  task: string;
  lambda: number;
  angleDeg: number;
  grid: string;
  onnxPath: string;
  hasSelection: boolean;
  running: boolean;
  hasResult: boolean;
}>();

const emit = defineEmits<{
  (e: "update:mode", v: string): void;
  (e: "update:task", v: string): void;
  (e: "update:lambda", v: number): void;
  (e: "update:angleDeg", v: number): void;
  (e: "update:grid", v: string): void;
  (e: "update:onnxPath", v: string): void;
  (e: "pick", files: File[]): void;
  (e: "calib", f: File): void;
  (e: "calibClear"): void;
  (e: "deltas", f: File): void;
  (e: "deltasClear"): void;
  (e: "run"): void;
  (e: "save"): void;
}>();

const modes = [
  { value: "auto", label: "自动盲估计", hint: "内置权重 · 无需标定" },
  { value: "fisheye", label: "除法模型", hint: "手动 λ / 标定文件" },
  { value: "tps", label: "RP-TPS", hint: "控制点网格" },
  { value: "checkpoint", label: "UniRect 权重", hint: "实验性" },
];

const tasks = [
  { value: "t1", label: "T1 人像校正" },
  { value: "t2", label: "T2 广角整形" },
  { value: "t3", label: "T3 拼接整形" },
  { value: "t4", label: "T4 旋转校正" },
];

const calibName = ref("");
const deltasName = ref("");
const pickRef = ref<HTMLInputElement>();
const calibRef = ref<HTMLInputElement>();
const deltasRef = ref<HTMLInputElement>();

const modeHint = computed(() => modes.find((m) => m.value === props.mode)?.hint ?? "");
const dragging = ref(false);

function onPickChange(e: Event) {
  const files = Array.from((e.target as HTMLInputElement).files ?? []);
  if (files.length) emit("pick", files);
  (e.target as HTMLInputElement).value = "";
}
function onCalibChange(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) {
    calibName.value = f.name;
    emit("calib", f);
  }
  (e.target as HTMLInputElement).value = "";
}
function onDeltasChange(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) {
    deltasName.value = f.name;
    emit("deltas", f);
  }
  (e.target as HTMLInputElement).value = "";
}
function onDrop(e: DragEvent) {
  dragging.value = false;
  const files = Array.from(e.dataTransfer?.files ?? []).filter((f) => f.type.startsWith("image/"));
  if (files.length) emit("pick", files);
}
</script>

<template>
  <aside class="rail">
    <el-scrollbar>
      <section
        class="dropzone"
        :class="{ drag: dragging }"
        @click="pickRef?.click()"
        @dragover.prevent="dragging = true"
        @dragleave="dragging = false"
        @drop.prevent="onDrop"
      >
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.6">
          <rect x="3" y="3" width="18" height="18" rx="3" />
          <circle cx="9" cy="9" r="2" />
          <path d="M3 17l5-5 4 4 3-3 6 6" />
        </svg>
        <strong>{{ hasSelection ? "继续添加图片" : "点击 / 拖入图片" }}</strong>
        <span class="mono">支持多选批量 · PNG/JPG</span>
        <input ref="pickRef" type="file" accept="image/*" multiple hidden @change="onPickChange" />
      </section>

      <section class="card">
        <h3 class="mono">01 · 模式</h3>
        <div class="modes">
          <button
            v-for="m in modes"
            :key="m.value"
            class="modebtn"
            :class="{ on: mode === m.value }"
            type="button"
            @click="emit('update:mode', m.value)"
          >
            <strong>{{ m.label }}</strong>
            <span class="mono">{{ m.hint }}</span>
          </button>
        </div>
        <p class="hint mono">{{ modeHint }}</p>
      </section>

      <section v-if="mode !== 'auto'" class="card">
        <h3 class="mono">02 · 任务与参数</h3>
        <el-select :model-value="task" style="width: 100%" @update:model-value="emit('update:task', String($event))">
          <el-option v-for="t in tasks" :key="t.value" :value="t.value" :label="t.label" />
        </el-select>

        <template v-if="mode === 'fisheye'">
          <div class="fieldrow">
            <label class="mono">λ</label>
            <el-slider
              :model-value="lambda"
              :min="0"
              :max="0.6"
              :step="0.01"
              @update:model-value="emit('update:lambda', Number($event))"
            />
            <span class="readout mono">{{ lambda.toFixed(2) }}</span>
          </div>
          <div class="fieldrow">
            <label class="mono">T4倾角</label>
            <el-input-number
              :model-value="angleDeg"
              :step="0.5"
              size="small"
              @update:model-value="emit('update:angleDeg', Number($event ?? 0))"
            />
            <span class="readout mono">deg</span>
          </div>
          <div class="uploadrow">
            <el-button size="small" @click="calibRef?.click()">calib.json</el-button>
            <span class="mono filename">{{ calibName || "未载入" }}</span>
            <el-button v-if="calibName" link type="danger" size="small" @click="((calibName = ''), emit('calibClear'))">清除</el-button>
            <input ref="calibRef" type="file" accept=".json" hidden @change="onCalibChange" />
          </div>
        </template>

        <template v-if="mode === 'tps'">
          <div class="uploadrow">
            <el-button size="small" @click="deltasRef?.click()">deltas.json</el-button>
            <span class="mono filename">{{ deltasName || "未载入" }}</span>
            <el-button v-if="deltasName" link type="danger" size="small" @click="((deltasName = ''), emit('deltasClear'))">清除</el-button>
            <input ref="deltasRef" type="file" accept=".json" hidden @change="onDeltasChange" />
          </div>
          <div class="fieldrow">
            <label class="mono">grid</label>
            <el-input :model-value="grid" size="small" style="width: 100px" @update:model-value="emit('update:grid', String($event))" />
          </div>
        </template>
      </section>

      <section class="card">
        <h3 class="mono">03 · 权重路径</h3>
        <el-input
          :model-value="onnxPath"
          size="small"
          placeholder="weights/autolambda.onnx（默认内置）"
          @update:model-value="emit('update:onnxPath', String($event))"
        />
      </section>
    </el-scrollbar>

    <section class="actions">
      <el-button size="large" class="runbtn main" :loading="running" @click="emit('run')">
        {{ running ? "推理中…" : "开始去畸变" }}
      </el-button>
      <el-button size="large" :disabled="!hasResult" @click="emit('save')">保存当前结果</el-button>
    </section>
  </aside>
</template>

<style scoped>
.rail { width: 296px; flex: none; display: flex; flex-direction: column; min-height: 0; gap: 12px; }
.rail :deep(.el-scrollbar) { flex: 1; min-height: 0; }
.rail :deep(.el-scrollbar__view) { display: flex; flex-direction: column; gap: 12px; padding-right: 4px; }

.dropzone {
  display: flex; flex-direction: column; align-items: center; gap: 6px;
  padding: 20px 12px;
  border: 1.5px dashed var(--line);
  border-radius: var(--radius);
  color: var(--ink-dim);
  cursor: pointer;
  transition: all 0.18s ease;
  background: rgba(255, 255, 255, 0.015);
}
.dropzone strong { font-size: 13.5px; font-weight: 700; color: var(--ink); }
.dropzone .mono { font-size: 10.5px; color: var(--ink-faint); }
.dropzone:hover, .dropzone.drag { border-color: var(--amber); color: var(--amber); background: rgba(255, 180, 84, 0.05); }
.dropzone.drag strong { color: var(--amber); }

.card {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), transparent 40%), var(--bg-1);
  border: 1px solid var(--line-soft);
  border-radius: var(--radius);
  padding: 14px 14px 12px;
}
.card h3 {
  margin: 0 0 10px;
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.18em;
  color: var(--amber);
}

.modes { display: flex; flex-direction: column; gap: 6px; }
.modebtn {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 9px 12px;
  background: var(--bg-2);
  border: 1px solid var(--line-soft);
  border-radius: 8px;
  color: var(--ink-dim);
  cursor: pointer;
  transition: all 0.15s ease;
  font-family: var(--font-body);
}
.modebtn strong { font-size: 13px; font-weight: 600; color: inherit; }
.modebtn .mono { font-size: 10px; color: var(--ink-faint); }
.modebtn:hover { border-color: var(--ink-faint); color: var(--ink); }
.modebtn.on {
  background: rgba(255, 180, 84, 0.08);
  border-color: var(--amber-deep);
  color: var(--ink);
  box-shadow: inset 3px 0 0 var(--amber);
}
.modebtn.on .mono { color: var(--amber); }
.hint { margin: 10px 2px 0; font-size: 10.5px; color: var(--ink-faint); }

.fieldrow { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
.fieldrow :deep(.el-slider) { flex: 1; }
.fieldrow label { width: 46px; font-size: 11px; color: var(--ink-dim); flex: none; }
.readout { font-size: 12px; color: var(--cyan); min-width: 40px; text-align: right; }
.uploadrow { display: flex; align-items: center; gap: 8px; margin-top: 12px; }
.filename { font-size: 11px; color: var(--ink-faint); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }

.actions { display: flex; flex-direction: column; gap: 8px; padding-bottom: 4px; }
.runbtn { width: 100%; }
.runbtn.main {
  font-weight: 700;
  background: linear-gradient(135deg, var(--amber), var(--amber-deep));
  border: none;
  color: #1a1206;
  animation: glowpulse 2.6s infinite;
}
.runbtn.main:hover { filter: brightness(1.08); }
</style>
