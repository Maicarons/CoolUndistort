<script setup lang="ts">
// Compare canvas: draggable before/after divider + readouts (AGPL-3.0-or-later).
import { computed, ref, watch } from "vue";
import type { QueueItem } from "../App.vue";

const props = defineProps<{ item: QueueItem | null; mode: string }>();

const pos = ref(50);
const wrap = ref<HTMLDivElement>();
const dragging = ref(false);

const hasImage = computed(() => !!props.item);

function onMove(e: PointerEvent) {
  if (!dragging.value || !wrap.value) return;
  const r = wrap.value.getBoundingClientRect();
  pos.value = Math.min(97, Math.max(3, ((e.clientX - r.left) / r.width) * 100));
}
function onDown(e: PointerEvent) {
  dragging.value = true;
  onMove(e);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
}
function onUp() {
  dragging.value = false;
  window.removeEventListener("pointermove", onMove);
  window.removeEventListener("pointerup", onUp);
}

watch(
  () => props.item?.id,
  () => {
    pos.value = 50;
  }
);
</script>

<template>
  <div class="canvas">
    <div v-if="!item" class="empty">
      <svg viewBox="0 0 64 64" width="72" height="72" fill="none" stroke-width="1.4">
        <circle cx="32" cy="32" r="26" stroke="var(--line)" />
        <circle cx="32" cy="32" r="14" stroke="var(--line)" />
        <path d="M32 6v10M32 48v10M6 32h10M48 32h10" stroke="var(--line)" />
        <circle cx="32" cy="32" r="4" fill="var(--amber)" stroke="none" opacity="0.9" />
      </svg>
      <p>把畸变图拖进来，看看它本来的样子</p>
      <span class="mono">DROP A DISTORTED IMAGE TO BEGIN</span>
    </div>

    <div v-else ref="wrap" class="viewer" @pointerdown="onDown">
      <img class="before" :src="item.url" alt="before" draggable="false" />
      <img
        v-if="item.resultUrl"
        class="after"
        :src="item.resultUrl"
        alt="after"
        draggable="false"
        :style="{ clipPath: `inset(0 0 0 ${pos}%)` }"
      />
      <div v-if="item.resultUrl" class="divider" :style="{ left: pos + '%' }">
        <div class="knob mono">⇔</div>
      </div>

      <div class="tag tl mono">原图 · DISTORTED</div>
      <div v-if="item.resultUrl" class="tag tr mono">结果 · RECTIFIED</div>

      <div v-if="item.resultUrl" class="readouts mono">
        <span v-if="mode === 'auto' && item.lambda != null" class="amber">λ ≈ {{ item.lambda.toFixed(4) }}</span>
        <span :class="item.failed ? 'bad' : 'good'">{{ item.failed ? "FAILED" : item.done ? "DONE" : "…" }}</span>
        <span>{{ item.file.name }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.canvas { flex: 1; min-width: 0; display: flex; border: 1px solid var(--line-soft); border-radius: var(--radius); overflow: hidden; background: var(--bg-1); }

.empty {
  flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px;
  animation: rise 0.6s 0.2s both;
}
.empty p { margin: 0; color: var(--ink-dim); font-size: 14px; }
.empty .mono { font-size: 10px; letter-spacing: 0.22em; color: var(--ink-faint); }

.viewer { position: relative; flex: 1; min-height: 0; user-select: none; background: #07080b; }
.viewer img {
  position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; pointer-events: none;
}
.viewer img.after { z-index: 2; }

.divider {
  position: absolute; top: 0; bottom: 0; width: 2px; z-index: 3;
  background: var(--amber);
  box-shadow: 0 0 14px rgba(255, 180, 84, 0.55);
  cursor: ew-resize;
}
.knob {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  width: 30px; height: 30px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  background: var(--bg-2); border: 1.5px solid var(--amber);
  color: var(--amber); font-size: 13px;
  box-shadow: 0 2px 14px rgba(0, 0, 0, 0.5);
}

.tag {
  position: absolute; top: 10px; z-index: 4;
  font-size: 10px; letter-spacing: 0.14em;
  padding: 4px 9px; border-radius: 5px;
  background: rgba(11, 13, 17, 0.72); border: 1px solid var(--line);
  color: var(--ink-dim); backdrop-filter: blur(6px);
}
.tag.tl { left: 10px; }
.tag.tr { right: 10px; color: var(--amber); border-color: var(--amber-deep); }

.readouts {
  position: absolute; left: 10px; bottom: 10px; z-index: 4;
  display: flex; gap: 14px; align-items: center;
  font-size: 11.5px; padding: 7px 12px; border-radius: 7px;
  background: rgba(11, 13, 17, 0.78); border: 1px solid var(--line);
  backdrop-filter: blur(6px);
}
.readouts .amber { color: var(--amber); font-weight: 600; }
.readouts .good { color: var(--good); }
.readouts .bad { color: var(--bad); }
.readouts span:last-child { color: var(--ink-faint); max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
