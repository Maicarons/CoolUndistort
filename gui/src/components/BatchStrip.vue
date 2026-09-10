<script setup lang="ts">
// Bottom batch queue strip: thumbnails, per-item state, progress (AGPL-3.0-or-later).
import type { QueueItem } from "../App.vue";

defineProps<{
  items: QueueItem[];
  activeId: number | null;
  progress: number;
  running: boolean;
}>();

const emit = defineEmits<{ (e: "select", id: number): void; (e: "remove", id: number): void; (e: "clear"): void }>();

function name(f: File): string {
  const n = f.name;
  return n.length > 18 ? n.slice(0, 8) + "…" + n.slice(-8) : n;
}
</script>

<template>
  <footer class="strip">
    <div class="head">
      <span class="mono label">QUEUE · {{ items.length }}</span>
      <el-progress
        v-if="running || progress > 0"
        :percentage="progress"
        :stroke-width="6"
        :show-text="false"
        :duration="8"
        style="flex: 1; max-width: 300px"
        status="success"
      />
      <el-button v-if="items.length" link size="small" type="danger" @click="emit('clear')">清空</el-button>
    </div>

    <div v-if="items.length" class="thumbs">
      <div
        v-for="it in items"
        :key="it.id"
        class="thumb"
        :class="{ active: it.id === activeId, done: it.done, failed: it.failed }"
        @click="emit('select', it.id)"
      >
        <img :src="it.resultUrl ?? it.url" alt="" draggable="false" />
        <span class="mono name">{{ name(it.file) }}</span>
        <span v-if="it.lambda != null" class="mono lam">λ{{ it.lambda.toFixed(2) }}</span>
        <span v-if="it.done || it.failed" class="state mono" :class="it.failed ? 'bad' : 'good'">{{ it.failed ? "✕" : "✓" }}</span>
        <button class="rm" type="button" title="移除" @click.stop="emit('remove', it.id)">×</button>
      </div>
    </div>
    <div v-else class="mono placeholder">队列为空 — 从左上角拖入或点击添加图片</div>
  </footer>
</template>

<style scoped>
.strip {
  flex: none;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius);
  background: var(--bg-1);
  padding: 8px 12px 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.head { display: flex; align-items: center; gap: 14px; }
.label { font-size: 10px; letter-spacing: 0.2em; color: var(--ink-faint); }

.thumbs { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 2px; }
.thumb {
  position: relative; flex: none;
  width: 84px; cursor: pointer;
  border: 1.5px solid var(--line-soft); border-radius: 8px;
  overflow: hidden; background: var(--bg-2);
  transition: border-color 0.15s ease, transform 0.15s ease;
}
.thumb:hover { border-color: var(--ink-faint); transform: translateY(-2px); }
.thumb.active { border-color: var(--amber); box-shadow: 0 0 0 1px var(--amber-deep), 0 4px 14px rgba(255, 180, 84, 0.12); }
.thumb img { display: block; width: 100%; height: 56px; object-fit: cover; }
.name { display: block; font-size: 9px; color: var(--ink-faint); padding: 3px 6px 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lam { position: absolute; left: 5px; bottom: 22px; font-size: 9px; color: var(--amber); background: rgba(11, 13, 17, 0.8); padding: 1px 5px; border-radius: 4px; }
.state { position: absolute; top: 4px; right: 4px; font-size: 10px; width: 16px; height: 16px; border-radius: 50%; display: flex; align-items: center; justify-content: center; background: rgba(11, 13, 17, 0.85); }
.state.good { color: var(--good); border: 1px solid var(--good); }
.state.bad { color: var(--bad); border: 1px solid var(--bad); }
.rm {
  position: absolute; top: 3px; left: 3px;
  width: 16px; height: 16px; border-radius: 4px; border: none;
  background: rgba(11, 13, 17, 0.85); color: var(--ink-dim);
  font-size: 11px; line-height: 1; cursor: pointer; opacity: 0;
  transition: opacity 0.12s ease;
}
.thumb:hover .rm { opacity: 1; }
.rm:hover { color: var(--bad); }

.placeholder { font-size: 11px; color: var(--ink-faint); padding: 6px 2px; }
</style>
