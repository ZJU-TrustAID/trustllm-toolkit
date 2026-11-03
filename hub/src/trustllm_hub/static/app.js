/* TrustLLM Toolkit Hub — 共享工具：API 封装、公共参数存取、轻量 markdown 渲染 */
"use strict";

const HUB = {
  async api(path, options) {
    const response = await fetch(path, Object.assign({
      headers: { "Content-Type": "application/json" },
    }, options || {}));
    let payload = null;
    try { payload = await response.json(); } catch (e) { /* non-JSON */ }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
      throw new Error(detail);
    }
    return payload;
  },
  get(path) { return this.api(path); },
  post(path, body) {
    return this.api(path, { method: "POST", body: JSON.stringify(body || {}) });
  },
};

/* ---- 公共参数（localStorage 持久化，跨页面共享） ---- */
const CommonStore = {
  KEY: "hub.common.v1",
  SOURCES: ["local_path", "hf", "api"],
  defaults() {
    return {
      model: { source: "local_path", path: "", repo: "", api: { base_url: "", api_key_env: "OPENAI_API_KEY", model: "" } },
      device: "auto",
      dtype: "auto",
      seed: 0,
    };
  },
  load() {
    const base = this.defaults();
    try {
      const value = JSON.parse(localStorage.getItem(this.KEY));
      if (value && typeof value === "object") {
        // 浅合并顶层、深合并 model：任何旧版本/残缺的存储都不会让页面缺字段。
        const merged = Object.assign(base, value);
        merged.model = Object.assign(this.defaults().model, value.model || {});
        merged.model.api = Object.assign(this.defaults().model.api, value.model && value.model.api || {});
        if (this.SOURCES.indexOf(merged.model.source) < 0) merged.model.source = "local_path";
        return merged;
      }
    } catch (e) { /* 存储损坏时回退默认值 */ }
    return base;
  },
  save(common) {
    localStorage.setItem(this.KEY, JSON.stringify(common));
    // 章节页的参数卡片需要感知顶栏公共参数变化（继承项实时跟随）。
    window.dispatchEvent(new CustomEvent("hub:common-changed"));
  },
  entry(common) {
    const entry = { source: common.model.source };
    if (common.model.source === "local_path") entry.path = (common.model.path || "").trim();
    if (common.model.source === "hf") entry.repo = (common.model.repo || "").trim();
    if (common.model.source === "api") entry.api = Object.assign({}, common.model.api);
    return entry;
  },
};

/* 全页共享的单例公共参数（shell.js / index.js / chapter.js 都引用它） */
const common = CommonStore.load();

/* ---- 状态徽章 ---- */
const STATE_CHIP = {
  running: ["amber", "运行中"],
  succeeded: ["green", "已完成"],
  failed: ["red", "失败"],
  timeout: ["red", "超时"],
  stopped: ["gray", "已停止"],
  ready: ["green", "就绪"],
  starting: ["amber", "启动中"],
  stopped_server: ["gray", "未启动"],
  unknown: ["gray", "未知"],
  unsupported: ["gray", "—"],
};
function stateChip(state) {
  const [cls, label] = STATE_CHIP[state] || STATE_CHIP.unknown;
  return `<span class="chip ${cls}">${label}</span>`;
}

/* ---- 极简 markdown 渲染（转义后构建，无外部依赖） ---- */
function escapeHtml(text) {
  return String(text == null ? "" : text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function inlineMd(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function mdToHtml(markdown) {
  const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const buf = [];
      i += 1;
      while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i += 1; }
      i += 1;
      html.push(`<pre><code>${escapeHtml(buf.join("\n"))}</code></pre>`);
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      const level = Math.min(heading[1].length + 1, 5);
      html.push(`<h${level}>${inlineMd(heading[2])}</h${level}>`);
      i += 1; continue;
    }
    if (/^\s*\|(.+)\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const header = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|(.+)\|\s*$/.test(lines[i])) { rows.push(cells(lines[i])); i += 1; }
      html.push("<table><thead><tr>" + header.map((c) => `<th>${inlineMd(c)}</th>`).join("") + "</tr></thead><tbody>"
        + rows.map((r) => "<tr>" + r.map((c) => `<td>${inlineMd(c)}</td>`).join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }
    if (/^\s*([-*])\s+/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*([-*])\s+/.test(lines[i])) { buf.push(lines[i].replace(/^\s*([-*])\s+/, "")); i += 1; }
      html.push("<ul>" + buf.map((item) => `<li>${inlineMd(item)}</li>`).join("") + "</ul>");
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) { buf.push(lines[i].replace(/^\s*\d+\.\s+/, "")); i += 1; }
      html.push("<ol>" + buf.map((item) => `<li>${inlineMd(item)}</li>`).join("") + "</ol>");
      continue;
    }
    if (/^\s*>/.test(line)) {
      html.push(`<blockquote>${inlineMd(line.replace(/^\s*>\s?/, ""))}</blockquote>`);
      i += 1; continue;
    }
    if (line.trim() === "") { i += 1; continue; }
    html.push(`<p>${inlineMd(line)}</p>`);
    i += 1;
  }
  return html.join("\n");
}

/* ---- 其他工具 ---- */
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (key === "class") node.className = value;
      else if (key === "html") node.innerHTML = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value);
    }
  }
  for (const child of children) {
    if (child == null) continue;
    node.append(child.nodeType ? child : document.createTextNode(child));
  }
  return node;
}
function toast(message, isError) {
  document.querySelectorAll(".toast").forEach((node) => node.remove());
  const node = el("div", { class: "toast" + (isError ? " err" : "") }, message);
  document.body.append(node);
  setTimeout(() => node.remove(), isError ? 4200 : 2400);
}
function fmtElapsed(startedAt, endedAt) {
  if (!startedAt) return "";
  const start = new Date(startedAt).getTime();
  const end = endedAt ? new Date(endedAt).getTime() : Date.now();
  let seconds = Math.max(0, Math.round((end - start) / 1000));
  const h = Math.floor(seconds / 3600); seconds %= 3600;
  const m = Math.floor(seconds / 60); const s = seconds % 60;
  return (h ? `${h}:` : "") + `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
function qs(name) { return new URLSearchParams(location.search).get(name); }

/* ---- 设备下拉（auto/cpu + nvidia-smi 枚举的 GPU，含空闲显存） ---- */
function fillDeviceOptions(select) {
  select.append(el("option", { value: "auto" }, "auto（自动·选空闲显存最大的卡）"));
  select.append(el("option", { value: "cpu" }, "cpu"));
  return HUB.get("/api/devices").then((info) => {
    for (const gpu of info.gpus || []) {
      const value = `cuda:${gpu.index}`;
      const free = (gpu.memory_free_mb / 1024).toFixed(1);
      const total = (gpu.memory_total_mb / 1024).toFixed(0);
      select.append(el("option", { value }, `${value} · ${gpu.name}（空闲 ${free} GB / ${total} GB）`));
    }
  }).catch(() => {
    for (let i = 0; i < 8; i++) select.append(el("option", { value: `cuda:${i}` }, `cuda:${i}`));
  });
}
/* 选中存在的选项；不存在（如换主机后残留的 cuda:N）时回退 fallback */
function selectValue(select, value, fallback) {
  const hit = [...select.options].some((option) => option.value === value);
  select.value = hit ? value : (fallback === undefined ? "auto" : fallback);
}

/* ---- 「已检测到的本地模型」下拉：models/ 目录扫描，选中写入 input 并回调 ---- */
async function attachLocalModelSelect(container, input, onPick, label = "已检测到的本地模型") {
  let models;
  try { models = await HUB.get("/api/models/local"); } catch (err) { return; }
  if (!models || !models.length) return;
  const select = el("select", { style: "margin-top:8px" });
  select.append(el("option", { value: "" }, "从 models/ 目录选择已下载的模型…"));
  for (const model of models) select.append(el("option", { value: model.path }, `${model.name}（${model.path}）`));
  if (input && models.some((m) => m.path === input.value)) select.value = input.value;
  select.addEventListener("change", () => {
    if (!select.value) return;
    if (input) input.value = select.value;
    if (onPick) onPick(select.value);
  });
  container.append(el("div", { class: "field-label", style: "margin-top:10px" }, label), select);
}

/* 结果块渲染（result.json 的 blocks 数组） */
function renderBlocks(container, blocks) {
  for (const block of blocks || []) {
    const wrap = el("div", { class: "result-block" });
    if (block.title) wrap.append(el("h3", null, block.title));
    const body = el("div", { class: "md" });
    if (block.type === "markdown") body.innerHTML = mdToHtml(block.content);
    else if (block.type === "table") {
      const head = block.header || [];
      body.innerHTML = "<table><thead><tr>" + head.map((c) => `<th>${escapeHtml(c)}</th>`).join("") +
        "</tr></thead><tbody>" + (block.rows || []).map((row) =>
          "<tr>" + row.map((c) => `<td>${escapeHtml(c)}</td>`).join("") + "</tr>").join("") + "</tbody></table>";
    } else if (block.type === "code") {
      body.innerHTML = `<pre><code>${escapeHtml(block.content)}</code></pre>`;
    } else { body.innerHTML = mdToHtml(String(block.content == null ? "" : block.content)); }
    wrap.append(body);
    container.append(wrap);
  }
}
