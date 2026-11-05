/* Signal Desk 真后端工作台（对话式）：发送提示词 → 模型回答 → 选方法 → 参数 → 结果卡片 */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (t) => String(t == null ? "" : t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const PRESET_PROMPTS = [
  "请介绍蓝鲸的食性。",
  "阿根廷一共赢过几次世界杯？",
  "纸是谁发明的？",
  "给爱尔兰代码 999 的国家是哪里？",
];

const TABS = {
  whitebox: {
    title: "白盒检测", panelTitle: "选择方法", runLabel: "开始检测",
    methods: {
      mtp: { title: "最小词元概率", params: {} },
      lntp: { title: "长度归一化词元概率", params: {} },
      token_entropy: { title: "词元熵", params: {} },
      eigenscore: { title: "EigenScore", params: {} },
      effective_rank: { title: "Effective Rank", params: {} },
    },
    groups: [
      { title: "基于词元概率检测", keys: ["mtp", "lntp"] },
      { title: "基于熵值检测", keys: ["token_entropy"] },
      { title: "基于表征分析检测", keys: ["eigenscore", "effective_rank"] },
    ],
  },
  blackbox: {
    title: "黑盒检测", panelTitle: "选择方法", runLabel: "开始检测",
    methods: {
      p_true: { title: "P(True)", params: {} },
      exact_match: { title: "精准匹配率", params: {} },
      response_cosine: { title: "余弦相似度", params: {} },
      semantic_entropy: { title: "语义熵", params: {} },
      selfcheck_ngram: { title: "SelfCheck-Ngram", params: { n: { label: "n-gram", type: "int", def: 1 } } },
      selfcheck_nli: { title: "SelfCheck-NLI", params: {} },
      bertscore: { title: "SelfCheck-BERTScore", params: {} },
    },
    groups: [
      { title: "基于模型自评估检测", keys: ["p_true"] },
      { title: "基于一致性检测", keys: ["exact_match", "response_cosine", "semantic_entropy", "selfcheck_ngram", "selfcheck_nli", "bertscore"] },
    ],
  },
  mitigation: {
    title: "缓解方法", panelTitle: "选择方法", runLabel: "开始缓解",
    methods: {
      dola: { title: "DoLa 层间对比解码", params: { relative_top: { label: "relative_top", type: "float", def: 0.1 } } },
      iti: { title: "ITI 推理激活干预", params: { probe: { label: "probe 产物路径", type: "text", def: "" }, top_k: { label: "top_k", type: "int", def: 48 }, alpha: { label: "alpha", type: "float", def: 15.0 } } },
      cad: { title: "CAD 上下文对比解码", params: { alpha: { label: "alpha", type: "float", def: 0.5 }, filter_top_p: { label: "filter_top_p", type: "float", def: 1.0 } } },
      cove: { title: "CoVe 验证链", params: { max_questions: { label: "验证问题数", type: "int", def: 6 }, max_answer_tokens: { label: "单答长度", type: "int", def: 96 } } },
    },
    groups: [
      { title: "事实性增强解码", keys: ["dola", "iti"] },
      { title: "忠实性增强解码", keys: ["cad"] },
      { title: "事后修正", keys: ["cove"] },
    ],
  },
};

/* 方法 → 论文出处与教材小节锚点（教材 6-5：幻觉的检测与缓解） */
const PAPER_REFS = {
  "whitebox.mtp": { title: "A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-confidence Generation (Varshney et al., 2023)", secNum: "6.5.1.2 基于词元概率检测", section: "#基于词元概率检测" },
  "whitebox.lntp": { title: "A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-confidence Generation (Varshney et al., 2023)", secNum: "6.5.1.2 基于词元概率检测", section: "#基于词元概率检测" },
  "whitebox.token_entropy": { title: "On Hallucination and Predictive Uncertainty in Conditional Language Generation (Xiao & Wang, 2021)", secNum: "6.5.1.3 基于熵值检测", section: "#基于熵值检测" },
  "whitebox.eigenscore": { title: "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection (Chen et al., 2024)", secNum: "6.5.1.4 基于表征分析检测", section: "#基于表征分析检测" },
  "whitebox.effective_rank": { title: "Revisiting Hallucination Detection with Effective Rank-based Uncertainty (Wang et al., 2025)", secNum: "6.5.1.4 基于表征分析检测", section: "#基于表征分析检测" },
  "blackbox.p_true": { title: "Language Models (Mostly) Know What They Know (Kadavath et al., 2022)", secNum: "6.5.1.7 基于模型自评估检测", section: "#基于模型自评估检测" },
  "blackbox.exact_match": { title: "Measuring and Narrowing the Compositionality Gap in Language Models (Press et al., 2023)", secNum: "6.5.1.8 基于一致性检测", section: "#基于一致性检测" },
  "blackbox.response_cosine": { title: "SelfCheckGPT: Zero-resource Black-box Hallucination Detection for Generative Large Language Models (Manakul et al., 2023)", secNum: "6.5.1.8 基于一致性检测", section: "#基于一致性检测" },
  "blackbox.semantic_entropy": { title: "Detecting Hallucinations in Large Language Models Using Semantic Entropy (Farquhar et al., 2024)", secNum: "6.5.1.8 基于一致性检测", section: "#基于一致性检测" },
  "blackbox.selfcheck_ngram": { title: "SelfCheckGPT: Zero-resource Black-box Hallucination Detection for Generative Large Language Models (Manakul et al., 2023)", secNum: "6.5.1.8 基于一致性检测", section: "#基于一致性检测" },
  "blackbox.selfcheck_nli": { title: "SelfCheckGPT: Zero-resource Black-box Hallucination Detection for Generative Large Language Models (Manakul et al., 2023)", secNum: "6.5.1.8 基于一致性检测", section: "#基于一致性检测" },
  "blackbox.bertscore": { title: "SelfCheckGPT: Zero-resource Black-box Hallucination Detection for Generative Large Language Models (Manakul et al., 2023)", secNum: "6.5.1.8 基于一致性检测", section: "#基于一致性检测" },
  "mitigation.dola": { title: "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models (Chuang et al., 2024)", secNum: "6.5.2.1 层间对比解码", section: "#层间对比解码" },
  "mitigation.iti": { title: "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (Li et al., 2023)", secNum: "6.5.2.2 推理激活干预", section: "#推理激活干预" },
  "mitigation.cad": { title: "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding (Shi et al., 2024)", secNum: "6.5.3 忠实性增强解码", section: "#忠实性增强解码" },
  "mitigation.cove": { title: "Chain-of-Verification Reduces Hallucination in Large Language Models (Dhuliawala et al., 2024)", secNum: "6.5.3 忠实性增强解码", section: "#忠实性增强解码" },
};

const state = {
  tab: "whitebox",
  method: null,
  records: [],          // 全部问答记录 [{ id, prompt, response, num_tokens, latency_seconds }]
  targetId: null,       // 手动选中的回答记录 id；null = 默认使用最新一条
  response: null,       // 当前目标回答（与 target 同步，供抽样等旧逻辑引用）
  samples: null,        // 黑盒文本样本
  samplesPrompt: null,  // 样本对应的提问（用于校验样本与所选回答是否配套）
  hiddenSampleKey: null, // 表征分析的抽样缓存键
  resultsByRecord: {},  // 每条回答的检测结果列表 { [recordId]: [{ title, html }] }
  activeResultIdx: {},  // 每条回答当前查看的结果下标（切回时停在上次查看处）
  busy: false,
};

/* 生成完成：登记问答记录，绑定气泡点选，回到默认「检测最新」 */
function commitAssistantAnswer(bubble, { prompt, response, num_tokens, latency_seconds }) {
  const record = {
    id: `rec-${state.records.length + 1}-${Date.now()}`,
    prompt, response, num_tokens, latency_seconds,
  };
  state.records.push(record);
  state.targetId = null; // 生成新回答后回到默认「检测最新」
  state.response = response;
  bubble.dataset.recordId = record.id;
  const row = bubble.closest(".rd-msg-row");
  row.classList.add("rd-targetable");
  row.addEventListener("click", () => {
    state.targetId = state.targetId === record.id ? null : record.id; // 再点一次取消，恢复检测最新
    renderTargetMarks();
  });
  renderTargetMarks();
}

/* 当前检测/缓解目标：手动选中的回答优先，否则默认最新一条 */
function currentTarget() {
  if (state.targetId) return state.records.find((r) => r.id === state.targetId) || null;
  return state.records[state.records.length - 1] || null;
}

/* 回答选中态：手动选中的气泡打勾高亮；未手动选时默认目标为最新一条 */
function renderTargetMarks() {
  const latest = state.records[state.records.length - 1];
  document.querySelectorAll("#rd-msglist .rd-msg-row").forEach((row) => {
    const bubble = row.querySelector(".rd-msg.assistant");
    const pick = row.querySelector(".rd-pick");
    if (!bubble || !pick || !bubble.dataset.recordId) return;
    const manual = state.targetId === bubble.dataset.recordId;
    row.classList.toggle("rd-picked", manual);
    row.classList.toggle("rd-latest", !manual && !!latest && latest.id === bubble.dataset.recordId);
  });
  const hint = $("rd-run-hint");
  const target = currentTarget();
  if (!target) hint.textContent = "先在左侧发送提示词，再选择方法运行。";
  else if (state.targetId) hint.textContent = `将检测/缓解你选中的第 ${state.records.indexOf(target) + 1} 轮回答；再次点击该气泡取消，恢复为检测最新。`;
  else hint.textContent = `默认将检测/缓解最新一次回答（第 ${state.records.length} 轮）；点击任意模型回答气泡可改选。`;
  renderResultsPanel(); // 目标回答变化 → 结果面板随之切换到该回答的结果集
}

/* ---------------- 基础 UI ---------------- */
function setBusy(busy) {
  state.busy = busy;
  for (const id of ["rd-btn-generate", "rd-btn-sample"]) $(id).disabled = busy;
  $("rd-btn-run").disabled = busy;
  if (!busy) updateRunButton(); // 解除忙碌后仍需遵守参数校验结果
}

/* 回答区（固定高度内滚）：用户消息回显 / 模型回答 */
function msgAppend(node) {
  const empty = document.querySelector("#rd-msglist .rd-empty");
  if (empty) empty.remove();
  $("rd-msglist").append(node);
  $("rd-msglist").scrollTop = $("rd-msglist").scrollHeight;
}
function addUserMsg(text) {
  msgAppend(el2("div", { class: "rd-msg user" }, esc(text)));
}
function beginAssistantMsg() {
  /* 回答行 = 左侧单选圆圈 + 气泡；生成完成后可点选为检测目标 */
  const row = el2("div", { class: "rd-msg-row" });
  const pick = el2("span", { class: "rd-pick", title: "点击选中此回答作为检测/缓解对象" });
  const node = el2("div", { class: "rd-msg assistant" }, "生成中…");
  row.append(pick, node);
  msgAppend(row);
  return node;
}
function el2(tag, attrs, html) {
  const node = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (html !== undefined) node.innerHTML = html;
  return node;
}

/* 结果显示在下方「检测结果」区域；清空时恢复占位提示 */
/* 「检测结果」区域：按回答记录保存结果，顶部的标签可在同一条回答的不同方法结果间切换。
   切换回答 → 展示该回答的结果集（没有则占位提示）；同一方法重跑 → 覆盖旧结果。 */
function renderResultsPanel() {
  const tabs = $("rd-result-tabs");
  const box = $("rd-result");
  const empty = $("rd-result-empty");
  const target = currentTarget();
  const list = target ? (state.resultsByRecord[target.id] || []) : [];
  tabs.innerHTML = "";
  if (!target || !list.length) {
    tabs.hidden = true;
    box.style.display = "none";
    box.innerHTML = "";
    empty.style.display = "";
    empty.textContent = target
      ? "对该回答运行方法后，检测结果展示在这里；不同方法的结果都会保留，可在上方标签间切换查看。"
      : "选择方法并运行后，检测结果展示在这里。";
    return;
  }
  const idx = state.activeResultIdx[target.id];
  const active = (typeof idx === "number" && idx >= 0 && idx < list.length) ? idx : list.length - 1;
  state.activeResultIdx[target.id] = active;
  list.forEach((item, i) => {
    const pill = el2("button", { class: "rd-result-tab" + (i === active ? " is-active" : "") }, esc(item.title));
    pill.addEventListener("click", () => { state.activeResultIdx[target.id] = i; renderResultsPanel(); });
    tabs.append(pill);
  });
  tabs.hidden = false;
  empty.style.display = "none";
  box.style.display = "";
  box.innerHTML = `<div class="rd-result-card">${list[active].html}</div>`;
}

function showResult(html, title) {
  const target = currentTarget();
  if (!target) return;
  const list = (state.resultsByRecord[target.id] ||= []);
  const item = { title: title || "结果", html };
  const existing = list.findIndex((it) => it.title === item.title);
  if (existing >= 0) list[existing] = item; // 同一方法重跑覆盖旧结果
  else list.push(item);
  state.activeResultIdx[target.id] = list.indexOf(item);
  renderResultsPanel();
}
function showError(message) {
  // 运行错误即时显示，不写入结果列表
  renderResultsPanel();
  const box = $("rd-result");
  box.style.display = "";
  box.innerHTML = `<div class="rd-result-card"><div class="rd-error">${esc(message)}</div></div>`;
}

async function api(action, payload) {
  const response = await fetch(`api/real/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

/* ---------------- 模型与辅助模型 ----------------
   模型/辅助模型来源由 hub 的章节参数卡片统一配置（服务启动时注入），
   首次真实调用时自动懒加载；这里只把「当前方法需要哪些辅助模型」
   实时同步给 hub 页面，用于启用/锁定卡片里对应的输入框。 */
const AUX_NEEDS = {
  nli: ["semantic_entropy", "selfcheck_nli"],
  embedding: ["response_cosine"],
};

function postModelNeeds() {
  if (window.parent === window) return;
  const needs = {};
  for (const [role, methods] of Object.entries(AUX_NEEDS)) {
    needs[role] = state.tab === "blackbox" && methods.includes(state.method);
  }
  window.parent.postMessage({ type: "trustllm-hub-model-needs", needs }, "*");
}

function toast(message) {
  const node = el2("div", { class: "rd-toast" }, esc(message));
  document.body.append(node);
  setTimeout(() => node.remove(), 2400);
}

/* ---------------- 生成（对话式） ---------------- */
async function generate() {
  const prompt = $("rd-prompt").value.trim();
  if (!prompt) return toast("请先输入提示词");
  addUserMsg(prompt);
  const bubble = beginAssistantMsg();
  setBusy(true);
  let acc = "";
  try {
    const response = await fetch("api/real/generate_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_new_tokens: parseInt($("rd-max-tokens").value || "128", 10) }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", final = null;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const chunk = JSON.parse(line);
        if (chunk.error) throw new Error(chunk.error);
        if (chunk.token !== undefined) {
          acc += chunk.token;
          bubble.textContent = acc;
          $("rd-msglist").scrollTop = $("rd-msglist").scrollHeight; // 回答区滚动
        }
        if (chunk.done) final = chunk;
      }
    }
    if (!final) throw new Error("生成流意外结束");
    bubble.innerHTML =
      `${esc(final.response)}<div class="rd-note">共 ${final.num_tokens} tokens · ${final.latency_seconds}s</div>`;
    commitAssistantAnswer(bubble, { prompt, response: final.response, num_tokens: final.num_tokens, latency_seconds: final.latency_seconds });
  } catch (err) {
    bubble.innerHTML = `<span style="color:#c0392b">生成失败：${esc(err.message)}</span>`;
  } finally {
    setBusy(false);
  }
}

/* ---------------- 抽样（黑盒 / 表征分析统一） ---------------- */
function samplingMode() {
  if (state.tab === "blackbox") return "text";
  if (state.tab === "whitebox" && (state.method === "eigenscore" || state.method === "effective_rank")) return "hidden";
  return null;
}
function updateSamplingCard() {
  const mode = samplingMode();
  $("rd-sampling").style.display = mode ? "" : "none";
  $("rd-layer-wrap").style.display = mode === "hidden" ? "" : "none";
  // 「输出结果」框只在需要抽样的方法下出现
  $("rd-samples-panel").hidden = !mode;
  document.querySelector(".tut-shell").classList.toggle("no-samples", !mode);
}

async function sample() {
  // 抽样针对「当前检测目标」的提问进行（未生成回答时退回输入框内容），保证样本与回答配套
  const target = currentTarget();
  const prompt = (target ? target.prompt : "") || $("rd-prompt").value.trim();
  if (!prompt) return toast("请先输入提示词");
  state.samplesPrompt = prompt;
  const mode = samplingMode();
  if (!mode) return;
  setBusy(true);
  const samplesBox = $("rd-samples");
  samplesBox.className = "rd-samples-empty";
  samplesBox.textContent = "抽样中…";
  try {
    if (mode === "hidden") {
      const data = await api("sample", {
        mode: "hidden", prompt,
        n: parseInt($("rd-num-samples").value || "5", 10),
        layer: parseInt($("rd-hidden-layer").value || "-1", 10),
        max_new_tokens: parseInt($("rd-max-tokens").value || "128", 10),
      });
      state.hiddenSampleKey = data.sample_key;
      renderSampleList(data.texts);
    } else {
      const texts = await api("sample", {
        mode: "text", prompt,
        n: parseInt($("rd-num-samples").value || "5", 10),
        max_new_tokens: parseInt($("rd-max-tokens").value || "64", 10),
      });
      state.samples = texts.texts;
      renderSampleList(texts.texts);
    }
  } catch (err) {
    if (mode === "hidden") state.hiddenSampleKey = null; else state.samples = null;
    samplesBox.textContent = "抽样失败：" + err.message;
  } finally { setBusy(false); }
}
function renderSampleList(texts) {
  const samplesBox = $("rd-samples");
  samplesBox.className = "rd-samples-list";
  samplesBox.innerHTML = texts.map((s, i) =>
    `<div class="rd-note"><b>样本 ${i + 1}</b>：${esc(s.slice(0, 120))}${s.length > 120 ? "…" : ""}</div>`).join("");
}

/* ---------------- 方法与参数 ---------------- */
function renderMethods() {
  const tab = TABS[state.tab];
  const box = $("rd-methods");
  box.innerHTML = `<p class="rd-panel-title">${tab.panelTitle}</p>`;
  if (!state.method || !tab.methods[state.method] || tab.methods[state.method].disabled) state.method = null;
  for (const group of tab.groups) {
    box.append(el2("p", { class: "rd-group-title" }, esc(group.title)));
    for (const key of group.keys) {
      const spec = tab.methods[key];
      const node = document.createElement("div");
      node.className = "rd-method" + (state.method === key ? " selected" : "") + (spec.disabled ? " disabled" : "");
      node.innerHTML = esc(spec.title) +
        (spec.disabled ? `<span class="rd-sub">${esc(spec.disabled)}</span>` : "");
      node.addEventListener("click", () => {
        if (spec.disabled) { toast(spec.disabled); return; }
        state.method = key;
        renderMethods(); renderParams(); updateSamplingCard(); postModelNeeds();
      });
      box.append(node);
    }
  }
  if (!state.method) box.append(el2("div", {}, `<p class="rd-hint">点击方法后填写参数。</p>`));
}

/* 选中方法后，在「开始检测 / 缓解」下方展示一行论文出处，点击跳教材对应小节 */
function updatePaperRef() {
  const box = $("rd-paper");
  const ref = state.method ? PAPER_REFS[`${state.tab}.${state.method}`] : null;
  if (!ref) { box.hidden = true; return; }
  box.hidden = false;
  const link = $("rd-paper-link");
  $("rd-paper-title").textContent = ref.title;
  $("rd-paper-sec").textContent = `教材 ${ref.secNum} ↗`;
  link.href = "https://trustllm.trustaid.net/docs/llm/chapters/chapter-6/6-5/" + ref.section;
  link.title = ref.title; // 悬停查看完整论文题名
}

function renderParams() {
  const tab = TABS[state.tab];
  const box = $("rd-params");
  const panel = box.closest(".rd-panel-params");
  box.innerHTML = "";
  $("rd-context-card").style.display = state.tab === "mitigation" ? "" : "none";
  $("rd-btn-run").textContent = tab.runLabel;
  updatePaperRef();
  const head = $("rd-params-for");
  head.hidden = true;
  head.textContent = "";
  if (!state.method) {
    panel.classList.add("rd-params-empty"); // 无内容时在框内水平上下居中
    box.innerHTML = `<p class="rd-hint">未选择方法。选中方法并完成参数设置后，「${esc(tab.runLabel)}」才会解锁。</p>`;
    updateRunButton();
    return;
  }
  const spec = tab.methods[state.method];
  const params = spec.params || {};
  const stored = JSON.parse(localStorage.getItem(`rd.params2.${state.tab}.${state.method}`) || "{}");
  const needsSampling = !!samplingMode();
  // 选中方法后，在「参数设置」标题下、参数上方标注「方法名」额外参数
  if (Object.keys(params).length || needsSampling) {
    head.hidden = false;
    head.textContent = `「${spec.title}」额外参数`;
  }
  if (!Object.keys(params).length) {
    // 无方法参数：需抽样时上方已展示抽样参数（此处留空），否则居中一行提示
    if (needsSampling) panel.classList.remove("rd-params-empty");
    else {
      panel.classList.add("rd-params-empty");
      box.innerHTML = `<p class="rd-hint">该方法无需额外参数。</p>`;
    }
    updateRunButton();
    return;
  }
  panel.classList.remove("rd-params-empty");
  for (const [key, p] of Object.entries(params)) {
    // 默认值不直接填入，仅作灰色占位提示；只有用户此前真正填写过的值才回填
    const value = stored[key] !== undefined && stored[key] !== "" ? String(stored[key]) : "";
    const id = `rd-p-${key}`;
    const wrap = document.createElement("label");
    wrap.className = "field" + (p.type === "text" ? "" : " field-num");
    if (p.type === "int") wrap.innerHTML = `${esc(p.label)}<input type="number" id="${id}" step="1" placeholder="${esc(p.def)}" value="${esc(value)}" />`;
    else if (p.type === "float") wrap.innerHTML = `${esc(p.label)}<input type="number" step="0.05" id="${id}" placeholder="${esc(p.def)}" value="${esc(value)}" />`;
    else wrap.innerHTML = `${esc(p.label)}<textarea id="${id}" rows="2" placeholder="${esc(p.def)}">${esc(value)}</textarea>`;
    wrap.querySelector(`#${id}`).addEventListener("input", updateRunButton);
    box.append(wrap);
  }
  updateRunButton();
}

/* 当前方法的抽样是否已完成（产出过可用样本） */
function samplingDone() {
  const mode = samplingMode();
  if (mode === "text") return Array.isArray(state.samples) && state.samples.length > 0;
  if (mode === "hidden") return !!state.hiddenSampleKey;
  return true;
}

/* 「开始检测 / 缓解」解锁条件：已选方法 + 需要抽样的方法完成抽样 + 数值参数全部填写
  （默认值仅作灰色占位提示，文本参数如产物路径可选）。 */
function updateRunButton() {
  const btn = $("rd-btn-run");
  if (!state.method) { btn.disabled = true; return; }
  if (!samplingDone()) { btn.disabled = true; return; }
  const params = TABS[state.tab].methods[state.method].params || {};
  const missing = Object.entries(params).some(([key, p]) =>
    p.type !== "text" && String($(`rd-p-${key}`)?.value ?? "").trim() === "");
  btn.disabled = missing;
}

function collectParams() {
  const tab = TABS[state.tab];
  if (!state.method) return null;
  const params = {};
  for (const [key, p] of Object.entries((tab.methods[state.method].params || {}))) {
    const node = $(`rd-p-${key}`);
    if (!node) continue;
    const raw = node.value.trim();
    if (raw === "") continue; // 未填写的参数不回传、不落库，由后端自行取默认值
    params[key] = p.type === "int" ? parseInt(raw, 10)
      : p.type === "float" ? parseFloat(raw) : raw;
  }
  localStorage.setItem(`rd.params2.${state.tab}.${state.method}`, JSON.stringify(params));
  return params;
}

/* ---------------- 运行 ---------------- */
async function run() {
  const tab = TABS[state.tab];
  const target = currentTarget();
  if (!target) return toast("请先发送提示词，等模型生成回答");
  if (!state.method) return toast("请先在右侧选择方法");
  const params = collectParams();

  if (state.tab === "blackbox" && !state.samples) {
    setBusy(false);
    const num = $("rd-num-samples").value || "5";
    return showError(`黑盒一致性方法需要多个独立样本。请先点击「抽样」按钮（当前设置：${num} 条），再运行检测。`);
  }
  if (state.tab === "blackbox" && state.samplesPrompt !== target.prompt) {
    return showError("「输出结果」中的样本对应其他提问；请点击「抽样」按所选回答的提问重新抽样后再运行。");
  }
  if (state.tab === "whitebox" && (state.method === "eigenscore" || state.method === "effective_rank")) {
    if (!state.hiddenSampleKey) {
      setBusy(false);
      return showError("表征分析方法需要先抽样：请点击抽样卡片里的「抽样」按钮，生成 n 条独立回答后再运行。");
    }
    if (state.samplesPrompt !== target.prompt) {
      return showError("已抽出的隐状态样本对应其他提问；请点击「抽样」按所选回答的提问重新抽样后再运行。");
    }
  }

  setBusy(true);
  try {
    if (state.tab === "whitebox") {
      const data = await api("detect", {
        task: "whitebox", method: state.method, prompt: target.prompt, params,
        sample_key: state.hiddenSampleKey || null,
      });
      renderDetection(data, tab);
    } else if (state.tab === "blackbox") {
      const data = await api("detect", {
        task: "blackbox", method: state.method, prompt: target.prompt, response: target.response, samples: state.samples, params,
      });
      renderDetection(data, tab);
    } else {
      const data = await api("mitigate", {
        method: state.method, prompt: target.prompt,
        context: $("rd-context").value.trim() || null, params,
      });
      renderMitigation(data);
    }
  } catch (err) {
    showError(tab.runLabel + "失败：" + err.message);
  } finally {
    setBusy(false);
  }
}

/* ---------------- 结果渲染 ---------------- */
function renderDetection(data, tab) {
  const f = data.final;
  const badge = f.is_hallucinated
    ? '<span class="rd-badge bad">判定：幻觉风险较高 (risk ≥ 0.5)</span>'
    : '<span class="rd-badge ok">判定：风险较低 (risk < 0.5)</span>';
  let html = `
    <div class="rd-final">
      <div><div class="big-num">${f.score.toFixed(4)}</div><div class="big-label">score（置信度）</div></div>
      <div><div class="big-num">${f.risk.toFixed(4)}</div><div class="big-label">risk（幻觉风险）</div></div>
      ${badge}
      <div><div class="big-num" style="font-size:16px">${data.latency_seconds}s</div><div class="big-label">耗时</div></div>
    </div>`;
  const inter = data.intermediates || {};
  if (inter.tokens) {
    html += `<div class="rd-section-title">逐 token 数据</div><table class="rd-table"><tr><th>#</th><th>token</th><th>log p</th><th>top-1 备选</th></tr>`;
    inter.tokens.forEach((token, i) => {
      const lp = inter.token_logprobs[i];
      const lpText = lp == null ? "-∞" : Number(lp).toFixed(3);
      const triples = (inter.top_logprobs && inter.top_logprobs[i]) || [];
      let top = null;
      for (const triple of triples) {
        const logp = triple && triple[2];
        if (logp != null && (top === null || logp > top[2])) top = triple;
      }
      const topText = top ? `${esc(top[1])} (${top[2] == null ? "-∞" : Number(top[2]).toFixed(2)})` : "-";
      html += `<tr><td>${i + 1}</td><td>${esc(token)}</td><td>${lpText}</td><td>${topText}</td></tr>`;
    });
    html += "</table>";
  }
  if (inter.per_token_entropy) {
    html += `<p class="rd-note">平均 entropy（=-log p）: <b>${inter.mean_entropy}</b></p>`;
  }
  if (inter.note) html += `<p class="rd-note">${esc(inter.note)}</p>`;
  if (inter.sentences) {
    html += `<div class="rd-section-title">逐句风险</div><table class="rd-table"><tr><th>句子</th><th>risk</th></tr>`;
    inter.sentences.forEach((sentence, i) => {
      html += `<tr><td>${esc(sentence)}</td><td>${(inter.sentence_risk[i] || 0).toFixed(4)}</td></tr>`;
    });
    html += `</table>`;
  }
  if (inter.effective_rank !== undefined) {
    html += `<div class="rd-section-title">谱分析</div>
      <table class="rd-table">
        <tr><th>有效秩 effective rank</th><td>${inter.effective_rank}</td></tr>
        <tr><th>谱熵 spectral entropy</th><td>${inter.spectral_entropy}</td></tr>
        <tr><th>Gram 特征值谱</th><td>${(inter.eigenvalues_gram || []).join(", ")}</td></tr>
      </table>`;
  }
  if (inter.samples) {
    html += `<div class="rd-section-title">使用的样本（${inter.samples.length} 条）</div>`;
    html += inter.samples.map((s, i) => `<p class="rd-note"><b>${i + 1}.</b> ${esc(s.slice(0, 90))}${s.length > 90 ? "…" : ""}</p>`).join("");
  }
  showResult(html, `${tab.title} · ${f.method}`);
}

function renderMitigation(data) {
  let html = `
    <div class="rd-final">
      <div><div class="big-num" style="font-size:16px">${data.latency_seconds}s</div><div class="big-label">耗时</div></div>
    </div>
    <div class="rd-section-title">基线回答（未缓解）</div>
    <div class="rd-response">${esc(data.baseline || "（空）")}</div>
    <div class="rd-section-title" style="margin-top:12px">缓解后回答</div>
    <div class="rd-response" style="border-left-color: var(--deep)">${esc(data.mitigated || "（空）")}</div>`;
  if (data.warnings && data.warnings.length) {
    html += `<p class="rd-note">⚠ ${esc(data.warnings.join("；"))}</p>`;
  }
  if (data.trace && Object.keys(data.trace).length) {
    html += `<div class="rd-section-title">方法轨迹（trace）</div><table class="rd-table">`;
    for (const [key, value] of Object.entries(flatten(data.trace))) {
      html += `<tr><td>${esc(key)}</td><td>${esc(shorten(value))}</td></tr>`;
    }
    html += `</table>`;
  }
  showResult(html, `缓解方法 · ${data.method}`);
}
function flatten(obj, prefix) {
  const out = {};
  for (const [key, value] of Object.entries(obj || {})) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) Object.assign(out, flatten(value, path));
    else out[path] = Array.isArray(value) ? JSON.stringify(value) : value;
  }
  return out;
}
function shorten(value) {
  const text = String(value);
  return text.length > 90 ? text.slice(0, 89) + "…" : text;
}

/* ---------------- 初始化 ---------------- */
function switchTab(tab) {
  state.tab = tab;
  state.method = null;
  document.querySelectorAll(".tut-page-tab").forEach((node) => {
    node.classList.toggle("is-active", node.dataset.tab === tab);
  });
  // 检测结果按回答记录保存，切页签不清空
  renderMethods();
  renderParams();
  updateSamplingCard();
  postModelNeeds();
}

function init() {
  document.querySelectorAll(".tut-page-tab").forEach((node) => {
    node.addEventListener("click", () => switchTab(node.dataset.tab));
  });
  $("rd-btn-generate").addEventListener("click", generate);
  $("rd-btn-sample").addEventListener("click", sample);
  $("rd-btn-run").addEventListener("click", run);
  $("rd-prompt-preset").addEventListener("change", () => {
    if ($("rd-prompt-preset").value) $("rd-prompt").value = $("rd-prompt-preset").value;
  });
  $("rd-prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); }
  });
  const preset = $("rd-prompt-preset");
  preset.append(new Option("选择预设提示词…", ""));
  for (const p of PRESET_PROMPTS) preset.append(new Option(p, p));
  renderMethods();
  renderParams();
  updateSamplingCard();
  postModelNeeds();
  renderTargetMarks();
  renderResultsPanel();
}

init();
