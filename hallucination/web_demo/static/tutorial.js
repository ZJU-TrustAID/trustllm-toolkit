/* Tutorial mode, MarkSec-style: three pages (white-box detection, black-box
   detection, mitigation). Each page has a prompt dialog on the left — click
   it to pick one of the preset prompts, the choice is echoed back into the
   box — and the page's method option group on the right. 生成 simulates the
   generation process (streamed output) before rendering the method result.
   Zero dependencies; everything renders from /api/catalog + /api/method. */
"use strict";

const tutState = {
  catalog: null,
  pageId: null,
  themeId: null,
  methodKey: null,
  coloring: true,
  generating: false,
};

const PAGE_DEFS = [
  {
    id: "whitebox",
    nav: "白盒幻觉检测",
    number: "01",
    group: "detection",
    methodsKey: "detection_whitebox",
    title: "白盒检测方法",
    note: "利用模型内部状态（词元概率、熵、隐藏层表征）估计置信度，适合白盒开源模型。",
  },
  {
    id: "blackbox",
    nav: "黑盒幻觉检测",
    number: "02",
    group: "detection",
    methodsKey: "detection_blackbox",
    title: "黑盒检测方法",
    note: "只依赖输入与输出：采样一致性、证据比对、模型自评估，适合黑盒闭源模型。",
  },
  {
    id: "mitigation",
    nav: "缓解方法",
    number: "03",
    group: "mitigation",
    methodsKey: "mitigation",
    title: "幻觉缓解方法",
    note: "推理时干预解码、激活或后处理修正，改变模型输出。",
  },
];

/* 每个页签的方法分组(键引用 /api/catalog 返回的分组目录)。 */
const PAGE_GROUP_KEYS = {
  whitebox: "detection_whitebox_groups",
  blackbox: "detection_blackbox_groups",
  mitigation: "mitigation_groups",
};

const SIGNAL_LABELS = {
  generation_confidence: "生成置信",
  self_consistency: "自一致性",
  evidence_support: "证据支持",
  self_reported_confidence: "自报告置信",
  judge_assessment: "评审评估",
};

function tutEl(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function tutPercent(value) {
  return `${Math.round(Number(value) * 1000) / 10}%`;
}

function tutFormat(format, value) {
  if (format === "percent") return tutPercent(value);
  if (format === "value3") return Number(value).toFixed(3);
  return Number(value).toFixed(2);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function currentPage() {
  return PAGE_DEFS.find((page) => page.id === tutState.pageId) || PAGE_DEFS[0];
}

function currentTheme() {
  return tutState.catalog.themes.find((theme) => theme.id === tutState.themeId) || null;
}

function currentMethod() {
  const page = currentPage();
  const theme = currentTheme();
  return (
    tutState.catalog[page.methodsKey].find(
      (method) => method.key === tutState.methodKey && methodAvailable(method.key, page, theme),
    ) || null
  );
}

function themesForPage(page) {
  return tutState.catalog.themes.filter((theme) => theme.available[page.group].length > 0);
}

function methodAvailable(methodKey, page, theme) {
  return Boolean(theme && theme.available[page.group].includes(methodKey));
}

/* ---------------- boot + page tabs ---------------- */

async function bootTutorial() {
  try {
    const response = await fetch("api/catalog");
    if (!response.ok) throw new Error("目录加载失败");
    tutState.catalog = await response.json();
    renderPageTabs();
    selectPage("whitebox");
  } catch (error) {
    document.querySelector("#tut-page").replaceChildren(
      tutEl("p", "tut-message", `目录不可用: ${error.message}`),
    );
  }
}

function renderPageTabs() {
  const bar = document.querySelector("#tut-pages");
  bar.replaceChildren();
  PAGE_DEFS.forEach((page) => {
    const tab = tutEl("button", "tut-page-tab");
    tab.type = "button";
    tab.dataset.page = page.id;
    tab.append(tutEl("small", null, page.number));
    tab.append(document.createTextNode(page.nav));
    tab.addEventListener("click", () => selectPage(page.id));
    bar.append(tab);
  });
}

function selectPage(id) {
  tutState.pageId = id;
  document.querySelectorAll(".tut-page-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.page === id);
  });
  const page = currentPage();
  const themes = themesForPage(page);
  if (!themes.some((theme) => theme.id === tutState.themeId)) {
    tutState.themeId = themes[0] ? themes[0].id : null;
  }
  /* 方法须由用户显式选择,不自动回退;当前页没有可选方法时保持未选状态。 */
  const methods = tutState.catalog[page.methodsKey];
  if (!methods.some((method) => method.key === tutState.methodKey)) {
    tutState.methodKey = null;
  }
  renderPage(page);
}

/* ---------------- page layout ---------------- */

function renderPage(page) {
  const container = document.querySelector("#tut-page");
  container.replaceChildren();

  const layout = tutEl("div", "tut-layout");

  /* left: prompt dialog + generate + streamed output */
  const panel = tutEl("div", "tut-panel");
  panel.append(tutEl("p", "tut-options-title", "提示词"));
  panel.append(tutEl("p", "tut-options-note", "点击对话框选择预设提示词。"));

  const dialog = tutEl("div", "tut-dialog");
  dialog.setAttribute("role", "button");
  dialog.setAttribute("tabindex", "0");
  dialog.setAttribute("aria-label", "选择预设提示词");
  const dialogText = tutEl("span", "tut-dialog-text");
  const dialogChevron = tutEl("span", "tut-dialog-chevron", "▾");
  dialog.append(dialogText, dialogChevron);
  const dialogWrap = tutEl("div", "tut-dialog-wrap");
  const popover = tutEl("div", "tut-popover");
  popover.hidden = true;
  dialogWrap.append(dialog, popover);
  panel.append(dialogWrap);

  const runRow = tutEl("div", "tut-run-row");
  const run = tutEl("button", "tut-run");
  run.type = "button";
  run.append(tutEl("span", "tut-run-label", "生成"));
  run.addEventListener("click", () => runGenerate(run));
  const message = tutEl("p", "tut-message");
  runRow.append(run, message);
  panel.append(runRow);

  const output = tutEl("div", "tut-output");
  output.append(tutEl("p", "tut-options-title", "模型输出"));
  const outputBox = tutEl("div", "tut-output-box");
  const outputText = tutEl("span", "tut-output-text");
  const caret = tutEl("span", "tut-caret", "▌");
  outputBox.append(outputText, caret);
  output.append(outputBox);
  const outputStatus = tutEl("p", "tut-output-status");
  output.append(outputStatus);
  panel.append(output);
  layout.append(panel);

  /* right: this page's method option group */
  layout.append(renderOptions(page));
  container.append(layout);

  const sample = tutEl("section", "tut-sample");
  sample.hidden = true;
  container.append(sample);

  /* dialog interactions */
  const togglePopover = () => {
    if (tutState.generating) return;
    renderPopover(popover, page);
    popover.hidden = !popover.hidden;
  };
  dialog.addEventListener("click", togglePopover);
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      togglePopover();
    }
  });
  document.addEventListener("click", (event) => {
    if (!dialogWrap.contains(event.target)) popover.hidden = true;
  });

  syncPage();
}

function renderPopover(popover, page) {
  popover.replaceChildren();
  popover.append(tutEl("p", "tut-pop-title", "选择提示词"));
  themesForPage(page).forEach((theme) => {
    const item = tutEl("button", "tut-pop-item");
    item.type = "button";
    item.dataset.theme = theme.id;
    item.append(tutEl("span", "tut-pop-prompt", theme.prompt));
    item.append(tutEl("span", "tut-pop-hint", theme.label));
    item.addEventListener("click", () => {
      tutState.themeId = theme.id;
      popover.hidden = true;
      syncPage();
    });
    popover.append(item);
  });
}

function renderOptions(page) {
  const options = tutEl("aside", "tut-options");
  options.id = "tut-options";
  options.setAttribute("aria-label", page.title);
  options.append(tutEl("p", "tut-options-title", page.title));
  options.append(tutEl("p", "tut-options-note", page.note));

  const groups = tutState.catalog[PAGE_GROUP_KEYS[page.id]] || [
    { title: page.title, methods: tutState.catalog[page.methodsKey] },
  ];
  groups.forEach((group) => {
    options.append(tutEl("p", "tut-options-subtitle", group.title));
    const list = tutEl("div", "tut-option-group");
    group.methods.forEach((method) => {
      const option = tutEl("button", "tut-option", method.name);
      option.type = "button";
      option.dataset.method = method.key;
      option.addEventListener("click", () => {
        tutState.methodKey = method.key;
        syncPage();
      });
      list.append(option);
    });
    options.append(list);
  });

  /* 选中方法的说明卡(名称、教材出处、原理与前置条件),由 syncPage 填充。 */
  const methodNote = tutEl("div", "tut-method-note");
  methodNote.id = "tut-method-note";
  methodNote.hidden = true;
  options.append(methodNote);
  return options;
}

function renderMethodNote(method) {
  const note = document.querySelector("#tut-method-note");
  if (!note) return;
  note.replaceChildren();
  if (!method) {
    note.hidden = true;
    return;
  }
  note.hidden = false;
  note.append(tutEl("small", "tut-method-ref", method.book_ref));
  note.append(tutEl("strong", "tut-method-name", method.name));
  note.append(tutEl("p", null, method.summary));
  note.append(tutEl("p", null, method.needs));
}

/* sync dialog text, method availability and result visibility */
function syncPage() {
  const page = currentPage();
  const theme = currentTheme();
  const method = currentMethod();

  const dialogText = document.querySelector(".tut-dialog-text");
  if (dialogText) {
    if (theme) {
      dialogText.replaceChildren(document.createTextNode(theme.prompt));
      dialogText.classList.remove("placeholder");
    } else {
      dialogText.textContent = "点击选择提示词…";
      dialogText.classList.add("placeholder");
    }
  }

  document.querySelectorAll(".tut-pop-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.theme === tutState.themeId);
  });

  document.querySelectorAll(".tut-option").forEach((option) => {
    option.classList.toggle("active", option.dataset.method === tutState.methodKey);
    option.disabled = !methodAvailable(option.dataset.method, page, theme);
    const available = methodAvailable(option.dataset.method, page, theme);
    const tagged = tutState.catalog[page.methodsKey].find((method) => method.key === option.dataset.method);
    option.title = available ? (tagged ? tagged.tag : "") : "该提示词场景不演示此方法";
  });

  renderMethodNote(method);

  /* 生成进行中不改状态栏、不清结果,避免打断流式输出。 */
  if (tutState.generating) return;

  const sample = document.querySelector(".tut-sample");
  if (sample && !sample.hidden) {
    sample.hidden = true;
    sample.replaceChildren();
  }
  const status = document.querySelector(".tut-output-status");
  if (status) {
    status.textContent = method
      ? "选择提示词后点击「生成」。"
      : `请在右侧选择${page.group === "detection" ? "检测" : "缓解"}方法，再点击「生成」。`;
  }
  const outputText = document.querySelector(".tut-output-text");
  if (outputText) outputText.textContent = "";
}

/* ---------------- generate: simulated streaming + method computation ---------------- */

async function runGenerate(runButton) {
  if (tutState.generating) return;
  const theme = currentTheme();
  const method = currentMethod();
  const page = currentPage();
  const message = document.querySelector(".tut-message");
  if (!method) {
    if (message) {
      message.textContent = `请先在右侧选择${page.group === "detection" ? "检测" : "缓解"}方法。`;
    }
    return;
  }
  if (message) message.textContent = "";
  tutState.generating = true;
  runButton.disabled = true;
  runButton.querySelector(".tut-run-label").textContent = "生成中…";

  const outputText = document.querySelector(".tut-output-text");
  const caret = document.querySelector(".tut-caret");
  const status = document.querySelector(".tut-output-status");
  const sample = document.querySelector(".tut-sample");
  sample.hidden = true;
  sample.replaceChildren();
  outputText.textContent = "";
  caret.classList.add("on");

  try {
    status.textContent = "正在生成回答…";
    await sleep(450);
    for (const ch of theme.response) {
      outputText.textContent += ch;
      await sleep(26);
    }
    caret.classList.remove("on");

    status.textContent = `正在用 ${method.name} 计算(不调用模型)…`;
    const payload = await fetchMethod(method.key);

    status.textContent = "完成。";
    renderResults(page, method, theme, payload);
  } catch (error) {
    status.textContent = `出错: ${error.message}`;
  } finally {
    caret.classList.remove("on");
    runButton.disabled = false;
    runButton.querySelector(".tut-run-label").textContent = "生成";
    tutState.generating = false;
  }
}

async function fetchMethod(methodKey) {
  const response = await fetch("api/method", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method: methodKey, scenario: tutState.themeId }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "请求失败");
  return payload;
}

/* ---------------- result rendering ---------------- */

function renderResults(page, method, theme, payload) {
  const sample = document.querySelector(".tut-sample");
  sample.hidden = false;
  sample.replaceChildren();

  /* 头部:提示词标题 + 页签/方法/场景 chips(对齐教科书 HallucinationLab)。 */
  const head = tutEl("div", "tut-sample-head");
  head.append(tutEl("h3", "tut-sample-title", theme.prompt));
  const chips = tutEl("div", "tut-sample-chips");
  chips.append(tutEl("span", "tut-chip", page.nav));
  chips.append(tutEl("span", "tut-chip accent", method.name));
  chips.append(tutEl("span", "tut-chip", theme.label));
  head.append(chips);
  sample.append(head);

  const section = tutEl("div", "tut-sample-group");
  if (payload.context) {
    const ctx = tutEl("div", "tut-block tut-context");
    ctx.append(tutEl("p", "tut-section-title", "上下文"));
    ctx.append(tutEl("p", "tut-context-text", payload.context));
    section.append(ctx);
  }
  if (payload.token_marks && payload.token_marks.baseline) {
    section.append(renderOutputCompare(payload.compare, payload.token_marks));
  } else {
    section.append(renderCompare(payload.compare));
    if (payload.token_marks) section.append(renderTokenMarks(payload.token_marks));
  }
  if (payload.viz.length) {
    section.append(tutEl("p", "tut-section-title", page.group === "detection" ? "检测过程" : "缓解机制"));
  }
  payload.viz.forEach((block) => section.append(renderVizBlock(block)));
  if (payload.method_scores && payload.method_scores.length) {
    section.append(renderScoreTable(payload.method_scores));
  }
  sample.append(section);

  if (payload.warnings && payload.warnings.length) sample.append(renderNotice(payload.warnings));
  sample.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderOutputCompare(compare, marks) {
  const section = tutEl("div", "tut-block");
  const wrap = tutEl("div", "tut-out-compare");
  const cards = compare.cards;
  const rows = [
    ["baseline", cards[0], marks.baseline],
    ["mitigated", cards[1], marks.mitigated],
  ];
  const colored = [];
  rows.forEach(([cls, card, mark]) => {
    const block = tutEl("div", `tut-out-block ${cls}`);
    const caption = tutEl("div", "tut-out-caption");
    caption.append(tutEl("span", "tut-out-name", card.title));
    caption.append(tutEl("span", "tut-out-verdict", card.verdict));
    block.append(caption);
    const line = tutEl("p", "tut-tokens");
    mark.tokens.forEach((token) => {
      const span = tutEl("span", `tok tok-${token.cls}`, token.text);
      span.title = token.tip || "";
      if (!tutState.coloring) span.classList.add("coloring-off");
      line.append(span);
    });
    block.append(line);
    const legend = tutEl("p", "tut-legend");
    mark.legend.forEach((item) => {
      legend.append(tutEl("span", `legend-dot tok-${item.cls}`));
      legend.append(tutEl("span", "legend-label", item.label));
    });
    block.append(legend);
    colored.push({ block, tokens: [...line.querySelectorAll(".tok")] });
    wrap.append(block);
  });
  const head = tutEl("div", "tut-block-head");
  head.append(tutEl("p", "tut-section-title", "输出对照 · 词元标注"));
  const toggle = tutEl("button", "tut-toggle", tutState.coloring ? "着色 开" : "着色 关");
  toggle.type = "button";
  toggle.addEventListener("click", () => {
    tutState.coloring = !tutState.coloring;
    toggle.textContent = tutState.coloring ? "着色 开" : "着色 关";
    colored.forEach(({ tokens }) => {
      tokens.forEach((node) => node.classList.toggle("coloring-off", !tutState.coloring));
    });
  });
  head.append(toggle);
  section.append(head);
  section.append(wrap);
  section.append(tutEl("p", "tut-footnote", compare.threshold_note));
  return section;
}

function renderCompare(compare) {
  const single = compare.cards.length === 1;
  const wrap = tutEl("div", single ? "tut-compare single" : "tut-compare");
  compare.cards.forEach((card) => {
    const node = tutEl("article", `tut-compare-card tone-${card.tone}`);
    if (card.active) node.classList.add("active");
    const head = tutEl("div", "tut-compare-head");
    head.append(tutEl("span", "tut-compare-title", card.title));
    head.append(tutEl("span", "tut-verdict", card.verdict));
    node.append(head);
    if (card.risk !== undefined) {
      node.append(tutEl("p", "tut-compare-risk", `risk ${tutPercent(card.risk)} · 置信 ${tutPercent(card.confidence)}`));
    }
    node.append(tutEl("p", "tut-compare-text", card.text));
    if (card.reference) {
      const ref = tutEl("p", "tut-card-reference");
      ref.append(tutEl("span", "tut-reference-label", "正确答案"));
      ref.append(document.createTextNode(card.reference));
      node.append(ref);
    }
    if (card.formula) {
      node.append(tutEl("p", "tut-card-formula", card.formula));
    }
    if (card.metrics && card.metrics.length) {
      const metrics = tutEl("div", "tut-metrics");
      card.metrics.forEach((metric) => {
        const item = tutEl("div", "tut-metric");
        item.append(tutEl("span", "tut-metric-label", metric.label));
        item.append(tutEl("strong", "tut-metric-value", metric.value));
        metrics.append(item);
      });
      node.append(metrics);
    }
    wrap.append(node);
  });
  const section = tutEl("div", "tut-block");
  section.append(tutEl("p", "tut-section-title", single ? "检测结论" : "对照结论"));
  section.append(wrap);
  section.append(tutEl("p", "tut-footnote", compare.threshold_note));
  return section;
}

function renderTokenMarks(marks) {
  const section = tutEl("div", "tut-block");
  const head = tutEl("div", "tut-block-head");
  head.append(tutEl("p", "tut-section-title", "词元级标注"));
  const toggle = tutEl("button", "tut-toggle", tutState.coloring ? "着色 开" : "着色 关");
  toggle.type = "button";
  toggle.addEventListener("click", () => {
    tutState.coloring = !tutState.coloring;
    toggle.textContent = tutState.coloring ? "着色 开" : "着色 关";
    section.querySelectorAll(".tok").forEach((node) => {
      node.classList.toggle("coloring-off", !tutState.coloring);
    });
  });
  head.append(toggle);
  section.append(head);
  const text = tutEl("p", "tut-tokens");
  marks.tokens.forEach((token) => {
    const span = tutEl("span", `tok tok-${token.cls}`, token.text);
    span.title = token.tip || "";
    if (!tutState.coloring) span.classList.add("coloring-off");
    text.append(span);
  });
  section.append(text);
  const legend = tutEl("p", "tut-legend");
  marks.legend.forEach((item) => {
    legend.append(tutEl("span", `legend-dot tok-${item.cls}`));
    legend.append(tutEl("span", "legend-label", item.label));
  });
  section.append(legend);
  return section;
}

function renderNotice(warnings) {
  const notice = tutEl("div", "tut-notice");
  notice.append(tutEl("span", "tut-notice-mark", "i"));
  const body = tutEl("div");
  body.append(tutEl("strong", null, "结果说明"));
  body.append(tutEl("p", null, warnings.join(" ")));
  notice.append(body);
  return notice;
}

function renderScoreTable(items) {
  const section = tutEl("div", "tut-block");
  section.append(tutEl("p", "tut-section-title", "指标明细"));
  const table = tutEl("div", "tut-score-table");
  items.forEach((item) => {
    const row = tutEl("div", "tut-score-row");
    const name = tutEl("div", "tut-score-name");
    name.append(tutEl("span", null, item.method));
    name.append(tutEl("small", null, `${item.level} · ${item.signal_type}`));
    const risk = tutEl("div", "tut-score-value risk", `${tutPercent(item.risk)} 风险`);
    const confidence = tutEl("div", "tut-score-value confidence", `${tutPercent(item.score)} 置信`);
    row.append(name, risk, confidence);
    table.append(row);
  });
  section.append(table);
  return section;
}

/* ---------------- viz block dispatch ---------------- */

function renderVizBlock(block) {
  const section = tutEl("div", "tut-block");
  const head = tutEl("div", "tut-block-head");
  head.append(tutEl("p", "tut-section-title", block.title));
  section.append(head);
  const renderers = {
    "token-bars": renderBars,
    "layer-curve": renderCurve,
    "logit-table": renderTable,
    "steps-table": renderTable,
    "cluster-view": renderClusters,
    "claim-table": renderClaims,
    "qa-chain": renderQaChain,
    matrix: renderMatrix,
    "unit-cards": renderUnitCards,
    "prompt-view": renderPromptView,
    "head-grid": renderHeadGrid,
    "pair-view": renderPairView,
    "sample-badges": renderSampleBadges,
  };
  const renderer = renderers[block.type];
  if (renderer) section.append(renderer(block));
  if (block.note) section.append(tutEl("p", "tut-footnote", block.note));
  return section;
}

function renderBars(block) {
  const wrap = tutEl("div", "tut-bars");
  const values = block.items.map((item) => Number(item.value));
  const max = Math.max(...values, 1e-9);
  block.items.forEach((item) => {
    const row = tutEl("div", "tut-bar-row");
    row.append(tutEl("span", "tut-bar-label", item.label));
    const track = tutEl("span", "tut-bar-track");
    const fill = tutEl("span", "tut-bar-fill");
    fill.style.width = `${Math.max(2, (Number(item.value) / max) * 100)}%`;
    if (item.highlight) fill.classList.add("highlight");
    track.append(fill);
    row.append(track);
    row.append(tutEl("span", "tut-bar-value", tutFormat(block.format, item.value)));
    if (item.note) row.append(tutEl("span", "tut-bar-note", item.note));
    wrap.append(row);
  });
  return wrap;
}

function renderCurve(block) {
  const wrap = tutEl("div", "tut-curve");
  const values = block.items.map((item) => Number(item.value));
  const max = Math.max(...values, 1e-9);
  const width = 320;
  const height = 96;
  const stepX = width / Math.max(1, values.length - 1);
  const points = values
    .map((value, index) => `${(index * stepX).toFixed(1)},${(height - 12 - (value / max) * (height - 28)).toFixed(1)}`)
    .join(" ");
  const svgNs = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNs, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "tut-svg");
  const polyline = document.createElementNS(svgNs, "polyline");
  polyline.setAttribute("points", points);
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", "currentColor");
  polyline.setAttribute("stroke-width", "2");
  svg.append(polyline);
  block.items.forEach((item, index) => {
    const [x, y] = points.split(" ")[index].split(",");
    const dot = document.createElementNS(svgNs, "circle");
    dot.setAttribute("cx", x);
    dot.setAttribute("cy", y);
    dot.setAttribute("r", item.highlight ? "5" : "3");
    dot.setAttribute("class", item.highlight ? "tut-dot highlight" : "tut-dot");
    svg.append(dot);
    const label = document.createElementNS(svgNs, "text");
    label.setAttribute("x", x);
    label.setAttribute("y", height - 1);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "tut-svg-label");
    label.textContent = item.label.replace("早层 ", "");
    svg.append(label);
  });
  wrap.append(svg);
  return wrap;
}

function renderTable(block) {
  const table = tutEl("table", "tut-table");
  const thead = tutEl("thead");
  const headRow = tutEl("tr");
  block.columns.forEach((column) => headRow.append(tutEl("th", null, column)));
  thead.append(headRow);
  table.append(thead);
  const tbody = tutEl("tbody");
  block.rows.forEach((row) => {
    const tr = tutEl("tr");
    row.forEach((cell, index) => {
      const td = tutEl("td", null, String(cell));
      const text = String(cell);
      if (index === row.length - 1 || block.columns[index] === "状态" || block.columns[index] === "评价") {
        if (text.includes("选中")) td.classList.add("st-ok");
        else if (text.includes("剪除") || text.includes("抑制") || text.includes("过冲")) td.classList.add("st-mask");
        else if (text.includes("基线") || text.includes("不足")) td.classList.add("st-base");
      }
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(tbody);
  return table;
}

function renderClusters(block) {
  const wrap = tutEl("div", "tut-clusters");
  block.clusters.forEach((cluster) => {
    const card = tutEl("div", "tut-cluster");
    const head = tutEl("div", "tut-cluster-head");
    head.append(tutEl("span", "tut-cluster-label", cluster.label));
    head.append(tutEl("span", "tut-cluster-prob", `${tutPercent(cluster.probability)} · ${cluster.members.length} 票`));
    card.append(head);
    const track = tutEl("span", "tut-bar-track");
    const fill = tutEl("span", "tut-bar-fill");
    fill.style.width = `${Math.max(2, Number(cluster.probability) * 100)}%`;
    if (cluster.probability < 0.4) fill.classList.add("warn");
    track.append(fill);
    card.append(track);
    const members = tutEl("ul", "tut-members");
    cluster.members.forEach((member) => members.append(tutEl("li", null, member)));
    card.append(members);
    wrap.append(card);
  });
  return wrap;
}

const CLAIM_LABELS = { supported: "支持", contradicted: "矛盾", unknown: "未知" };

function renderClaims(block) {
  const wrap = tutEl("div", "tut-claims");
  block.rows.forEach((row) => {
    const card = tutEl("article", `tut-claim claim-${row.label}`);
    const head = tutEl("div", "tut-claim-head");
    head.append(tutEl("span", "tut-claim-text", row.claim));
    head.append(tutEl("span", "tut-claim-label", (block.labels && block.labels[row.label]) || CLAIM_LABELS[row.label] || row.label));
    card.append(head);
    const evidence = tutEl("p", "tut-claim-evidence", `证据《${row.evidence_title}》: ${row.evidence_snippet || "无"}`);
    card.append(evidence);
    card.append(tutEl("p", "tut-claim-values", `蕴含 ${tutPercent(row.entailment)} · 矛盾 ${tutPercent(row.contradiction)}`));
    wrap.append(card);
  });
  return wrap;
}

function renderQaChain(block) {
  const wrap = tutEl("div", "tut-qa");
  if (block.stages) {
    block.stages.forEach((stage) => {
      const card = tutEl("div", "tut-qa-stage");
      card.append(tutEl("p", "tut-qa-title", stage.title));
      if (stage.content) card.append(tutEl("p", "tut-qa-content", stage.content));
      if (stage.questions) {
        stage.questions.forEach((item) => {
          const row = tutEl("div", "tut-qa-pair");
          row.append(tutEl("p", "tut-qa-q", `问: ${item.question}`));
          row.append(tutEl("p", "tut-qa-a", `答: ${item.answer}`));
          card.append(row);
        });
      }
      wrap.append(card);
    });
    return wrap;
  }
  (block.questions || []).forEach((item) => {
    const card = tutEl("article", `tut-qa-item ${item.conflict ? "conflict" : ""}`);
    card.append(tutEl("p", "tut-qa-q", `问: ${item.question}`));
    card.append(tutEl("p", "tut-qa-a", `独立回答: ${item.answer}`));
    const badge = tutEl("span", "tut-qa-badge", item.conflict ? "与草稿矛盾" : "无明显矛盾");
    badge.classList.add(item.conflict ? "bad" : "ok");
    card.append(badge);
    if (item.contradiction !== undefined) {
      card.append(tutEl("p", "tut-claim-values", `NLI 矛盾 ${tutPercent(item.contradiction)}`));
    }
    wrap.append(card);
  });
  return wrap;
}

function renderMatrix(block) {
  const table = tutEl("table", "tut-table tut-matrix");
  const thead = tutEl("thead");
  const headRow = tutEl("tr");
  headRow.append(tutEl("th", null, ""));
  block.col_labels.forEach((label) => headRow.append(tutEl("th", null, label)));
  thead.append(headRow);
  table.append(thead);
  const tbody = tutEl("tbody");
  block.values.forEach((rowValues, rowIndex) => {
    const tr = tutEl("tr");
    tr.append(tutEl("th", null, block.row_labels[rowIndex]));
    rowValues.forEach((value) => {
      const td = tutEl("td", "tut-matrix-cell", tutFormat(block.format, value));
      td.style.backgroundColor = `rgba(16, 122, 87, ${Math.max(0.06, Number(value) * 0.55)})`;
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(tbody);
  return table;
}

function renderUnitCards(block) {
  const wrap = tutEl("div", "tut-units");
  block.units.forEach((unit) => {
    const card = tutEl("article", "tut-unit");
    const head = tutEl("div", "tut-unit-head");
    head.append(tutEl("p", "tut-unit-text", unit.text));
    head.append(tutEl("span", "tut-unit-risk", `${tutPercent(unit.risk)} 风险`));
    card.append(head);
    card.append(tutEl("p", "tut-unit-values", `NLI 蕴含 ${tutPercent(unit.mean_entailment)} · 矛盾 ${tutPercent(unit.mean_contradiction)}`));
    const details = tutEl("details", "tut-unit-details");
    details.append(tutEl("summary", null, "查看逐采样判定"));
    (unit.comparisons || []).forEach((comparison) => {
      details.append(tutEl("div", "tut-detail-line",
        `采样 ${comparison.sample_index + 1}:蕴含 ${tutPercent(comparison.entailment)} · 矛盾 ${tutPercent(comparison.contradiction)} · 中性 ${tutPercent(comparison.neutral)}`));
    });
    card.append(details);
    wrap.append(card);
  });
  return wrap;
}

function renderPromptView(block) {
  return tutEl("pre", "tut-prompt", block.text);
}

function renderHeadGrid(block) {
  const selected = new Set(block.selected.map(([layer, head]) => `${layer}-${head}`));
  const wrap = tutEl("div", "tut-head-grid");
  for (let layer = 0; layer < block.layers; layer += 1) {
    for (let head = 0; head < block.heads; head += 1) {
      const value = Number(block.values[layer][head]);
      const cell = tutEl("div", "tut-head-cell", `${(value * 100).toFixed(0)}`);
      cell.style.backgroundColor = `rgba(16, 122, 87, ${Math.max(0.05, (value - 0.45) * 1.6)})`;
      cell.title = `L${layer} · H${head} · 准确率 ${tutPercent(value)}`;
      if (selected.has(`${layer}-${head}`)) cell.classList.add("selected");
      wrap.append(cell);
    }
  }
  wrap.style.gridTemplateColumns = `repeat(${block.heads}, 1fr)`;
  return wrap;
}

function renderPairView(block) {
  const wrap = tutEl("div", "tut-pairs");
  const pairs = tutEl("div", "tut-pair-list");
  block.pairs.forEach((pair) => {
    const card = tutEl("div", "tut-pair");
    card.append(tutEl("p", "tut-pair-prompt", pair.prompt));
    card.append(tutEl("p", "tut-pair-pos", `+: ${pair.positive}`));
    card.append(tutEl("p", "tut-pair-neg", `−: ${pair.negative}`));
    pairs.append(card);
  });
  wrap.append(pairs);
  if (block.stats && block.stats.length) {
    const stats = tutEl("div", "tut-metrics");
    block.stats.forEach((stat) => {
      const item = tutEl("div", "tut-metric");
      item.append(tutEl("span", "tut-metric-label", stat.label));
      item.append(tutEl("strong", "tut-metric-value", stat.value));
      stats.append(item);
    });
    wrap.append(stats);
  }
  return wrap;
}

function renderSampleBadges(block) {
  const wrap = tutEl("div", "tut-badges");
  block.items.forEach((item) => {
    const card = tutEl("div", `tut-badge ${item.matched ? "ok" : "bad"}`);
    const head = tutEl("div", "tut-badge-head");
    head.append(tutEl("span", "tut-badge-label", item.label));
    head.append(tutEl("span", "tut-badge-verdict", item.matched ? "✓ 一致" : "✗ 不一致"));
    card.append(head);
    card.append(tutEl("p", "tut-badge-text", item.text));
    if (item.note) card.append(tutEl("p", "tut-badge-note", item.note));
    wrap.append(card);
  });
  return wrap;
}

bootTutorial();
