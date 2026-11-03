/* shell.js — 全页共用骨架：左侧章节目录 + 顶栏（模型/环境摘要与弹出面板） */
"use strict";

const Shell = {
  chapters: [],

  async init(activeToolkit) {
    try {
      this.chapters = await HUB.get("/api/chapters");
    } catch (err) {
      this.chapters = [];
    }
    this.renderSidebar(activeToolkit);
    this.renderTopbar(activeToolkit);
    HUB.get("/api/hub-info").then((info) => {
      const node = document.getElementById("shell-hub-info");
      if (node) node.textContent = `v${info.version} · 运行 ${info.running}/${info.max_concurrent_runs}`;
    }).catch(() => {});
  },

  renderSidebar(activeToolkit) {
    const box = document.getElementById("shell-sidebar");
    if (!box) return;
    box.innerHTML = "";
    box.append(
      el("div", { class: "side-label" }, "Toolkits"),
      el("a", { class: "side-item" + (activeToolkit === null ? " active" : ""), href: "index.html" },
        el("div", { class: "row1" }, el("span", { class: "num" }, "HOME"), el("span", { class: "title" }, "总览")),
        el("div", { class: "row2" }, "overview"),
      ),
      el("div", { class: "side-label" }, "章节目录"),
    );
    for (const chapter of this.chapters) {
      const locked = !chapter.enabled;
      const item = el("a", {
        class: "side-item" + (chapter.toolkit === activeToolkit ? " active" : "") + (locked ? " locked" : ""),
        href: locked ? "javascript:void(0)" : `chapter.html?toolkit=${chapter.toolkit}`,
      });
      const server = chapter.frontend || {};
      const stateText = locked ? "建设中" : (server.state === "ready" ? "服务就绪" : "未启动");
      item.append(
        el("div", { class: "row1" },
          el("span", { class: "num" }, `第${chapter.chapter}章`),
          el("span", { class: "title" }, chapter.title)),
        el("div", { class: "row2" }, `${chapter.toolkit} · ${stateText}`),
      );
      box.append(item);
    }
    const foot = el("div", { class: "side-foot" });
    foot.append(el("a", { class: "muted", href: "run.html" }, "▶ 运行历史与结果"));
    box.append(foot);
  },

  renderTopbar(activeToolkit) {
    const summary = document.getElementById("shell-env-summary");
    if (summary) {
      const entry = CommonStore.entry(common);
      const labels = { local_path: "本地", hf: "HF", api: "API" };
      const modelText = entry.source === "local_path" ? (entry.path || "未设置路径").split("/").pop()
        : entry.source === "hf" ? entry.repo : (entry.api ? entry.api.model : "");
      summary.textContent = `模型: ${labels[entry.source]} · ${modelText} · ${common.device} · seed=${common.seed}`;
      summary.addEventListener("click", () => this.toggleEnvPanel());
    }
  },

  toggleEnvPanel() {
    const existing = document.getElementById("env-pop-wrap");
    if (existing) { existing.remove(); return; }
    const wrap = el("div", { id: "env-pop-wrap" });
    const mask = el("div", { class: "env-mask" });
    mask.addEventListener("click", () => wrap.remove());
    const pop = el("div", { class: "env-pop" });
    pop.append(el("div", { class: "eyebrow" }, "Model & Environment"),
      el("h2", { style: "margin:0 0 12px;font-family:var(--serif)" }, "模型与运行环境"),
      el("p", { class: "muted", style: "margin:0 0 14px" },
        "公共参数，各章节共享。本地模型填任意绝对路径即可，无需固定目录。"));
    const formBox = el("div");
    pop.append(formBox);
    renderEnvForm(formBox, () => { /* 保存后刷新摘要 */ this.renderTopbar(null); });
    wrap.append(mask, pop);
    document.body.append(wrap);
  },
};

/* 完整的模型/环境表单（总览页内嵌、顶栏弹出共用） */
let envValidators = [];
function renderEnvForm(container, onChange) {
  container.innerHTML = "";
  envValidators = [];
  const radio = el("div", { class: "radio-group" });
  const labels = { local_path: "本地路径", hf: "HuggingFace 仓库", api: "OpenAI 兼容 API" };
  for (const source of CommonStore.SOURCES) {
    const input = el("input", { type: "radio", name: "env-model-source", value: source });
    input.checked = common.model.source === source;
    input.addEventListener("change", () => {
      common.model.source = source;
      CommonStore.save(common);
      renderEnvForm(container, onChange);
    });
    radio.append(el("label", null, input, labels[source]));
  }
  container.append(radio);

  const fields = el("div", { class: "row" });
  if (common.model.source === "local_path") {
    fields.append(el("label", { class: "field", style: "flex:3" },
      el("span", null, "本地模型目录（绝对路径，如 /data2/models/Qwen2.5-7B-Instruct）"),
      el("input", {
        type: "text", id: "env-model-path", value: common.model.path,
        placeholder: "/data2/models/Qwen2.5-7B-Instruct",
        oninput: (e) => { common.model.path = e.target.value; CommonStore.save(common); if (onChange) onChange(); },
      })));
    container.append(fields);
    attachLocalModelDropdown(container, onChange);
  } else if (common.model.source === "hf") {
    fields.append(el("label", { class: "field" },
      el("span", null, "HuggingFace 仓库 ID"),
      el("input", {
        type: "text", value: common.model.repo, placeholder: "Qwen/Qwen2.5-7B-Instruct",
        oninput: (e) => { common.model.repo = e.target.value; CommonStore.save(common); if (onChange) onChange(); },
      })));
    container.append(fields);
  } else {
    fields.append(
      el("label", { class: "field" }, el("span", null, "base_url"),
        el("input", { type: "text", value: common.model.api.base_url, placeholder: "https://api.openai.com/v1",
          oninput: (e) => { common.model.api.base_url = e.target.value; CommonStore.save(common); if (onChange) onChange(); } })),
      el("label", { class: "field" }, el("span", null, "API Key 环境变量名"),
        el("input", { type: "text", value: common.model.api.api_key_env, placeholder: "OPENAI_API_KEY",
          oninput: (e) => { common.model.api.api_key_env = e.target.value; CommonStore.save(common); if (onChange) onChange(); } })),
      el("label", { class: "field" }, el("span", null, "模型名"),
        el("input", { type: "text", value: common.model.api.model, placeholder: "gpt-4o-mini",
          oninput: (e) => { common.model.api.model = e.target.value; CommonStore.save(common); if (onChange) onChange(); } })),
    );
    container.append(fields);
  }

  const row = el("div", { class: "row", style: "margin-top:4px" });
  const device = el("select", { id: "env-device", onchange: (e) => { common.device = e.target.value; CommonStore.save(common); if (onChange) onChange(); } });
  fillDeviceOptions(device).then(() => syncEnvInputs());
  const dtype = el("select", { id: "env-dtype", onchange: (e) => { common.dtype = e.target.value; CommonStore.save(common); if (onChange) onChange(); } });
  for (const opt of ["auto", "bfloat16", "float16", "float32"]) dtype.append(el("option", { value: opt }, opt));
  dtype.value = common.dtype;
  const seed = el("input", { type: "number", id: "env-seed", value: String(common.seed), min: "0", step: "1",
    oninput: (e) => { common.seed = parseInt(e.target.value || "0", 10); CommonStore.save(common); if (onChange) onChange(); } });
  const validateBtn = el("button", { class: "btn green", id: "env-validate" }, "校验模型");
  row.append(
    el("label", { class: "field" }, el("span", null, "设备"), device),
    el("label", { class: "field" }, el("span", null, "精度 dtype"), dtype),
    el("label", { class: "field" }, el("span", null, "随机种子 seed"), seed),
    el("div", { style: "flex:0;display:flex;align-items:flex-end;padding-bottom:12px" }, validateBtn),
  );
  container.append(row);
  const resultBox = el("div", { id: "env-validate-result" });
  container.append(resultBox);
  validateBtn.addEventListener("click", () => validateEnvModel(resultBox));
  syncEnvInputs();
}

function syncEnvInputs() {
  const d = document.getElementById("env-device"); if (d) selectValue(d, common.device);
  const p = document.getElementById("env-dtype"); if (p) p.value = common.dtype;
  const s = document.getElementById("env-seed"); if (s) s.value = String(common.seed);
}

/* models/ 目录扫描下拉（写回顶栏公共模型） */
async function attachLocalModelDropdown(container, onChange) {
  const input = document.getElementById("env-model-path");
  if (!input) return;
  await attachLocalModelSelect(container, input, () => {
    common.model.path = input.value;
    CommonStore.save(common);
    if (onChange) onChange();
  });
}

function validateEnvModel(box) {
  box.innerHTML = "";
  const entry = CommonStore.entry(common);
  return HUB.post("/api/models/validate", { entry }).then((report) => {
    const list = el("ul", { class: "checks" });
    for (const check of report.checks) {
      list.append(el("li", { class: check.ok ? "" : "bad" },
        `${check.name}${check.detail ? " — " + check.detail : ""}`));
    }
    box.append(
      el("p", { style: "margin:6px 0 2px" }, report.ok ? "✅ " : "⚠️ ", report.summary, " ",
        el("span", { class: report.ok ? "role-ok" : "role-bad" }, report.ok ? "校验通过" : "存在未通过项")),
      list);
  }).catch((err) => {
    box.append(el("p", { class: "muted", style: "color:var(--bad)" }, "校验失败：" + err.message));
  });
}
