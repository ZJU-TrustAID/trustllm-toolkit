/* 章节工作台：主区 = 该章自带前端（iframe）；右侧可展开执行配置面板 */
"use strict";

const toolkit = qs("toolkit");
let chapter = null;         // manifest public view
let serverState = "stopped_server";

/* ---- 模型角色状态 ----
   roles = [{role, title, accepts, required, useCommon, entry}]
   useCommon/entry 按章节持久化（localStorage），默认继承公共模型/清单默认值。 */
const roles = [];

/* ---- 章节级环境覆盖（持久化）：默认完全跟随顶栏公共参数 ---- */
const ChapterEnvStore = {
  key() { return `hub.env.${toolkit}.v1`; },
  defaults() {
    return {
      env: { enabled: false, device: "auto", dtype: "auto", seed: 0 },
      roles: {},
      collapsed: true,
    };
  },
  load() {
    const base = this.defaults();
    try {
      const value = JSON.parse(localStorage.getItem(this.key()));
      if (value && typeof value === "object") return {
        env: Object.assign(base.env, value.env || {}),
        roles: (value.roles && typeof value.roles === "object") ? value.roles : {},
        collapsed: !!value.collapsed,
      };
    } catch (e) { /* 损坏时回退默认 */ }
    return base;
  },
  save(state) { localStorage.setItem(this.key(), JSON.stringify(state)); },
};
const chapterEnv = ChapterEnvStore.load();

/* 本章真正生效的运行环境：未开启覆盖时与顶栏公共参数一致 */
function effectiveEnv() {
  if (!chapterEnv.env.enabled) return { device: common.device, dtype: common.dtype, seed: common.seed };
  return { device: chapterEnv.env.device, dtype: chapterEnv.env.dtype, seed: chapterEnv.env.seed };
}

function saveChapterEnv() {
  ChapterEnvStore.save({
    env: chapterEnv.env,
    roles: Object.fromEntries(roles.map((role) => [role.role, { useCommon: role.useCommon, entry: role.entry }])),
    collapsed: chapterEnv.collapsed,
  });
}

function defaultEntry(spec) {
  const d = spec.default || {};
  return { source: d.source || "local_path", path: d.path || "", repo: d.repo || "", api: { base_url: "", api_key_env: "OPENAI_API_KEY", model: "" } };
}

/* ---- 环境覆盖控件 ---- */
function renderEnvOverride() {
  const box = document.getElementById("env-override");
  box.innerHTML = "";
  const toggle = el("label", { style: "display:flex;align-items:center;gap:6px;font-size:12.5px;margin-bottom:8px" },
    el("input", {
      type: "checkbox",
      onchange: (e) => {
        chapterEnv.env.enabled = e.target.checked;
        if (chapterEnv.env.enabled && !chapterEnv.env._touched) {
          // 首次开启覆盖时，从当前公共参数取值作为起点，避免突变为 auto/0。
          chapterEnv.env.device = common.device;
          chapterEnv.env.dtype = common.dtype;
          chapterEnv.env.seed = common.seed;
          chapterEnv.env._touched = true;
        }
        saveChapterEnv();
        renderEnvOverride();
        refreshEnvCard();
      },
    }),
    "覆盖本章运行环境");
  toggle.querySelector("input").checked = !!chapterEnv.env.enabled;
  box.append(toggle);

  if (!chapterEnv.env.enabled) {
    box.append(el("p", { class: "muted", style: "margin:0" },
      `→ 跟随公共参数：设备 ${common.device} · dtype ${common.dtype} · seed ${common.seed}`));
    return;
  }
  const device = el("select", { onchange: (e) => { chapterEnv.env.device = e.target.value; saveChapterEnv(); refreshEnvCard(); } });
  fillDeviceOptions(device).then(() => selectValue(device, chapterEnv.env.device));
  const dtype = el("select", { onchange: (e) => { chapterEnv.env.dtype = e.target.value; saveChapterEnv(); refreshEnvCard(); } });
  for (const opt of ["auto", "bfloat16", "float16", "float32"]) dtype.append(el("option", { value: opt }, opt));
  selectValue(dtype, chapterEnv.env.dtype);
  const seed = el("input", { type: "number", value: String(chapterEnv.env.seed), min: "0", step: "1",
    oninput: (e) => { chapterEnv.env.seed = parseInt(e.target.value || "0", 10); saveChapterEnv(); refreshEnvCard(); } });
  box.append(el("div", { class: "row" },
    el("label", { class: "field" }, el("span", null, "设备"), device),
    el("label", { class: "field" }, el("span", null, "精度 dtype"), dtype),
    el("label", { class: "field" }, el("span", null, "随机种子 seed"), seed)));
}

/* ---- 模型角色卡片 ---- */
function renderRoles() {
  const box = document.getElementById("roles");
  box.innerHTML = "";
  roles.forEach((role) => {
    const block = el("div", { class: "role-block" });
    block.append(el("div", { class: "role-title" },
      role.title || role.role,
      role.required ? "" : el("span", { class: "muted" }, "（可选）")));
    const useCommon = el("label", { style: "display:flex;align-items:center;gap:6px;font-size:12.5px;margin-bottom:8px" },
      el("input", {
        type: "checkbox",
        onchange: (e) => { role.useCommon = e.target.checked; saveChapterEnv(); refreshEnvCard(); renderRoleFields(role, fields); },
      }),
      "同公共模型");
    if (role.useCommon) useCommon.querySelector("input").checked = true;
    block.append(useCommon);
    const fields = el("div", { "data-role-fields": role.role });
    block.append(fields);
    renderRoleFields(role, fields);
    box.append(block);
  });
}

function renderRoleFields(role, container) {
  container.innerHTML = "";
  if (role.useCommon) {
    const entry = CommonStore.entry(common);
    container.append(el("p", { class: "muted", style: "margin:0" }, "→ ", describeEntry(entry)));
    return;
  }
  const disabled = !!role.disabled;
  if (disabled) {
    container.append(el("p", { class: "muted", style: "margin:0 0 8px" },
      "⏸ 下方当前选择的方法不需要该模型，输入已锁定。"));
  }
  const e = role.entry;
  const radio = el("div", { class: "radio-group" });
  const labels = { local_path: "本地路径", hf: "HuggingFace", api: "API" };
  for (const source of role.accepts || ["local_path", "hf"]) {
    const attrs = { type: "radio", name: `role-source-${role.role}`, value: source };
    if (disabled) attrs.disabled = "";
    const input = el("input", attrs);
    input.checked = e.source === source;
    input.addEventListener("change", () => {
      e.source = input.value;
      saveChapterEnv(); refreshEnvCard();
      renderRoleFields(role, container);
    });
    radio.append(el("label", null, input, labels[source] || source));
  }
  container.append(radio);

  const bindEntryInput = (key, path, placeholder, after) => {
    const attrs = {
      type: "text",
      value: (key === "api" ? e.api[path] : e[path]) || "",
      placeholder,
      oninput: (ev) => {
        if (key === "api") e.api[path] = ev.target.value; else e[path] = ev.target.value;
        saveChapterEnv();
        if (after) refreshEnvCard();
      },
    };
    if (disabled) attrs.disabled = "";
    return el("input", attrs);
  };
  if (e.source === "local_path") {
    const pathInput = bindEntryInput("entry", "path", "/data2/models/… 或 models/…", true);
    container.append(el("label", { class: "field" }, el("span", null, "本地模型目录（绝对路径）"), pathInput));
    if (!disabled) {
      attachLocalModelSelect(container, pathInput, () => {
        e.path = pathInput.value;
        saveChapterEnv(); refreshEnvCard();
      });
    }
  } else if (e.source === "hf") {
    container.append(el("label", { class: "field" },
      el("span", null, "HuggingFace 仓库 ID"), bindEntryInput("entry", "repo", "org/model-name", true)));
  } else {
    container.append(el("div", { class: "row" },
      el("label", { class: "field" }, el("span", null, "base_url"), bindEntryInput("api", "base_url", "https://api.openai.com/v1", false)),
      el("label", { class: "field" }, el("span", null, "API Key 环境变量名"), bindEntryInput("api", "api_key_env", "OPENAI_API_KEY", false)),
      el("label", { class: "field" }, el("span", null, "模型名"), bindEntryInput("api", "model", "gpt-4o-mini", true))));
  }
}

function describeEntry(entry) {
  if (entry.source === "local_path") return `本地模型 ${entry.path || "（未填）"}`;
  if (entry.source === "hf") return `HuggingFace ${entry.repo || "（未填）"}`;
  return `API ${entry.api ? entry.api.model : ""}`;
}

function entryUsable(entry) {
  if (!entry || !entry.source) return false;
  if (entry.source === "local_path") return !!(entry.path || "").trim();
  if (entry.source === "hf") return !!(entry.repo || "").trim();
  if (entry.source === "api") return !!((entry.api && entry.api.model) || "").trim();
  return false;
}

function collectRoles() {
  const models = {};
  for (const role of roles) {
    const entry = role.useCommon ? CommonStore.entry(common) : role.entry;
    // 可选角色（如辅助模型）留空时不进 YAML，由章节侧回退默认。
    if (!role.required && !entryUsable(entry)) continue;
    models[role.role] = entry;
  }
  return models;
}

/* ---- 折叠卡片：摘要 + 展开/收起 ---- */
function refreshEnvCard() {
  const env = effectiveEnv();
  const mode = document.getElementById("env-card-mode");
  mode.textContent = chapterEnv.env.enabled ? "（本章覆盖）" : "（继承公共参数）";
  const roleText = roles.map((role) => {
    const entry = role.useCommon ? CommonStore.entry(common) : role.entry;
    const off = role.disabled && !role.useCommon ? "（下方方法未使用）" : "";
    return `${role.title || role.role}：${describeEntry(entry)}${off}`;
  }).join("；");
  document.getElementById("env-card-summary").textContent =
    `设备 ${env.device} · dtype ${env.dtype} · seed ${env.seed} · ${roleText}`;
}

function toggleEnvCard(expand) {
  const card = document.getElementById("env-card");
  const body = document.getElementById("env-card-body");
  chapterEnv.collapsed = expand === undefined ? !chapterEnv.collapsed : !expand;
  body.hidden = chapterEnv.collapsed;
  card.classList.toggle("open", !chapterEnv.collapsed);
  saveChapterEnv();
}

function validateChapterEnv(box) {
  box.innerHTML = "";
  const models = collectRoles();
  const checks = Object.entries(models).map(([role, entry]) =>
    HUB.post("/api/models/validate", { entry, toolkit, role })
      .then((report) => ({ role, report }))
      .catch((err) => ({ role, error: err })));
  Promise.all(checks).then((results) => {
    const list = el("ul", { class: "checks" });
    let allOk = true;
    const titleOf = (role) => {
      const spec = roles.find((item) => item.role === role);
      return (spec && spec.title) || role;
    };
    for (const result of results) {
      if (result.error) {
        allOk = false;
        list.append(el("li", { class: "bad" }, `${titleOf(result.role)} — ${result.error.message}`));
        continue;
      }
      if (!result.report.ok) allOk = false;
      for (const check of result.report.checks) {
        list.append(el("li", { class: check.ok ? "" : "bad" },
          `${titleOf(result.role)} · ${check.name}${check.detail ? " — " + check.detail : ""}`));
      }
    }
    box.append(el("p", { style: "margin:6px 0 2px" }, allOk ? "✅ " : "⚠️ ",
      el("span", { class: allOk ? "role-ok" : "role-bad" }, allOk ? "校验通过" : "存在未通过项")));
    box.append(list);
  });
}

/* ---- 工作台方法联动：章节前端通过 postMessage 声明当前方法需要哪些模型角色 ----
   消息格式 {type: "trustllm-hub-model-needs", needs: {<role>: true|false}}，
   来源必须是本章 iframe；未声明的角色保持可编辑。 */
let modelNeeds = {};

function applyModelNeeds() {
  let changed = false;
  for (const role of roles) {
    const need = modelNeeds[role.role];
    const disabled = need === false;
    if (role.disabled !== disabled) {
      role.disabled = disabled;
      changed = true;
    }
  }
  if (!changed) return;
  refreshEnvCard();
  for (const role of roles) {
    if (role.useCommon) continue;
    const fields = document.querySelector(`#roles [data-role-fields="${role.role}"]`);
    if (fields) renderRoleFields(role, fields);
  }
}

/* ---- 章节参数表单 ---- */
const paramValues = {};

function renderParams() {
  const box = document.getElementById("params");
  box.innerHTML = "";
  for (const spec of chapter.params || []) {
    const key = spec.key;
    if (!(key in paramValues)) paramValues[key] = spec.default !== undefined ? spec.default : "";
    const wrap = el("label", { class: "field", "data-when-key": (spec.when && Object.keys(spec.when)[0]) || "", "data-when-value": spec.when ? Object.values(spec.when)[0] : "" });
    const type = spec.type || "text";
    let input;
    if (type === "enum") {
      input = el("select", { onchange: (e) => { paramValues[key] = e.target.value; syncWhen(); } });
      for (const option of spec.options) input.append(el("option", { value: option }, option));
      input.value = String(paramValues[key]);
      wrap.append(el("span", null, spec.title || key), input);
    } else if (type === "bool") {
      input = el("input", { type: "checkbox", onchange: (e) => { paramValues[key] = e.target.checked; } });
      input.checked = !!paramValues[key];
      wrap.style.display = "flex";
      wrap.style.alignItems = "center";
      wrap.style.gap = "6px";
      wrap.append(input, el("span", { style: "margin:0" }, spec.title || key));
      box.append(wrap);
      continue;
    } else if (type === "int" || type === "float") {
      input = el("input", { type: "number", step: type === "int" ? "1" : "any", value: String(paramValues[key]),
        oninput: (e) => { paramValues[key] = type === "int" ? parseInt(e.target.value || "0", 10) : parseFloat(e.target.value || "0"); } });
      wrap.append(el("span", null, spec.title || key), input);
    } else {
      input = el("textarea", { rows: key === "prompt" || key === "text" || key === "response" ? 3 : 2,
        oninput: (e) => { paramValues[key] = e.target.value; } });
      input.value = String(paramValues[key] == null ? "" : paramValues[key]);
      wrap.append(el("span", null, spec.title || key), input);
    }
    box.append(wrap);
  }
  syncWhen();
}

function syncWhen() {
  document.querySelectorAll("#params .field[data-when-key]").forEach((node) => {
    const key = node.getAttribute("data-when-key");
    if (!key) return;
    const value = node.getAttribute("data-when-value");
    node.style.display = String(paramValues[key]) === value ? "" : "none";
  });
}

/* ---- 组合统一 YAML ---- */
function composeConfig() {
  return {
    version: 1,
    chapter: chapter.chapter,
    toolkit: chapter.toolkit,
    common: effectiveEnv(),
    models: collectRoles(),
    chapter_params: JSON.parse(JSON.stringify(paramValues)),
  };
}

function openYamlModal() {
  const payload = composeConfig();
  HUB.post("/api/yaml", payload).then((result) => {
    document.getElementById("yaml-text").value = result.yaml;
    document.getElementById("yaml-error").textContent = "";
    document.getElementById("yaml-modal").style.display = "flex";
  }).catch((err) => toast("生成 YAML 失败：" + err.message, true));
}

async function revalidateYaml() {
  const text = document.getElementById("yaml-text").value;
  try {
    const result = await HUB.post("/api/yaml", { yaml: text });
    document.getElementById("yaml-text").value = result.yaml;
    document.getElementById("yaml-error").textContent = "";
    toast("YAML 校验通过");
  } catch (err) {
    document.getElementById("yaml-error").textContent = "校验失败：" + err.message;
  }
}

async function executeRun() {
  const text = document.getElementById("yaml-text").value;
  let session;
  try {
    const result = await HUB.post("/api/runs", { yaml: text });
    session = result.session;
  } catch (err) {
    document.getElementById("yaml-error").textContent = "执行失败：" + err.message;
    return;
  }
  location.href = `run.html?session=${session}`;
}

/* ---- 交互前端服务 ---- */
async function refreshServer(keepFrame) {
  const status = await HUB.get(`/api/chapters/${toolkit}/server`);
  serverState = status.state;
  const chip = document.getElementById("server-chip");
  chip.innerHTML = stateChip(status.state === "stopped" ? "stopped_server" : status.state);
  document.getElementById("btn-server").disabled = status.state === "ready" || status.state === "starting";
  document.getElementById("btn-server-stop").disabled = !status.owned;
  const sub = document.getElementById("chapter-sub");
  if (status.state === "starting") {
    const tail = status.log_tail ? ` · ${status.log_tail}` : "";
    sub.textContent = `启动中${serverStartedAt ? `（已 ${Math.round((Date.now() - serverStartedAt) / 1000)}s）` : ""}${tail}`;
  } else if (status.state === "ready") {
    sub.textContent = `第${chapter.chapter}章 · ${chapter.toolkit} · 以该章自带前端为主体`;
  }
  if (status.state === "ready") {
    serverStartedAt = null;
    const area = document.getElementById("embed-area");
    const existing = document.getElementById("chapter-iframe");
    if (existing && keepFrame) return;
    area.innerHTML = "";
    area.append(el("iframe", { id: "chapter-iframe", class: "embed-frame", src: status.url }));
  } else if (!keepFrame) {
    const area = document.getElementById("embed-area");
    if (status.state !== "starting") {
      area.innerHTML = "";
      area.append(el("div", { class: "embed-placeholder" },
        el("div", { class: "big" }, "交互前端尚未启动"),
        el("div", null, "点击上方「启动交互服务」，该章自带前端会显示在这里，方法都在其中选择。")));
    }
  }
}

let serverStartedAt = null;
let serverStartTimer = null;

async function startServer() {
  const button = document.getElementById("btn-server");
  button.disabled = true;
  serverStartedAt = Date.now();
  // 启动期间每 2s 轮询一次，把 vLLM 等后端的加载进度显示在工具栏。
  clearInterval(serverStartTimer);
  serverStartTimer = setInterval(() => {
    HUB.get(`/api/chapters/${toolkit}/server`).then((s) => {
      if (s.state === "starting") {
        const tail = s.log_tail ? ` · ${s.log_tail}` : "";
        document.getElementById("chapter-sub").textContent =
          `启动中（已 ${Math.round((Date.now() - serverStartedAt) / 1000)}s）${tail}`;
      }
    }).catch(() => {});
  }, 2000);
  try {
    const status = await HUB.post(`/api/chapters/${toolkit}/server/start`, {
      models: collectRoles(),
      common: effectiveEnv(),
    });
    clearInterval(serverStartTimer);
    if (status.state === "ready") { await refreshServer(); toast("交互服务已就绪"); }
    else { await refreshServer(); }
  } catch (err) {
    clearInterval(serverStartTimer);
    await refreshServer();
    toast("启动失败：" + err.message, true);
  } finally {
    button.disabled = false;
  }
}

/* ---- 初始化 ---- */
async function init() {
  if (!toolkit) { location.href = "index.html"; return; }
  await Shell.init(toolkit);
  const chapters = Shell.chapters.length ? Shell.chapters : await HUB.get("/api/chapters");
  chapter = chapters.find((item) => item.toolkit === toolkit);
  if (!chapter) { toast("未找到该章节", true); setTimeout(() => { location.href = "index.html"; }, 800); return; }
  document.title = `${chapter.title} · TrustLLM Toolkit Hub`;
  document.getElementById("chapter-title").textContent = chapter.title;
  document.getElementById("chapter-sub").textContent = `第${chapter.chapter}章 · ${chapter.toolkit} · 以该章自带前端为主体`;

  for (const spec of chapter.models || []) {
    const persisted = chapterEnv.roles[spec.role] || {};
    roles.push({
      role: spec.role, title: spec.title, accepts: spec.accepts, required: spec.required !== false,
      useCommon: persisted.useCommon !== undefined ? persisted.useCommon : spec.role === "default",
      entry: Object.assign(defaultEntry(spec), persisted.entry || {}),
    });
  }
  renderRoles();
  renderEnvOverride();
  refreshEnvCard();
  toggleEnvCard(!chapterEnv.collapsed);
  renderParams();
  refreshServer();

  document.getElementById("env-card-toggle").addEventListener("click", () => toggleEnvCard());
  document.getElementById("btn-env-validate").addEventListener("click", (e) => {
    validateChapterEnv(document.getElementById("chapter-env-validate-result"));
    e.target.blur();
  });
  /* 顶栏公共参数变化时，继承项（未开启覆盖的环境、勾选「同公共模型」的角色）实时跟随 */
  window.addEventListener("hub:common-changed", () => {
    renderEnvOverride();
    refreshEnvCard();
    for (const role of roles) {
      if (!role.useCommon) continue;
      const fields = document.querySelector(`#roles [data-role-fields="${role.role}"]`);
      if (fields) renderRoleFields(role, fields);
    }
  });

  document.getElementById("btn-yaml").addEventListener("click", openYamlModal);
  document.getElementById("btn-run").addEventListener("click", () => {
    document.getElementById("config-panel").classList.add("open");
    openYamlModal();
  });
  document.getElementById("btn-yaml-close").addEventListener("click", () => { document.getElementById("yaml-modal").style.display = "none"; });
  document.getElementById("btn-yaml-revalidate").addEventListener("click", revalidateYaml);
  document.getElementById("btn-yaml-download").addEventListener("click", () => {
    const blob = new Blob([document.getElementById("yaml-text").value], { type: "text/yaml" });
    const link = el("a", { href: URL.createObjectURL(blob), download: `hub-${toolkit}.yaml` });
    link.click();
  });
  document.getElementById("btn-run-modal").addEventListener("click", executeRun);
  document.getElementById("btn-server").addEventListener("click", startServer);
  document.getElementById("btn-server-stop").addEventListener("click", async () => {
    await HUB.post(`/api/chapters/${toolkit}/server/stop`);
    refreshServer();
  });
  document.getElementById("btn-panel").addEventListener("click", () => {
    document.getElementById("config-panel").classList.toggle("open");
  });

  /* 轻量章节（manifest 声明 autostart: true）进入工作台即自动启动交互前端。 */
  if (chapter.frontend && chapter.frontend.autostart && chapter.frontend.state) {
    HUB.get(`/api/chapters/${toolkit}/server`).then((status) => {
      if (status.state === "stopped") startServer();
    }).catch(() => {});
  }

  /* 扩展点：章节前端可通过 postMessage 把内部配置回传给 hub。
     消息格式 {type: "trustllm-hub-params", chapter_params: {...}}，来源须为当前章节服务。 */
  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || typeof data.type !== "string") return;
    const frame = document.getElementById("chapter-iframe");
    const fromChapter = frame && event.source === frame.contentWindow;
    if (data.type === "trustllm-hub-params" && data.chapter_params && fromChapter) {
      Object.assign(paramValues, data.chapter_params);
      renderParams();
      document.getElementById("config-panel").classList.add("open");
      toast("已接收章节前端回传的参数");
      return;
    }
    if (data.type === "trustllm-hub-model-needs" && data.needs && fromChapter) {
      modelNeeds = data.needs;
      applyModelNeeds();
    }
  });
}

init().catch((err) => toast("页面初始化失败：" + err.message, true));
