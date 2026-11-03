/* 总览页：环境表单（shell 共用渲染）、章节卡片、运行历史 */
"use strict";

function describeEntry(entry) {
  if (entry.source === "local_path") return `本地模型 ${entry.path || "（未填）"}`;
  if (entry.source === "hf") return `HuggingFace ${entry.repo || "（未填）"}`;
  return `API ${entry.api ? entry.api.model : ""}`;
}

async function loadChapters() {
  const grid = document.getElementById("chapter-grid");
  grid.innerHTML = "<p class='muted'>加载中…</p>";
  let chapters;
  try {
    chapters = Shell.chapters.length ? Shell.chapters : await HUB.get("/api/chapters");
  } catch (err) {
    grid.innerHTML = `<p class="muted">章节加载失败：${escapeHtml(err.message)}</p>`;
    return;
  }
  grid.innerHTML = "";
  for (const chapter of chapters) {
    const card = el("div", { class: "chapter-card" + (chapter.enabled ? "" : " disabled") });
    card.append(
      el("div", { class: "num" }, `Chapter ${chapter.chapter}`),
      el("div", { class: "name" }, chapter.title),
      el("div", { class: "brief" }, chapter.brief || ""),
      el("div", { class: "meta" },
        el("span", null, chapter.toolkit),
        chapter.enabled
          ? el("span", { class: "chip dark" }, "进入 →")
          : el("span", { class: "chip gray" }, "建设中")),
    );
    if (chapter.enabled) {
      card.addEventListener("click", () => { location.href = `chapter.html?toolkit=${chapter.toolkit}`; });
    }
    grid.append(card);
  }
  if (!chapters.length) grid.innerHTML = "<p class='muted'>未发现任何章节工具包（检查各章目录下的 chapter.yaml）。</p>";
}

async function loadHistory() {
  const box = document.getElementById("history");
  let runs;
  try {
    runs = await HUB.get("/api/runs?limit=8");
  } catch (err) {
    box.innerHTML = `<p class="muted">历史加载失败：${escapeHtml(err.message)}</p>`;
    return;
  }
  box.innerHTML = "";
  if (!runs.length) {
    box.append(el("p", { class: "empty" }, "暂无运行记录。进入章节工作台生成 YAML 并执行后，这里会显示历史。"));
    return;
  }
  for (const run of runs) {
    box.append(el("div", { class: "history-item" },
      el("span", { html: stateChip(run.state) }),
      el("span", { class: "sess" }, run.session),
      el("span", { class: "muted" }, `第${run.chapter}章 · ${run.toolkit || ""}`),
      el("span", { class: "spacer" }),
      el("a", { href: `run.html?session=${run.session}` }, "查看运行")));
  }
}

function init() {
  Shell.init(null);
  renderEnvForm(document.getElementById("env-form"), () => Shell.renderTopbar(null));
  loadChapters();
  loadHistory();
}

init();
