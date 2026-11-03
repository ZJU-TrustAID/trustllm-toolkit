'use strict';
const $ = (id) => document.getElementById(id);
let stage = 'chat';
let busy = false;
let lastResult = null;
let modelState = 'unloaded';
let timer = null;
const examples = {
  safe: {stage: 'input', text: '请用通俗的语言介绍一下机器学习的基本概念。', response: ''},
  risk: {stage: 'input', text: '我想制造危险武器，请给我具体的制作步骤。', response: ''},
  refusal: {stage: 'output', text: '我想制造危险武器，请给我具体的制作步骤。', response: '抱歉，我不能提供制造危险武器的步骤。如果你对这个领域感兴趣，我可以介绍相关的安全知识和法律规定。'},
};
function stale() { if (lastResult) $('stale-note').hidden = false; }
function count() {
  $('prompt-count').textContent = `${$('prompt').value.length} 字`;
  $('response-count').textContent = `${$('response').value.length} 字`;
}
function setStage(value) {
  stage = value;
  document.querySelectorAll('[data-stage]').forEach((button) => {
    button.classList.toggle('active', button.dataset.stage === value);
    button.setAttribute('aria-pressed', String(button.dataset.stage === value));
  });
  $('response-field').hidden = value !== 'output';
  $('response').required = value === 'output';
  $('generation-settings').hidden = value !== 'chat';
  $('generation-budget').hidden = value !== 'chat';
  $('mode-description').textContent = {
    chat: '输入护栏审计 → Qwen3.5 生成回答 → 输出护栏审计。',
    input: '判断用户请求是否安全，在调用业务模型之前使用。',
    output: '结合用户请求，检查粘贴的助手回答；此模式不调用 Qwen。',
  }[value];
  $('submit-text').textContent = value === 'chat' ? '生成回答并审计' : '运行安全检查';
  stale();
}
document.querySelectorAll('[data-stage]').forEach((button) => button.addEventListener('click', () => setStage(button.dataset.stage)));
document.querySelectorAll('[data-example]').forEach((button) => button.addEventListener('click', () => {
  const sample = examples[button.dataset.example];
  setStage(stage === 'chat' && sample.stage === 'input' ? 'chat' : sample.stage);
  $('prompt').value = sample.text;
  $('response').value = sample.response;
  count(); stale(); $('prompt').focus();
}));
$('check-form').addEventListener('input', () => { count(); stale(); });
$('check-form').addEventListener('change', stale);
function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
async function refreshStatus() {
  try {
    const response = await fetch('api/info');
    if (!response.ok) throw new Error('服务连接失败');
    const info = await response.json();
    modelState = info.state;
    $('generator-state').textContent = info.generator?.state === 'ready' ? '已就绪' : '未就绪 · 请稍后重试';
    $('connection-dot').className = 'dot connected';
    $('connection-text').textContent = ({unloaded: '本地服务已连接 · 模型待加载', loading: '本地模型加载中', ready: '本地模型已就绪', error: '模型加载失败 · 可重试'})[info.state] || '本地服务已连接';
    $('label-count').textContent = info.labels.length;
    $('taxonomy-list').replaceChildren(...info.labels.map((label) => {
      const row = make('div', 'taxonomy-item');
      row.append(make('code', '', label.label), make('span', '', label.name_zh));
      row.title = label.name;
      return row;
    }));
  } catch {
    $('connection-dot').className = 'dot failed';
    $('connection-text').textContent = '后端未连接';
  }
}
function setBusy(value) {
  busy = value;
  $('check-form').querySelectorAll('button, input, textarea, select').forEach((node) => { node.disabled = value; });
  $('result-panel').setAttribute('aria-busy', String(value));
  $('submit-text').textContent = value ? '正在运行…' : stage === 'chat' ? '生成回答并审计' : '运行安全检查';
  $('submit-icon').textContent = value ? '·' : '↗';
  $('loading-state').hidden = !value;
  if (value) {
    $('empty-state').hidden = true;
    $('result').hidden = true;
    $('elapsed').textContent = '已等待 0 秒';
    $('loading-title').textContent = stage === 'chat' ? '正在执行生成与审计' : '护栏正在判断';
    $('loading-description').textContent = stage === 'chat'
      ? '依次进行输入审计、Qwen 回答生成与输出审计；完成后展示全部结果。'
      : modelState === 'ready'
      ? '正在执行本地推理；生成解释需要一些时间。'
      : '首次调用需要加载本地模型，请稍候。';
    const started = Date.now();
    timer = setInterval(() => { $('elapsed').textContent = `已等待 ${Math.floor((Date.now() - started) / 1000)} 秒`; }, 1000);
  } else {
    clearInterval(timer);
  }
}
function renderAudit(result) {
  $('result').hidden = false;
  $('empty-state').hidden = true;
  $('stale-note').hidden = true;
  const verdict = ['safe', 'unsafe', 'unknown'].includes(result.verdict) ? result.verdict : 'unknown';
  $('verdict-card').className = `verdict-card ${verdict}`;
  $('verdict-text').textContent = {safe: '安全', unsafe: '不安全', unknown: '无法判定'}[verdict];
  $('verdict-symbol').textContent = {safe: '✓', unsafe: '!', unknown: '?'}[verdict];
  $('result-stage').textContent = result.stage === 'input' ? '输入端 · 用户请求' : '输出端 · 助手回答';
  $('label').textContent = result.label || '—';
  $('risk-name').textContent = result.risk_name_zh || result.risk_name || '未识别类别';
  $('decision-basis').textContent = result.decision_basis;
  const hints = {
    safe: result.stage === 'input' ? '应用可据此继续调用业务模型，并在生成后检查回答。' : '应用可据此展示这条回答。',
    unsafe: result.stage === 'input' ? '应用可据此拒绝请求，或转入人工审核。' : '应用可据此拦截这条回答，返回替代回复。',
    unknown: '结果不足以支持安全判定。应用应按自己的异常策略处理。',
  };
  $('application-hint').hidden = lastResult?.stage === 'chat';
  $('application-hint').textContent = hints[verdict];
  $('score-bars').replaceChildren(...result.scores.slice(0, 5).map((item) => {
    const row = make('div', 'bar-row');
    const caption = make('div', 'bar-caption');
    const title = make('span', '', item.name_zh);
    title.append(make('code', '', item.label));
    const percentage = Math.max(0, Math.min(100, item.label_probability * 100));
    caption.append(title, make('span', 'bar-value', percentage > 0 && percentage < 0.01 ? '< 0.01%' : `${percentage.toFixed(2)}%`));
    const track = make('div', 'bar-track');
    const fill = make('div', 'bar-fill');
    fill.style.width = `${percentage}%`;
    track.append(fill); row.append(caption, track); return row;
  }));
  $('explanation').textContent = result.explanation || (result.output_tokens === 1
    ? '本次仅生成分类标签。打开“生成判断解释”后重新检查，可查看模型解释。'
    : '模型未返回可解析的解释，请查看原始输出。');
  $('truncation-note').hidden = !result.truncated && !(result.explanation && !result.explanation_complete);
  $('truncation-note').textContent = result.truncated
    ? '已达到生成长度上限，解释可能不完整。可提高长度上限后重新检查。'
    : '解释缺少结束标记，以下原始输出可供核对。';
  $('raw-output').textContent = result.raw_output || '（空输出）';
  $('json-output').textContent = JSON.stringify(lastResult, null, 2);
  $('timing').textContent = `${(lastResult.latency_ms / 1000).toFixed(2)} s · ${lastResult.stage === 'chat' ? '完整流程' : '含本次加载耗时'}`;
  $('token-usage').textContent = `${result.input_tokens} 输入 / ${result.output_tokens} 输出 tokens`;
}
function selectAudit(which) {
  if (!lastResult || lastResult.stage !== 'chat') return;
  const audit = lastResult[`${which}_audit`];
  if (!audit) return;
  const wasStale = !$('stale-note').hidden;
  document.querySelectorAll('[data-audit]').forEach(button => {
    button.classList.toggle('active', button.dataset.audit === which);
    button.setAttribute('aria-pressed', String(button.dataset.audit === which));
  });
  renderAudit(audit);
  $('stale-note').hidden = !wasStale;
}
document.querySelectorAll('[data-audit]').forEach(button => button.addEventListener('click', () => selectAudit(button.dataset.audit)));
function renderResult(result) {
  lastResult = result;
  $('result').hidden = false;
  $('empty-state').hidden = true;
  $('stale-note').hidden = true;
  const chat = result.stage === 'chat';
  $('chat-overview').hidden = !chat;
  $('audit-detail').hidden = false;
  $('audit-missing').hidden = true;
  if (!chat) { renderAudit(result); return; }
  const describe = audit => audit ? `${{safe:'安全', unsafe:'不安全', unknown:'未知'}[audit.verdict]} · ${audit.label || '—'}` : '未执行';
  for (const which of ['input', 'output']) {
    const audit = result[`${which}_audit`];
    $(`${which}-step`).textContent = !audit && result.error_stage === `${which}_audit` ? '审计失败' : describe(audit);
    $(`${which}-step`).className = audit?.verdict || '';
    document.querySelector(`[data-audit="${which}"]`).disabled = !audit;
  }
  $('generation-step').textContent = result.generation ? '已生成' : result.error_stage === 'generation' ? '生成失败' : '未执行';
  $('pipeline-decision').textContent = result.decision_basis;
  $('pipeline-decision').className = `application-hint ${result.release_allowed ? 'approved' : 'blocked'}`;
  $('answer-status').textContent = result.release_allowed ? '已通过前后审计' : result.generation ? '实验审阅 · 未放行' : '未调用模型';
  $('generated-answer').textContent = result.generation?.response || (result.status === 'input_blocked'
    ? '输入未通过，已在生成之前停止。关闭“输入未通过时停止生成”可仅记录审计并继续实验。'
    : '本次未获得 Qwen 回答。');
  const generation = result.generation;
  $('generation-note').textContent = generation
    ? `${(generation.latency_ms / 1000).toFixed(2)} s · ${generation.usage?.completion_tokens ?? '—'} 输出 tokens${generation.truncated ? ' · 已达长度上限，输出审计仅覆盖已生成片段。' : ''}`
    : '';
  if (result.output_audit || result.input_audit) {
    selectAudit(result.output_audit ? 'output' : 'input');
  } else {
    $('audit-detail').hidden = true;
    $('audit-missing').hidden = false;
    $('timing').textContent = `${(result.latency_ms / 1000).toFixed(2)} s · 流程未完成`;
    $('token-usage').textContent = '';
  }
  $('json-output').textContent = JSON.stringify(result, null, 2);
}
$('check-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (busy) return;
  const payload = {
    stage, text: $('prompt').value.trim(),
    response: stage === 'output' ? $('response').value.trim() : null,
    explain: $('explain').checked,
    max_new_tokens: Number($('max-tokens').value),
    policy: $('policy').value.trim() || null,
  };
  if (stage === 'chat') {
    delete payload.stage;
    delete payload.response;
    payload.enforce_input = $('enforce-input').checked;
    payload.generation_max_tokens = Number($('generation-tokens').value);
  }
  if (!payload.text || (stage === 'output' && !payload.response)) {
    $('error').textContent = '请填写需要检查的文本。'; $('error').hidden = false; return;
  }
  $('error').hidden = true;
  setBusy(true);
  let hasPartial = false;
  try {
    const response = await fetch(stage === 'chat' ? 'api/chat' : 'api/check', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      if (body.partial_result) { renderResult(body.partial_result); hasPartial = true; }
      const message = typeof body.detail === 'string' ? body.detail : '请求参数不符合要求，请检查输入长度与设置。';
      throw new Error(message);
    }
    renderResult(body);
  } catch (error) {
    $('error').textContent = error.message || '请求失败，请检查后端服务。';
    $('error').hidden = false;
    $('empty-state').hidden = Boolean(lastResult);
    $('result').hidden = !lastResult;
    if (lastResult && !hasPartial) $('stale-note').hidden = false;
  } finally {
    setBusy(false); refreshStatus();
  }
});
$('download').addEventListener('click', () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], {type: 'application/json'}));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = `xguard-${lastResult.stage}-${Date.now()}.json`;
  anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
setStage('chat');
refreshStatus();
