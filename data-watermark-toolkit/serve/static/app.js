/* DataMark 数据集水印实验台 —— 逻辑移植自 toolkit/toolkit.html，仅改用相对路径 API
 * （独立部署与 hub /proxy/datamark/ 代理路径下都成立）。 */
const S = { meta: null, method: null, preset: 'strong', job: null, timer: null };
const $id = (id) => document.getElementById(id);
const esc = (s) => String(s === undefined || s === null ? '' : s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const fmtP = (p) => (p === null || p === undefined) ? '—' : (p < 0.001 ? Number(p).toExponential(1) : String(Number(p.toFixed(4))));

const LABELS = {
  repeats: ['trap 重复次数', '同一 trap 在文档中重复出现的次数'],
  trap_tokens: ['trap 长度（token）', '生成的 trap 序列 token 数'],
  delta: ['KGW δ', '绿名单 logit 加成，越大水印越强'],
  max_new_tokens: ['生成长度', '改写 / 水印文本的最大 token 数'],
  gamma: ['green 比例 γ', 'KGW 绿名单比例'],
  window_k: ['窗口 k', 'KGW 上下文窗口长度'],
  n_entities: ['虚构实体数', '语义知识型的实体数量'],
  n_attributes: ['每个实体属性数', '每个实体的属性数量'],
  passage_tokens: ['段落长度', '生成资料段的最大 token 数'],
  n_watermarked_texts: ['水印文本数', '分布统计型生成的带水印训练文本数'],
  detect_tokens: ['检测生成长度', '检测时每条 prompt 的生成 token 数'],
  n_verify_docs: ['验证文档数', 'Double-I 的 held-out 验证文档数'],
  epochs: ['训练轮数', 'SFT epochs'],
  lr: ['学习率', 'LoRA / 全参微调学习率'],
  max_length: ['最大序列长度', '训练截断长度'],
};
const SHARED = [
  { key: 'fraction', label: '带水印比例 ρ', hint: '0–1；0 表示负对照', step: 0.05, min: 0, max: 1 },
  { key: 'seed', label: '随机种子', hint: '同一配置 + 同一种子可复现', step: 1, min: 0 },
];
const PRESETS = {
  strong: { fraction: 0.5, overrides: {} },
  weak: {
    fraction: 0.1,
    overrides: { repeats: 1, delta: 0.8, n_watermarked_texts: 16, epochs: 3 },
  },
  clean: { fraction: 0.0, overrides: {} },
};

function metaLoad() {
  fetch('api/meta').then((r) => r.json()).then((meta) => {
    if (meta.error) {
      $id('env-info').innerHTML = '<b>环境异常：</b>' + esc(meta.error);
    }
    S.meta = meta;
    S.method = Object.keys(meta.methods)[0];
    renderMethods(); renderEnv(); renderRuntime(); applyPreset(S.preset); syncFinetuneForMethod();
  }).catch((err) => {
    $id('env-info').innerHTML = '后端未启动：' + esc(err.message) +
      '<br/>请从 hub 章节页启动本章前端服务。';
  });
}

function renderMethods() {
  const host = $id('methods'); host.innerHTML = '';
  for (const [id, m] of Object.entries(S.meta.methods)) {
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'method';
    btn.setAttribute('aria-pressed', String(id === S.method));
    btn.innerHTML = '<span>' + esc(m.family) + '</span><span class="access">' + esc(accessZh(m.access)) + '</span>';
    btn.addEventListener('click', () => {
      S.method = id; renderMethods(); renderFields(); syncFinetuneForMethod();
    });
    host.appendChild(btn);
  }
  renderInfo();
}

function accessZh(a) {
  return ({ 'white-box': '白盒', 'gray-box': '灰盒', 'black-box': '黑盒' })[a] || a;
}

function renderInfo() {
  const m = S.meta.methods[S.method];
  if (!m) return;
  $id('method-info').innerHTML = '<b>' + esc(m.family) + '</b> · ' + esc(accessZh(m.access)) +
    '<br/>论文：' + esc(m.reference) +
    '<br/>配置：<code>' + esc(m.config) + '</code>';
}

function renderEnv() {
  const rt = S.meta.runtime || {};
  const ver = rt.service_versions || {};
  $id('env-info').innerHTML =
    '服务进程：<b>Python ' + esc(rt.python_version || '—') + '</b>' +
    '<br/>torch ' + esc(ver.torch || '—') + ' · transformers ' + esc(ver.transformers || '—') +
    '<br/>模型：<b>' + esc((rt.model || {}).name || '—') + '</b>' +
    '<br/>设备：<b>' + esc((rt.model || {}).device || '—') +
    '</b> · 微调：<b>' + esc((rt.model || {}).finetune || '—') + '</b>' +
    '<br/>公共 seed：<b>' + esc(rt.seed !== undefined ? rt.seed : '—') + '</b>' +
    '<br/>输出目录：<code>' + esc(S.meta.runs_dir) + '</code>';
}

function textField(id, label, value, hint, cls) {
  return '<div class="field ' + (cls || '') + '"><label for="' + id + '">' + esc(label) + '</label>' +
    '<input id="' + id + '" type="text" value="' + esc(value === undefined || value === null ? '' : value) + '"/>' +
    (hint ? '<span class="hint">' + esc(hint) + '</span>' : '') + '</div>';
}

function selectField(id, label, value, options) {
  const opts = options.map((o) => {
    const [v, text] = Array.isArray(o) ? o : [o, o];
    return '<option value="' + esc(v) + '"' + (v === value ? ' selected' : '') + '>' + esc(text) + '</option>';
  }).join('');
  return '<div class="field"><label for="' + id + '">' + esc(label) + '</label>' +
    '<select id="' + id + '">' + opts + '</select></div>';
}

function checkField(id, label, checked, hint) {
  return '<div class="field check"><input id="' + id + '" type="checkbox"' + (checked ? ' checked' : '') + '/>' +
    '<label for="' + id + '">' + esc(label) + '</label>' +
    (hint ? '<span class="hint">' + esc(hint) + '</span>' : '') + '</div>';
}

function renderRuntime() {
  const rt = S.meta.runtime;
  if (!rt) return;
  const model = rt.model || {};
  $id('runtime-summary').textContent = (model.name || '').split('/').pop() +
    ' · ' + (model.device || '') + ' · ' + (model.finetune || '');
  $id('env-python-fields').innerHTML =
    textField('e-python', 'Python 解释器', rt.python, '目标解释器需已安装 torch / transformers / peft') +
    textField('e-pythonpath', 'PYTHONPATH 覆盖层', rt.pythonpath, '可选；多个路径用 : 分隔') +
    textField('e-cuda', 'CUDA_VISIBLE_DEVICES', rt.cuda_visible_devices, '留空则继承服务进程') +
    textField('e-hfhome', 'HF_HOME', rt.hf_home, '模型缓存目录') +
    checkField('e-offline', '离线加载', rt.offline !== false, '仅读取本地缓存');
  $id('env-model-fields').innerHTML =
    textField('e-model', '模型目录 / HF 仓库', model.name, '本地目录需含 config.json；仓库名需关闭离线加载', 'wide') +
    textField('e-device', '设备', model.device, 'cuda:0 或 cpu') +
    selectField('e-dtype', '数据类型', model.dtype, ['bfloat16', 'float16', 'float32']) +
    selectField('e-finetune', '微调方式', model.finetune, [['lora', 'LoRA'], ['full', '全参数']]) +
    textField('e-lora-r', 'LoRA r', model.lora_r, '仅 LoRA 生效') +
    textField('e-lora-alpha', 'LoRA alpha', model.lora_alpha, '仅 LoRA 生效') +
    checkField('e-chat', '使用 chat 模板', model.use_chat_template) +
    checkField('e-gc', '梯度检查点', model.gradient_checkpointing);
  $id('e-finetune').addEventListener('change', toggleLora);
  toggleLora();
}

function toggleLora() {
  const lora = $id('e-finetune').value === 'lora';
  for (const id of ['e-lora-r', 'e-lora-alpha']) {
    const el = $id(id);
    if (el) el.closest('.field').style.display = lora ? '' : 'none';
  }
}

function syncFinetuneForMethod() {
  if (!$id('e-finetune')) return;
  const want = S.method === 'token_statistics' ? 'full' : 'lora';
  if ($id('e-finetune').value !== want) {
    $id('e-finetune').value = want;
    toggleLora();
  }
  const model = ((S.meta.runtime || {}).model || {});
  $id('runtime-summary').textContent = (model.name || '').split('/').pop() +
    ' · ' + (model.device || '') + ' · ' + $id('e-finetune').value;
}

function fieldHtml(spec, value, opts = {}) {
  const step = opts.step !== undefined ? opts.step : 'any';
  return '<div class="field"><label for="f-' + spec.key + '">' + esc(spec.label) + '</label>' +
    '<input id="f-' + spec.key + '" data-key="' + spec.key + '" type="number" step="' + step + '"' +
    (opts.min !== undefined ? ' min="' + opts.min + '"' : '') +
    (opts.max !== undefined ? ' max="' + opts.max + '"' : '') +
    ' value="' + value + '"/>' +
    (spec.hint ? '<span class="hint">' + esc(spec.hint) + '</span>' : '') + '</div>';
}

function renderFields() {
  const m = S.meta.methods[S.method];
  if (!m) return;
  const preset = PRESETS[S.preset];
  $id('shared-fields').innerHTML = SHARED.map((s) =>
    fieldHtml(s, s.key === 'fraction' ? preset.fraction : ((S.meta.runtime || {}).seed !== undefined ? S.meta.runtime.seed : 42),
      { step: s.step, min: s.min, max: s.max })).join('');
  const keys = m.allowed.filter((k) => LABELS[k]);
  const defaults = m.defaults || {};
  $id('method-fields').innerHTML = keys.map((k) => {
    const spec = { key: k, label: LABELS[k][0], hint: LABELS[k][1] };
    const override = preset.overrides[k];
    const value = override !== undefined ? override : (defaults[k] !== undefined ? defaults[k] : 1);
    return fieldHtml(spec, value);
  }).join('') || '<div class="empty">该方法没有可调参数。</div>';
  renderInfo();
}

function applyPreset(name) {
  S.preset = name;
  for (const chip of document.querySelectorAll('#presets .chip-btn')) {
    chip.classList.toggle('on', chip.dataset.preset === name);
  }
  renderFields();
}

function collect() {
  const overrides = {};
  for (const el of document.querySelectorAll('#method-fields input')) {
    overrides[el.dataset.key] = Number(el.value);
  }
  return {
    method: S.method,
    fraction: Number($id('f-fraction').value),
    seed: Number($id('f-seed').value),
    overrides,
    env: {
      python: $id('e-python').value.trim(),
      pythonpath: $id('e-pythonpath').value.trim(),
      cuda_visible_devices: $id('e-cuda').value.trim(),
      hf_home: $id('e-hfhome').value.trim(),
      offline: $id('e-offline').checked,
      model: {
        name: $id('e-model').value.trim(),
        device: $id('e-device').value.trim(),
        dtype: $id('e-dtype').value,
        finetune: $id('e-finetune').value,
        lora_r: Number($id('e-lora-r').value),
        lora_alpha: Number($id('e-lora-alpha').value),
        lora_dropout: 0.05,
        use_chat_template: $id('e-chat').checked,
        gradient_checkpointing: $id('e-gc').checked,
      },
    },
  };
}

async function run() {
  const body = collect();
  $id('run-btn').disabled = true;
  $id('stop-btn').disabled = false;
  $id('open-btn').disabled = true;
  $id('json').hidden = true;
  $id('metrics').innerHTML = '';
  $id('status').textContent = '正在提交…';
  try {
    const r = await fetch('api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || '提交失败');
    S.job = data.id;
    const rt = data.runtime || {};
    $id('status').innerHTML = '实验已启动：<b>' + esc(data.id) + '</b>' +
      (rt.python ? ' · ' + esc(rt.python.split('/').pop()) : '') +
      (rt.model ? ' · ' + esc(rt.model.name.split('/').pop()) : '') +
      (rt.note ? ' · ' + esc(rt.note) : '');
    $id('log').hidden = false;
    poll();
  } catch (err) {
    $id('status').innerHTML = '<b class="err">失败：</b>' + esc(err.message);
    $id('run-btn').disabled = false;
    $id('stop-btn').disabled = true;
  }
}

function poll() {
  clearTimeout(S.timer);
  if (!S.job) return;
  fetch('api/job?id=' + encodeURIComponent(S.job)).then((r) => r.json()).then((job) => {
    if (job.error) { $id('status').textContent = job.error; return; }
    $id('status').innerHTML = '实验 <b>' + esc(job.id) + '</b> · 状态 <b>' + esc(job.status) +
      '</b> · 已用 ' + job.elapsed + ' s';
    if (job.log_tail) { $id('log').textContent = job.log_tail; $id('log').scrollTop = 1e6; }
    if (job.report) {
      const p = job.report.p;
      const det = (p !== null && p !== undefined && p < 0.01) ? '检出' : '未检出';
      $id('metrics').innerHTML =
        '<span>p 值 <b>' + fmtP(p) + '</b></span>' +
        '<span>效应量 <b>' + Number(job.report.score).toFixed(3) + '</b></span>' +
        '<span>样本量 <b>' + job.report.n + '</b></span>' +
        '<span class="' + (det === '检出' ? 'on' : '') + '">判定 α=1% <b>' + det + '</b></span>';
      fetch('api/job/' + encodeURIComponent(job.id) + '/report')
        .then((r) => r.json()).then((rep) => {
          $id('json').hidden = false;
          $id('json').textContent = JSON.stringify({
            method: rep.method, fraction: rep.fraction, seed: rep.seed,
            train: rep.train, detection: rep.detection,
          }, null, 2);
        });
      $id('open-btn').disabled = false;
    }
    if (job.status === 'running') {
      S.timer = setTimeout(poll, 2000);
    } else {
      $id('run-btn').disabled = false;
      $id('stop-btn').disabled = true;
    }
  }).catch(() => { S.timer = setTimeout(poll, 3000); });
}

$id('run-btn').addEventListener('click', run);
$id('stop-btn').addEventListener('click', () => {
  if (!S.job) return;
  fetch('api/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: S.job }) }).then(() => poll());
});
$id('open-btn').addEventListener('click', () => {
  if (S.job) window.open('api/job/' + encodeURIComponent(S.job) + '/report', '_blank');
});
for (const chip of document.querySelectorAll('#presets .chip-btn')) {
  chip.addEventListener('click', () => applyPreset(chip.dataset.preset));
}
metaLoad();
