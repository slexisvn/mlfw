import { TeraRuntime, formatValue, CsvStreamParser } from './dist/mlfw-lang.esm.js';
import { createChartApi, isChartSpec, renderChart } from './chart/index.js';
import { CHART_METHOD_DOCS, chartMethodOwner } from './chart/docs.js';
import { highlightHtml } from './highlight.js';

const STORAGE_KEY = 'mlfw-notebook-v1';
const THEME_KEY = 'mlfw-notebook-theme';

const KEYWORDS = [
  'model', 'forward', 'train', 'validate', 'optimizer', 'return', 'fn',
  'if', 'else', 'for', 'in', 'while', 'break', 'continue',
  'and', 'or', 'not', 'true', 'false', 'null',
];
const PAIR = { '(': ')', '[': ']', '{': '}', '"': '"', "'": "'" };
const OPENERS = new Set(['(', '[', '{']);
const QUOTES = new Set(['"', "'"]);
const CLOSERS = new Set([')', ']', '}']);

function highlight(cell) {
  cell.pre.innerHTML = highlightHtml(cell.editor.value);
}

const SEED = [
  `a = tensor([[1, 2], [3, 4]])\nb = tensor([[5, 6], [7, 8]])\na @ b`,
  `x = randn([3, 4])\nprint(x.shape)\nx.relu().mean()`,
  `metrics = DataFrame(epoch=[1, 2, 3, 4], loss=[1.0, 0.72, 0.48, 0.31], val_loss=[1.1, 0.81, 0.6, 0.44])\nchart.line(metrics, x="epoch", y=["loss", "val_loss"], title="Training")`,
  `model MLP(input, hidden, output):\n  fc1 = Linear(input, hidden)\n  fc2 = Linear(hidden, output)\n\n  forward x:\n    x = fc1(x).relu()\n    return fc2(x)\n\nnet = MLP(2, 4, 1)\nnet(randn([8, 2]))`,
  `fn fib(n):\n  if n < 2:\n    return n\n  return fib(n - 1) + fib(n - 2)\n\nfib(12)`,
];

const listEl = document.getElementById('cells');
const kernelStatus = document.getElementById('kernel-status');

let runtime;
let execCount = 0;
let activeOutput = [];
const cells = [];

function makeRuntime() {
  execCount = 0;
  activeOutput = [];
  runtime = new TeraRuntime({ output: (t) => activeOutput.push(String(t)) });
  runtime.registerGlobal('chart', createChartApi());
}

function setKernel(text, busy) {
  kernelStatus.textContent = 'kernel: ' + text;
  kernelStatus.classList.toggle('busy', !!busy);
}

const uploadedFiles = new Map();

function csvVarName(filename) {
  let base = filename.replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_]/g, '_');
  if (!/^[A-Za-z_]/.test(base)) base = '_' + base;
  return base || 'data';
}

function fmtSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

const BATCH_ROWS = 16384;

function parseCsvInWorker(file, onBatch, onProgress) {
  return new Promise((resolve, reject) => {
    let worker;
    try {
      worker = new Worker(new URL('./csv-worker.js', import.meta.url), { type: 'module' });
    } catch (err) {
      reject(err);
      return;
    }
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'batch') onBatch(m.rows);
      else if (m.type === 'progress') onProgress(m.read);
      else if (m.type === 'done') { worker.terminate(); resolve({ rowCount: m.rowCount }); }
      else if (m.type === 'error') { worker.terminate(); reject(new Error(m.message)); }
    };
    worker.onerror = (err) => { worker.terminate(); reject(new Error(err.message || 'worker failed')); };
    worker.postMessage({ file, separator: ',' });
  });
}

async function parseCsvOnMainThread(file, onBatch, onProgress) {
  const parser = new CsvStreamParser(',');
  const reader = file.stream().getReader();
  const decoder = new TextDecoder();
  let read = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    read += value.byteLength;
    parser.feed(decoder.decode(value, { stream: true }));
    if (parser.pending.length >= BATCH_ROWS) onBatch(parser.drain());
    onProgress(read);
  }
  const { rowCount } = parser.finish();
  const last = parser.drain();
  if (last.length) onBatch(last);
  return { rowCount };
}

async function uploadCsv(file) {
  const onProgress = (read) => {
    const pct = file.size ? Math.round((read / file.size) * 100) : 0;
    setKernel(`loading ${file.name} ${pct}%`, true);
  };
  setKernel(`loading ${file.name}…`, true);
  const handle = runtime.beginUploadedCsv(file.name);
  let appended = false;
  const onBatch = (rows) => { appended = true; handle.appendRows(rows); };
  let result;
  try {
    result = await parseCsvInWorker(file, onBatch, onProgress);
  } catch (err) {
    if (appended) throw err;
    result = await parseCsvOnMainThread(file, onBatch, onProgress);
  }
  handle.finish();
  uploadedFiles.set(file.name, { rowCount: result.rowCount, size: file.size });
  renderFiles();
}

async function uploadCsvFiles(fileList) {
  for (const file of fileList) {
    try {
      await uploadCsv(file);
    } catch (err) {
      setKernel(`error in ${file.name}: ${err.message || err}`);
      return;
    }
  }
  setKernel('ready');
}

function removeFile(name) {
  uploadedFiles.delete(name);
  runtime.removeUploadedCsv(name);
  renderFiles();
}

function renderFiles() {
  const list = document.getElementById('files-list');
  const empty = document.getElementById('files-empty');
  list.innerHTML = '';
  empty.style.display = uploadedFiles.size ? 'none' : 'block';
  for (const [name, meta] of uploadedFiles) {
    const li = document.createElement('li');
    li.className = 'file-item';
    const open = document.createElement('button');
    open.className = 'file-open';
    open.title = `Insert load_csv("${name}") cell`;
    const nameEl = document.createElement('span');
    nameEl.className = 'file-name';
    nameEl.textContent = name;
    const metaEl = document.createElement('span');
    metaEl.className = 'file-meta';
    metaEl.textContent = `${meta.rowCount} rows · ${fmtSize(meta.size)}`;
    open.append(nameEl, metaEl);
    open.addEventListener('click', () => createCell(`${csvVarName(name)} = load_csv("${name}")`, { focus: true }));
    const del = document.createElement('button');
    del.className = 'file-del';
    del.textContent = '×';
    del.title = 'Remove file';
    del.addEventListener('click', () => removeFile(name));
    li.append(open, del);
    list.append(li);
  }
}

function save() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cells.map((c) => c.editor.value)));
}

function autoSize(ta) {
  ta.style.height = 'auto';
  ta.style.height = Math.max(ta.scrollHeight, 24) + 'px';
}

function createCell(code = '', { focus = false, before = null } = {}) {
  const root = document.createElement('div');
  root.className = 'cell';

  const gutter = document.createElement('div');
  gutter.className = 'gutter';
  const runBtn = document.createElement('button');
  runBtn.className = 'run';
  runBtn.textContent = '▶';
  runBtn.title = 'Run this cell';
  const count = document.createElement('span');
  count.className = 'count';
  count.textContent = '[ ]';
  gutter.append(runBtn, count);

  const main = document.createElement('div');
  main.className = 'main';
  const wrap = document.createElement('div');
  wrap.className = 'editor-wrap';
  const pre = document.createElement('pre');
  pre.className = 'highlight';
  pre.setAttribute('aria-hidden', 'true');
  const editor = document.createElement('textarea');
  editor.className = 'editor';
  editor.spellcheck = false;
  editor.value = code;
  editor.rows = 1;
  wrap.append(pre, editor);
  const output = document.createElement('div');
  output.className = 'output';
  main.append(wrap, output);

  const tools = document.createElement('div');
  tools.className = 'cell-tools';
  const upBtn = document.createElement('button');
  upBtn.textContent = '↑ up';
  upBtn.title = 'Move cell up';
  const downBtn = document.createElement('button');
  downBtn.textContent = '↓ down';
  downBtn.title = 'Move cell down';
  const addBtn = document.createElement('button');
  addBtn.textContent = '＋ below';
  const delBtn = document.createElement('button');
  delBtn.textContent = '🗑 delete';
  tools.append(upBtn, downBtn, addBtn, delBtn);

  root.append(gutter, main, tools);

  const cell = { root, editor, output, count, pre, chartCleanup: null };

  runBtn.addEventListener('click', () => runCell(cell));
  addBtn.addEventListener('click', () => {
    const c = createCell('', { focus: true, before: nextSibling(cell) });
    return c;
  });
  delBtn.addEventListener('click', () => deleteCell(cell));
  upBtn.addEventListener('click', () => moveCell(cell, -1));
  downBtn.addEventListener('click', () => moveCell(cell, 1));
  editor.addEventListener('input', () => { autoSize(editor); highlight(cell); save(); updateAutocomplete(cell); });
  editor.addEventListener('keydown', (e) => onEditorKey(e, cell));
  editor.addEventListener('blur', () => setTimeout(() => { if (ac.cell === cell) closeAutocomplete(); }, 120));
  editor.addEventListener('scroll', () => { pre.scrollTop = editor.scrollTop; pre.scrollLeft = editor.scrollLeft; });
  editor.addEventListener('mousemove', (e) => onEditorHover(e, cell));
  editor.addEventListener('mouseleave', hideHover);

  if (before) {
    listEl.insertBefore(root, before);
    const idx = cells.findIndex((c) => c.root === before);
    cells.splice(idx === -1 ? cells.length : idx, 0, cell);
  } else {
    listEl.append(root);
    cells.push(cell);
  }

  autoSize(editor);
  highlight(cell);
  if (focus) editor.focus();
  save();
  return cell;
}

function nextSibling(cell) {
  const idx = cells.indexOf(cell);
  return idx >= 0 && idx + 1 < cells.length ? cells[idx + 1].root : null;
}

function deleteCell(cell) {
  const idx = cells.indexOf(cell);
  if (idx === -1) return;
  clearCellOutput(cell);
  cell.root.remove();
  cells.splice(idx, 1);
  if (cells.length === 0) createCell('', { focus: true });
  save();
}

function moveCell(cell, dir) {
  const idx = cells.indexOf(cell);
  const target = idx + dir;
  if (idx === -1 || target < 0 || target >= cells.length) return;
  const other = cells[target];
  cells[target] = cell;
  cells[idx] = other;
  if (dir < 0) listEl.insertBefore(cell.root, other.root);
  else listEl.insertBefore(cell.root, other.root.nextSibling);
  hideHover();
  closeAutocomplete();
  save();
  cell.editor.focus();
}

function onEditorKey(e, cell) {
  hideHover();
  if (autocompleteOpen()) {
    if (e.key === 'ArrowDown') { e.preventDefault(); moveAutocomplete(1); return; }
    if (e.key === 'ArrowUp') { e.preventDefault(); moveAutocomplete(-1); return; }
    if (e.key === 'Escape') { e.preventDefault(); closeAutocomplete(); return; }
    if (e.key === 'Tab') { e.preventDefault(); acceptAutocomplete(); return; }
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) { e.preventDefault(); acceptAutocomplete(); return; }
    if (e.key === 'Enter') closeAutocomplete();
  }
  if (e.key.length === 1 || e.key === 'Backspace') {
    if (handleAutoPairs(e, cell)) return;
  }
  if (e.key === 'Enter' && (e.shiftKey || e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    runCell(cell).then(() => {
      if (e.shiftKey && !e.ctrlKey && !e.metaKey) {
        const idx = cells.indexOf(cell);
        if (idx === cells.length - 1) createCell('', { focus: true });
        else cells[idx + 1].editor.focus();
      }
    });
    return;
  }
  if (e.key === 'Tab') {
    e.preventDefault();
    insertText(e.target, '  ');
    return;
  }
  if (e.key === 'Enter') {
    const ta = e.target;
    const upto = ta.value.slice(0, ta.selectionStart);
    const line = upto.slice(upto.lastIndexOf('\n') + 1);
    const indent = (line.match(/^\s*/) || [''])[0];
    const extra = /:\s*$/.test(line) ? '  ' : '';
    if (indent || extra) {
      e.preventDefault();
      insertText(ta, '\n' + indent + extra);
    }
  }
}

function afterEdit(cell) {
  autoSize(cell.editor);
  highlight(cell);
  save();
  updateAutocomplete(cell);
}

function handleAutoPairs(e, cell) {
  if (e.ctrlKey || e.metaKey || e.altKey) return false;
  const ta = cell.editor;
  const s = ta.selectionStart;
  const en = ta.selectionEnd;
  const key = e.key;

  if (OPENERS.has(key) || QUOTES.has(key)) {
    const close = PAIR[key];
    if (s !== en) {
      const sel = ta.value.slice(s, en);
      e.preventDefault();
      ta.value = ta.value.slice(0, s) + key + sel + close + ta.value.slice(en);
      ta.selectionStart = s + 1;
      ta.selectionEnd = en + 1;
      afterEdit(cell);
      return true;
    }
    if (QUOTES.has(key) && ta.value[s] === key) {
      e.preventDefault();
      ta.selectionStart = ta.selectionEnd = s + 1;
      return true;
    }
    e.preventDefault();
    ta.value = ta.value.slice(0, s) + key + close + ta.value.slice(en);
    ta.selectionStart = ta.selectionEnd = s + 1;
    afterEdit(cell);
    return true;
  }

  if (CLOSERS.has(key) && s === en && ta.value[s] === key) {
    e.preventDefault();
    ta.selectionStart = ta.selectionEnd = s + 1;
    return true;
  }

  if (key === 'Backspace' && s === en && s > 0) {
    const before = ta.value[s - 1];
    if (PAIR[before] && ta.value[s] === PAIR[before]) {
      e.preventDefault();
      ta.value = ta.value.slice(0, s - 1) + ta.value.slice(s + 1);
      ta.selectionStart = ta.selectionEnd = s - 1;
      afterEdit(cell);
      return true;
    }
  }
  return false;
}

function insertText(ta, text) {
  const start = ta.selectionStart;
  const end = ta.selectionEnd;
  ta.value = ta.value.slice(0, start) + text + ta.value.slice(end);
  ta.selectionStart = ta.selectionEnd = start + text.length;
  autoSize(ta);
  const pre = ta.previousElementSibling;
  if (pre && pre.classList.contains('highlight')) pre.innerHTML = highlightHtml(ta.value);
  save();
}

async function runCell(cell) {
  closeAutocomplete();
  const code = cell.editor.value;
  clearCellOutput(cell);
  if (!code.trim()) { cell.count.textContent = '[ ]'; return; }

  cell.count.textContent = '[*]';
  setKernel('running…', true);
  activeOutput = [];

  let result;
  try {
    // In the notebook, `df.show()` is the idiomatic "display this frame" call —
    // drop a trailing .show(...) so the DataFrame itself renders as a paginated table.
    const value = await runtime.execute(code.replace(SHOW_TAIL, ''));
    result = { ok: true, prints: activeOutput.slice(), value };
  } catch (err) {
    result = { ok: false, prints: activeOutput.slice(), error: (err && err.message) || String(err) };
  }

  execCount += 1;
  cell.count.textContent = '[' + execCount + ']';
  renderOutput(cell, result);
  setKernel('ready');
  save();
  return result;
}

function renderOutput(cell, result) {
  const out = cell.output;
  out.innerHTML = '';
  for (const line of result.prints) {
    const pre = document.createElement('div');
    pre.className = 'print';
    pre.textContent = line;
    out.append(pre);
  }
  if (!result.ok) {
    const err = document.createElement('div');
    err.className = 'error';
    err.textContent = result.error;
    out.append(err);
    return;
  }
  if (isDataFrame(result.value)) {
    renderDataFrameTable(out, result.value);
    return;
  }
  if (isChartSpec(result.value)) {
    cell.chartCleanup = renderChart(out, result.value);
    return;
  }
  const text = result.value === undefined ? '' : formatValue(result.value);
  if (text) {
    const res = document.createElement('div');
    res.className = 'result';
    res.textContent = text;
    out.append(res);
  }
}

const SHOW_TAIL = /\.show\s*\([^()]*\)\s*;?\s*$/;
const DF_PAGE_SIZE = 25;

function isDataFrame(v) {
  return v && typeof v.limit === 'function' && typeof v.collect === 'function' && typeof v.count === 'function';
}

async function renderDataFrameTable(out, df) {
  const view = document.createElement('div');
  view.className = 'df-view';
  out.append(view);
  let offset = 0;
  let total = null;
  let columns = null;

  async function load() {
    view.classList.add('df-loading');
    try {
      const rows = await df.limit(DF_PAGE_SIZE, offset).collect();
      if (total === null) total = await df.count();
      if (columns === null) columns = await df.columns();
      render(rows);
    } catch (e) {
      view.innerHTML = '';
      const err = document.createElement('div');
      err.className = 'error';
      err.textContent = (e && e.message) || String(e);
      view.append(err);
    } finally {
      view.classList.remove('df-loading');
    }
  }

  function render(rows) {
    view.innerHTML = '';
    const scroll = document.createElement('div');
    scroll.className = 'df-scroll';
    const table = document.createElement('table');
    table.className = 'df-grid';
    const thead = document.createElement('thead');
    const htr = document.createElement('tr');
    for (const c of columns) {
      const th = document.createElement('th');
      th.textContent = c;
      htr.append(th);
    }
    thead.append(htr);
    table.append(thead);
    const tbody = document.createElement('tbody');
    for (const row of rows) {
      const tr = document.createElement('tr');
      for (const c of columns) {
        const td = document.createElement('td');
        const v = row[c];
        if (v === null || v === undefined) { td.textContent = 'NULL'; td.className = 'df-null'; }
        else td.textContent = String(v);
        tr.append(td);
      }
      tbody.append(tr);
    }
    table.append(tbody);
    scroll.append(table);
    view.append(scroll);

    const pager = document.createElement('div');
    pager.className = 'df-pager';
    const from = total === 0 ? 0 : offset + 1;
    const to = offset + rows.length;
    const info = document.createElement('span');
    info.className = 'df-info';
    info.textContent = `${from}–${to} of ${total} · ${columns.length} cols`;
    const prev = document.createElement('button');
    prev.textContent = '‹ Prev';
    prev.disabled = offset <= 0;
    prev.addEventListener('click', () => { offset = Math.max(0, offset - DF_PAGE_SIZE); load(); });
    const next = document.createElement('button');
    next.textContent = 'Next ›';
    next.disabled = to >= total;
    next.addEventListener('click', () => { offset += DF_PAGE_SIZE; load(); });
    pager.append(prev, info, next);
    view.append(pager);
  }

  await load();
}

async function runAll() {
  for (const cell of cells) {
    const r = await runCell(cell);
    if (r && !r.ok) break;
  }
}

function restart() {
  makeRuntime();
  for (const cell of cells) {
    cell.count.textContent = '[ ]';
    clearCellOutput(cell);
  }
  setKernel('ready');
}

function clearOutputs() {
  for (const cell of cells) {
    cell.count.textContent = '[ ]';
    clearCellOutput(cell);
  }
}

function clearCellOutput(cell) {
  cell.chartCleanup?.();
  cell.chartCleanup = null;
  cell.output.innerHTML = '';
}

function load() {
  let saved;
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch { saved = null; }
  const source = Array.isArray(saved) && saved.length ? saved : SEED;
  for (const code of source) createCell(code);
  if (cells.length === 0) createCell('', { focus: true });
}

const themeBtn = document.getElementById('theme-toggle');

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  themeBtn.textContent = theme === 'dark' ? '☀ Light' : '🌙 Dark';
  localStorage.setItem(THEME_KEY, theme);
}

function initTheme() {
  let theme = localStorage.getItem(THEME_KEY);
  if (!theme) theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  applyTheme(theme);
}

themeBtn.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

const ac = { el: null, items: [], index: 0, cell: null, start: 0, end: 0 };

function closeAutocomplete() {
  if (ac.el) { ac.el.remove(); ac.el = null; }
  ac.items = [];
  ac.cell = null;
}

function autocompleteOpen() {
  return !!ac.el;
}

function caretCoordinates(ta, position) {
  const mirror = document.createElement('div');
  const style = getComputedStyle(ta);
  const props = [
    'boxSizing', 'width', 'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
    'borderTopWidth', 'borderRightWidth', 'borderBottomWidth', 'borderLeftWidth',
    'fontFamily', 'fontSize', 'fontWeight', 'lineHeight', 'letterSpacing', 'tabSize',
  ];
  for (const p of props) mirror.style[p] = style[p];
  mirror.style.position = 'absolute';
  mirror.style.visibility = 'hidden';
  mirror.style.whiteSpace = 'pre';
  mirror.style.width = 'auto';
  mirror.style.top = '0';
  mirror.style.left = '0';
  mirror.textContent = ta.value.slice(0, position);
  const marker = document.createElement('span');
  marker.textContent = '​';
  mirror.append(marker);
  document.body.append(mirror);
  const top = marker.offsetTop;
  const left = marker.offsetLeft;
  mirror.remove();
  return { top, left };
}

function completionCandidates(ta) {
  const before = ta.value.slice(0, ta.selectionStart);
  const member = before.match(/([A-Za-z_]\w*)\.(\w*)$/);
  if (member) {
    const obj = runtime.getVariable(member[1]);
    if (obj == null || typeof obj !== 'object') return null;
    const prefix = member[2];
    const keys = Object.keys(obj)
      .filter((k) => !k.startsWith('_') && k !== 'constructor' && k.startsWith(prefix))
      .sort();
    if (!keys.length) return null;
    return { start: ta.selectionStart - prefix.length, items: keys.map((name) => ({ name, kind: 'attr' })) };
  }
  const word = before.match(/([A-Za-z_]\w*)$/);
  if (!word || word[1].length < 1) return null;
  const prefix = word[1];
  const names = runtime.getCompletionNames();
  const seen = new Set();
  const items = [];
  for (const name of names) {
    if (name.startsWith(prefix) && !seen.has(name)) { seen.add(name); items.push({ name, kind: 'name' }); }
  }
  for (const kw of KEYWORDS) {
    if (kw.startsWith(prefix) && !seen.has(kw)) { seen.add(kw); items.push({ name: kw, kind: 'keyword' }); }
  }
  items.sort((a, b) => a.name.localeCompare(b.name));
  if (!items.length || (items.length === 1 && items[0].name === prefix)) return null;
  return { start: ta.selectionStart - prefix.length, items };
}

function updateAutocomplete(cell) {
  const ta = cell.editor;
  const data = completionCandidates(ta);
  if (!data) { closeAutocomplete(); return; }

  ac.items = data.items;
  ac.index = 0;
  ac.cell = cell;
  ac.start = data.start;
  ac.end = ta.selectionStart;

  if (!ac.el) {
    ac.el = document.createElement('div');
    ac.el.className = 'autocomplete';
    document.body.append(ac.el);
  }
  renderAutocomplete();

  const coords = caretCoordinates(ta, ta.selectionStart);
  const rect = ta.getBoundingClientRect();
  const lh = parseFloat(getComputedStyle(ta).lineHeight) || 20;
  ac.el.style.left = window.scrollX + rect.left + coords.left - ta.scrollLeft + 'px';
  ac.el.style.top = window.scrollY + rect.top + coords.top - ta.scrollTop + lh + 4 + 'px';
}

function renderAutocomplete() {
  ac.el.innerHTML = '';
  ac.items.forEach((item, i) => {
    const row = document.createElement('div');
    row.className = 'item' + (i === ac.index ? ' active' : '');
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = item.name;
    const kind = document.createElement('span');
    kind.className = 'kind';
    kind.textContent = item.kind;
    row.append(name, kind);
    row.addEventListener('mousedown', (e) => { e.preventDefault(); acceptAutocomplete(i); });
    ac.el.append(row);
  });
}

function moveAutocomplete(delta) {
  ac.index = (ac.index + delta + ac.items.length) % ac.items.length;
  renderAutocomplete();
  const active = ac.el.children[ac.index];
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function acceptAutocomplete(i = ac.index) {
  const item = ac.items[i];
  const cell = ac.cell;
  if (!item || !cell) { closeAutocomplete(); return; }
  const ta = cell.editor;
  ta.value = ta.value.slice(0, ac.start) + item.name + ta.value.slice(ac.end);
  const caret = ac.start + item.name.length;
  ta.selectionStart = ta.selectionEnd = caret;
  closeAutocomplete();
  autoSize(ta);
  highlight(cell);
  save();
  ta.focus();
}

const docs = new Map();
const memberDocs = new Map();
let hoverEl = null;

async function loadDocs() {
  try {
    const res = await fetch('./dist/language-data.json');
    if (!res.ok) return;
    const data = await res.json();
    for (const b of data.builtins || []) {
      docs.set(b.name, {
        display: (b.signature && b.signature.display) || b.name,
        kind: b.kind || null,
        description: b.description || null,
      });
    }
    for (const [group, names] of Object.entries(data.keywordGroups || {})) {
      for (const name of names) {
        if (!docs.has(name)) docs.set(name, { display: name, kind: group + ' keyword', description: null });
      }
    }
    for (const [typeName, methods] of Object.entries(data.pseudoTypes || {})) {
      for (const m of methods) {
        if (memberDocs.has(m.name)) continue;
        memberDocs.set(m.name, {
          display: typeName + '.' + ((m.signature && m.signature.display) || m.name),
          kind: (m.isGetter ? 'property of ' : 'method of ') + typeName,
          description: m.description || null,
        });
      }
    }
  } catch { /* hover docs unavailable */ }
}

let hoverSpan = null;

function spanAtPoint(pre, x, y) {
  const spans = pre.querySelectorAll('span');
  for (const span of spans) {
    const r = span.getBoundingClientRect();
    if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return span;
  }
  return null;
}

function showHoverAt(info, rect) {
  if (!hoverEl) {
    hoverEl = document.createElement('div');
    hoverEl.className = 'hover-doc';
    document.body.append(hoverEl);
  }
  hoverEl.innerHTML = '';
  const title = document.createElement('div');
  title.className = 'hd-title';
  title.textContent = info.display;
  hoverEl.append(title);
  if (info.kind) {
    const kind = document.createElement('div');
    kind.className = 'hd-kind';
    kind.textContent = info.kind;
    hoverEl.append(kind);
  }
  if (info.description) {
    const desc = document.createElement('div');
    desc.className = 'hd-desc';
    desc.textContent = info.description;
    hoverEl.append(desc);
  }
  hoverEl.style.display = 'block';
  const box = hoverEl.getBoundingClientRect();
  let left = rect.left;
  let top = rect.top - box.height - 6;
  if (top < 8) top = rect.bottom + 6;
  if (left + box.width > window.innerWidth - 12) left = window.innerWidth - box.width - 12;
  hoverEl.style.left = window.scrollX + Math.max(8, left) + 'px';
  hoverEl.style.top = window.scrollY + top + 'px';
}

function hideHover() {
  if (hoverEl) hoverEl.style.display = 'none';
  hoverSpan = null;
}

function onEditorHover(e, cell) {
  if (autocompleteOpen()) { hideHover(); return; }
  const span = spanAtPoint(cell.pre, e.clientX, e.clientY);
  if (!span) { hideHover(); return; }
  if (span === hoverSpan && hoverEl && hoverEl.style.display === 'block') return;
  const isMember = span.classList.contains('tok-method') || span.classList.contains('tok-prop');
  const owner = isMember ? chartMethodOwner(cell.pre, span) : null;
  const info = owner === 'chart' ? CHART_METHOD_DOCS.get(span.textContent) : isMember ? memberDocs.get(span.textContent) : docs.get(span.textContent);
  if (!info) { hideHover(); return; }
  hoverSpan = span;
  showHoverAt(info, span.getBoundingClientRect());
}

document.getElementById('run-all').addEventListener('click', runAll);
document.getElementById('add-cell').addEventListener('click', () => createCell('', { focus: true }));
document.getElementById('restart').addEventListener('click', restart);
document.getElementById('clear-out').addEventListener('click', clearOutputs);
const csvInput = document.getElementById('csv-file');
document.getElementById('upload-csv').addEventListener('click', () => csvInput.click());
csvInput.addEventListener('change', () => { uploadCsvFiles([...csvInput.files]); csvInput.value = ''; });

const sidebarEl = document.querySelector('.sidebar');
const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes('Files');
window.addEventListener('dragover', (e) => { if (hasFiles(e)) { e.preventDefault(); sidebarEl.classList.add('drag-over'); } });
window.addEventListener('dragleave', (e) => { if (e.relatedTarget === null) sidebarEl.classList.remove('drag-over'); });
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  sidebarEl.classList.remove('drag-over');
  const files = [...e.dataTransfer.files].filter((f) => /\.csv$/i.test(f.name) || f.type === 'text/csv');
  if (files.length) uploadCsvFiles(files);
});
document.addEventListener('click', (e) => { if (ac.el && !ac.el.contains(e.target)) closeAutocomplete(); });
document.addEventListener('scroll', hideHover, true);

initTheme();
loadDocs();
makeRuntime();
renderFiles();
load();
