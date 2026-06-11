import { TeraRuntime, formatValue } from './dist/mlfw-lang.esm.js';

const STORAGE_KEY = 'mlfw-notebook-v1';
const THEME_KEY = 'mlfw-notebook-theme';

const KEYWORDS = [
  'model', 'forward', 'train', 'validate', 'optimizer', 'return', 'fn',
  'if', 'elif', 'else', 'for', 'in', 'while', 'break', 'continue',
  'and', 'or', 'not', 'true', 'false', 'null',
];
const KEYWORD_SET = new Set(KEYWORDS);

const BUILTIN_SET = new Set([
  'tensor', 'zeros', 'ones', 'empty', 'full', 'randn', 'arange', 'eye', 'linspace', 'randperm',
  'zerosLike', 'onesLike', 'emptyLike', 'fullLike', 'randnLike',
  'add', 'sub', 'mul', 'div', 'neg', 'pow', 'remainder', 'maximum', 'minimum',
  'exp', 'log', 'sqrt', 'rsqrt', 'abs', 'sin', 'cos', 'tanh', 'sigmoid', 'relu',
  'gelu', 'silu', 'sign', 'floor', 'ceil', 'eq', 'ne', 'lt', 'le', 'gt', 'ge',
  'where', 'matmul', 'dot', 'cat', 'stack', 'clone', 'softmax', 'log_softmax',
  'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'prod',
  'reshape', 'transpose', 'permute', 'expand', 'slice', 'unsqueeze', 'squeeze',
  'narrow', 'select', 'contiguous', 'detach', 'requires_grad', 'grad', 'backward',
  'range', 'len', 'shape', 'dtype', 'print', 'trace', 'graph', 'compile',
]);

const TOKEN_RE = /#[^\n]*|"(?:\\.|[^"\n])*"|'(?:\\.|[^'\n])*'|\b\d+(?:\.\d+)?\b|[A-Za-z_]\w*/g;

const PAIR = { '(': ')', '[': ']', '{': '}', '"': '"', "'": "'" };
const OPENERS = new Set(['(', '[', '{']);
const QUOTES = new Set(['"', "'"]);
const CLOSERS = new Set([')', ']', '}']);

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => (c === '&' ? '&amp;' : c === '<' ? '&lt;' : '&gt;'));
}

function tokenClass(tok) {
  if (tok[0] === '#') return 'tok-com';
  if (tok[0] === '"' || tok[0] === "'") return 'tok-str';
  if (tok[0] >= '0' && tok[0] <= '9') return 'tok-num';
  if (KEYWORD_SET.has(tok)) return 'tok-kw';
  if (BUILTIN_SET.has(tok)) return 'tok-builtin';
  if (tok[0] >= 'A' && tok[0] <= 'Z') return 'tok-type';
  return null;
}

function highlightHtml(code) {
  let out = '';
  let last = 0;
  let m;
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(code))) {
    out += escapeHtml(code.slice(last, m.index));
    const cls = tokenClass(m[0]);
    out += cls ? `<span class="${cls}">${escapeHtml(m[0])}</span>` : escapeHtml(m[0]);
    last = m.index + m[0].length;
  }
  out += escapeHtml(code.slice(last));
  if (code.endsWith('\n') || code === '') out += '​';
  return out;
}

function highlight(cell) {
  cell.pre.innerHTML = highlightHtml(cell.editor.value);
}

const SEED = [
  `a = tensor([[1, 2], [3, 4]])\nb = tensor([[5, 6], [7, 8]])\na @ b`,
  `x = randn([3, 4])\nprint(shape(x))\nmean(relu(x))`,
  `model MLP(input, hidden, output):\n  fc1 = Linear(input, hidden)\n  fc2 = Linear(hidden, output)\n\n  forward x:\n    x = relu(fc1(x))\n    return fc2(x)\n\nnet = MLP(2, 4, 1)\nnet(randn([8, 2]))`,
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
}

function setKernel(text, busy) {
  kernelStatus.textContent = 'kernel: ' + text;
  kernelStatus.classList.toggle('busy', !!busy);
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

  const cell = { root, editor, output, count, pre };

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
  cell.output.innerHTML = '';
  if (!code.trim()) { cell.count.textContent = '[ ]'; return; }

  cell.count.textContent = '[*]';
  setKernel('running…', true);
  activeOutput = [];

  let result;
  try {
    const value = await runtime.execute(code);
    result = { ok: true, prints: activeOutput.slice(), text: formatValue(value) };
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
  if (result.text) {
    const res = document.createElement('div');
    res.className = 'result';
    res.textContent = result.text;
    out.append(res);
  }
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
    cell.output.innerHTML = '';
  }
  setKernel('ready');
}

function clearOutputs() {
  for (const cell of cells) {
    cell.count.textContent = '[ ]';
    cell.output.innerHTML = '';
  }
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
  const info = docs.get(span.textContent);
  if (!info) { hideHover(); return; }
  hoverSpan = span;
  showHoverAt(info, span.getBoundingClientRect());
}

document.getElementById('run-all').addEventListener('click', runAll);
document.getElementById('add-cell').addEventListener('click', () => createCell('', { focus: true }));
document.getElementById('restart').addEventListener('click', restart);
document.getElementById('clear-out').addEventListener('click', clearOutputs);
document.addEventListener('click', (e) => { if (ac.el && !ac.el.contains(e.target)) closeAutocomplete(); });
document.addEventListener('scroll', hideHover, true);

initTheme();
loadDocs();
makeRuntime();
load();
