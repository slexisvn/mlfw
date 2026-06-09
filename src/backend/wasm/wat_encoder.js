const WASM_MAGIC = [0x00, 0x61, 0x73, 0x6d];
const WASM_VERSION = [0x01, 0x00, 0x00, 0x00];
const SEC_TYPE = 1, SEC_IMPORT = 2, SEC_FUNC = 3, SEC_MEMORY = 5, SEC_EXPORT = 7, SEC_CODE = 10;
const T_I32 = 0x7f, T_F32 = 0x7d, T_V128 = 0x7b, T_FUNC = 0x60, T_VOID = 0x40;

function uleb(v) { const r = []; do { let b = v & 0x7f; v >>>= 7; if (v) b |= 0x80; r.push(b); } while (v); return r; }
function sleb(v) { const r = []; let more = true; while (more) { let b = v & 0x7f; v >>= 7; if ((v === 0 && !(b & 0x40)) || (v === -1 && (b & 0x40))) more = false; else b |= 0x80; r.push(b); } return r; }
function encStr(s) { const e = new TextEncoder().encode(s); return [...uleb(e.length), ...e]; }
function encF32(v) { const b = new ArrayBuffer(4); new Float32Array(b)[0] = v; return [...new Uint8Array(b)]; }
function section(id, data) { return [id, ...uleb(data.length), ...data]; }
function vec(items) { const f = []; for (const i of items) f.push(...i); return [...uleb(items.length), ...f]; }

const INSTR = new Map([
  ['i32.const', [0x41]], ['f32.const', [0x43]],
  ['local.get', [0x20]], ['local.set', [0x21]],
  ['i32.add', [0x6a]], ['i32.sub', [0x6b]], ['i32.mul', [0x6c]], ['i32.div_s', [0x6d]], ['i32.rem_s', [0x6f]],
  ['i32.eq', [0x46]], ['i32.ne', [0x47]], ['i32.lt_s', [0x48]], ['i32.gt_s', [0x4a]], ['i32.le_s', [0x4c]], ['i32.ge_s', [0x4e]], ['i32.eqz', [0x45]],
  ['i32.trunc_f32_s', [0xa8]],
  ['f32.add', [0x92]], ['f32.sub', [0x93]], ['f32.mul', [0x94]], ['f32.div', [0x95]],
  ['f32.neg', [0x8c]], ['f32.abs', [0x8b]], ['f32.ceil', [0x8d]], ['f32.floor', [0x8e]], ['f32.sqrt', [0x91]], ['f32.min', [0x96]], ['f32.max', [0x97]],
  ['f32.eq', [0x5b]], ['f32.ne', [0x5c]], ['f32.lt', [0x5d]], ['f32.gt', [0x5e]], ['f32.le', [0x5f]], ['f32.ge', [0x60]],
  ['f32.convert_i32_s', [0xb2]],
  ['select', [0x1b]],
  ['f32.load', [0x2a, 0x02, 0x00]], ['f32.store', [0x38, 0x02, 0x00]],
  ['i32.load', [0x28, 0x02, 0x00]], ['i32.store', [0x36, 0x02, 0x00]],
  ['i32.load8_s', [0x2c, 0x00, 0x00]], ['i32.store8', [0x3a, 0x00, 0x00]],
  ['v128.load', [0xfd, ...uleb(0), 0x04, 0x00]], ['v128.store', [0xfd, ...uleb(11), 0x04, 0x00]],
  ['v128.bitselect', [0xfd, ...uleb(0x52)]], ['v128.and', [0xfd, ...uleb(0x4e)]], ['v128.or', [0xfd, ...uleb(0x50)]], ['v128.not', [0xfd, ...uleb(0x4d)]],
  ['f32x4.splat', [0xfd, ...uleb(0x13)]], ['i32x4.splat', [0xfd, ...uleb(0x11)]],
  ['f32x4.add', [0xfd, ...uleb(0xe4)]], ['f32x4.sub', [0xfd, ...uleb(0xe5)]],
  ['f32x4.mul', [0xfd, ...uleb(0xe6)]], ['f32x4.div', [0xfd, ...uleb(0xe7)]],
  ['f32x4.neg', [0xfd, ...uleb(0xe1)]], ['f32x4.abs', [0xfd, ...uleb(0xe0)]],
  ['f32x4.sqrt', [0xfd, ...uleb(0xe3)]],
  ['f32x4.ceil', [0xfd, ...uleb(0x67)]], ['f32x4.floor', [0xfd, ...uleb(0x68)]],
  ['f32x4.min', [0xfd, ...uleb(0xe8)]], ['f32x4.max', [0xfd, ...uleb(0xe9)]],
  ['f32x4.eq', [0xfd, ...uleb(0x41)]], ['f32x4.ne', [0xfd, ...uleb(0x42)]],
  ['f32x4.lt', [0xfd, ...uleb(0x43)]], ['f32x4.gt', [0xfd, ...uleb(0x44)]],
  ['f32x4.le', [0xfd, ...uleb(0x45)]], ['f32x4.ge', [0xfd, ...uleb(0x46)]],
  ['f32x4.extract_lane', [0xfd, ...uleb(0x1f)]], ['f32x4.replace_lane', [0xfd, ...uleb(0x20)]],
  ['i32x4.add', [0xfd, ...uleb(0xae)]], ['i32x4.sub', [0xfd, ...uleb(0xb1)]],
  ['i32x4.mul', [0xfd, ...uleb(0xb5)]],
  ['i32x4.abs', [0xfd, ...uleb(0xa0)]],
  ['i32x4.min_s', [0xfd, ...uleb(0xb6)]], ['i32x4.max_s', [0xfd, ...uleb(0xb8)]],
  ['i32x4.eq', [0xfd, ...uleb(0x37)]], ['i32x4.ne', [0xfd, ...uleb(0x38)]],
  ['i32x4.lt_s', [0xfd, ...uleb(0x39)]], ['i32x4.gt_s', [0xfd, ...uleb(0x3a)]],
  ['i32x4.le_s', [0xfd, ...uleb(0x3b)]], ['i32x4.ge_s', [0xfd, ...uleb(0x3c)]],
  ['i32x4.extract_lane', [0xfd, ...uleb(0x1b)]], ['i32x4.replace_lane', [0xfd, ...uleb(0x1c)]],
]);

function tokenize(wat) {
  const tokens = [];
  let i = 0;
  while (i < wat.length) {
    const c = wat[i];
    if (c <= ' ') { i++; continue; }
    if (c === ';' && wat[i + 1] === ';') { i = wat.indexOf('\n', i); if (i < 0) i = wat.length; continue; }
    if (c === '(' && wat[i + 1] === ';') { i = wat.indexOf(';)', i + 2); i = i < 0 ? wat.length : i + 2; continue; }
    if (c === '(' || c === ')') { tokens.push(c); i++; continue; }
    if (c === '"') { let j = i + 1; while (j < wat.length && wat[j] !== '"') j++; tokens.push(wat.substring(i, j + 1)); i = j + 1; continue; }
    let j = i; while (j < wat.length && wat[j] > ' ' && wat[j] !== '(' && wat[j] !== ')') j++;
    tokens.push(wat.substring(i, j)); i = j;
  }
  return tokens;
}

function parseModule(tokens) {
  let p = 0;
  const eat = () => tokens[p++];
  const peek = () => tokens[p];
  const expect = t => { if (eat() !== t) throw new Error('expected ' + t + ' at ' + (p - 1)); };

  expect('('); expect('module');

  const imports = [];
  let memMin = 1, memMax = 256;
  let funcExportName = '';
  const funcParams = [];
  const funcLocals = [];
  const funcLocalNames = [];
  let bodyStart = -1, bodyEnd = -1;

  while (peek() !== ')') {
    if (peek() !== '(') { p++; continue; }
    p++;
    const kw = eat();

    if (kw === 'memory') {
      while (peek() === '(') skipSExpr();
      if (peek() !== ')') memMin = parseInt(eat()) || 1;
      if (peek() !== ')') memMax = parseInt(eat()) || 256;
      expect(')');
    } else if (kw === 'import') {
      const mod = eat().replace(/"/g, '');
      const name = eat().replace(/"/g, '');
      expect('('); expect('func');
      if (peek().startsWith('$')) p++;
      const params = [];
      const results = [];
      while (peek() === '(') {
        p++;
        const inner = eat();
        if (inner === 'param') { while (peek() !== ')') { const t = eat(); if (t === 'f32') params.push(T_F32); else if (t === 'i32') params.push(T_I32); else if (t === 'v128') params.push(T_V128); } }
        else if (inner === 'result') { while (peek() !== ')') { const t = eat(); if (t === 'f32') results.push(T_F32); else if (t === 'i32') results.push(T_I32); else if (t === 'v128') results.push(T_V128); } }
        expect(')');
      }
      expect(')'); expect(')');
      imports.push({ module: mod, name, params, results });
    } else if (kw === 'func') {
      while (peek() === '(') {
        const saved = p;
        p++;
        const inner = eat();
        if (inner === 'export') { funcExportName = eat().replace(/"/g, ''); expect(')'); }
        else if (inner === 'param') { while (peek() !== ')') { const t = eat(); if (t === 'i32') funcParams.push(T_I32); else if (t === 'f32') funcParams.push(T_F32); else if (t === 'v128') funcParams.push(T_V128); } expect(')'); }
        else if (inner === 'result') { while (peek() !== ')') eat(); expect(')'); }
        else if (inner === 'local') {
          while (peek() !== ')') {
            const t = eat();
            if (t.startsWith('$')) { funcLocalNames.push(t.replace('$', '')); }
            else if (t === 'i32') funcLocals.push(T_I32);
            else if (t === 'f32') funcLocals.push(T_F32);
            else if (t === 'v128') funcLocals.push(T_V128);
          }
          expect(')');
        } else { p = saved; break; }
      }
      bodyStart = p;
      let depth = 1;
      while (depth > 0) { const t = eat(); if (t === '(') depth++; else if (t === ')') depth--; }
      bodyEnd = p - 1;
    } else {
      skipSExpr();
    }
  }

  function skipSExpr() {
    if (peek() !== '(') { p++; return; }
    let d = 0;
    do { const t = eat(); if (t === '(') d++; else if (t === ')') d--; } while (d > 0);
  }

  return { imports, memMin, memMax, funcExportName, funcParams, funcLocals, funcLocalNames, bodyTokens: tokens.slice(bodyStart, bodyEnd) };
}

function encodeBody(bodyTokens, localMap, importMap) {
  const bytes = [];
  let p = 0;
  const peek = () => bodyTokens[p];
  const eat = () => bodyTokens[p++];

  const labelStack = [];

  function localIdx(name) {
    const clean = name.replace('$', '');
    if (localMap.has(clean)) return localMap.get(clean);
    const num = parseInt(clean, 10);
    if (!isNaN(num)) return num;
    const nextIdx = localMap.size > 0 ? Math.max(...localMap.values()) + 1 : 0;
    localMap.set(clean, nextIdx);
    return nextIdx;
  }
  function eatLabel() { return (peek() && peek().startsWith('$')) ? eat().replace('$', '') : ''; }
  function extractName(tok, prefix) { const i = tok.indexOf(prefix); return i >= 0 ? tok.substring(i + prefix.length) : ''; }

  function emitBlock() {
    while (p < bodyTokens.length && peek() !== ')') {
      const t = peek();

      if (t === '(') {
        p++;
        const kw = eat();
        if (kw === 'i32.const') { bytes.push(0x41); bytes.push(...sleb(parseInt(eat()))); expect(')'); }
        else if (kw === 'f32.const') { bytes.push(0x43); bytes.push(...encF32(parseFloat(eat()))); expect(')'); }
        else if (kw === 'local.get') { bytes.push(0x20); bytes.push(...uleb(localIdx(eat()))); expect(')'); }
        else if (kw === 'local.set') { bytes.push(0x21); bytes.push(...uleb(localIdx(eat()))); expect(')'); }
        else if (kw === 'block') {
          const label = eat().replace('$', '');
          labelStack.push(label);
          bytes.push(0x02, T_VOID);
          emitBlock();
          bytes.push(0x0b);
          labelStack.pop();
          expect(')');
        }
        else if (kw === 'loop') {
          const label = eat().replace('$', '');
          labelStack.push(label);
          bytes.push(0x03, T_VOID);
          emitBlock();
          bytes.push(0x0b);
          labelStack.pop();
          expect(')');
        }
        else if (kw === 'if') {
          let blockType = T_VOID;
          if (peek() === '(') {
            const saved = p;
            p++;
            if (peek() === 'result') { eat(); const rt = eat(); blockType = rt === 'f32' ? T_F32 : rt === 'v128' ? T_V128 : T_I32; expect(')'); }
            else { p = saved; }
          }
          bytes.push(0x04, blockType);
          labelStack.push('_if');
          while (p < bodyTokens.length && peek() !== ')') {
            if (peek() === '(') {
              const saved = p;
              p++;
              const inner = eat();
              if (inner === 'then') { emitBlock(); expect(')'); }
              else if (inner === 'else') { bytes.push(0x05); emitBlock(); expect(')'); }
              else { p = saved; break; }
            } else break;
          }
          bytes.push(0x0b);
          labelStack.pop();
          expect(')');
        }
        else { p -= 2; skipSExpr(); }
        continue;
      }

      p++;

      if (t === 'br_if') {
        bytes.push(0x0d);
        bytes.push(...uleb(resolveBr(eatLabel())));
      }
      else if (t === 'br') {
        bytes.push(0x0c);
        bytes.push(...uleb(resolveBr(eatLabel())));
      }
      else if (t.startsWith('call')) {
        bytes.push(0x10);
        const fn = extractName(t, 'math_') || (peek() && peek().startsWith('$') ? extractName(eat(), 'math_') : '');
        bytes.push(...uleb(importMap.get(fn) ?? 0));
      }
      else if (t === 'local.get') {
        bytes.push(0x20);
        bytes.push(...uleb(localIdx(eatLabel())));
      }
      else if (t === 'local.set') {
        bytes.push(0x21);
        bytes.push(...uleb(localIdx(eatLabel())));
      }
      else if (t.endsWith('.extract_lane') || t.endsWith('.replace_lane')) {
        bytes.push(...INSTR.get(t));
        bytes.push(parseInt(eat(), 10));
      }
      else if (INSTR.has(t)) {
        bytes.push(...INSTR.get(t));
      }
    }
  }

  function resolveBr(label) {
    for (let i = labelStack.length - 1; i >= 0; i--) {
      if (labelStack[i] === label) return labelStack.length - 1 - i;
    }
    return 0;
  }

  function expect(t) { if (eat() !== t) throw new Error('expect ' + t); }
  function skipSExpr() { if (peek() !== '(') { p++; return; } let d = 0; do { const t = eat(); if (t === '(') d++; else if (t === ')') d--; } while (d > 0); }

  emitBlock();
  return bytes;
}

export function encodeWat(wat) {
  const tokens = tokenize(wat);
  const mod = parseModule(tokens);

  const localMap = new Map();
  const paramCount = mod.funcParams.length;
  for (let i = 0; i < mod.funcLocalNames.length; i++) {
    localMap.set(mod.funcLocalNames[i], paramCount + i);
  }

  const importMap = new Map();
  for (let i = 0; i < mod.imports.length; i++) importMap.set(mod.imports[i].name, i);

  const typeSigs = [];
  const typeMap = new Map();
  function getType(params, results) {
    const key = params.join(',') + '>' + results.join(',');
    if (typeMap.has(key)) return typeMap.get(key);
    const i = typeSigs.length;
    typeSigs.push({ params, results });
    typeMap.set(key, i);
    return i;
  }

  for (const imp of mod.imports) imp.typeIdx = getType(imp.params, imp.results);
  const mainTypeIdx = getType(mod.funcParams, []);

  const typeSec = section(SEC_TYPE, vec(typeSigs.map(s => [T_FUNC, ...uleb(s.params.length), ...s.params, ...uleb(s.results.length), ...s.results])));

  let importSec = [];
  if (mod.imports.length > 0) {
    importSec = section(SEC_IMPORT, vec(mod.imports.map(imp => [...encStr(imp.module), ...encStr(imp.name), 0x00, ...uleb(imp.typeIdx)])));
  }

  const funcSec = section(SEC_FUNC, vec([[...uleb(mainTypeIdx)]]));
  const memSec = section(SEC_MEMORY, vec([[0x01, ...uleb(mod.memMin), ...uleb(mod.memMax)]]));

  const funcIdx = mod.imports.length;
  const exportSec = section(SEC_EXPORT, vec([
    [...encStr('memory'), 0x02, ...uleb(0)],
    [...encStr(mod.funcExportName), 0x00, ...uleb(funcIdx)],
  ]));

  const localDecls = [];
  if (mod.funcLocals.length > 0) {
    const groups = [];
    let cur = mod.funcLocals[0], cnt = 1;
    for (let i = 1; i < mod.funcLocals.length; i++) {
      if (mod.funcLocals[i] === cur) cnt++;
      else { groups.push([...uleb(cnt), cur]); cur = mod.funcLocals[i]; cnt = 1; }
    }
    groups.push([...uleb(cnt), cur]);
    localDecls.push(...vec(groups));
  } else {
    localDecls.push(0x00);
  }

  const body = encodeBody(mod.bodyTokens, localMap, importMap);
  const codeBody = [...localDecls, ...body, 0x0b];
  const codeSec = section(SEC_CODE, vec([[...uleb(codeBody.length), ...codeBody]]));

  return new Uint8Array([...WASM_MAGIC, ...WASM_VERSION, ...typeSec, ...importSec, ...funcSec, ...memSec, ...exportSec, ...codeSec]);
}
