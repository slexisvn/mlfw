import { wasmBytes } from '../../backend/dtype_map.js';

export class WasmKernelInstance {
  constructor(wasmInstance, memory, bufferOffsets, paramNames, bufferMap) {
    this._instance = wasmInstance;
    this._memory = memory;
    this._bufferOffsets = bufferOffsets;
    this._paramNames = paramNames;
    this._bufferMap = bufferMap;
    this._func = wasmInstance.exports[Object.keys(wasmInstance.exports).find(k => k !== 'memory')];
  }

  run(...tensorArgs) {
    const paramBufs = [...this._bufferMap.values()];
    for (let i = 0; i < tensorArgs.length && i < paramBufs.length; i++) {
      const buf = paramBufs[i];
      const offset = this._bufferOffsets.get(buf.name);
      if (offset === undefined) continue;
      const data = tensorArgs[i].data || tensorArgs[i];
      const dst = new Float32Array(this._memory.buffer, offset, data.length);
      dst.set(data);
    }

    this._func();

    for (let i = 0; i < tensorArgs.length && i < paramBufs.length; i++) {
      const buf = paramBufs[i];
      const offset = this._bufferOffsets.get(buf.name);
      if (offset === undefined) continue;
      const data = tensorArgs[i].data || tensorArgs[i];
      const src = new Float32Array(this._memory.buffer, offset, data.length);
      data.set(src);
    }
  }
}

export async function instantiateWasmKernel(compiledKernel) {
  const { source: wat, metadata } = compiledKernel;
  const { memoryPages, bufferOffsets, imports: mathImports, bufferMap } = metadata;

  const memory = new WebAssembly.Memory({ initial: memoryPages || 1 });

  const importObject = { math: {} };
  if (mathImports) {
    for (const [name] of mathImports) {
      importObject.math[name] = Math[name] || (x => x);
    }
  }

  const wasmModule = await compileWat(wat);
  const instance = await WebAssembly.instantiate(wasmModule, {
    ...importObject,
    env: { memory }
  });

  return new WasmKernelInstance(
    instance,
    instance.exports.memory || memory,
    bufferOffsets,
    metadata.params || [],
    bufferMap || new Map()
  );
}

async function compileWat(watText) {
  const binary = watToWasm(watText);
  return WebAssembly.compile(binary);
}

function watToWasm(wat) {
  const encoder = new WatEncoder();
  return encoder.encode(wat);
}

class WatEncoder {
  encode(wat) {
    const tokens = this._tokenize(wat);
    const ast = this._parse(tokens);
    return this._emit(ast);
  }

  _tokenize(src) {
    const tokens = [];
    let i = 0;
    while (i < src.length) {
      const ch = src[i];
      if (ch === '(' || ch === ')') { tokens.push(ch); i++; continue; }
      if (ch === ';' && src[i+1] === ';') { while (i < src.length && src[i] !== '\n') i++; continue; }
      if (/\s/.test(ch)) { i++; continue; }
      if (ch === '"') {
        let j = i + 1;
        while (j < src.length && src[j] !== '"') j++;
        tokens.push(src.substring(i, j + 1));
        i = j + 1;
        continue;
      }
      let j = i;
      while (j < src.length && !/[\s()]/.test(src[j])) j++;
      tokens.push(src.substring(i, j));
      i = j;
    }
    return tokens;
  }

  _parse(tokens) {
    let pos = 0;
    const parse = () => {
      if (tokens[pos] === '(') {
        pos++;
        const list = [];
        while (pos < tokens.length && tokens[pos] !== ')') {
          list.push(parse());
        }
        pos++;
        return list;
      }
      return tokens[pos++];
    };
    const result = [];
    while (pos < tokens.length) result.push(parse());
    return result;
  }

  _emit(ast) {
    const sections = { types: [], imports: [], funcs: [], memories: [], exports: [], codes: [] };
    const module = ast[0];
    if (!Array.isArray(module) || module[0] !== 'module') return new Uint8Array(0);

    for (let i = 1; i < module.length; i++) {
      const node = module[i];
      if (!Array.isArray(node)) continue;
      switch (node[0]) {
        case 'memory': this._emitMemory(node, sections); break;
        case 'import': this._emitImport(node, sections); break;
        case 'func': this._emitFunc(node, sections); break;
      }
    }

    return this._buildBinary(sections);
  }

  _emitMemory(node, sections) {
    let pages = 1;
    let exported = null;
    for (let i = 1; i < node.length; i++) {
      if (Array.isArray(node[i]) && node[i][0] === 'export') exported = this._stripQuotes(node[i][1]);
      else if (typeof node[i] === 'string' && /^\d+$/.test(node[i])) pages = parseInt(node[i]);
    }
    sections.memories.push({ pages, exported });
  }

  _emitImport(node, sections) {
    const moduleName = this._stripQuotes(node[1]);
    const fieldName = this._stripQuotes(node[2]);
    const desc = node[3];
    if (Array.isArray(desc) && desc[0] === 'func') {
      const funcName = desc[1];
      const params = [];
      const results = [];
      for (let i = 2; i < desc.length; i++) {
        if (Array.isArray(desc[i])) {
          if (desc[i][0] === 'param') {
            for (let j = 1; j < desc[i].length; j++) params.push(this._valtype(desc[i][j]));
          }
          if (desc[i][0] === 'result') {
            for (let j = 1; j < desc[i].length; j++) results.push(this._valtype(desc[i][j]));
          }
        }
      }
      sections.imports.push({ module: moduleName, field: fieldName, funcName, params, results });
    }
  }

  _emitFunc(node, sections) {
    let funcName = null;
    let exported = null;
    const params = [];
    const locals = [];
    const bodyNodes = [];
    let idx = 1;

    if (typeof node[idx] === 'string' && node[idx].startsWith('$')) { funcName = node[idx]; idx++; }
    while (idx < node.length) {
      const item = node[idx];
      if (Array.isArray(item)) {
        if (item[0] === 'export') { exported = this._stripQuotes(item[1]); }
        else if (item[0] === 'param') {
          for (let j = 1; j < item.length; j++) {
            if (typeof item[j] === 'string' && !item[j].startsWith('$')) params.push(this._valtype(item[j]));
          }
        }
        else if (item[0] === 'local') {
          for (let j = 1; j < item.length; j++) {
            if (typeof item[j] === 'string' && item[j].startsWith('$')) {
              locals.push({ name: item[j], type: this._valtype(item[j+1] || 'i32') });
              j++;
            } else if (typeof item[j] === 'string') {
              locals.push({ name: null, type: this._valtype(item[j]) });
            }
          }
        }
        else { bodyNodes.push(item); }
      } else {
        bodyNodes.push(item);
      }
      idx++;
    }

    sections.funcs.push({ funcName, exported, params, locals, bodyNodes });
  }

  _buildBinary(sections) {
    const bytes = [];
    bytes.push(0x00, 0x61, 0x73, 0x6d);
    bytes.push(0x01, 0x00, 0x00, 0x00);

    const importFuncCount = sections.imports.length;
    const funcCount = sections.funcs.length;

    const typeSection = [];
    const typeMap = new Map();
    const funcTypeIndices = [];

    for (const imp of sections.imports) {
      const sig = this._typeSig(imp.params, imp.results);
      if (!typeMap.has(sig)) typeMap.set(sig, typeMap.size);
      funcTypeIndices.push(typeMap.get(sig));
    }
    for (const func of sections.funcs) {
      const results = [];
      const sig = this._typeSig(func.params, results);
      if (!typeMap.has(sig)) typeMap.set(sig, typeMap.size);
      funcTypeIndices.push(typeMap.get(sig));
    }

    const types = [];
    for (const [sig] of typeMap) {
      const [params, results] = sig.split('->');
      const p = params ? params.split(',').filter(Boolean).map(Number) : [];
      const r = results ? results.split(',').filter(Boolean).map(Number) : [];
      types.push({ params: p, results: r });
    }

    this._appendSection(bytes, 1, this._encodeTypeSection(types));
    this._appendSection(bytes, 2, this._encodeImportSection(sections.imports, funcTypeIndices));
    this._appendSection(bytes, 3, this._encodeFuncSection(funcTypeIndices.slice(importFuncCount)));
    this._appendSection(bytes, 5, this._encodeMemorySection(sections.memories));
    this._appendSection(bytes, 7, this._encodeExportSection(sections.memories, sections.funcs, importFuncCount));
    this._appendSection(bytes, 10, this._encodeCodeSection(sections.funcs, sections.imports));

    return new Uint8Array(bytes);
  }

  _typeSig(params, results) {
    return params.join(',') + '->' + results.join(',');
  }

  _encodeTypeSection(types) {
    const body = [];
    this._writeU32(body, types.length);
    for (const t of types) {
      body.push(0x60);
      this._writeU32(body, t.params.length);
      for (const p of t.params) body.push(p);
      this._writeU32(body, t.results.length);
      for (const r of t.results) body.push(r);
    }
    return body;
  }

  _encodeImportSection(imports, funcTypeIndices) {
    if (imports.length === 0) return null;
    const body = [];
    this._writeU32(body, imports.length);
    for (let i = 0; i < imports.length; i++) {
      const imp = imports[i];
      this._writeString(body, imp.module);
      this._writeString(body, imp.field);
      body.push(0x00);
      this._writeU32(body, funcTypeIndices[i]);
    }
    return body;
  }

  _encodeFuncSection(typeIndices) {
    const body = [];
    this._writeU32(body, typeIndices.length);
    for (const idx of typeIndices) this._writeU32(body, idx);
    return body;
  }

  _encodeMemorySection(memories) {
    const body = [];
    this._writeU32(body, memories.length);
    for (const mem of memories) {
      body.push(0x00);
      this._writeU32(body, mem.pages);
    }
    return body;
  }

  _encodeExportSection(memories, funcs, importFuncCount) {
    const body = [];
    const exports = [];
    for (let i = 0; i < memories.length; i++) {
      if (memories[i].exported) exports.push({ name: memories[i].exported, kind: 0x02, idx: i });
    }
    for (let i = 0; i < funcs.length; i++) {
      if (funcs[i].exported) exports.push({ name: funcs[i].exported, kind: 0x00, idx: importFuncCount + i });
    }
    this._writeU32(body, exports.length);
    for (const exp of exports) {
      this._writeString(body, exp.name);
      body.push(exp.kind);
      this._writeU32(body, exp.idx);
    }
    return body;
  }

  _encodeCodeSection(funcs, imports) {
    const body = [];
    this._writeU32(body, funcs.length);
    for (const func of funcs) {
      const funcBody = this._encodeFuncBody(func, imports);
      this._writeU32(body, funcBody.length);
      body.push(...funcBody);
    }
    return body;
  }

  _encodeFuncBody(func, imports) {
    const body = [];
    const localGroups = [];
    let currentType = -1;
    let count = 0;
    for (const local of func.locals) {
      if (local.type === currentType) { count++; }
      else {
        if (count > 0) localGroups.push({ count, type: currentType });
        currentType = local.type;
        count = 1;
      }
    }
    if (count > 0) localGroups.push({ count, type: currentType });

    this._writeU32(body, localGroups.length);
    for (const g of localGroups) {
      this._writeU32(body, g.count);
      body.push(g.type);
    }

    const nameMap = new Map();
    for (let i = 0; i < func.params.length; i++) nameMap.set(`$p${i}`, i);
    for (let i = 0; i < func.locals.length; i++) {
      if (func.locals[i].name) nameMap.set(func.locals[i].name, func.params.length + i);
    }

    const funcMap = new Map();
    for (let i = 0; i < imports.length; i++) {
      funcMap.set(imports[i].funcName, i);
    }

    for (const node of func.bodyNodes) {
      this._encodeInstruction(body, node, nameMap, funcMap);
    }

    body.push(0x0b);
    return body;
  }

  _encodeInstruction(body, node, nameMap, funcMap) {
    if (typeof node === 'string') {
      this._encodeSingleInstruction(body, node, nameMap, funcMap);
      return;
    }
    if (!Array.isArray(node)) return;

    const head = node[0];
    if (typeof head !== 'string') return;

    if (head === 'block' || head === 'loop') {
      const label = typeof node[1] === 'string' && node[1].startsWith('$') ? node[1] : null;
      body.push(head === 'block' ? 0x02 : 0x03);
      body.push(0x40);
      const start = label ? 2 : 1;
      for (let i = start; i < node.length; i++) {
        this._encodeInstruction(body, node[i], nameMap, funcMap);
      }
      body.push(0x0b);
      return;
    }

    if (head === 'if') {
      let resultType = null;
      let startIdx = 1;
      if (Array.isArray(node[1]) && node[1][0] === 'result') {
        resultType = this._valtype(node[1][1]);
        startIdx = 2;
      }
      body.push(0x04);
      body.push(resultType !== null ? resultType : 0x40);
      for (let i = startIdx; i < node.length; i++) {
        const child = node[i];
        if (Array.isArray(child) && child[0] === 'then') {
          for (let j = 1; j < child.length; j++) this._encodeInstruction(body, child[j], nameMap, funcMap);
        } else if (Array.isArray(child) && child[0] === 'else') {
          body.push(0x05);
          for (let j = 1; j < child.length; j++) this._encodeInstruction(body, child[j], nameMap, funcMap);
        } else {
          this._encodeInstruction(body, child, nameMap, funcMap);
        }
      }
      body.push(0x0b);
      return;
    }

    for (let i = 1; i < node.length; i++) {
      if (Array.isArray(node[i])) {
        this._encodeInstruction(body, node[i], nameMap, funcMap);
      }
    }
    this._encodeSingleInstruction(body, head, nameMap, funcMap, node);
  }

  _encodeSingleInstruction(body, instr, nameMap, funcMap, fullNode = null) {
    if (instr.startsWith('local.get')) {
      body.push(0x20);
      const name = fullNode ? this._findArg(fullNode, '$') : instr.split(' ')[1];
      this._writeU32(body, nameMap.get(name) ?? 0);
      return;
    }
    if (instr.startsWith('local.set')) {
      body.push(0x21);
      const name = fullNode ? this._findArg(fullNode, '$') : instr.split(' ')[1];
      this._writeU32(body, nameMap.get(name) ?? 0);
      return;
    }
    if (instr.startsWith('br_if')) {
      body.push(0x0d);
      this._writeU32(body, 0);
      return;
    }
    if (instr.startsWith('br ') || instr === 'br') {
      body.push(0x0c);
      this._writeU32(body, 0);
      return;
    }
    if (instr.startsWith('call')) {
      body.push(0x10);
      const name = fullNode ? this._findArg(fullNode, '$') : instr.split(' ')[1];
      this._writeU32(body, funcMap.get(name) ?? 0);
      return;
    }
    if (instr.startsWith('i32.const')) {
      body.push(0x41);
      const val = fullNode ? parseInt(this._findNumArg(fullNode)) : parseInt(instr.split(' ')[1]);
      this._writeI32(body, val);
      return;
    }
    if (instr.startsWith('f32.const')) {
      body.push(0x43);
      const val = fullNode ? parseFloat(this._findNumArg(fullNode)) : parseFloat(instr.split(' ')[1]);
      this._writeF32(body, val);
      return;
    }

    const opcodes = {
      'i32.add': 0x6a, 'i32.sub': 0x6b, 'i32.mul': 0x6c, 'i32.div_s': 0x6d, 'i32.rem_s': 0x6f,
      'i32.eq': 0x46, 'i32.ne': 0x47, 'i32.lt_s': 0x48, 'i32.gt_s': 0x4a, 'i32.le_s': 0x4c, 'i32.ge_s': 0x4e,
      'i32.eqz': 0x45,
      'f32.add': 0x92, 'f32.sub': 0x93, 'f32.mul': 0x94, 'f32.div': 0x95,
      'f32.eq': 0x5b, 'f32.ne': 0x5c, 'f32.lt': 0x5d, 'f32.gt': 0x5e, 'f32.le': 0x5f, 'f32.ge': 0x60,
      'f32.abs': 0x8b, 'f32.neg': 0x8c, 'f32.ceil': 0x8d, 'f32.floor': 0x8e, 'f32.sqrt': 0x91,
      'f32.min': 0x96, 'f32.max': 0x97,
      'f32.convert_i32_s': 0xb2, 'i32.trunc_f32_s': 0xa8,
      'f32.load': [0x2a, 2, 0], 'f32.store': [0x38, 2, 0],
      'i32.load': [0x28, 2, 0], 'i32.store': [0x36, 2, 0],
      'i32.load8_s': [0x2c, 0, 0], 'i32.load8_u': [0x2d, 0, 0], 'i32.store8': [0x3a, 0, 0],
      'i32.load16_s': [0x2e, 1, 0], 'i32.store16': [0x3b, 1, 0],
      'f64.load': [0x2b, 3, 0], 'f64.store': [0x39, 3, 0],
    };

    const simdOpcodes = {
      'v128.load': { prefix: 0xfd, sub: 0, memarg: [4, 0] },
      'v128.store': { prefix: 0xfd, sub: 11, memarg: [4, 0] },
      'v128.bitselect': { prefix: 0xfd, sub: 0x52 }, 'v128.and': { prefix: 0xfd, sub: 0x4e }, 'v128.or': { prefix: 0xfd, sub: 0x50 }, 'v128.not': { prefix: 0xfd, sub: 0x4d },
      'f32x4.splat': { prefix: 0xfd, sub: 0x13 }, 'i32x4.splat': { prefix: 0xfd, sub: 0x11 },
      'f32x4.add': { prefix: 0xfd, sub: 0xe4 }, 'f32x4.sub': { prefix: 0xfd, sub: 0xe5 },
      'f32x4.mul': { prefix: 0xfd, sub: 0xe6 }, 'f32x4.div': { prefix: 0xfd, sub: 0xe7 },
      'f32x4.neg': { prefix: 0xfd, sub: 0xe1 }, 'f32x4.abs': { prefix: 0xfd, sub: 0xe0 },
      'f32x4.sqrt': { prefix: 0xfd, sub: 0xe3 },
      'f32x4.ceil': { prefix: 0xfd, sub: 0x67 }, 'f32x4.floor': { prefix: 0xfd, sub: 0x68 },
      'f32x4.min': { prefix: 0xfd, sub: 0xe8 }, 'f32x4.max': { prefix: 0xfd, sub: 0xe9 },
      'f32x4.eq': { prefix: 0xfd, sub: 0x41 }, 'f32x4.ne': { prefix: 0xfd, sub: 0x42 },
      'f32x4.lt': { prefix: 0xfd, sub: 0x43 }, 'f32x4.gt': { prefix: 0xfd, sub: 0x44 },
      'f32x4.le': { prefix: 0xfd, sub: 0x45 }, 'f32x4.ge': { prefix: 0xfd, sub: 0x46 },
      'f32x4.extract_lane': { prefix: 0xfd, sub: 0x1f, lane: true },
      'f32x4.replace_lane': { prefix: 0xfd, sub: 0x20, lane: true },
      'i32x4.add': { prefix: 0xfd, sub: 0xae }, 'i32x4.sub': { prefix: 0xfd, sub: 0xb1 },
      'i32x4.mul': { prefix: 0xfd, sub: 0xb5 },
      'i32x4.abs': { prefix: 0xfd, sub: 0xa0 },
      'i32x4.min_s': { prefix: 0xfd, sub: 0xb6 }, 'i32x4.max_s': { prefix: 0xfd, sub: 0xb8 },
      'i32x4.eq': { prefix: 0xfd, sub: 0x37 }, 'i32x4.ne': { prefix: 0xfd, sub: 0x38 },
      'i32x4.lt_s': { prefix: 0xfd, sub: 0x39 }, 'i32x4.gt_s': { prefix: 0xfd, sub: 0x3a },
      'i32x4.le_s': { prefix: 0xfd, sub: 0x3b }, 'i32x4.ge_s': { prefix: 0xfd, sub: 0x3c },
      'i32x4.extract_lane': { prefix: 0xfd, sub: 0x1b, lane: true },
      'i32x4.replace_lane': { prefix: 0xfd, sub: 0x1c, lane: true },
    };

    const op = opcodes[instr];
    if (op !== undefined) {
      if (Array.isArray(op)) {
        body.push(op[0]);
        this._writeU32(body, op[1]);
        this._writeU32(body, op[2]);
      } else {
        body.push(op);
      }
      return;
    }

    const baseName = instr.endsWith('.extract_lane') || instr.endsWith('.replace_lane') ? instr : instr;
    const simd = simdOpcodes[baseName];
    if (simd) {
      body.push(simd.prefix);
      this._writeU32(body, simd.sub);
      if (simd.memarg) {
        this._writeU32(body, simd.memarg[0]);
        this._writeU32(body, simd.memarg[1]);
      }
      if (simd.lane && fullNode) {
        const laneVal = parseInt(this._findNumArg(fullNode));
        body.push(laneVal);
      }
    }
  }

  _findArg(node, prefix) {
    for (let i = 1; i < node.length; i++) {
      if (typeof node[i] === 'string' && node[i].startsWith(prefix)) return node[i];
    }
    return null;
  }

  _findNumArg(node) {
    for (let i = 1; i < node.length; i++) {
      if (typeof node[i] === 'string' && !node[i].startsWith('$') && !node[i].startsWith('(')) return node[i];
    }
    return '0';
  }

  _valtype(s) {
    switch (s) {
      case 'i32': return 0x7f;
      case 'i64': return 0x7e;
      case 'f32': return 0x7d;
      case 'f64': return 0x7c;
      case 'v128': return 0x7b;
      default: return 0x7f;
    }
  }

  _stripQuotes(s) {
    if (s && s.startsWith('"') && s.endsWith('"')) return s.slice(1, -1);
    return s;
  }

  _appendSection(bytes, id, content) {
    if (!content) return;
    bytes.push(id);
    this._writeU32(bytes, content.length);
    bytes.push(...content);
  }

  _writeU32(bytes, value) {
    let v = value >>> 0;
    do {
      let b = v & 0x7f;
      v >>>= 7;
      if (v !== 0) b |= 0x80;
      bytes.push(b);
    } while (v !== 0);
  }

  _writeI32(bytes, value) {
    let v = value;
    let more = true;
    while (more) {
      let b = v & 0x7f;
      v >>= 7;
      if ((v === 0 && (b & 0x40) === 0) || (v === -1 && (b & 0x40) !== 0)) more = false;
      else b |= 0x80;
      bytes.push(b);
    }
  }

  _writeF32(bytes, value) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    const arr = new Uint8Array(buf);
    bytes.push(arr[0], arr[1], arr[2], arr[3]);
  }

  _writeString(bytes, str) {
    const encoded = new TextEncoder().encode(str);
    this._writeU32(bytes, encoded.length);
    bytes.push(...encoded);
  }
}
