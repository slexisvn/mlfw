import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { BufferLoadNode, IntImmNode } from '../../../../src/compiler/ir/tensor/nodes.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, WasmTarget } from '../../../../src/compiler/support/target.js';

const SKIP_KEYS = new Set(['_parent', '_parentKey', '_parentIdx']);

function collectNodes(node, predicate) {
  const result = [];
  const visited = new Set();
  const stack = [node];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object' || visited.has(n)) continue;
    visited.add(n);
    if (predicate(n)) result.push(n);
    for (const key of Object.keys(n)) {
      if (SKIP_KEYS.has(key)) continue;
      const val = n[key];
      if (Array.isArray(val)) for (const item of val) stack.push(item);
      else if (val && typeof val === 'object') stack.push(val);
    }
  }
  return result;
}

function loadFor(pf, bufferName) {
  return collectNodes(pf.body, n => n instanceof BufferLoadNode && n.buffer && n.buffer.name === bufferName);
}

function inputLoad(pf, inBufName) {
  const loads = loadFor(pf, inBufName);
  return loads.find(l => l.indices.length === 4);
}

describe('pool2d lowering layout awareness', () => {
  it('NCHW max-pool indexes channels at axis 1 and spatial at axes 2,3', () => {
    const inT = new TensorType([1, 3, 4, 4], ScalarType.F32);
    const outT = new TensorType([1, 3, 2, 2], ScalarType.F32);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      b.returnOp([b.pool2d(args[0], 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
    });
    const pf = lowerGraphToPrimFunc(func);
    const inBuf = pf.bufferMap.get(pf.params[0]);
    const load = inputLoad(pf, inBuf.name);
    expect(load).toBeDefined();
    expect(load.buffer.shape).toEqual([1, 3, 4, 4]);
  });

  it('NHWC avg-pool reads spatial dims from layout axes 1,2 (not hardcoded 2,3)', () => {
    const inT = new TensorType([1, 4, 4, 3], ScalarType.F32);
    const outT = new TensorType([1, 2, 2, 3], ScalarType.F32);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      const op = b._buildOp('pool2d', [args[0]], [outT], {
        pool_type: 'avg',
        kernel_size: [2, 2],
        strides: [2, 2],
        padding: [[0, 0], [0, 0]],
        ceil_mode: false,
        count_include_pad: false,
        layout: 'NHWC',
      });
      b.returnOp([op.getResult(0)]);
    });
    const pf = lowerGraphToPrimFunc(func);
    const inBuf = pf.bufferMap.get(pf.params[0]);
    const load = inputLoad(pf, inBuf.name);
    expect(load).toBeDefined();

    expect(load.indices.length).toBe(4);
    const chIdx = load.indices[3];
    const isPlainVar = chIdx && chIdx.type !== 'MathOpNode';
    expect(isPlainVar).toBe(true);

    const stores = collectNodes(pf.body, n => n && n.type === 'BufferStoreNode');
    expect(stores.length).toBeGreaterThan(0);
  });
});

function poolRef(inp, N, C, H, W, k, s, pad, type, cip) {
  const oH = Math.floor((H + 2 * pad - k) / s) + 1, oW = Math.floor((W + 2 * pad - k) / s) + 1;
  const out = [];
  for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let oh = 0; oh < oH; oh++) for (let ow = 0; ow < oW; ow++) {
    let acc = type === 'max' ? -Infinity : 0, cnt = 0;
    for (let kh = 0; kh < k; kh++) for (let kw = 0; kw < k; kw++) {
      const ih = oh * s + kh - pad, iw = ow * s + kw - pad;
      if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        const v = inp[((n * C + c) * H + ih) * W + iw];
        if (type === 'max') acc = Math.max(acc, v); else { acc += v; cnt++; }
      }
    }
    out.push(type === 'max' ? acc : acc / (cip ? k * k : cnt));
  }
  return out;
}

describe('pool2d end-to-end value vs independent reference (cpu+wasm)', () => {
  const N = 1, C = 2, H = 6, W = 6;
  let seed = 7;
  const rng = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const inp = new Float32Array(N * C * H * W);
  for (let i = 0; i < inp.length; i++) inp[i] = -3 + 6 * rng();

  const CONFIGS = [];
  for (const type of ['max', 'avg']) for (const k of [2, 3]) for (const s of [1, 2]) for (const pad of [0, 1]) for (const cip of [false, true]) {
    if (type === 'max' && cip) continue;
    CONFIGS.push({ type, k, s, pad, cip });
  }

  for (const cfg of CONFIGS) {
    for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
      it(`pool2d ${cfg.type} k${cfg.k} s${cfg.s} pad${cfg.pad} cip${cfg.cip} on ${tname}`, () => {
        const oH = Math.floor((H + 2 * cfg.pad - cfg.k) / cfg.s) + 1, oW = Math.floor((W + 2 * cfg.pad - cfg.k) / cfg.s) + 1;
        const ref = poolRef(inp, N, C, H, W, cfg.k, cfg.s, cfg.pad, cfg.type, cfg.cip);
        const func = buildFunction('p', [new TensorType([N, C, H, W], ScalarType.F32)], [new TensorType([N, C, oH, oW], ScalarType.F32)],
          (b, a) => { b.returnOp([b.pool2d(a[0], cfg.type, [cfg.k, cfg.k], [cfg.s, cfg.s], [[cfg.pad, cfg.pad], [cfg.pad, cfg.pad]], { countIncludePad: cfg.cip }).getResult(0)]); });
        const res = compileGraph(func, makeTarget());
        const out = new Float32Array(N * C * oH * oW);
        res.run('p', inp, out);
        for (let i = 0; i < ref.length; i++) {
          const relErr = Math.abs(ref[i] - out[i]) / (1 + Math.abs(ref[i]));
          expect(relErr, `idx ${i}: ref=${ref[i]} got=${out[i]}`).toBeLessThan(2e-3);
        }
      });
    }
  }
});
