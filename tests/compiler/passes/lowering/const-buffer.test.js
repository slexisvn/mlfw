import { describe, it, expect } from 'vitest';
import { tensor, manual_seed, Linear, Sequential, ReLU } from '../../../../src/index.js';
import { compile, _traceCore } from '../../../../src/tracing/compile.js';
import { foldWeightParams, weightPredicate } from '../../../../src/tracing/fold_params.js';
import { tensorToContiguous } from '../../../../src/dispatcher/jit_dispatch.js';
import { CPUTarget, WasmTarget, CUDATarget, WebGPUTarget } from '../../../../src/backend/target.js';
import { QuantizationScheme } from '../../../../src/compiler/ir/graph/quantization_types.js';
import { FuncAttr } from '../../../../src/compiler/ir/func_attrs.js';
import { BufferStoreNode } from '../../../../src/compiler/ir/tensor/nodes.js';
import { collect } from '../../../../src/compiler/ir/ir_visitor.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { LegalizeConstBuffersPass } from '../../../../src/compiler/passes/lowering/const_buffer_pass.js';

const D = 48;
const B = 5;

function mlp() {
  manual_seed(4231);
  return new Sequential(new Linear(D, D, false), new ReLU(), new Linear(D, D, true), new ReLU(), new Linear(D, D, false));
}

function inputs() {
  const rows = [];
  for (let i = 0; i < B; i++) {
    const r = [];
    for (let d = 0; d < D; d++) r.push(Math.sin(i * 0.37 + d * 0.11));
    rows.push(r);
  }
  return tensor(rows);
}

function constBuffersOf(compiled) {
  const result = compiled.result();
  const all = [];
  for (const name of result.listKernels()) {
    const cbs = result.module.getKernelMetadata(name).constBuffers;
    if (cbs) all.push(...cbs);
  }
  return all;
}

async function compiled(model, x, opts) {
  const c = compile(model, [x], opts);
  if (c._ready) await c._ready;
  return c;
}

function storesInto(primFunc, buffer) {
  return collect(primFunc, n => n instanceof BufferStoreNode && n.buffer === buffer).length;
}

describe('folded weights lower to constant buffers', () => {
  it('binds each folded weight as a trailing kernel parameter carrying its data', async () => {
    const model = mlp();
    const x = inputs();
    const c = await compiled(model, x, { target: CPUTarget(), foldWeights: true });

    const cbs = constBuffersOf(c);
    expect(cbs.length).toBe(3);

    const weights = [model[0].weight, model[2].weight, model[4].weight];
    for (const w of weights) {
      const match = cbs.find(cb => cb.data.length === w.numel);
      expect(match).toBeDefined();
      expect(match.dtype).toBe(w.dtype);
    }

    const source = c.source();
    for (const cb of cbs) expect(source).toContain(cb.name);
  });

  it('emits no per-element immediate store for a folded weight', async () => {
    const x = inputs();
    const folded = await compiled(mlp(), x, { target: CPUTarget(), foldWeights: true });
    const plain = await compiled(mlp(), x, { target: CPUTarget(), foldWeights: false });

    const weightElements = 3 * D * D;
    expect(folded.source().length).toBeLessThan(plain.source().length + weightElements);
  });

  it('produces the same result as the unfolded compile on every constant-buffer target', async () => {
    const model = mlp();
    const x = inputs();
    const reference = model.forward(x);

    for (const target of [CPUTarget(), WasmTarget()]) {
      const c = await compiled(model, x, { target, foldWeights: true });
      expect(constBuffersOf(c).length).toBe(3);
      const out = await c(x);
      expect(out.shape).toEqual(reference.shape);
      for (let i = 0; i < reference.data.length; i++) {
        expect(out.data[i]).toBeCloseTo(reference.data[i], 6);
      }
    }
  });

  it('folds weights larger than the immediate-store cap when the target links constants', async () => {
    manual_seed(4231);
    const big = new Sequential(new Linear(128, 128, false));
    const rows = [];
    for (let i = 0; i < 4; i++) {
      const r = [];
      for (let d = 0; d < 128; d++) r.push(Math.cos(i * 0.9 + d * 0.013));
      rows.push(r);
    }
    const x = tensor(rows);
    const reference = big.forward(x);

    const c = await compiled(big, x, { target: CPUTarget(), foldWeights: true });
    const cbs = constBuffersOf(c);
    expect(cbs.length).toBe(1);
    expect(cbs[0].data.length).toBe(128 * 128);

    const out = await c(x);
    for (let i = 0; i < reference.data.length; i++) {
      expect(out.data[i]).toBeCloseTo(reference.data[i], 5);
    }
  });

  it('leaves the unfolded compile free of constant buffers', async () => {
    const c = await compiled(mlp(), inputs(), { target: CPUTarget() });
    expect(constBuffersOf(c).length).toBe(0);
  });

  it('keeps weight-only int8 quantization working on a folded Linear(128, 128)', async () => {
    const lin = new Linear(128, 128, false);
    for (let i = 0; i < lin.weight.data.length; i++) {
      lin.weight.data[i] = Math.sin(i * 0.017) * 0.3;
    }
    const model = new Sequential(lin);
    const rows = [];
    for (let i = 0; i < 4; i++) {
      const r = [];
      for (let d = 0; d < 128; d++) r.push(Math.cos(i * 1.3 + d * 0.021));
      rows.push(r);
    }
    const x = tensor(rows);

    const float = await compiled(model, x, { target: CPUTarget() });
    const quant = await compiled(model, x, {
      target: CPUTarget(),
      foldWeights: true,
      quantization: { enabled: true, scheme: QuantizationScheme.PER_CHANNEL, quantizableOps: new Set(['dot']) },
    });

    const fo = await float(x);
    const qo = await quant(x);
    let maxAbs = 0;
    for (let i = 0; i < fo.data.length; i++) maxAbs = Math.max(maxAbs, Math.abs(fo.data[i]));
    for (let i = 0; i < fo.data.length; i++) {
      expect(Math.abs(qo.data[i] - fo.data[i])).toBeLessThan(maxAbs * 0.05);
    }
  });
});

describe('LegalizeConstBuffersPass', () => {
  it('expands constant buffers back into stores for targets that cannot bind them', async () => {
    const small = new Sequential(new Linear(16, 16, false));
    const rows = [];
    for (let i = 0; i < 3; i++) {
      const r = [];
      for (let d = 0; d < 16; d++) r.push(Math.sin(i + d * 0.2));
      rows.push(r);
    }
    const x = tensor(rows);

    const gpu = await compiled(small, x, { target: WebGPUTarget(), foldWeights: true });
    expect(constBuffersOf(gpu).length).toBe(0);

    const literals = gpu.source().match(/buf_\d+\[\d+\] = /g) || [];
    expect(literals.length).toBeGreaterThanOrEqual(16 * 16);
  });

  it('moves the constant buffer out of the parameter list and into the body', () => {
    const model = new Sequential(new Linear(8, 8, false));
    const traced = foldWeightParams(
      _traceCore((...a) => model.forward(...a), [tensor([[1, 2, 3, 4, 5, 6, 7, 8]])]),
      tensorToContiguous,
      weightPredicate(Infinity),
    );
    const pf = lowerGraphToPrimFunc(traced.graph.functions().next().value, CPUTarget());

    const declared = pf.getAttr(FuncAttr.CONST_BUFFERS);
    expect(declared.length).toBe(1);
    const constBuffer = declared[0].buffer;
    expect([...pf.bufferMap.values()]).toContain(constBuffer);
    expect(storesInto(pf, constBuffer)).toBe(0);
    expect(pf.params.length).toBe(pf.bufferMap.size + pf.shapeParams.length);

    new LegalizeConstBuffersPass().run(pf);

    expect(pf.getAttr(FuncAttr.CONST_BUFFERS)).toBe(null);
    expect([...pf.bufferMap.values()]).not.toContain(constBuffer);
    expect(pf.params.length).toBe(pf.bufferMap.size + pf.shapeParams.length);
    expect(storesInto(pf, constBuffer)).toBe(64);
  });
});

describe('constant-buffer capability is a target property', () => {
  it('is declared by the targets whose launch path binds trailing buffer arguments', () => {
    expect(CPUTarget().supportsConstBuffers).toBe(true);
    expect(WasmTarget().supportsConstBuffers).toBe(true);
    expect(CUDATarget().supportsConstBuffers).toBe(true);
    expect(WebGPUTarget().supportsConstBuffers).toBe(false);
  });
});
