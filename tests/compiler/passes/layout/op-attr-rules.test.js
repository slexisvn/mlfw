import { describe, it, expect, afterEach } from 'vitest';
import { registry } from '../../../../src/compiler/ir/graph/ops.js';
import { OpDef } from '../../../../src/compiler/ir/graph/op_registry.js';
import { OpAttrKey, hasLibraryOp } from '../../../../src/compiler/ir/graph/op_traits.js';
import { LayoutPreference } from '../../../../src/compiler/ir/graph/layout_pref.js';
import { LayoutPolicy } from '../../../../src/compiler/passes/layout/layout_policy.js';
import { Layout, TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { Operation } from '../../../../src/compiler/ir/graph/operation.js';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, CUDATarget } from '../../../../src/compiler/support/target.js';

const CUSTOM_OP = 'test_layout_sensitive_op';
const f32 = (shape) => new TensorType(shape, ScalarType.F32);
const opsOf = (func) => [...func.ops()].filter(o => o.opName !== 'return');

afterEach(() => {
  registry.unregister(CUSTOM_OP);
});

function registerCustomOp(opAttrs) {
  registry.register(new OpDef({ name: CUSTOM_OP, numOperands: 1, numResults: 1, opAttrs }));
  return new Operation(CUSTOM_OP, [], [f32([16, 16])]);
}

describe('layout rules live on the op registry', () => {
  it('honours a layout rule declared by a newly registered op', () => {
    const colMajor = Layout.columnMajor(2);
    const op = registerCustomOp({
      [OpAttrKey.INFER_LAYOUT]: () => new LayoutPreference([colMajor], [colMajor]),
    });

    const pref = new LayoutPolicy(CPUTarget()).getPreference(op);
    expect(pref.inputs[0]).toBe(colMajor);
    expect(pref.outputs[0]).toBe(colMajor);
  });

  it('returns no preference for an op that declares none', () => {
    const op = registerCustomOp({});
    expect(new LayoutPolicy(CPUTarget()).getPreference(op)).toBe(null);
  });

  it('passes the target through so a rule can specialise per backend', () => {
    const seen = [];
    const op = registerCustomOp({
      [OpAttrKey.INFER_LAYOUT]: (_op, target) => {
        seen.push(target.kind);
        return null;
      },
    });
    new LayoutPolicy(CPUTarget()).getPreference(op);
    new LayoutPolicy(CUDATarget()).getPreference(op);
    expect(seen).toEqual(['cpu', 'cuda']);
  });

  it('lets a per-compilation override win over the registry rule', () => {
    const fromRegistry = Layout.rowMajor(2);
    const fromOverride = Layout.columnMajor(2);
    const op = registerCustomOp({
      [OpAttrKey.INFER_LAYOUT]: () => new LayoutPreference([fromRegistry], [fromRegistry]),
    });

    const policy = new LayoutPolicy(CPUTarget());
    policy.registerRule(CUSTOM_OP, () => new LayoutPreference([fromOverride], [fromOverride]));
    expect(policy.getPreference(op).inputs[0]).toBe(fromOverride);
  });
});

describe('layout benefit weighting comes from op attributes', () => {
  it('scales the benefit by the declared layout sensitivity', () => {
    const op = registerCustomOp({ [OpAttrKey.LAYOUT_SENSITIVITY]: 7 });
    const policy = new LayoutPolicy(CPUTarget());
    expect(policy.estimateBenefit(op, f32([10, 10]), 3)).toBe(100 * 7 * 3);
  });

  it('keeps the built-in matmul and reduce weights', () => {
    const t = f32([16, 16]);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const dot = opsOf(func).find(o => o.opName === 'dot');
    const policy = new LayoutPolicy(CPUTarget());
    expect(policy.estimateBenefit(dot, f32([4, 4]), 1)).toBe(16 * 4);
  });

  it('falls back to the cache-line heuristic for ops with no declared sensitivity', () => {
    const op = registerCustomOp({});
    const policy = new LayoutPolicy(CPUTarget());
    expect(policy.estimateBenefit(op, f32([8, 8]), 1)).toBe(0);
    expect(policy.estimateBenefit(op, f32([64, 64]), 1)).toBe(Math.floor(4096 * 0.5));
  });
});

describe('target library kernels are keyed by launch-boundary class', () => {
  it('matches every op in a class the target declares a library for', () => {
    const cuda = CUDATarget();
    expect(hasLibraryOp(cuda, 'dot')).toBe(true);
    expect(hasLibraryOp(cuda, 'cublas_gemm')).toBe(true);
    expect(hasLibraryOp(cuda, 'conv')).toBe(true);
    expect(hasLibraryOp(cuda, 'quantized_conv')).toBe(true);
    expect(hasLibraryOp(cuda, 'reduce')).toBe(false);
    expect(hasLibraryOp(cuda, 'add')).toBe(false);
  });

  it('reports no library kernels on a target that declares none', () => {
    const cpu = CPUTarget();
    expect(hasLibraryOp(cpu, 'dot')).toBe(false);
    expect(hasLibraryOp(CPUTarget({ libraryClasses: new Set(['matmul']) }), 'dot')).toBe(true);
  });
});
