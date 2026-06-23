import { describe, it, expect } from 'vitest';
import { registry } from '../../src/compiler/ir/graph/ops.js';
import { buildFunction } from '../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../src/compiler/ir/graph/types.js';
import { mathOp, IntImmNode } from '../../src/compiler/ir/tensor/nodes.js';

const t = (s, dt = ScalarType.F32) => new TensorType(s, dt);

describe('second-pass review fixes', () => {
  it('arithmetic/exp/quant folds bail on tensor (array) constants instead of corrupting them', () => {
    expect(registry.get('add').fold([2, 3])).toBe(5);
    expect(registry.get('add').fold([new Float32Array([1, 2]), new Float32Array([3, 4])])).toBeUndefined();
    expect(registry.get('mul').fold([new Float32Array([1, 2]), 3])).toBeUndefined();
    expect(registry.get('div').fold([new Float32Array([1]), new Float32Array([2])])).toBeUndefined();
    expect(registry.get('neg').fold([new Float32Array([1, 2])])).toBeUndefined();
    expect(registry.get('exp').fold([new Float32Array([0, 1])])).toBeUndefined();
    expect(registry.get('exp').fold([0])).toBe(1);
  });

  it('mathOp does not fold integer // or % by zero', () => {
    expect(mathOp('//', new IntImmNode(4), new IntImmNode(0)).type).toBe('MathOpNode');
    expect(mathOp('%', new IntImmNode(4), new IntImmNode(0)).type).toBe('MathOpNode');
    expect(mathOp('//', new IntImmNode(7), new IntImmNode(2)).type).toBe('IntImmNode');
    expect(mathOp('%', new IntImmNode(7), new IntImmNode(3)).type).toBe('IntImmNode');
  });

  it('where requires matching branch dtypes (mirrors select)', () => {
    const where = registry.get('where');
    expect(where.inferResultTypes([t([2], ScalarType.BOOL), t([2], ScalarType.F32), t([2], ScalarType.I32)])).toBeNull();
    expect(where.inferResultTypes([t([2], ScalarType.BOOL), t([2], ScalarType.F32), t([2], ScalarType.F32)])).not.toBeNull();
  });

  it('scanOp rejects xs inputs with mismatched leading lengths', () => {
    expect(() => buildFunction('s', [t([3, 2]), t([4, 2]), t([2])], [t([2])], (b, [xs1, xs2, init]) => {
      const s = b.scanOp([xs1, xs2], [init], (bb, xt, cy) => {
        const nc = bb.add(cy[0], xt[0]).getResult(0);
        return [[nc], [nc]];
      });
      b.returnOp([s.getResult(0)]);
    })).toThrow(/leading length/);
  });
});
