import { describe, it, expect } from 'vitest';
import { registry } from '../../../../src/compiler/ir/graph/ops.js';
import { hasLoweringRule } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { hasDecomposition } from '../../../../src/compiler/passes/decompose/decomposition_pass.js';
import { OpTrait } from '../../../../src/compiler/ir/graph/op_registry.js';

const STRUCTURAL = new Set([
  'return', 'yield', 'constant', 'scalar_constant', 'fusion', 'scan', 'if', 'while', 'custom_call',
  'tuple', 'get_tuple_element', 'call',
]);

describe('lowering coverage (build-time gap report)', () => {
  it('every non-opaque, non-decomposed, non-structural op has a lowering rule', () => {
    const gaps = [];
    for (const name of registry.names()) {
      if (STRUCTURAL.has(name)) continue;
      const def = registry.get(name);
      if (def.hasTrait(OpTrait.OPAQUE)) continue;
      if (hasDecomposition(name)) continue;
      if (!hasLoweringRule(name)) gaps.push(name);
    }
    expect(gaps).toEqual([]);
  });
});
