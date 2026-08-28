import { describe, it, expect } from 'vitest';
import { planCommonSubexprs } from '../../src/backend/expr_cse.js';
import {
  MathOpNode, CompareNode, IfThenElseNode, CastNode,
  BufferLoadNode, VariableNode, IntImmNode, FloatImmNode,
} from '../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../src/compiler/ir/tensor/buffer.js';

const iv = (name, dtype = 'int32') => new VariableNode(name, dtype);
const add = (a, b) => new MathOpNode('+', a, b);
const mul = (a, b) => new MathOpNode('*', a, b);
const floorDiv = (a, b) => new MathOpNode('//', a, new IntImmNode(b));
const floorMod = (a, b) => new MathOpNode('%', a, new IntImmNode(b));

function decomposition(base) {
  return add(add(mul(floorDiv(base, 7), new IntImmNode(8)), floorMod(base, 7)), new IntImmNode(3));
}

const hoistedNodes = (plan) => plan.hoisted.map(h => h.node);

describe('planCommonSubexprs', () => {
  it('merges structurally equal but distinct subtrees into one class', () => {
    const base = iv('idx');
    const left = decomposition(base);
    const right = decomposition(base);
    expect(left).not.toBe(right);

    const plan = planCommonSubexprs(add(left, right), 4);

    expect(plan.ids.get(left)).toBe(plan.ids.get(right));
    expect(hoistedNodes(plan)).toContain(left);
  });

  it('leaves an expression with no repetition untouched', () => {
    const plan = planCommonSubexprs(add(decomposition(iv('a')), decomposition(iv('b'))), 4);
    expect(plan.hoisted).toEqual([]);
  });

  it('orders hoisted classes so every class precedes its users', () => {
    const base = iv('idx');
    const plan = planCommonSubexprs(add(decomposition(base), decomposition(base)), 2);
    const position = new Map(plan.hoisted.map((h, i) => [h.id, i]));

    for (const { id, node } of plan.hoisted) {
      for (const child of [node.a, node.b]) {
        const childId = child && plan.ids.get(child);
        if (childId !== undefined && position.has(childId)) {
          expect(position.get(childId)).toBeLessThan(position.get(id));
        }
      }
    }
  });

  it('respects the minimum size so trivial repeats are not bound', () => {
    const shared = add(iv('i'), new IntImmNode(1));
    const root = add(shared, shared);
    expect(planCommonSubexprs(root, 2).hoisted.length).toBe(1);
    expect(planCommonSubexprs(root, 8).hoisted).toEqual([]);
  });

  it('never hoists a buffer load, so guarded out-of-range reads stay guarded', () => {
    const buffer = new Buffer('src', [16], 'f32', 'global');
    const i = iv('i');
    const load = new BufferLoadNode(buffer, [decomposition(i)]);
    const guard = new CompareNode('lt', i, new IntImmNode(16));
    const root = add(new IfThenElseNode(guard, load, new FloatImmNode(0)), new IfThenElseNode(guard, load, new FloatImmNode(0)));

    const plan = planCommonSubexprs(root, 2);

    expect(hoistedNodes(plan)).not.toContain(load);
    expect(hoistedNodes(plan).every(n => n.type === 'MathOpNode')).toBe(true);
  });

  it('hoists the pure index shared between two guarded branches', () => {
    const buffer = new Buffer('src', [16], 'f32', 'global');
    const i = iv('i');
    const index = decomposition(i);
    const guard = new CompareNode('lt', i, new IntImmNode(16));
    const root = add(
      new IfThenElseNode(guard, new BufferLoadNode(buffer, [index]), new FloatImmNode(0)),
      new IfThenElseNode(guard, new BufferLoadNode(buffer, [decomposition(i)]), new FloatImmNode(0)),
    );

    expect(hoistedNodes(planCommonSubexprs(root, 4))).toContain(index);
  });

  it('never hoists float arithmetic, which would be truncated by an integer binding', () => {
    const x = new VariableNode('x', 'f32');
    const shared = mul(add(x, new FloatImmNode(1)), add(x, new FloatImmNode(2)));
    const plan = planCommonSubexprs(add(shared, shared), 2);
    expect(plan.hoisted).toEqual([]);
  });

  it('never hoists a division whose divisor is not a non-zero constant', () => {
    const base = iv('idx');
    const risky = new MathOpNode('//', mul(base, new IntImmNode(4)), iv('denom'));
    expect(planCommonSubexprs(add(risky, risky), 2).hoisted).toEqual([]);

    const byZero = new MathOpNode('//', mul(base, new IntImmNode(4)), new IntImmNode(0));
    expect(planCommonSubexprs(add(byZero, byZero), 2).hoisted).toEqual([]);
  });

  it('never hoists through a cast, whose source may not be integral', () => {
    const cast = new CastNode(mul(new VariableNode('f', 'f32'), new FloatImmNode(2)), 'f32', 'i32');
    const pure = decomposition(iv('idx'));
    const root = add(add(mul(cast, new IntImmNode(3)), pure), add(mul(cast, new IntImmNode(3)), decomposition(iv('idx'))));

    const hoisted = hoistedNodes(planCommonSubexprs(root, 4));

    expect(hoisted).toContain(pure);
    expect(hoisted).not.toContain(cast);
    expect(hoisted.some(n => n.a === cast || n.b === cast)).toBe(false);
  });

  it('treats loads from different buffers as different classes', () => {
    const i = iv('i');
    const a = new BufferLoadNode(new Buffer('a', [16], 'f32', 'global'), [i]);
    const b = new BufferLoadNode(new Buffer('b', [16], 'f32', 'global'), [i]);
    const plan = planCommonSubexprs(add(a, b), 1);
    expect(plan.ids.get(a)).not.toBe(plan.ids.get(b));
  });
});
