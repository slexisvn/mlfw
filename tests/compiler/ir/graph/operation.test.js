import { describe, it, expect } from 'vitest';
import { Operation } from '../../../../src/compiler/ir/graph/operation.js';
import { Block, Region } from '../../../../src/compiler/ir/graph/block.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';

function f32(shape) {
  return new TensorType(shape, ScalarType.F32);
}

describe('Operation.clone region value remap', () => {
  it('remaps cloned inner ops to new block arguments', () => {
    const t = f32([4]);
    const region = new Region();
    const block = new Block([t, t]);
    region.addBlock(block);

    const inner = new Operation('add', [block.arguments[0], block.arguments[1]], [t]);
    block.pushOp(inner);

    const outer = new Operation('fusion', [], [t], null, [region]);
    const cloned = outer.clone();

    const clonedRegion = cloned.regions[0];
    const clonedBlock = clonedRegion.blocks[0];
    const clonedInner = clonedBlock.firstOp;

    expect(clonedInner.getOperand(0)).toBe(clonedBlock.arguments[0]);
    expect(clonedInner.getOperand(1)).toBe(clonedBlock.arguments[1]);
    expect(clonedInner.getOperand(0)).not.toBe(block.arguments[0]);
  });

  it('remaps inner ops that consume earlier cloned results', () => {
    const t = f32([4]);
    const region = new Region();
    const block = new Block([t]);
    region.addBlock(block);

    const first = new Operation('neg', [block.arguments[0]], [t]);
    block.pushOp(first);
    const second = new Operation('abs', [first.results[0]], [t]);
    block.pushOp(second);

    const outer = new Operation('fusion', [], [t], null, [region]);
    const cloned = outer.clone();
    const clonedBlock = cloned.regions[0].blocks[0];
    const clonedFirst = clonedBlock.firstOp;
    const clonedSecond = clonedFirst._next;

    expect(clonedSecond.getOperand(0)).toBe(clonedFirst.results[0]);
    expect(clonedSecond.getOperand(0)).not.toBe(first.results[0]);
  });

  it('keeps original block arguments referenced by the original op', () => {
    const t = f32([4]);
    const region = new Region();
    const block = new Block([t]);
    region.addBlock(block);
    const inner = new Operation('neg', [block.arguments[0]], [t]);
    block.pushOp(inner);
    const outer = new Operation('fusion', [], [t], null, [region]);

    outer.clone();

    expect(inner.getOperand(0)).toBe(block.arguments[0]);
  });
});

describe('IRBuilder result-type inference errors carry operand context', () => {
  const f32 = (shape) => new TensorType(shape, ScalarType.F32);

  it('reports op name, reason, and operand shape:dtype on inference failure', () => {
    let err = null;
    try {
      buildFunction('t', [f32([3]), f32([4])], [f32([3])],
        (b, a) => { b.returnOp([b.add(a[0], a[1]).getResult(0)]); });
    } catch (e) { err = e; }

    expect(err).not.toBeNull();
    expect(err.message).toContain("op 'add'");
    expect(err.message).toContain('[3]:f32');
    expect(err.message).toContain('[4]:f32');
  });
});

describe('the use list is maintained by construction', () => {
  it('records one use per operand position and drops it on erase', () => {
    const t = f32([4]);
    const func = buildFunction('uses', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([sum.getResult(0)]);
    });
    const add = func.findOp(o => o.opName === 'add');
    const [lhs, rhs] = func.args;

    expect(lhs.useCount).toBe(1);
    expect(lhs.getUsers()).toEqual([add]);
    expect(rhs.getUsers()).toEqual([add]);

    const self = new Operation('mul', [lhs, lhs], [t]);
    expect(lhs.useCount).toBe(3);
    self.dropAllOperands();
    expect(lhs.useCount).toBe(1);
  });

  it('erase refuses while a result still has users, and succeeds once it does not', () => {
    const t = f32([4]);
    const func = buildFunction('erase', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([sum.getResult(0)]);
    });
    const add = func.findOp(o => o.opName === 'add');

    expect(() => add.erase()).toThrow(/result 0 still has uses/);

    add.getResult(0).replaceAllUsesWith(func.args[0]);
    expect(add.getResult(0).hasUses).toBe(false);
    add.erase();
    expect(func.findOp(o => o.opName === 'add')).toBe(null);
  });

  it('replaceAllUsesWith rewires every consumer and bumps the function version', () => {
    const t = f32([4]);
    const func = buildFunction('rauw', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const twice = b.mul(sum.getResult(0), sum.getResult(0));
      b.returnOp([twice.getResult(0)]);
    });
    const add = func.findOp(o => o.opName === 'add');
    const mul = func.findOp(o => o.opName === 'mul');
    const before = func.version;

    expect(add.getResult(0).useCount).toBe(2);
    add.getResult(0).replaceAllUsesWith(func.args[0]);

    expect(mul.getOperand(0)).toBe(func.args[0]);
    expect(mul.getOperand(1)).toBe(func.args[0]);
    expect(add.getResult(0).useCount).toBe(0);
    expect(func.args[0].useCount).toBe(3);
    expect(func.version).toBeGreaterThan(before);
  });

  it('setAttr and removeAttr bump the function version, so an analysis cache cannot outlive them', () => {
    const t = f32([4]);
    const func = buildFunction('attrs', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const add = func.findOp(o => o.opName === 'add');

    const beforeSet = func.version;
    add.setAttr('marker', 1);
    expect(func.version).toBeGreaterThan(beforeSet);

    const beforeRemove = func.version;
    expect(add.removeAttr('marker')).toBe(true);
    expect(func.version).toBeGreaterThan(beforeRemove);
  });

  it('removing an absent attribute is not a mutation and leaves the version alone', () => {
    const t = f32([4]);
    const func = buildFunction('attrs', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const add = func.findOp(o => o.opName === 'add');
    const before = func.version;

    expect(add.removeAttr('never_set')).toBe(false);
    expect(func.version).toBe(before);
  });
});

describe('structural identity is what CSE compares', () => {
  const t = f32([4]);

  function twoOps(opName) {
    let ops = null;
    buildFunction('pair', [t, t], [t], (b, args) => {
      const first = b._buildOp(opName, [args[0], args[1]], [t]);
      const second = b._buildOp(opName, [args[1], args[0]], [t]);
      ops = [first, second];
      b.returnOp([first.getResult(0)]);
    });
    return ops;
  }

  it('a commutative op is equal to itself with the operands swapped', () => {
    const [a, b] = twoOps('add');
    expect(a.structuralEquals(b)).toBe(true);
    expect(a.structuralHash()).toBe(b.structuralHash());
  });

  it('a non-commutative op is not', () => {
    const [a, b] = twoOps('sub');
    expect(a.structuralEquals(b)).toBe(false);
  });

  it('two operations carrying regions are never structurally equal', () => {
    const region = () => {
      const r = new Region();
      const block = new Block([t]);
      r.addBlock(block);
      block.pushOp(new Operation('neg', [block.arguments[0]], [t]));
      return r;
    };
    const a = new Operation('fusion', [], [t], null, [region()]);
    const b = new Operation('fusion', [], [t], null, [region()]);
    expect(a.structuralEquals(b)).toBe(false);
  });
});
