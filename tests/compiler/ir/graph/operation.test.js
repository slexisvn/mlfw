import { describe, it, expect } from 'vitest';
import { Operation } from '../../../../src/compiler/ir/graph/operation.js';
import { Block, Region } from '../../../../src/compiler/ir/graph/block.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';

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
