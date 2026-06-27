import { describe, it, expect } from 'vitest';
import { Block, Region } from '../../../../src/compiler/ir/graph/block.js';
import { Operation } from '../../../../src/compiler/ir/graph/operation.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';

function t() { return new TensorType([4], ScalarType.F32); }
function mkOp(operands = []) { return new Operation('neg', operands, [t()]); }

describe('Block/Region mutation invariants', () => {
  it('pushOp rejects an operation already attached to a block', () => {
    const block = new Block([]);
    const op = mkOp();
    block.pushOp(op);
    expect(() => block.pushOp(op)).toThrow(/already attached/);

    const other = new Block([]);
    expect(() => other.pushOp(op)).toThrow(/already attached/);
  });

  it('insertBefore/insertAfter reject an already-attached operation', () => {
    const block = new Block([]);
    const a = mkOp();
    const b = mkOp();
    block.pushOp(a);
    block.pushOp(b);
    expect(() => block.insertBefore(b, a)).toThrow(/already attached/);
    expect(() => block.insertAfter(b, a)).toThrow(/already attached/);
  });

  it('insertBefore rejects a reference operation from another block', () => {
    const block = new Block([]);
    const block2 = new Block([]);
    const ref = mkOp();
    block2.pushOp(ref);
    expect(() => block.insertBefore(mkOp(), ref)).toThrow(/not in this block/);
  });

  it('an operation can be re-inserted after removeOp restores detached state', () => {
    const block = new Block([]);
    const op = mkOp();
    block.pushOp(op);
    block.removeOp(op);
    expect(() => block.pushOp(op)).not.toThrow();
    expect(block.size).toBe(1);
  });

  it('removeArguments rejects a still-used block argument', () => {
    const block = new Block([t()]);
    const arg = block.getArgument(0);
    block.pushOp(mkOp([arg]));
    expect(() => block.removeArguments([0])).toThrow(/still has uses/);
  });

  it('removeArguments removes an unused block argument', () => {
    const block = new Block([t()]);
    block.removeArguments([0]);
    expect(block.arguments.length).toBe(0);
  });

  it('addBlock/insertBlock reject a block already owned by a region', () => {
    const region = new Region();
    const block = new Block([]);
    region.addBlock(block);
    expect(() => new Region().addBlock(block)).toThrow(/already belongs/);
    expect(() => new Region().insertBlock(0, block)).toThrow(/already belongs/);
  });

  it('replaceOperand rejects an out-of-range index or a non-Value operand', () => {
    const block = new Block([t()]);
    const arg = block.getArgument(0);
    const op = mkOp([arg]);
    block.pushOp(op);
    expect(() => op.replaceOperand(5, arg)).toThrow(/out of range/);
    expect(() => op.replaceOperand(0, {})).toThrow(/not a Value/);
    expect(() => op.replaceOperand(0, arg)).not.toThrow();
  });

  it('replaceAllResultsWith rejects a mismatched result count', () => {
    const op = mkOp();
    expect(() => op.replaceAllResultsWith([])).toThrow(/results/);
  });
});
