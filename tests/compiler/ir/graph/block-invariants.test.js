import { describe, it, expect } from 'vitest';
import { Block, Region } from '../../../../src/compiler/ir/graph/block.js';
import { Operation } from '../../../../src/compiler/ir/graph/operation.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { AnalysisManager } from '../../../../src/compiler/analysis/analysis_manager.js';
import { UseDefAnalysis } from '../../../../src/compiler/analysis/use_def.js';

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

describe('IR version tracking from mutation primitives', () => {
  function addFunc() {
    const ty = new TensorType([4], ScalarType.F32);
    return buildFunction('f', [ty, ty], [ty], (b, [x, y]) => {
      const a = b.add(x, y);
      b.returnOp([b.mul(a.getResult(0), x).getResult(0)]);
    });
  }

  it('a structural mutation bumps the function version directly', () => {
    const func = addFunc();
    const v0 = func.version;
    func.getReturnOp().replaceOperand(0, func.args[0]);
    expect(func.version).toBeGreaterThan(v0);
  });

  it('a mutation invalidates the analysis cache without any pass reporting CHANGED', () => {
    const func = addFunc();
    const am = new AnalysisManager();
    const a1 = am.getAnalysis(UseDefAnalysis, func);
    expect(am.getAnalysis(UseDefAnalysis, func)).toBe(a1);

    func.getReturnOp().replaceOperand(0, func.args[0]);

    const a2 = am.getAnalysis(UseDefAnalysis, func);
    expect(a2).not.toBe(a1);
  });
});
