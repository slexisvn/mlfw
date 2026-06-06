import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { DecompositionPass } from '../../../src/compiler/passes/decompose/decomposition_pass.js';

function run(func) {
  return new DecompositionPass().run(func);
}

function retVal(func) {
  return func.getReturnOp().getOperand(0);
}

describe('softmax on 1D input', () => {
  it('reduce on axis=0 produces scalar, broadcast restores 1D shape', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 0).getResult(0)]);
    });

    run(func);

    const reduces = func.findOps(op => op.opName === 'reduce');
    for (const r of reduces) {
      expect(r.getResult(0).type.shape).toEqual([]);
    }

    const broadcasts = func.findOps(op => op.opName === 'broadcast_in_dim')
      .filter(bc => bc.getResult(0).type.shape.length === 1);
    for (const bc of broadcasts) {
      expect(bc.getResult(0).type.shape).toEqual([8]);
    }

    expect(retVal(func).type.shape).toEqual([8]);
  });
});

describe('softmax operand chain integrity', () => {
  it('exp receives shifted (x - max) input, NOT raw x', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 1).getResult(0)]);
    });

    run(func);

    const expOp = func.findOp(op => op.opName === 'exp');
    const expInput = expOp.getOperand(0).definingOp;
    expect(expInput.opName).toBe('sub');
    expect(expInput.getOperand(0)).toBe(func.args[0]);
  });

  it('div numerator is exp output, denominator is broadcast of sum', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 1).getResult(0)]);
    });

    run(func);

    const divOp = retVal(func).definingOp;
    expect(divOp.opName).toBe('div');
    expect(divOp.getOperand(0).definingOp.opName).toBe('exp');

    const denomBcast = divOp.getOperand(1).definingOp;
    expect(denomBcast.opName).toBe('broadcast_in_dim');
    expect(denomBcast.getOperand(0).definingOp.opName).toBe('reduce');
  });
});

describe('sigmoid with scalar input', () => {
  it('all intermediate values are scalar-shaped', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.sigmoid(args[0]).getResult(0)]);
    });

    run(func);

    for (const op of func.ops()) {
      for (let i = 0; i < op.numResults; i++) {
        const v = op.getResult(i);
        if (v.type instanceof TensorType) {
          expect(v.type.shape).toEqual([]);
        }
      }
    }
  });

  it('div(1, 1+exp(-x)) structure is intact on scalar', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.sigmoid(args[0]).getResult(0)]);
    });

    run(func);

    const divOp = retVal(func).definingOp;
    expect(divOp.opName).toBe('div');

    const denom = divOp.getOperand(1).definingOp;
    expect(denom.opName).toBe('add');

    const expNeg = denom.getOperand(1).definingOp;
    expect(expNeg.opName).toBe('exp');
    expect(expNeg.getOperand(0).definingOp.opName).toBe('neg');
    expect(expNeg.getOperand(0).definingOp.getOperand(0)).toBe(func.args[0]);
  });
});

describe('log_softmax operand chain', () => {
  it('final result is sub(shifted, broadcast(log(sum(exp(shifted)))))', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.logSoftmax(args[0], 1).getResult(0)]);
    });

    run(func);

    const subOp = retVal(func).definingOp;
    expect(subOp.opName).toBe('sub');

    const logBcast = subOp.getOperand(1).definingOp;
    expect(logBcast.opName).toBe('broadcast_in_dim');
    expect(logBcast.getOperand(0).definingOp.opName).toBe('log');

    const logInput = logBcast.getOperand(0).definingOp.getOperand(0).definingOp;
    expect(logInput.opName).toBe('reduce');
  });
});

describe('gelu operand chain', () => {
  it('scaled value feeds into neg, which feeds into exp, tracing back to input', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.gelu(args[0]).getResult(0)]);
    });

    run(func);

    const negOps = func.findOps(op => op.opName === 'neg');
    const neg = negOps[0];
    const scaled = neg.getOperand(0).definingOp;
    expect(scaled.opName).toBe('mul');

    const coeffBcast = scaled.getOperand(0).definingOp;
    expect(coeffBcast.opName).toBe('broadcast_in_dim');
    expect(coeffBcast.getOperand(0).definingOp.getAttr('value')).toBe(1.702);

    expect(scaled.getOperand(1)).toBe(func.args[0]);
  });
});

describe('layer_norm variance computation', () => {
  it('variance uses squared centered values — mul(centered, centered)', () => {
    const inputT = new TensorType([4, 8], ScalarType.F32);
    const paramT = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [inputT, paramT, paramT], [inputT], (b, args) => {
      b.returnOp([b.layernorm(args[0], args[1], args[2], 1, 1e-5).getResult(0)]);
    });

    run(func);

    const muls = func.findOps(op => op.opName === 'mul');
    const selfMul = muls.find(m => m.getOperand(0) === m.getOperand(1));
    expect(selfMul).toBeDefined();
    expect(selfMul.getOperand(0).definingOp.opName).toBe('sub');
  });
});

describe('worklist skips erased ops', () => {
  it('two composites: second still decomposes after first is erased', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const sig = b.sigmoid(args[0]);
      b.returnOp([b.silu(sig.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'sigmoid')).toBeNull();
    expect(func.findOp(op => op.opName === 'silu')).toBeNull();

    const negOps = func.findOps(op => op.opName === 'neg');
    expect(negOps.length).toBeGreaterThanOrEqual(2);
  });
});
