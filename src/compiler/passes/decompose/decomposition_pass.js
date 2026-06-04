import { FunctionPass, PassResult } from '../pass.js';
import { IRBuilder, broadcastDimsExcluding } from '../../ir/graph/builder.js';
import { ScalarType } from '../../ir/graph/types.js';

const decompositionRules = new Map();

export function registerDecomposition(opName, ruleFn) {
  decompositionRules.set(opName, ruleFn);
}

export function hasDecomposition(opName) {
  return decompositionRules.has(opName);
}

export class DecompositionPass extends FunctionPass {
  constructor() {
    super('DecompositionPass');
  }

  run(func) {
    const worklist = [];
    for (const op of func.ops()) {
      if (decompositionRules.has(op.opName)) worklist.push(op);
    }
    if (worklist.length === 0) return PassResult.UNCHANGED;

    const builder = new IRBuilder(func);

    for (const op of worklist) {
      if (!op.parentBlock) continue;
      const rule = decompositionRules.get(op.opName);
      builder.block = op.parentBlock;
      builder.setInsertionPoint(op);
      rule(op, builder);
    }

    return PassResult.CHANGED;
  }
}

registerDecomposition('softmax', (op, b) => {
  const input = op.getOperand(0);
  const axis = op.getAttr('axis');
  const rank = input.type.rank;
  const dtype = input.type.dtype;
  const shape = input.type.shape;
  const bcastDims = broadcastDimsExcluding(rank, axis);

  const maxVal = b.reduce(input, b.scalarConstant(-Infinity, dtype).getResult(0), [axis], 'max');
  const bcastMax = b.broadcast(maxVal.getResult(0), shape, bcastDims);
  const shifted = b.sub(input, bcastMax.getResult(0));
  const exps = b.exp(shifted.getResult(0));
  const sumVal = b.reduce(exps.getResult(0), b.scalarConstant(0, dtype).getResult(0), [axis], 'sum');
  const bcastSum = b.broadcast(sumVal.getResult(0), shape, bcastDims);
  const result = b.div(exps.getResult(0), bcastSum.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('log_softmax', (op, b) => {
  const input = op.getOperand(0);
  const axis = op.getAttr('axis');
  const rank = input.type.rank;
  const dtype = input.type.dtype;
  const shape = input.type.shape;
  const bcastDims = broadcastDimsExcluding(rank, axis);

  const maxVal = b.reduce(input, b.scalarConstant(-Infinity, dtype).getResult(0), [axis], 'max');
  const bcastMax = b.broadcast(maxVal.getResult(0), shape, bcastDims);
  const shifted = b.sub(input, bcastMax.getResult(0));
  const exps = b.exp(shifted.getResult(0));
  const sumVal = b.reduce(exps.getResult(0), b.scalarConstant(0, dtype).getResult(0), [axis], 'sum');
  const logSum = b.log(sumVal.getResult(0));
  const bcastLogSum = b.broadcast(logSum.getResult(0), shape, bcastDims);
  const result = b.sub(shifted.getResult(0), bcastLogSum.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('sigmoid', (op, b) => {
  const x = op.getOperand(0);
  const dtype = x.type.dtype;
  const shape = x.type.shape;

  const negX = b.neg(x);
  const expNeg = b.exp(negX.getResult(0));
  const one = b.broadcast(b.scalarConstant(1, dtype).getResult(0), shape, []);
  const denom = b.add(one.getResult(0), expNeg.getResult(0));
  const result = b.div(one.getResult(0), denom.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('gelu', (op, b) => {
  const x = op.getOperand(0);
  const dtype = x.type.dtype;
  const shape = x.type.shape;

  const coeff = b.broadcast(b.scalarConstant(1.702, dtype).getResult(0), shape, []);
  const scaled = b.mul(coeff.getResult(0), x);
  const negScaled = b.neg(scaled.getResult(0));
  const expNeg = b.exp(negScaled.getResult(0));
  const one = b.broadcast(b.scalarConstant(1, dtype).getResult(0), shape, []);
  const denom = b.add(one.getResult(0), expNeg.getResult(0));
  const sig = b.div(one.getResult(0), denom.getResult(0));
  const result = b.mul(x, sig.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('silu', (op, b) => {
  const x = op.getOperand(0);
  const dtype = x.type.dtype;
  const shape = x.type.shape;

  const negX = b.neg(x);
  const expNeg = b.exp(negX.getResult(0));
  const one = b.broadcast(b.scalarConstant(1, dtype).getResult(0), shape, []);
  const denom = b.add(one.getResult(0), expNeg.getResult(0));
  const sig = b.div(one.getResult(0), denom.getResult(0));
  const result = b.mul(x, sig.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('layer_norm', (op, b) => {
  const input = op.getOperand(0);
  const gamma = op.getOperand(1);
  const beta = op.getOperand(2);
  const axis = op.getAttr('axis');
  const eps = op.getAttr('epsilon');
  const rank = input.type.rank;
  const dtype = input.type.dtype;
  const shape = input.type.shape;
  const bcastDims = broadcastDimsExcluding(rank, axis);

  const meanVal = b.reduce(input, b.scalarConstant(0, dtype).getResult(0), [axis], 'mean');
  const bcastMean = b.broadcast(meanVal.getResult(0), shape, bcastDims);
  const centered = b.sub(input, bcastMean.getResult(0));
  const sq = b.mul(centered.getResult(0), centered.getResult(0));
  const variance = b.reduce(sq.getResult(0), b.scalarConstant(0, dtype).getResult(0), [axis], 'mean');
  const epsConst = b.broadcast(b.scalarConstant(eps, dtype).getResult(0), variance.getResult(0).type.shape, []);
  const varPlusEps = b.add(variance.getResult(0), epsConst.getResult(0));
  const rstd = b.rsqrt(varPlusEps.getResult(0));
  const bcastRstd = b.broadcast(rstd.getResult(0), shape, bcastDims);
  const normalized = b.mul(centered.getResult(0), bcastRstd.getResult(0));
  const bcastGamma = b.broadcast(gamma, shape, [axis]);
  const scaled = b.mul(normalized.getResult(0), bcastGamma.getResult(0));
  const bcastBeta = b.broadcast(beta, shape, [axis]);
  const result = b.add(scaled.getResult(0), bcastBeta.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('batch_norm', (op, b) => {
  const input = op.getOperand(0);
  const gamma = op.getOperand(1);
  const beta = op.getOperand(2);
  const mean = op.getOperand(3);
  const variance = op.getOperand(4);
  const axis = op.getAttr('axis');
  const eps = op.getAttr('epsilon');
  const rank = input.type.rank;
  const dtype = input.type.dtype;
  const shape = input.type.shape;

  const epsConst = b.broadcast(b.scalarConstant(eps, dtype).getResult(0), variance.type.shape, []);
  const varPlusEps = b.add(variance, epsConst.getResult(0));
  const rstd = b.rsqrt(varPlusEps.getResult(0));
  const bcastMean = b.broadcast(mean, shape, [axis]);
  const centered = b.sub(input, bcastMean.getResult(0));
  const bcastRstd = b.broadcast(rstd.getResult(0), shape, [axis]);
  const normalized = b.mul(centered.getResult(0), bcastRstd.getResult(0));
  const bcastGamma = b.broadcast(gamma, shape, [axis]);
  const scaled = b.mul(normalized.getResult(0), bcastGamma.getResult(0));
  const bcastBeta = b.broadcast(beta, shape, [axis]);
  const result = b.add(scaled.getResult(0), bcastBeta.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});
