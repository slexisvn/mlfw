import { GraphFunction } from '../ir/graph/function.js';
import { IRBuilder } from '../ir/graph/builder.js';
import { UseDefAnalysis } from '../analysis/use_def.js';

const _jvpRules = new Map();

export function registerJVPRule(opName, fn) {
  _jvpRules.set(opName, fn);
}

export function getJVPRule(opName) {
  return _jvpRules.get(opName) || null;
}

export function buildForwardDiff(forwardFunc) {
  const topo = UseDefAnalysis.compute(forwardFunc).topologicalOrder;
  const returnOp = forwardFunc.getReturnOp();
  if (!returnOp) throw new Error('buildForwardDiff: forward function has no return op');
  const outputs = returnOp.operands;
  const inputs = forwardFunc.args;
  const argTypes = inputs.map((v) => v.type);

  const jvpFunc = new GraphFunction(`jvp_${forwardFunc.name}`, [...argTypes, ...argTypes], outputs.map((v) => v.type));
  const builder = new IRBuilder(jvpFunc);
  const jvpArgs = jvpFunc.args;

  const fwdMap = new Map();
  const tan = new Map();
  for (let i = 0; i < inputs.length; i++) {
    fwdMap.set(inputs[i], jvpArgs[i]);
    tan.set(inputs[i], jvpArgs[inputs.length + i]);
  }

  const zeroLike = (type) => builder.broadcast(builder.scalarConstant(0, type.dtype).getResult(0), type.shape, []).getResult(0);

  for (const op of topo) {
    if (op.opName === 'return') continue;

    const cloned = op.clone(fwdMap);
    builder.block.pushOp(cloned);

    if (op.opName === 'constant' || op.opName === 'scalar_constant') {
      for (let r = 0; r < op.numResults; r++) tan.set(op.getResult(r), zeroLike(op.getResult(r).type));
      continue;
    }

    const rule = getJVPRule(op.opName);
    if (!rule) {
      throw new Error(`buildForwardDiff: no JVP rule for op '${op.opName}' (forward-mode AD would otherwise emit a silently-wrong zero tangent)`);
    }

    const operands = op.operands.map((o) => fwdMap.get(o) || o);
    const operandTangents = op.operands.map((o) => tan.get(o) || zeroLike(o.type));
    const dy = rule({ builder, op, operands, operandTangents, results: cloned.results, attrs: op.attributes });
    const dys = Array.isArray(dy) ? dy : [dy];
    for (let r = 0; r < op.numResults; r++) {
      tan.set(op.getResult(r), dys[r] != null ? dys[r] : zeroLike(op.getResult(r).type));
    }
  }

  builder.returnOp(outputs.map((o) => tan.get(o) || zeroLike(o.type)));
  return jvpFunc;
}

const g = (x) => x.getResult(0);

registerJVPRule('add', (ctx) => g(ctx.builder.add(ctx.operandTangents[0], ctx.operandTangents[1])));
registerJVPRule('sub', (ctx) => g(ctx.builder.sub(ctx.operandTangents[0], ctx.operandTangents[1])));
registerJVPRule('neg', (ctx) => g(ctx.builder.neg(ctx.operandTangents[0])));
registerJVPRule('mul', (ctx) => {
  const [a, b] = ctx.operands;
  const [da, db] = ctx.operandTangents;
  return g(ctx.builder.add(g(ctx.builder.mul(da, b)), g(ctx.builder.mul(a, db))));
});
registerJVPRule('div', (ctx) => {
  const [, b] = ctx.operands;
  const [da, db] = ctx.operandTangents;
  const y = ctx.results[0];
  return g(ctx.builder.div(g(ctx.builder.sub(da, g(ctx.builder.mul(y, db)))), b));
});
registerJVPRule('exp', (ctx) => g(ctx.builder.mul(ctx.results[0], ctx.operandTangents[0])));
registerJVPRule('log', (ctx) => g(ctx.builder.div(ctx.operandTangents[0], ctx.operands[0])));
registerJVPRule('sqrt', (ctx) => {
  const da = ctx.operandTangents[0];
  const y = ctx.results[0];
  const two = g(ctx.builder.broadcast(g(ctx.builder.scalarConstant(2, y.type.dtype)), y.type.shape, []));
  return g(ctx.builder.div(da, g(ctx.builder.mul(two, y))));
});
registerJVPRule('tanh', (ctx) => {
  const da = ctx.operandTangents[0];
  const y = ctx.results[0];
  const one = g(ctx.builder.broadcast(g(ctx.builder.scalarConstant(1, y.type.dtype)), y.type.shape, []));
  const oneMinusSq = g(ctx.builder.sub(one, g(ctx.builder.mul(y, y))));
  return g(ctx.builder.mul(oneMinusSq, da));
});
