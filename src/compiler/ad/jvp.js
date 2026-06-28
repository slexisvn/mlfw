import { GraphFunction } from '../ir/graph/function.js';
import { IRBuilder } from '../ir/graph/builder.js';
import { UseDefAnalysis } from '../analysis/use_def.js';
import { isConstantOp } from '../ir/graph/op_traits.js';

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

    if (isConstantOp(op.opName)) {
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

function fillLike(ctx, type, value) {
  return g(ctx.builder.broadcast(g(ctx.builder.scalarConstant(value, type.dtype)), type.shape, []));
}

function cloneToTangent(ctx, valueForOperand) {
  const map = new Map();
  for (let i = 0; i < ctx.op.numOperands; i++) map.set(ctx.op.getOperand(i), valueForOperand(i));
  const cloned = ctx.op.clone(map);
  ctx.builder.block.pushOp(cloned);
  return cloned;
}

function linearJVP(ctx) {
  const cloned = cloneToTangent(ctx, (i) => ctx.operandTangents[i]);
  if (cloned.numResults === 1) return cloned.getResult(0);
  return cloned.results.map((_, i) => cloned.getResult(i));
}

function linearJVPKeep(ctx, keep) {
  return cloneToTangent(ctx, (i) => keep.has(i) ? ctx.operands[i] : ctx.operandTangents[i]).getResult(0);
}

function bilinearJVP(ctx) {
  const o0 = ctx.op.getOperand(0), o1 = ctx.op.getOperand(1);
  const c1 = ctx.op.clone(new Map([[o0, ctx.operandTangents[0]], [o1, ctx.operands[1]]]));
  ctx.builder.block.pushOp(c1);
  const c2 = ctx.op.clone(new Map([[o0, ctx.operands[0]], [o1, ctx.operandTangents[1]]]));
  ctx.builder.block.pushOp(c2);
  return g(ctx.builder.add(c1.getResult(0), c2.getResult(0)));
}

const LINEAR_OPS = ['reshape', 'transpose', 'broadcast_in_dim', 'broadcast', 'slice', 'concat', 'reverse', 'copy_to_device'];
for (const name of LINEAR_OPS) registerJVPRule(name, linearJVP);

const ZERO_OPS = ['compare', 'logical_not', 'logical_and', 'logical_or', 'argmax', 'argmin', 'iota', 'sign', 'floor', 'ceil'];
for (const name of ZERO_OPS) registerJVPRule(name, (ctx) => ctx.results.map((r) => fillLike(ctx, r.type, 0)));

registerJVPRule('pad', linearJVP);
registerJVPRule('dot', bilinearJVP);
registerJVPRule('conv', bilinearJVP);
registerJVPRule('select', (ctx) => linearJVPKeep(ctx, new Set([0])));
registerJVPRule('where', (ctx) => linearJVPKeep(ctx, new Set([0])));

registerJVPRule('convert', (ctx) => {
  const target = ctx.op.getAttr('target_dtype') || ctx.results[0].type.dtype;
  if (target === 'i8' || target === 'i16' || target === 'i32' || target === 'i64' || target === 'ui8' || target === 'bool') {
    return fillLike(ctx, ctx.results[0].type, 0);
  }
  return linearJVP(ctx);
});

registerJVPRule('reduce', (ctx) => {
  const rt = ctx.op.getAttr('reduce_type');
  if (rt === 'sum' || rt === 'mean') return linearJVP(ctx);
  throw new Error(`buildForwardDiff: reduce_type '${rt}' is not supported in forward mode (only sum/mean are linear)`);
});

registerJVPRule('maximum', (ctx) => {
  const [a, b] = ctx.operands;
  const [da, db] = ctx.operandTangents;
  const mask = g(ctx.builder.compare(a, b, 'ge'));
  return g(ctx.builder.select(mask, da, db));
});
registerJVPRule('minimum', (ctx) => {
  const [a, b] = ctx.operands;
  const [da, db] = ctx.operandTangents;
  const mask = g(ctx.builder.compare(a, b, 'le'));
  return g(ctx.builder.select(mask, da, db));
});

registerJVPRule('square', (ctx) => {
  const two = fillLike(ctx, ctx.operands[0].type, 2);
  return g(ctx.builder.mul(g(ctx.builder.mul(two, ctx.operands[0])), ctx.operandTangents[0]));
});
registerJVPRule('abs', (ctx) => g(ctx.builder.mul(g(ctx.builder.sign(ctx.operands[0])), ctx.operandTangents[0])));
registerJVPRule('reciprocal', (ctx) => {
  const y = ctx.results[0];
  return g(ctx.builder.neg(g(ctx.builder.mul(g(ctx.builder.mul(y, y)), ctx.operandTangents[0]))));
});
registerJVPRule('rsqrt', (ctx) => {
  const y = ctx.results[0];
  const half = fillLike(ctx, y.type, -0.5);
  const y3 = g(ctx.builder.mul(g(ctx.builder.mul(y, y)), y));
  return g(ctx.builder.mul(g(ctx.builder.mul(half, y3)), ctx.operandTangents[0]));
});
registerJVPRule('sigmoid', (ctx) => {
  const y = ctx.results[0];
  const one = fillLike(ctx, y.type, 1);
  const grad = g(ctx.builder.mul(y, g(ctx.builder.sub(one, y))));
  return g(ctx.builder.mul(grad, ctx.operandTangents[0]));
});
registerJVPRule('sin', (ctx) => g(ctx.builder.mul(g(ctx.builder.cos(ctx.operands[0])), ctx.operandTangents[0])));
registerJVPRule('cos', (ctx) => g(ctx.builder.neg(g(ctx.builder.mul(g(ctx.builder.sin(ctx.operands[0])), ctx.operandTangents[0])))));
registerJVPRule('pow', (ctx) => {
  const [a, b] = ctx.operands;
  const [da, db] = ctx.operandTangents;
  const y = ctx.results[0];
  const one = fillLike(ctx, a.type, 1);
  const term1 = g(ctx.builder.mul(g(ctx.builder.mul(b, g(ctx.builder.pow(a, g(ctx.builder.sub(b, one)))))), da));
  const term2 = g(ctx.builder.mul(g(ctx.builder.mul(y, g(ctx.builder.log(a)))), db));
  return g(ctx.builder.add(term1, term2));
});
