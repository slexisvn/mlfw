import {
  MathOpNode, FloatImmNode, CompareNode, IfThenElseNode,
  CallExternNode, CastNode
} from '../../../ir/tensor/nodes.js';
import { registry } from '../../../ir/graph/ops.js';
import { registerLoweringRule, lowerPointwise } from '../lowering_registry.js';
import type { TirNode } from '../../../ir/tensor/nodes.js';

export type SyntheticUnaryBuilder = (args: readonly TirNode[]) => TirNode;

const BINARY_ARITH = new Set(['+', '-', '*', '/']);
const BINARY_LOGICAL = new Set(['&&', '||']);
const UNARY_PREFIX = new Set(['!']);

const SYNTHETIC_UNARY: Record<string, SyntheticUnaryBuilder> = {
  'square': (args) => new MathOpNode('*', args[0], args[0]),
  'reciprocal': (args) => new MathOpNode('/', new FloatImmNode(1), args[0]),
};

const ELEMENTWISE_SCALAR_OP_ATTR = 'elementwiseScalarOp';

const ELEMENTWISE_SCALAR_OPS: Record<string, string> = {
  'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
  'max': 'max', 'min': 'min', 'exp': 'exp', 'log': 'log',
  'sqrt': 'sqrt', 'rsqrt': 'rsqrt', 'tanh': 'tanh', 'abs': 'abs',
  'ceil': 'ceil', 'floor': 'floor', 'neg': '-',
  'maximum': 'max', 'minimum': 'min',
  'sin': 'sin', 'cos': 'cos', 'round': 'round', 'sign': 'sign',
  'pow': 'pow', 'rem': 'fmod',
  'erf': 'erf', 'erfc': 'erfc', 'lgamma': 'lgamma', 'gamma': 'gamma', 'log2': 'log2', 'log10': 'log10', 'exp2': 'exp2',
  'square': 'square', 'reciprocal': 'reciprocal',
  'logical_not': '!', 'logical_and': '&&', 'logical_or': '||'
};

for (const [opName, jsOp] of Object.entries(ELEMENTWISE_SCALAR_OPS)) {
  if (registry.has(opName)) registry.registerOpAttr(opName, ELEMENTWISE_SCALAR_OP_ATTR, jsOp);
}

export function elementwiseScalarOp(opName: string): string | null {
  const def = registry.get(opName);
  return def ? def.getAttr<string>(ELEMENTWISE_SCALAR_OP_ATTR) as string : null;
}

export function elementwiseOpNames(): string[] {
  const names: string[] = [];
  for (const def of registry.allOps()) {
    if (def.hasAttr(ELEMENTWISE_SCALAR_OP_ATTR)) names.push(def.name);
  }
  return names;
}

export function buildElementwiseExpr(opName: string, loadArgs: readonly TirNode[], dtype: string): TirNode | null {
  const jsOp = elementwiseScalarOp(opName);
  if (!jsOp) return null;
  if (SYNTHETIC_UNARY[opName]) return SYNTHETIC_UNARY[opName](loadArgs);
  if (loadArgs.length === 2 && BINARY_ARITH.has(jsOp)) {
    return new MathOpNode(jsOp, loadArgs[0], loadArgs[1]);
  }
  if (loadArgs.length === 2 && BINARY_LOGICAL.has(jsOp)) {
    return new MathOpNode(jsOp, loadArgs[0], loadArgs[1]);
  }
  if (loadArgs.length === 1 && UNARY_PREFIX.has(jsOp)) {
    return new MathOpNode(jsOp, loadArgs[0]);
  }
  if (loadArgs.length === 1 && jsOp === '-') {
    return new MathOpNode('-', loadArgs[0]);
  }
  return new CallExternNode(jsOp, loadArgs as TirNode[], dtype);
}

export function register(): void {
  for (const opName of elementwiseOpNames()) {
    registerLoweringRule(opName, (ctx, op, inputs, outputs) =>
      lowerPointwise(ctx, op, inputs, outputs, (o, loads, dtype) => buildElementwiseExpr(o.opName, loads, dtype) as TirNode)
    );
  }

  registerLoweringRule('compare', (ctx, op, inputs, outputs) =>
    lowerPointwise(ctx, op, inputs, outputs, (o, loads) =>
      new CompareNode(o.getAttr<string>('direction') || 'eq', loads[0], loads[1]))
  );

  registerLoweringRule('select', (ctx, op, inputs, outputs) =>
    lowerPointwise(ctx, op, inputs, outputs, (o, loads) =>
      new IfThenElseNode(loads[0], loads[1], loads[2]))
  );

  registerLoweringRule('clamp', (ctx, op, inputs, outputs) =>
    lowerPointwise(ctx, op, inputs, outputs, (o, loads, dtype) =>
      new CallExternNode('min', [new CallExternNode('max', [loads[1], loads[0]], dtype), loads[2]], dtype))
  );

  registerLoweringRule('convert', (ctx, op, inputs, outputs) =>
    lowerPointwise(ctx, op, inputs, outputs, (o, loads) =>
      new CastNode(loads[0], inputs[0].dtype, outputs[0].dtype))
  );

  registerLoweringRule('copy_to_device', (ctx, op, inputs, outputs) =>
    lowerPointwise(ctx, op, inputs, outputs, (o, loads) => loads[0])
  );
}
