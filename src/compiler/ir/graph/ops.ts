import { OpRegistry } from './op_registry.js';
import { register as registerArithmetic } from './ops/arithmetic.js';
import { register as registerUnary } from './ops/unary.js';
import { register as registerComparison } from './ops/comparison.js';
import { register as registerShape } from './ops/shape.js';
import { register as registerReduction } from './ops/reduction.js';
import { register as registerLinalg } from './ops/linalg.js';
import { register as registerData } from './ops/data.js';
import { register as registerControlFlow } from './ops/control_flow.js';
import { register as registerLayout } from './ops/layout.js';
import { register as registerQuantization } from './ops/quantization.js';
import { register as registerComposite } from './ops/composite.js';
import { register as registerTransfer } from './ops/transfer.js';
import { register as registerPooling } from './ops/pooling.js';
import { register as registerResize } from './ops/resize.js';

export type DialectRegistrar = (registry: OpRegistry) => void;

export const DIALECTS: readonly DialectRegistrar[] = [
  registerArithmetic, registerUnary, registerComparison, registerShape,
  registerReduction, registerLinalg, registerData, registerControlFlow,
  registerLayout, registerQuantization, registerComposite, registerTransfer,
  registerPooling, registerResize,
];

export function buildRegistry(dialects: readonly DialectRegistrar[] = DIALECTS): OpRegistry {
  const reg = new OpRegistry();
  for (const register of dialects) register(reg);
  return reg;
}

export const registry: OpRegistry = buildRegistry();
