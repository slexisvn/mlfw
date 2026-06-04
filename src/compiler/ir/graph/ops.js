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

export const registry = new OpRegistry();

registerArithmetic(registry);
registerUnary(registry);
registerComparison(registry);
registerShape(registry);
registerReduction(registry);
registerLinalg(registry);
registerData(registry);
registerControlFlow(registry);
registerLayout(registry);
registerQuantization(registry);
registerComposite(registry);
