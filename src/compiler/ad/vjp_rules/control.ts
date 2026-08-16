import { registerVJPRule, registerGradientBarrier } from '../vjp_registry.js';

registerGradientBarrier('stop_gradient');
registerVJPRule('stop_gradient', () => [null]);

for (const op of ['compare', 'logical_not', 'logical_and', 'logical_or', 'argmax', 'argmin', 'iota', 'one_hot']) {
  registerGradientBarrier(op);
}
