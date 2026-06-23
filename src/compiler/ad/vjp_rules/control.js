import { registerVJPRule, registerGradientBarrier } from '../vjp_registry.js';

registerGradientBarrier('stop_gradient');
registerVJPRule('stop_gradient', () => [null]);

for (const op of ['compare', 'logical_not', 'argmax', 'argmin', 'iota']) {
  registerGradientBarrier(op);
}
