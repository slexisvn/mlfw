import { registerVJPRule, registerGradientBarrier } from '../vjp_registry.js';

registerGradientBarrier('stop_gradient');
registerVJPRule('stop_gradient', () => [null]);
