import { MathOpNode, FloatImmNode, CallExternNode, CastNode } from '../../ir/tensor/nodes.js';
import type { TirNode } from '../../ir/tensor/nodes.js';

export type QuantizeSpec = Readonly<{ scale: number; zeroPoint: number; targetDtype?: string }>;
export type DequantizeSpec = Readonly<{ scale: number; zeroPoint: number; srcDtype?: string; targetDtype?: string }>;

function clampRange(targetDtype: string): { cMin: number; cMax: number } {
  const isUnsigned = targetDtype === 'ui8';
  const bits = 8;
  return {
    cMin: isUnsigned ? 0 : -(1 << (bits - 1)),
    cMax: isUnsigned ? (1 << bits) - 1 : (1 << (bits - 1)) - 1,
  };
}

export function buildQuantizeExpr(input: TirNode, { scale, zeroPoint, targetDtype }: QuantizeSpec): TirNode {
  const tgt = targetDtype || 'i8';
  const { cMin, cMax } = clampRange(tgt);
  const scaled = new MathOpNode('/', input, new FloatImmNode(scale));
  const shifted = zeroPoint !== 0 ? new MathOpNode('+', scaled, new FloatImmNode(zeroPoint)) : scaled;
  const rounded = new CallExternNode('round', [shifted], 'f32');
  const clamped = new CallExternNode('min', [
    new CallExternNode('max', [rounded, new FloatImmNode(cMin)], 'f32'),
    new FloatImmNode(cMax),
  ], 'f32');
  return new CastNode(clamped, 'f32', tgt);
}

export function buildDequantizeExpr(input: TirNode, { scale, zeroPoint, srcDtype, targetDtype }: DequantizeSpec): TirNode {
  const tgt = targetDtype || 'f32';
  const asFloat = new CastNode(input, srcDtype || 'i8', tgt);
  const shifted = zeroPoint !== 0 ? new MathOpNode('-', asFloat, new FloatImmNode(zeroPoint)) : asFloat;
  return new MathOpNode('*', shifted, new FloatImmNode(scale));
}
