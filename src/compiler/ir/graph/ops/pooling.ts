import { OpDef, OpTrait } from '../op_registry.js';
import type { OpAttrRecord, OpRegistry } from '../op_registry.js';
import { TensorType } from '../types.js';

function computePoolOutputDim(inSize: number, kernelSize: number, stride: number, padLo: number, padHi: number, ceilMode: boolean): number {
  const padded = inSize + padLo + padHi - kernelSize;
  return (ceilMode ? Math.ceil(padded / stride) : Math.floor(padded / stride)) + 1;
}

export function register(registry: OpRegistry) {
  registry.register(new OpDef({
    name: 'pool2d',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'pool_type', type: 'string', required: true },
      { name: 'kernel_size', type: 'array', required: true },
      { name: 'strides', type: 'array', required: true },
      { name: 'padding', type: 'array', required: true },
      { name: 'ceil_mode', type: 'boolean', required: false },
      { name: 'count_include_pad', type: 'boolean', required: false },
      { name: 'layout', type: 'string', required: false }
    ],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length < 1) return null;
      const inp = operandTypes[0];
      if (!(inp instanceof TensorType) || inp.rank !== 4) return null;
      const ks = (attrs.get ? attrs.get('kernel_size') : (attrs as unknown as OpAttrRecord).kernel_size) as readonly number[];
      const st = (attrs.get ? attrs.get('strides') : (attrs as unknown as OpAttrRecord).strides) as readonly number[];
      const pad = (attrs.get ? attrs.get('padding') : (attrs as unknown as OpAttrRecord).padding) as readonly (readonly number[])[];
      const ceil = ((attrs.get ? attrs.get('ceil_mode') : (attrs as unknown as OpAttrRecord).ceil_mode) as boolean) || false;
      const outH = computePoolOutputDim(inp.shape[2] as number, ks[0], st[0], pad[0][0], pad[0][1], ceil);
      const outW = computePoolOutputDim(inp.shape[3] as number, ks[1], st[1], pad[1][0], pad[1][1], ceil);
      return [new TensorType([inp.shape[0], inp.shape[1], outH, outW], inp.dtype)];
    }
  }));
}
