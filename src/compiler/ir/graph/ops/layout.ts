import { OpDef, OpTrait } from '../op_registry.js';
import type { OpAttrRecord, OpRegistry } from '../op_registry.js';
import { TensorType, Layout } from '../types.js';
import * as pat from '../patterns.js';

export function register(registry: OpRegistry) {
  registry.register(new OpDef({
    name: 'layout_transform',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.VIEW],
    attrs: [
      { name: 'src_layout', type: 'array', required: true },
      { name: 'dst_layout', type: 'array', required: true },
      { name: 'src_block', type: 'array', required: false },
      { name: 'dst_block', type: 'array', required: false }
    ],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length !== 1) return null;
      const inp = operandTypes[0];
      if (!(inp instanceof TensorType)) return null;
      const read = (name: string) => (attrs.get ? attrs.get(name) : (attrs as unknown as OpAttrRecord)[name]) as readonly number[] | undefined;
      const dstOrder = read('dst_layout');
      if (!dstOrder) return null;
      const dstBlock = read('dst_block');
      const layout = dstBlock ? Layout.blocked(dstOrder, dstBlock[0], dstBlock[1]) : new Layout(dstOrder);
      return [new TensorType(inp.shape, inp.dtype, layout)];
    },
    verify(op) {
      const errs = [];
      if (op.numOperands !== 1) { errs.push('layout_transform expects 1 operand'); return errs; }
      if (!op.hasAttr('src_layout')) errs.push('layout_transform missing src_layout');
      if (!op.hasAttr('dst_layout')) errs.push('layout_transform missing dst_layout');
      if (errs.length === 0) {
        const inp = op.getOperand(0).type;
        const src = op.getAttr<readonly number[]>('src_layout')!;
        const dst = op.getAttr<readonly number[]>('dst_layout')!;
        if (inp instanceof TensorType) {
          if (src.length !== inp.rank) errs.push(`src_layout length ${src.length} != input rank ${inp.rank}`);
          if (dst.length !== inp.rank) errs.push(`dst_layout length ${dst.length} != input rank ${inp.rank}`);
        }
      }
      return errs;
    },
    getCanonicalizationPatterns() {
      return [new pat.LayoutTransformIdentity(), new pat.LayoutTransformCompose()];
    }
  }));
}
