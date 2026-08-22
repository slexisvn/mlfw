import { IntImmNode, MathOpNode, BufferLoadNode, SeqNode, CastNode } from '../../../ir/tensor/nodes.js';
import { registerLoweringRule, lowerPointwise, buildConvNest, emitMatmulInitAcc } from '../lowering_registry.js';
import { buildQuantizeExpr, buildDequantizeExpr } from '../quant_math.js';

export function register(): void {
  registerLoweringRule('quantize', (ctx, op, inputs, outputs) => {
    const scale = op.getAttr<number>('scale') as number;
    const zeroPoint = op.getAttr<number>('zero_point') as number;
    const targetDtype = op.getAttr<string>('target_dtype') || 'i8';
    return lowerPointwise(ctx, op, inputs, outputs, (_op, loads) =>
      buildQuantizeExpr(loads[0], { scale, zeroPoint, targetDtype }));
  });

  registerLoweringRule('dequantize', (ctx, op, inputs, outputs) => {
    const scale = op.getAttr<number>('scale') as number;
    const zeroPoint = op.getAttr<number>('zero_point') as number;
    const srcDtype = inputs[0].dtype || 'i8';
    const targetDtype = op.getAttr<string>('target_dtype') || 'f32';
    return lowerPointwise(ctx, op, inputs, outputs, (_op, loads) =>
      buildDequantizeExpr(loads[0], { scale, zeroPoint, srcDtype, targetDtype }));
  });

  registerLoweringRule('quantized_dot', (ctx, op, inputs, outputs) => {
    const lhs = inputs[0];
    const rhs = inputs[1];
    const lhsZP = op.getAttr<number>('lhs_zero_point') || 0;
    const rhsZP = op.getAttr<number>('rhs_zero_point') || 0;
    const { initBody, accBody } = emitMatmulInitAcc(ctx, op, lhs, rhs, outputs[0], {
      prefix: 'qdi',
      initBlockName: 'qmatmul_init',
      accBlockName: 'qmatmul_acc',
      initVal: () => new IntImmNode(0),
      accLeaf: (lhsLoad, rhsLoad) => {
        const cl = new CastNode(lhsLoad, lhs.dtype, 'i32');
        const cr = new CastNode(rhsLoad, rhs.dtype, 'i32');
        const ls = lhsZP !== 0 ? new MathOpNode('-', cl, new IntImmNode(lhsZP)) : cl;
        const rs = rhsZP !== 0 ? new MathOpNode('-', cr, new IntImmNode(rhsZP)) : cr;
        return new MathOpNode('*', ls, rs);
      },
    });
    return new SeqNode([initBody, accBody]);
  });

  registerLoweringRule('quantized_conv', (ctx, op, inputs, outputs) => {
    const inBuf = inputs[0];
    const kerBuf = inputs[1];
    const inputZP = op.getAttr<number>('input_zero_point') || 0;
    const kernelZP = op.getAttr<number>('kernel_zero_point') || 0;
    return buildConvNest(ctx, op, inBuf, kerBuf, outputs[0], {
      prefix: 'qc',
      blockPrefix: 'qconv',
      initVal: () => new IntImmNode(0),
      guardFill: () => new IntImmNode(0),
      leafBuilder: (inIdx, kerIdx) => {
        const loadIn = new CastNode(new BufferLoadNode(inBuf, inIdx), inBuf.dtype, 'i32');
        const loadKer = new CastNode(new BufferLoadNode(kerBuf, kerIdx), kerBuf.dtype, 'i32');
        const inSub = inputZP !== 0 ? new MathOpNode('-', loadIn, new IntImmNode(inputZP)) : loadIn;
        const kerSub = kernelZP !== 0 ? new MathOpNode('-', loadKer, new IntImmNode(kernelZP)) : loadKer;
        return new MathOpNode('*', inSub, kerSub);
      },
    });
  });
}
