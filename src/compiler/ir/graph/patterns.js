import { Pattern } from '../../passes/rewrite/pattern.js';
import { TensorType } from './types.js';
import { isDtypeInt } from '../../../backend/dtype_map.js';

function isConstantVal(op, val) {
  return op && op.opName === 'constant' && op.getAttr('value') === val;
}

export class FoldTrivialReshape extends Pattern {
  constructor() { super('fold_trivial_reshape', 10); this.rootOpName = 'reshape'; }
  match(op) {
    const input = op.getOperand(0).type;
    const output = op.getResult(0).type;
    return input instanceof TensorType && output instanceof TensorType && input.shapeEquals(output);
  }
  rewrite(op, builder) {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class ReshapeReshape extends Pattern {
  constructor() { super('reshape_reshape', 10); this.rootOpName = 'reshape'; }
  match(op) {
    const inputOp = op.getOperand(0).definingOp;
    return inputOp && inputOp.opName === 'reshape';
  }
  rewrite(op, builder) {
    const originalInput = op.getOperand(0).definingOp.getOperand(0);
    const newShape = op.getAttr('new_shape');
    const newReshape = builder.reshape(originalInput, newShape);
    op.replaceAllResultsWith([newReshape.getResult(0)]);
    op.erase();
    return true;
  }
}

export class FoldTrivialTranspose extends Pattern {
  constructor() { super('fold_trivial_transpose', 10); this.rootOpName = 'transpose'; }
  match(op) {
    const perm = op.getAttr('permutation');
    if (!perm) return false;
    for (let i = 0; i < perm.length; i++) {
      if (perm[i] !== i) return false;
    }
    return true;
  }
  rewrite(op, builder) {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class TransposeTranspose extends Pattern {
  constructor() { super('transpose_transpose', 10); this.rootOpName = 'transpose'; }
  match(op) {
    const inputOp = op.getOperand(0).definingOp;
    return inputOp && inputOp.opName === 'transpose';
  }
  rewrite(op, builder) {
    const originalInput = op.getOperand(0).definingOp.getOperand(0);
    const perm1 = op.getOperand(0).definingOp.getAttr('permutation');
    const perm2 = op.getAttr('permutation');
    const newPerm = new Array(perm2.length);
    for (let i = 0; i < perm2.length; i++) {
      newPerm[i] = perm1[perm2[i]];
    }
    const newTranspose = builder.transpose(originalInput, newPerm);
    op.replaceAllResultsWith([newTranspose.getResult(0)]);
    op.erase();
    return true;
  }
}

export class FoldTrivialPad extends Pattern {
  constructor() { super('fold_trivial_pad', 10); this.rootOpName = 'pad'; }
  match(op) {
    const low = op.getAttr('low');
    const high = op.getAttr('high');
    const interior = op.getAttr('interior') || [];
    if (low && low.some(x => x !== 0)) return false;
    if (high && high.some(x => x !== 0)) return false;
    if (interior && interior.some(x => x !== 0)) return false;
    return true;
  }
  rewrite(op, builder) {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class FoldTrivialSlice extends Pattern {
  constructor() { super('fold_trivial_slice', 10); this.rootOpName = 'slice'; }
  match(op) {
    const inputType = op.getOperand(0).type;
    const outputType = op.getResult(0).type;
    if (!(inputType instanceof TensorType) || !(outputType instanceof TensorType)) return false;
    const starts = op.getAttr('starts');
    const strides = op.getAttr('strides') || starts.map(() => 1);
    if (starts.some(x => x !== 0)) return false;
    if (strides.some(x => x !== 1)) return false;
    return inputType.shapeEquals(outputType);
  }
  rewrite(op, builder) {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class CommutativeConstantRight extends Pattern {
  constructor(opName = null) {
    super(`commutative_constant_right${opName ? '_' + opName : ''}`, 5);
    this.rootOpName = opName;
  }
  match(op) {
    if (op.numOperands !== 2) return false;
    const lhsDef = op.getOperand(0).definingOp;
    const rhsDef = op.getOperand(1).definingOp;
    return (lhsDef && lhsDef.opName === 'constant') && !(rhsDef && rhsDef.opName === 'constant');
  }
  rewrite(op, builder) {
    const lhs = op.getOperand(0);
    const rhs = op.getOperand(1);
    op.replaceOperand(0, rhs);
    op.replaceOperand(1, lhs);
    return true;
  }
}

export function commutativeConstantRightFor(opName) {
  return new CommutativeConstantRight(opName);
}

export class CanonicalizeCompare extends Pattern {
  constructor() { super('canonicalize_compare', 5); this.rootOpName = 'compare'; }
  match(op) {
    const lhsDef = op.getOperand(0).definingOp;
    const rhsDef = op.getOperand(1).definingOp;
    return (lhsDef && lhsDef.opName === 'constant') && !(rhsDef && rhsDef.opName === 'constant');
  }
  rewrite(op, builder) {
    const lhs = op.getOperand(0);
    const rhs = op.getOperand(1);
    op.replaceOperand(0, rhs);
    op.replaceOperand(1, lhs);
    const dir = op.getAttr('direction');
    const invert = { 'eq': 'eq', 'ne': 'ne', 'lt': 'gt', 'le': 'ge', 'gt': 'lt', 'ge': 'le' };
    op.setAttr('direction', invert[dir]);
    return true;
  }
}

export class AddZero extends Pattern {
  constructor() { super('add_zero', 5); this.rootOpName = 'add'; }
  match(op) {
    return isConstantVal(op.getOperand(1).definingOp, 0) || isConstantVal(op.getOperand(0).definingOp, 0);
  }
  rewrite(op, builder) {
    const keep = isConstantVal(op.getOperand(1).definingOp, 0) ? op.getOperand(0) : op.getOperand(1);
    if (keep.type.equals(op.getResult(0).type)) {
      op.replaceAllResultsWith([keep]);
      op.erase();
      return true;
    }
    return false;
  }
}

export class SubZero extends Pattern {
  constructor() { super('sub_zero', 5); this.rootOpName = 'sub'; }
  match(op) {
    return isConstantVal(op.getOperand(1).definingOp, 0);
  }
  rewrite(op, builder) {
    const keep = op.getOperand(0);
    if (keep.type.equals(op.getResult(0).type)) {
      op.replaceAllResultsWith([keep]);
      op.erase();
      return true;
    }
    return false;
  }
}

export class SubSelf extends Pattern {
  constructor(fastMath = false) { super('sub_self', 5); this.rootOpName = 'sub'; this.fastMath = fastMath; }
  match(op) {
    if (op.getOperand(0) !== op.getOperand(1)) return false;
    return isDtypeInt(op.getResult(0).type.dtype) || this.fastMath;
  }
  rewrite(op, builder) {
    const zero = builder.scalarConstant(0, op.getResult(0).type.dtype);
    let result = zero.getResult(0);
    const shape = op.getResult(0).type.shape;
    if (shape.length > 0) {
      result = builder.broadcast(result, shape, []).getResult(0);
    }
    op.replaceAllResultsWith([result]);
    op.erase();
    return true;
  }
}

export class MulOne extends Pattern {
  constructor() { super('mul_one', 5); this.rootOpName = 'mul'; }
  match(op) {
    return isConstantVal(op.getOperand(1).definingOp, 1) || isConstantVal(op.getOperand(0).definingOp, 1);
  }
  rewrite(op, builder) {
    const keep = isConstantVal(op.getOperand(1).definingOp, 1) ? op.getOperand(0) : op.getOperand(1);
    if (keep.type.equals(op.getResult(0).type)) {
      op.replaceAllResultsWith([keep]);
      op.erase();
      return true;
    }
    return false;
  }
}

export class MulZero extends Pattern {
  constructor(fastMath = false) { super('mul_zero', 5); this.rootOpName = 'mul'; this.fastMath = fastMath; }
  match(op) {
    if (!isDtypeInt(op.getResult(0).type.dtype) && !this.fastMath) return false;
    return isConstantVal(op.getOperand(1).definingOp, 0) || isConstantVal(op.getOperand(0).definingOp, 0);
  }
  rewrite(op, builder) {
    const zero = builder.scalarConstant(0, op.getResult(0).type.dtype);
    let result = zero.getResult(0);
    const shape = op.getResult(0).type.shape;
    if (shape.length > 0) {
      result = builder.broadcast(result, shape, []).getResult(0);
    }
    op.replaceAllResultsWith([result]);
    op.erase();
    return true;
  }
}

export class DivOne extends Pattern {
  constructor() { super('div_one', 5); this.rootOpName = 'div'; }
  match(op) {
    return isConstantVal(op.getOperand(1).definingOp, 1);
  }
  rewrite(op, builder) {
    const keep = op.getOperand(0);
    if (keep.type.equals(op.getResult(0).type)) {
      op.replaceAllResultsWith([keep]);
      op.erase();
      return true;
    }
    return false;
  }
}

export class DoubleNeg extends Pattern {
  constructor() { super('double_neg', 5); this.rootOpName = 'neg'; }
  match(op) {
    const input = op.getOperand(0).definingOp;
    return input && input.opName === 'neg';
  }
  rewrite(op, builder) {
    const original = op.getOperand(0).definingOp.getOperand(0);
    op.replaceAllResultsWith([original]);
    op.erase();
    return true;
  }
}

export class ExpLog extends Pattern {
  constructor(fastMath = false) { super('exp_log', 5); this.rootOpName = 'exp'; this.fastMath = fastMath; }
  match(op) {
    if (!this.fastMath) return false;
    const input = op.getOperand(0).definingOp;
    return input && input.opName === 'log';
  }
  rewrite(op, builder) {
    op.replaceAllResultsWith([op.getOperand(0).definingOp.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class LogExp extends Pattern {
  constructor(fastMath = false) { super('log_exp', 5); this.rootOpName = 'log'; this.fastMath = fastMath; }
  match(op) {
    if (!this.fastMath) return false;
    const input = op.getOperand(0).definingOp;
    return input && input.opName === 'exp';
  }
  rewrite(op, builder) {
    op.replaceAllResultsWith([op.getOperand(0).definingOp.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class DivSelf extends Pattern {
  constructor(fastMath = false) { super('div_self', 5); this.rootOpName = 'div'; this.fastMath = fastMath; }
  match(op) {
    if (!this.fastMath) return false;
    return op.getOperand(0) === op.getOperand(1);
  }
  rewrite(op, builder) {
    const one = builder.scalarConstant(1, op.getResult(0).type.dtype);
    let result = one.getResult(0);
    const shape = op.getResult(0).type.shape;
    if (shape.length > 0) {
      result = builder.broadcast(result, shape, []).getResult(0);
    }
    op.replaceAllResultsWith([result]);
    op.erase();
    return true;
  }
}

export class MulNegNeg extends Pattern {
  constructor() { super('mul_neg_neg', 4); this.rootOpName = 'mul'; }
  match(op) {
    const lDef = op.getOperand(0).definingOp;
    const rDef = op.getOperand(1).definingOp;
    return lDef && lDef.opName === 'neg' && rDef && rDef.opName === 'neg';
  }
  rewrite(op, builder) {
    const a = op.getOperand(0).definingOp.getOperand(0);
    const b = op.getOperand(1).definingOp.getOperand(0);
    const newMul = builder.mul(a, b);
    op.replaceAllResultsWith([newMul.getResult(0)]);
    op.erase();
    return true;
  }
}

export class AddNegToSub extends Pattern {
  constructor() { super('add_neg_to_sub', 4); this.rootOpName = 'add'; }
  match(op) {
    const rDef = op.getOperand(1).definingOp;
    return rDef && rDef.opName === 'neg';
  }
  rewrite(op, builder) {
    const a = op.getOperand(0);
    const b = op.getOperand(1).definingOp.getOperand(0);
    const newSub = builder.sub(a, b);
    op.replaceAllResultsWith([newSub.getResult(0)]);
    op.erase();
    return true;
  }
}

export class SubNegToAdd extends Pattern {
  constructor() { super('sub_neg_to_add', 4); this.rootOpName = 'sub'; }
  match(op) {
    const rDef = op.getOperand(1).definingOp;
    return rDef && rDef.opName === 'neg';
  }
  rewrite(op, builder) {
    const a = op.getOperand(0);
    const b = op.getOperand(1).definingOp.getOperand(0);
    const newAdd = builder.add(a, b);
    op.replaceAllResultsWith([newAdd.getResult(0)]);
    op.erase();
    return true;
  }
}

export class DoubleConvert extends Pattern {
  constructor() { super('double_convert', 6); this.rootOpName = 'convert'; }
  match(op) {
    const inputOp = op.getOperand(0).definingOp;
    if (!inputOp || inputOp.opName !== 'convert') return false;
    const origDtype = inputOp.getOperand(0).type.dtype;
    const finalDtype = op.getAttr('target_dtype');
    return origDtype === finalDtype;
  }
  rewrite(op, builder) {
    op.replaceAllResultsWith([op.getOperand(0).definingOp.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class LayoutTransformIdentity extends Pattern {
  constructor() { super('layout_transform_identity', 10); this.rootOpName = 'layout_transform'; }
  match(op) {
    const src = op.getAttr('src_layout');
    const dst = op.getAttr('dst_layout');
    if (!src || !dst || src.length !== dst.length) return false;
    for (let i = 0; i < src.length; i++) {
      if (src[i] !== dst[i]) return false;
    }
    return true;
  }
  rewrite(op, builder) {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class FoldTransposeIntoDot extends Pattern {
  constructor() { super('fold_transpose_into_dot', 10); this.rootOpName = 'dot'; }
  match(op) {
    for (let i = 0; i < 2; i++) {
      const def = op.getOperand(i).definingOp;
      if (!def || def.opName !== 'transpose') continue;
      const perm = def.getAttr('permutation');
      if (!perm || perm.length !== 2) continue;
      if (perm[0] !== 1 || perm[1] !== 0) continue;
      return true;
    }
    return false;
  }
  rewrite(op, builder) {
    const operands = [op.getOperand(0), op.getOperand(1)];
    const lhsC = [...op.getAttr('lhs_contracting')];
    const rhsC = [...op.getAttr('rhs_contracting')];
    const lhsB = [...(op.getAttr('lhs_batch') || [])];
    const rhsB = [...(op.getAttr('rhs_batch') || [])];
    for (let i = 0; i < 2; i++) {
      const def = operands[i].definingOp;
      if (!def || def.opName !== 'transpose') continue;
      const perm = def.getAttr('permutation');
      if (!perm || perm.length !== 2 || perm[0] !== 1 || perm[1] !== 0) continue;
      operands[i] = def.getOperand(0);
      const dims = i === 0 ? lhsC : rhsC;
      const batch = i === 0 ? lhsB : rhsB;
      for (let j = 0; j < dims.length; j++) dims[j] = dims[j] === 0 ? 1 : 0;
      for (let j = 0; j < batch.length; j++) batch[j] = batch[j] === 0 ? 1 : 0;
    }
    const newDot = builder.dot(operands[0], operands[1], lhsC, rhsC);
    if (lhsB.length > 0) newDot.setAttr('lhs_batch', lhsB);
    if (rhsB.length > 0) newDot.setAttr('rhs_batch', rhsB);
    op.replaceAllResultsWith([newDot.getResult(0)]);
    op.erase();
    return true;
  }
}

export class LayoutTransformCompose extends Pattern {
  constructor() { super('layout_transform_compose', 10); this.rootOpName = 'layout_transform'; }
  match(op) {
    const inputOp = op.getOperand(0).definingOp;
    return inputOp && inputOp.opName === 'layout_transform';
  }
  rewrite(op, builder) {
    const innerOp = op.getOperand(0).definingOp;
    const srcLayout = innerOp.getAttr('src_layout');
    const midLayout = innerOp.getAttr('dst_layout');
    const dstLayout = op.getAttr('dst_layout');
    const composed = new Array(srcLayout.length);
    for (let i = 0; i < dstLayout.length; i++) {
      composed[i] = srcLayout[midLayout.indexOf(dstLayout[i])];
    }
    const original = innerOp.getOperand(0);
    const newOp = builder._inferAndBuild('layout_transform', [original],
      { src_layout: srcLayout, dst_layout: composed });
    op.replaceAllResultsWith([newOp.getResult(0)]);
    op.erase();
    return true;
  }
}
