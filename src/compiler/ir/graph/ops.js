import { OpDef, OpRegistry, SideEffectKind, OpTrait } from './op_registry.js';
import { TensorType, TupleType, ScalarType, DYNAMIC, broadcastDim, isFloatType, isBoolType } from './types.js';
import * as pat from './patterns.js';

export const registry = new OpRegistry();

function inferBinaryElementwise(operandTypes) {
  if (operandTypes.length !== 2) return null;
  const lhs = operandTypes[0], rhs = operandTypes[1];
  if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return null;
  if (lhs.dtype !== rhs.dtype) return null;
  const shape = TensorType.broadcastShape(lhs.shape, rhs.shape);
  if (!shape) return null;
  return [new TensorType(shape, lhs.dtype)];
}

function inferUnaryElementwise(operandTypes) {
  if (operandTypes.length !== 1) return null;
  const inp = operandTypes[0];
  if (!(inp instanceof TensorType)) return null;
  return [new TensorType(inp.shape, inp.dtype)];
}

function inferCompare(operandTypes) {
  if (operandTypes.length !== 2) return null;
  const lhs = operandTypes[0], rhs = operandTypes[1];
  if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return null;
  const shape = TensorType.broadcastShape(lhs.shape, rhs.shape);
  if (!shape) return null;
  return [new TensorType(shape, ScalarType.BOOL)];
}

function verifyBinaryElementwise(op) {
  const errs = [];
  if (op.numOperands !== 2) { errs.push(`${op.opName} expects 2 operands, got ${op.numOperands}`); return errs; }
  const lt = op.getOperand(0).type, rt = op.getOperand(1).type;
  if (!(lt instanceof TensorType)) errs.push(`${op.opName} operand 0 is not tensor`);
  if (!(rt instanceof TensorType)) errs.push(`${op.opName} operand 1 is not tensor`);
  if (lt instanceof TensorType && rt instanceof TensorType && lt.dtype !== rt.dtype) {
    errs.push(`${op.opName} dtype mismatch: ${lt.dtype} vs ${rt.dtype}`);
  }
  return errs;
}

function verifyUnaryElementwise(op) {
  const errs = [];
  if (op.numOperands !== 1) { errs.push(`${op.opName} expects 1 operand, got ${op.numOperands}`); return errs; }
  if (!(op.getOperand(0).type instanceof TensorType)) errs.push(`${op.opName} operand is not tensor`);
  return errs;
}

const binaryArithTraits = [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE];
const commBinaryArithTraits = [...binaryArithTraits, OpTrait.COMMUTATIVE, OpTrait.ASSOCIATIVE];

registry.register(new OpDef({
  name: 'add',
  numOperands: 2,
  numResults: 1,
  traits: commBinaryArithTraits,
  inferResultTypes: inferBinaryElementwise,
  verify: verifyBinaryElementwise,
  getCanonicalizationPatterns() { return [pat.commutativeConstantRightFor('add'), new pat.AddZero()]; },
  fold(constValues, attrs) { return constValues[0] + constValues[1]; }
}));

registry.register(new OpDef({
  name: 'mul',
  numOperands: 2,
  numResults: 1,
  traits: commBinaryArithTraits,
  inferResultTypes: inferBinaryElementwise,
  verify: verifyBinaryElementwise,
  getCanonicalizationPatterns() { return [pat.commutativeConstantRightFor('mul'), new pat.MulOne(), new pat.MulZero()]; },
  fold(constValues, attrs) { return constValues[0] * constValues[1]; }
}));

registry.register(new OpDef({
  name: 'sub',
  numOperands: 2,
  numResults: 1,
  traits: binaryArithTraits,
  inferResultTypes: inferBinaryElementwise,
  verify: verifyBinaryElementwise,
  getCanonicalizationPatterns() { return [new pat.SubZero(), new pat.SubSelf()]; },
  fold(constValues, attrs) { return constValues[0] - constValues[1]; }
}));

registry.register(new OpDef({
  name: 'div',
  numOperands: 2,
  numResults: 1,
  traits: binaryArithTraits,
  inferResultTypes: inferBinaryElementwise,
  verify: verifyBinaryElementwise,
  getCanonicalizationPatterns() { return [new pat.DivOne()]; },
  fold(constValues, attrs) { return constValues[0] / constValues[1]; }
}));

for (const name of ['rem', 'pow']) {
  registry.register(new OpDef({
    name,
    numOperands: 2,
    numResults: 1,
    traits: binaryArithTraits,
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise
  }));
}

registry.register(new OpDef({
  name: 'neg',
  numOperands: 1,
  numResults: 1,
  traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
  inferResultTypes: inferUnaryElementwise,
  verify: verifyUnaryElementwise,
  getCanonicalizationPatterns() { return [new pat.DoubleNeg()]; },
  fold(constValues, attrs) { return -constValues[0]; }
}));

for (const name of ['maximum', 'minimum']) {
  registry.register(new OpDef({
    name,
    numOperands: 2,
    numResults: 1,
    traits: [...binaryArithTraits, OpTrait.COMMUTATIVE],
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise
  }));
}

for (const name of ['abs', 'floor', 'ceil', 'round', 'sign']) {
  registry.register(new OpDef({
    name,
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
    inferResultTypes: inferUnaryElementwise,
    verify: verifyUnaryElementwise
  }));
}


registry.register(new OpDef({
  name: 'exp',
  numOperands: 1,
  numResults: 1,
  traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
  inferResultTypes(operandTypes) {
    if (operandTypes.length !== 1) return null;
    const t = operandTypes[0];
    if (!(t instanceof TensorType) || !isFloatType(t.dtype)) return null;
    return [new TensorType(t.shape, t.dtype)];
  },
  getCanonicalizationPatterns() { return [new pat.ExpLog()]; },
  fold(constValues, attrs) { return Math.exp(constValues[0]); },
  verify(op) {
    const errs = verifyUnaryElementwise(op);
    if (errs.length === 0) {
      const dt = op.getOperand(0).type.dtype;
      if (!isFloatType(dt)) errs.push(`${op.opName} requires float type, got ${dt}`);
    }
    return errs;
  }
}));

for (const name of ['log', 'sqrt', 'rsqrt', 'tanh', 'sin', 'cos']) {
  registry.register(new OpDef({
    name,
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
    inferResultTypes(operandTypes) {
      if (operandTypes.length !== 1) return null;
      const t = operandTypes[0];
      if (!(t instanceof TensorType) || !isFloatType(t.dtype)) return null;
      return [new TensorType(t.shape, t.dtype)];
    },
    verify(op) {
      const errs = verifyUnaryElementwise(op);
      if (errs.length === 0) {
        const dt = op.getOperand(0).type.dtype;
        if (!isFloatType(dt)) errs.push(`${op.opName} requires float type, got ${dt}`);
      }
      return errs;
    }
  }));
}

registry.register(new OpDef({
  name: 'compare',
  numOperands: 2,
  numResults: 1,
  attrs: [{ name: 'direction', type: 'string', required: true }],
  traits: [OpTrait.ELEMENTWISE],
  inferResultTypes: inferCompare,
  verify(op) {
    const errs = [];
    if (op.numOperands !== 2) { errs.push('compare expects 2 operands'); return errs; }
    if (!op.hasAttr('direction')) errs.push('compare missing direction attr');
    else {
      const d = op.getAttr('direction');
      if (!['eq', 'ne', 'lt', 'le', 'gt', 'ge'].includes(d)) {
        errs.push(`compare invalid direction: ${d}`);
      }
    }
    return errs;
  }
}));

registry.register(new OpDef({
  name: 'select',
  numOperands: 3,
  numResults: 1,
  traits: [OpTrait.ELEMENTWISE],
  inferResultTypes(operandTypes) {
    if (operandTypes.length !== 3) return null;
    const [pred, onTrue, onFalse] = operandTypes;
    if (!(pred instanceof TensorType) || pred.dtype !== ScalarType.BOOL) return null;
    if (!(onTrue instanceof TensorType) || !(onFalse instanceof TensorType)) return null;
    if (onTrue.dtype !== onFalse.dtype) return null;
    const shape = TensorType.broadcastShape(pred.shape, onTrue.shape, onFalse.shape);
    if (!shape) return null;
    return [new TensorType(shape, onTrue.dtype)];
  }
}));

registry.register(new OpDef({
  name: 'clamp',
  numOperands: 3,
  numResults: 1,
  traits: [OpTrait.ELEMENTWISE],
  inferResultTypes(operandTypes) {
    if (operandTypes.length !== 3) return null;
    const [lo, x, hi] = operandTypes;
    if (!(x instanceof TensorType)) return null;
    return [new TensorType(x.shape, x.dtype)];
  }
}));

registry.register(new OpDef({
  name: 'broadcast_in_dim',
  numOperands: 1,
  numResults: 1,
  traits: [OpTrait.BROADCAST],
  attrs: [
    { name: 'broadcast_dimensions', type: 'array', required: true },
    { name: 'result_shape', type: 'array', required: true }
  ],
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length !== 1) return null;
    const inp = operandTypes[0];
    if (!(inp instanceof TensorType)) return null;
    const shape = attrs.get ? attrs.get('result_shape') : attrs.result_shape;
    if (!shape) return null;
    return [new TensorType(shape, inp.dtype)];
  },
  propagateSymbolicShapes(op, shapes) {
    const inShape = shapes.get(op.getOperand(0));
    if (!inShape) return null;
    const bDims = op.getAttr('broadcast_dimensions');
    const resultShape = op.getAttr('result_shape');
    if (!bDims || !resultShape) return null;
    const resShape = resultShape.map(d => d === DYNAMIC ? null : d);
    for (let j = 0; j < bDims.length; j++) {
      if (typeof inShape[j] !== 'number') {
        resShape[bDims[j]] = inShape[j];
      } else if (resShape[bDims[j]] === null) {
        resShape[bDims[j]] = inShape[j];
      }
    }
    return [resShape];
  },
  verify(op) {
    const errs = [];
    if (!op.hasAttr('broadcast_dimensions')) errs.push('broadcast_in_dim missing broadcast_dimensions');
    if (!op.hasAttr('result_shape')) errs.push('broadcast_in_dim missing result_shape');
    if (op.numOperands !== 1) errs.push('broadcast_in_dim expects 1 operand');
    if (errs.length === 0) {
      const dims = op.getAttr('broadcast_dimensions');
      const resultShape = op.getAttr('result_shape');
      const inp = op.getOperand(0).type;
      if (dims.length !== inp.rank) {
        errs.push(`broadcast_dimensions length ${dims.length} != input rank ${inp.rank}`);
      }
      for (let i = 0; i < dims.length; i++) {
        if (dims[i] < 0 || dims[i] >= resultShape.length) {
          errs.push(`broadcast_dimensions[${i}]=${dims[i]} out of range for result rank ${resultShape.length}`);
        } else if (inp instanceof TensorType && inp.shape[i] !== DYNAMIC && inp.shape[i] !== 1 && resultShape[dims[i]] !== DYNAMIC && inp.shape[i] !== resultShape[dims[i]]) {
          errs.push(`broadcast_in_dim: input dim ${i} size ${inp.shape[i]} incompatible with result dim ${dims[i]} size ${resultShape[dims[i]]}`);
        }
      }
      const seen = new Set();
      for (const d of dims) {
        if (seen.has(d)) errs.push(`broadcast_dimensions has duplicate: ${d}`);
        seen.add(d);
      }
    }
    return errs;
  }
}));

registry.register(new OpDef({
  name: 'reduce',
  numOperands: 2,
  numResults: 1,
  attrs: [
    { name: 'dimensions', type: 'array', required: true },
    { name: 'reduce_type', type: 'string', required: true }
  ],
  traits: [OpTrait.REDUCTION],
  hasRegions: true,
  numRegions: 1,
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length < 1) return null;
    const inp = operandTypes[0];
    if (!(inp instanceof TensorType)) return null;
    const dims = attrs.get ? attrs.get('dimensions') : attrs.dimensions;
    if (!dims) return null;
    const dimSet = new Set(dims);
    const newShape = [];
    for (let i = 0; i < inp.rank; i++) {
      if (!dimSet.has(i)) newShape.push(inp.shape[i]);
    }
    return [new TensorType(newShape, inp.dtype)];
  },
  verify(op) {
    const errs = [];
    if (!op.hasAttr('dimensions')) errs.push('reduce missing dimensions');
    if (!op.hasAttr('reduce_type')) errs.push('reduce missing reduce_type');
    else {
      const rt = op.getAttr('reduce_type');
      if (!['sum', 'max', 'min', 'prod', 'mean', 'and', 'or'].includes(rt)) {
        errs.push(`reduce invalid reduce_type: ${rt}`);
      }
    }
    return errs;
  }
}));

registry.register(new OpDef({
  name: 'dot',
  numOperands: 2,
  numResults: 1,
  traits: [OpTrait.OPAQUE],
  attrs: [
    { name: 'lhs_contracting', type: 'array', required: true },
    { name: 'rhs_contracting', type: 'array', required: true },
    { name: 'lhs_batch', type: 'array', required: false },
    { name: 'rhs_batch', type: 'array', required: false }
  ],
  getFlops(op) {
    const lhs = op.getOperand(0).type;
    const rhs = op.getOperand(1).type;
    if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return 0;
    const lhsC = op.getAttr('lhs_contracting') || [];
    let contractDim = 1;
    for (const d of lhsC) {
      if (lhs.shape[d] !== DYNAMIC) contractDim *= lhs.shape[d];
    }
    const result = op.getResult(0).type;
    if (!(result instanceof TensorType)) return 0;
    const outputElements = result.numel();
    if (outputElements === DYNAMIC) return 0;
    return 2 * outputElements * contractDim;
  },
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length !== 2) return null;
    const lhs = operandTypes[0], rhs = operandTypes[1];
    if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return null;
    if (lhs.dtype !== rhs.dtype) return null;
    const lhsC = new Set(attrs.get ? attrs.get('lhs_contracting') : attrs.lhs_contracting);
    const rhsC = new Set(attrs.get ? attrs.get('rhs_contracting') : attrs.rhs_contracting);
    const lhsB = new Set((attrs.get ? attrs.get('lhs_batch') : attrs.lhs_batch) || []);
    const rhsB = new Set((attrs.get ? attrs.get('rhs_batch') : attrs.rhs_batch) || []);
    const shape = [];
    for (let i = 0; i < lhs.rank; i++) {
      if (lhsB.has(i)) shape.push(lhs.shape[i]);
    }
    for (let i = 0; i < lhs.rank; i++) {
      if (!lhsC.has(i) && !lhsB.has(i)) shape.push(lhs.shape[i]);
    }
    for (let i = 0; i < rhs.rank; i++) {
      if (!rhsC.has(i) && !rhsB.has(i)) shape.push(rhs.shape[i]);
    }
    return [new TensorType(shape, lhs.dtype)];
  },
  verify(op) {
    const errs = [];
    if (op.numOperands !== 2) { errs.push('dot expects 2 operands'); return errs; }
    if (!op.hasAttr('lhs_contracting')) errs.push('dot missing lhs_contracting');
    if (!op.hasAttr('rhs_contracting')) errs.push('dot missing rhs_contracting');
    const lhs = op.getOperand(0).type, rhs = op.getOperand(1).type;
    if (lhs instanceof TensorType && rhs instanceof TensorType) {
      if (lhs.dtype !== rhs.dtype) {
        errs.push(`dot dtype mismatch: ${lhs.dtype} vs ${rhs.dtype}`);
      }
      const lhsC = op.getAttr('lhs_contracting') || [];
      const rhsC = op.getAttr('rhs_contracting') || [];
      if (lhsC.length !== rhsC.length) {
        errs.push(`dot contracting dimensions count mismatch: lhs ${lhsC.length} vs rhs ${rhsC.length}`);
      } else {
        for (let i = 0; i < lhsC.length; i++) {
          const ld = lhsC[i], rd = rhsC[i];
          if (ld >= lhs.rank) errs.push(`dot lhs_contracting[${i}]=${ld} out of range (rank ${lhs.rank})`);
          if (rd >= rhs.rank) errs.push(`dot rhs_contracting[${i}]=${rd} out of range (rank ${rhs.rank})`);
          if (ld < lhs.rank && rd < rhs.rank && lhs.shape[ld] !== DYNAMIC && rhs.shape[rd] !== DYNAMIC && lhs.shape[ld] !== rhs.shape[rd]) {
            errs.push(`dot contracting dim size mismatch at [${i}]: lhs dim ${ld} size ${lhs.shape[ld]} vs rhs dim ${rd} size ${rhs.shape[rd]}`);
          }
        }
      }
    }
    return errs;
  }
}));

registry.register(new OpDef({
  name: 'conv',
  numOperands: 2,
  numResults: 1,
  traits: [OpTrait.OPAQUE],
  attrs: [
    { name: 'strides', type: 'array', required: true },
    { name: 'padding', type: 'array', required: true },
    { name: 'dilation', type: 'array', required: false },
    { name: 'groups', type: 'number', required: false },
    { name: 'input_layout', type: 'string', required: true },
    { name: 'kernel_layout', type: 'string', required: true }
  ],
  getFlops(op) {
    const output = op.getResult(0).type;
    const kernel = op.getOperand(1).type;
    if (!(output instanceof TensorType) || !(kernel instanceof TensorType)) return 0;
    const outElements = output.numel();
    const kElements = kernel.numel();
    if (outElements === DYNAMIC || kElements === DYNAMIC) return 0;
    const inChannels = kernel.shape[1] || 1;
    return 2 * outElements * kElements / (kernel.shape[0] || 1);
  },
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length !== 2) return null;
    const inp = operandTypes[0], kernel = operandTypes[1];
    if (!(inp instanceof TensorType) || !(kernel instanceof TensorType)) return null;
    if (inp.dtype !== kernel.dtype) return null;
    const strides = attrs.get ? attrs.get('strides') : attrs.strides;
    const padding = attrs.get ? attrs.get('padding') : attrs.padding;
    const dilation = (attrs.get ? attrs.get('dilation') : attrs.dilation) || strides.map(() => 1);
    const spatialDims = strides.length;
    const batch = inp.shape[0];
    const outChannels = kernel.shape[0];
    const outSpatial = [];
    for (let i = 0; i < spatialDims; i++) {
      const inDim = inp.shape[i + 2];
      const kDim = kernel.shape[i + 2];
      const padTotal = padding[i][0] + padding[i][1];
      if (inDim === DYNAMIC || kDim === DYNAMIC) {
        outSpatial.push(DYNAMIC);
      } else {
        const effectiveK = (kDim - 1) * dilation[i] + 1;
        outSpatial.push(Math.floor((inDim + padTotal - effectiveK) / strides[i]) + 1);
      }
    }
    return [new TensorType([batch, outChannels, ...outSpatial], inp.dtype)];
  },
  verify(op) {
    const errs = [];
    if (op.numOperands !== 2) { errs.push('conv expects 2 operands'); return errs; }
    if (!op.hasAttr('strides')) errs.push('conv missing strides');
    if (!op.hasAttr('padding')) errs.push('conv missing padding');
    if (!op.hasAttr('input_layout')) errs.push('conv missing input_layout');
    if (!op.hasAttr('kernel_layout')) errs.push('conv missing kernel_layout');
    return errs;
  }
}));

registry.register(new OpDef({
  name: 'reshape',
  numOperands: 1,
  numResults: 1,
  traits: [OpTrait.VIEW],
  attrs: [{ name: 'new_shape', type: 'array', required: true }],
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length !== 1) return null;
    const inp = operandTypes[0];
    if (!(inp instanceof TensorType)) return null;
    const shape = attrs.get ? attrs.get('new_shape') : attrs.new_shape;
    if (!shape) return null;
    return [new TensorType(shape, inp.dtype)];
  },
  verify(op) {
    const errs = [];
    if (!op.hasAttr('new_shape')) errs.push('reshape missing new_shape');
    if (op.numOperands !== 1) errs.push('reshape expects 1 operand');
    if (errs.length === 0) {
      const inp = op.getOperand(0).type;
      const newShape = op.getAttr('new_shape');
      if (inp instanceof TensorType && inp.isFullyStatic) {
        const dynCount = newShape.filter(d => d === DYNAMIC).length;
        if (dynCount > 1) errs.push('reshape can have at most one dynamic dimension');
        if (dynCount === 0) {
          const inNumel = inp.numel();
          const outNumel = newShape.reduce((a, b) => a * b, 1);
          if (inNumel !== outNumel) {
            errs.push(`reshape numel mismatch: input ${inNumel} vs output ${outNumel}`);
          }
        }
      }
    }
    return errs;
  },
  propagateSymbolicShapes(op, shapes) {
    const inShape = shapes.get(op.getOperand(0));
    const newShape = op.getAttr('new_shape');
    const resShape = [];
    for (let d of newShape) {
      if (d === -1) {
        const symVar = inShape.find(id => typeof id !== 'number');
        resShape.push(symVar || -1);
      } else {
        resShape.push(d);
      }
    }
    return [resShape];
  },
  getCanonicalizationPatterns() { return [new pat.FoldTrivialReshape(), new pat.ReshapeReshape()]; },
  fold(constValues, attrs) { return constValues[0]; }
}));

registry.register(new OpDef({
  name: 'transpose',
  numOperands: 1,
  numResults: 1,
  traits: [OpTrait.VIEW],
  attrs: [{ name: 'permutation', type: 'array', required: true }],
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length !== 1) return null;
    const inp = operandTypes[0];
    if (!(inp instanceof TensorType)) return null;
    const perm = attrs.get ? attrs.get('permutation') : attrs.permutation;
    if (!perm) return null;
    const newShape = perm.map(i => inp.shape[i]);
    return [new TensorType(newShape, inp.dtype)];
  },
  getCanonicalizationPatterns() { return [new pat.FoldTrivialTranspose()]; },
  verify(op) {
    const errs = [];
    if (!op.hasAttr('permutation')) { errs.push('transpose missing permutation'); return errs; }
    if (op.numOperands !== 1) { errs.push('transpose expects 1 operand'); return errs; }
    const perm = op.getAttr('permutation');
    const inp = op.getOperand(0).type;
    if (inp instanceof TensorType && perm.length !== inp.rank) {
      errs.push(`transpose permutation length ${perm.length} != input rank ${inp.rank}`);
    }
    const seen = new Set();
    for (const p of perm) {
      if (seen.has(p)) errs.push(`transpose duplicate in permutation: ${p}`);
      seen.add(p);
    }
    return errs;
  }
}));

registry.register(new OpDef({
  name: 'slice',
  numOperands: 1,
  numResults: 1,
  traits: [OpTrait.VIEW],
  attrs: [
    { name: 'starts', type: 'array', required: true },
    { name: 'limits', type: 'array', required: true },
    { name: 'strides', type: 'array', required: false }
  ],
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length !== 1) return null;
    const inp = operandTypes[0];
    if (!(inp instanceof TensorType)) return null;
    const starts = attrs.get ? attrs.get('starts') : attrs.starts;
    const limits = attrs.get ? attrs.get('limits') : attrs.limits;
    const strides = (attrs.get ? attrs.get('strides') : attrs.strides) || starts.map(() => 1);
    const shape = [];
    for (let i = 0; i < starts.length; i++) {
      shape.push(Math.ceil((limits[i] - starts[i]) / strides[i]));
    }
    return [new TensorType(shape, inp.dtype)];
  }
}));

registry.register(new OpDef({
  name: 'concat',
  numOperands: -1,
  numResults: 1,
  attrs: [{ name: 'dimension', type: 'number', required: true }],
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length < 1) return null;
    const first = operandTypes[0];
    if (!(first instanceof TensorType)) return null;
    const dim = attrs.get ? attrs.get('dimension') : attrs.dimension;
    if (dim === undefined) return null;
    const shape = [...first.shape];
    for (let i = 1; i < operandTypes.length; i++) {
      const t = operandTypes[i];
      if (!(t instanceof TensorType) || t.dtype !== first.dtype) return null;
      if (t.rank !== first.rank) return null;
      if (shape[dim] === DYNAMIC || t.shape[dim] === DYNAMIC) {
        shape[dim] = DYNAMIC;
      } else {
        shape[dim] += t.shape[dim];
      }
    }
    return [new TensorType(shape, first.dtype)];
  }
}));

registry.register(new OpDef({
  name: 'pad',
  numOperands: 2,
  numResults: 1,
  traits: [OpTrait.INJECTIVE],
  attrs: [
    { name: 'low', type: 'array', required: true },
    { name: 'high', type: 'array', required: true },
    { name: 'interior', type: 'array', required: false }
  ],
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length < 1) return null;
    const inp = operandTypes[0];
    if (!(inp instanceof TensorType)) return null;
    const low = attrs.get ? attrs.get('low') : attrs.low;
    const high = attrs.get ? attrs.get('high') : attrs.high;
    const interior = (attrs.get ? attrs.get('interior') : attrs.interior) || low.map(() => 0);
    const shape = [];
    for (let i = 0; i < inp.rank; i++) {
      if (inp.shape[i] === DYNAMIC) {
        shape.push(DYNAMIC);
      } else {
        shape.push(low[i] + inp.shape[i] + (inp.shape[i] - 1) * interior[i] + high[i]);
      }
    }
    return [new TensorType(shape, inp.dtype)];
  }
}));

registry.register(new OpDef({
  name: 'gather',
  numOperands: 2,
  numResults: 1,
  traits: [OpTrait.INJECTIVE],
  attrs: [
    { name: 'offset_dims', type: 'array', required: true },
    { name: 'collapsed_slice_dims', type: 'array', required: true },
    { name: 'start_index_map', type: 'array', required: true },
    { name: 'slice_sizes', type: 'array', required: true },
    { name: 'index_vector_dim', type: 'number', required: true }
  ],
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length !== 2) return null;
    const operand = operandTypes[0], indices = operandTypes[1];
    if (!(operand instanceof TensorType) || !(indices instanceof TensorType)) return null;
    const offsetDims = attrs.get ? attrs.get('offset_dims') : attrs.offset_dims;
    const collapsedDims = new Set(attrs.get ? attrs.get('collapsed_slice_dims') : attrs.collapsed_slice_dims);
    const sliceSizes = attrs.get ? attrs.get('slice_sizes') : attrs.slice_sizes;
    const indexVectorDim = attrs.get ? attrs.get('index_vector_dim') : attrs.index_vector_dim;
    const batchDims = [];
    for (let i = 0; i < indices.rank; i++) {
      if (i !== indexVectorDim) batchDims.push(indices.shape[i]);
    }
    const offsetSizes = [];
    for (let i = 0; i < sliceSizes.length; i++) {
      if (!collapsedDims.has(i)) offsetSizes.push(sliceSizes[i]);
    }
    const offsetSet = new Set(offsetDims);
    const shape = [];
    let batchIdx = 0, offsetIdx = 0;
    const totalRank = batchDims.length + offsetSizes.length;
    for (let i = 0; i < totalRank; i++) {
      if (offsetSet.has(i)) {
        shape.push(offsetSizes[offsetIdx++]);
      } else {
        shape.push(batchDims[batchIdx++]);
      }
    }
    return [new TensorType(shape, operand.dtype)];
  }
}));

registry.register(new OpDef({
  name: 'scatter',
  numOperands: 3,
  numResults: 1,
  traits: [OpTrait.INJECTIVE],
  attrs: [
    { name: 'update_window_dims', type: 'array', required: true },
    { name: 'inserted_window_dims', type: 'array', required: true },
    { name: 'scatter_dims_to_operand_dims', type: 'array', required: true },
    { name: 'index_vector_dim', type: 'number', required: true }
  ],
  hasRegions: true,
  numRegions: 1,
  sideEffects: SideEffectKind.WRITE,
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length !== 1) return null;
    const inp = operandTypes[0];
    if (!(inp instanceof TensorType)) return null;
    return [new TensorType(attrs.new_shape, inp.dtype, inp.layout)];
  }
}));

registry.register(new OpDef({
  name: 'constant',
  numOperands: 0,
  numResults: 1,
  attrs: [
    { name: 'value', type: 'any', required: true },
    { name: 'tensor_type', type: 'object', required: true }
  ],
  traits: [OpTrait.CONSTANT],
  inferResultTypes(operandTypes, attrs) {
    const tt = attrs.get ? attrs.get('tensor_type') : attrs.tensor_type;
    if (!tt) return null;
    return [tt];
  }
}));

registry.register(new OpDef({
  name: 'iota',
  numOperands: 0,
  numResults: 1,
  attrs: [
    { name: 'iota_dimension', type: 'number', required: true },
    { name: 'tensor_type', type: 'object', required: true }
  ],
  inferResultTypes(operandTypes, attrs) {
    const tt = attrs.get ? attrs.get('tensor_type') : attrs.tensor_type;
    return tt ? [tt] : null;
  }
}));

registry.register(new OpDef({
  name: 'tuple',
  numOperands: -1,
  numResults: 1,
  inferResultTypes(operandTypes) {
    return [new TupleType(operandTypes)];
  }
}));

registry.register(new OpDef({
  name: 'get_tuple_element',
  numOperands: 1,
  numResults: 1,
  attrs: [{ name: 'index', type: 'number', required: true }],
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length !== 1) return null;
    const tup = operandTypes[0];
    if (!tup || !tup.elements) return null;
    const idx = attrs.get ? attrs.get('index') : attrs.index;
    if (idx === undefined || idx < 0 || idx >= tup.elements.length) return null;
    return [tup.elements[idx]];
  }
}));

registry.register(new OpDef({
  name: 'convert',
  numOperands: 1,
  numResults: 1,
  attrs: [{ name: 'target_dtype', type: 'string', required: true }],
  traits: [OpTrait.ELEMENTWISE],
  inferResultTypes(operandTypes, attrs) {
    if (operandTypes.length !== 1) return null;
    const inp = operandTypes[0];
    if (!(inp instanceof TensorType)) return null;
    const dt = attrs.get ? attrs.get('target_dtype') : attrs.target_dtype;
    return [new TensorType(inp.shape, dt)];
  }
}));

registry.register(new OpDef({
  name: 'return',
  numOperands: -1,
  numResults: 0,
  traits: [OpTrait.TERMINATOR]
}));

registry.register(new OpDef({
  name: 'yield',
  numOperands: -1,
  numResults: 0,
  traits: [OpTrait.TERMINATOR]
}));

registry.register(new OpDef({
  name: 'if',
  numOperands: 1,
  numResults: -1,
  hasRegions: true,
  numRegions: 2,
  sideEffects: SideEffectKind.CONTROL,
  inferResultTypes(operandTypes, attrs, resultTypes) {
    return resultTypes || null;
  }
}));

registry.register(new OpDef({
  name: 'while',
  numOperands: -1,
  numResults: -1,
  hasRegions: true,
  numRegions: 2,
  sideEffects: SideEffectKind.CONTROL,
  inferResultTypes(operandTypes) {
    return [...operandTypes];
  }
}));

registry.register(new OpDef({
  name: 'custom_call',
  numOperands: -1,
  numResults: -1,
  attrs: [
    { name: 'call_target_name', type: 'string', required: true },
    { name: 'backend_config', type: 'any', required: false }
  ],
  sideEffects: SideEffectKind.WRITE,
  inferResultTypes(operandTypes, attrs, resultTypes) {
    return resultTypes || null;
  }
}));

registry.register(new OpDef({
  name: 'fused_dot_epilogue',
  numOperands: -1,
  numResults: 1,
  traits: [OpTrait.OPAQUE],
  attrs: [
    { name: 'lhs_contracting', type: 'array', required: true },
    { name: 'rhs_contracting', type: 'array', required: true },
    { name: 'lhs_batch', type: 'array', required: false },
    { name: 'rhs_batch', type: 'array', required: false },
    { name: 'epilogue_ops', type: 'array', required: true },
    { name: 'epilogue_tags', type: 'array', required: true },
    { name: 'num_dot_operands', type: 'number', required: true },
    { name: 'num_extra_inputs', type: 'number', required: true }
  ],
  inferResultTypes(operandTypes, attrs, resultTypes) {
    return resultTypes || null;
  }
}));

registry.register(new OpDef({
  name: 'fusion',
  numOperands: -1,
  numResults: -1,
  hasRegions: true,
  numRegions: 1,
  attrs: [{ name: 'fusion_kind', type: 'string', required: false }],
  inferResultTypes(operandTypes, attrs, resultTypes) {
    return resultTypes || null;
  }
}));
