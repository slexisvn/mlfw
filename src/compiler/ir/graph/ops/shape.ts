import { OpDef, OpTrait } from '../op_registry.js';
import type { OpAttrRecord, OpRegistry, SymbolicDim } from '../op_registry.js';
import { TensorType, DYNAMIC } from '../types.js';
import type { AttrValue } from '../types.js';
import * as pat from '../patterns.js';

export function register(registry: OpRegistry) {
  registry.register(new OpDef({
    name: 'stop_gradient',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.VIEW],
    inferResultTypes(operandTypes) {
      if (operandTypes.length !== 1) return null;
      return [operandTypes[0]];
    },
    propagateSymbolicShapes(op, shapes) {
      const inShape = shapes.get(op.getOperand(0));
      return inShape ? [inShape] : null;
    }
  }));

  registry.register(new OpDef({
    name: 'reverse',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.INJECTIVE],
    attrs: [{ name: 'dimensions', type: 'array', required: true }],
    inferResultTypes(operandTypes) {
      if (operandTypes.length !== 1) return null;
      return [operandTypes[0]];
    },
    propagateSymbolicShapes(op, shapes) {
      const s = shapes.get(op.getOperand(0));
      return s ? [s] : null;
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
      const shape = (attrs.get ? attrs.get('result_shape') : (attrs as unknown as OpAttrRecord).result_shape) as readonly number[];
      if (!shape) return null;
      return [new TensorType(shape, inp.dtype)];
    },
    propagateSymbolicShapes(op, shapes) {
      const inShape = shapes.get(op.getOperand(0));
      if (!inShape) return null;
      const bDims = op.getAttr<readonly number[]>('broadcast_dimensions')!;
      const resultShape = op.getAttr<readonly number[]>('result_shape')!;
      if (!bDims || !resultShape) return null;
      const resShape: SymbolicDim[] = resultShape.map(d => d === DYNAMIC ? null : d);
      for (let j = 0; j < bDims.length; j++) {
        if (typeof inShape[j] !== 'number') {
          resShape[bDims[j]] = inShape[j];
        } else if (resShape[bDims[j]] === null) {
          resShape[bDims[j]] = inShape[j];
        }
      }
      return [resShape];
    },
    fold(values) {
      if (values.length === 1 && typeof values[0] === 'number') return values[0];
      return undefined;
    },
    verify(op) {
      const errs = [];
      if (!op.hasAttr('broadcast_dimensions')) errs.push('broadcast_in_dim missing broadcast_dimensions');
      if (!op.hasAttr('result_shape')) errs.push('broadcast_in_dim missing result_shape');
      if (op.numOperands !== 1) errs.push('broadcast_in_dim expects 1 operand');
      if (errs.length === 0) {
        const dims = op.getAttr<readonly number[]>('broadcast_dimensions')!;
        const resultShape = op.getAttr<readonly number[]>('result_shape')!;
        const inp = op.getOperand(0).type;
        if (dims.length !== (inp as TensorType).rank) {
          errs.push(`broadcast_dimensions length ${dims.length} != input rank ${(inp as TensorType).rank}`);
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
    name: 'reshape',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.VIEW],
    attrs: [{ name: 'new_shape', type: 'array', required: true }],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length !== 1) return null;
      const inp = operandTypes[0];
      if (!(inp instanceof TensorType)) return null;
      const shape = (attrs.get ? attrs.get('new_shape') : (attrs as unknown as OpAttrRecord).new_shape) as readonly number[];
      if (!shape) return null;
      return [new TensorType(shape, inp.dtype)];
    },
    verify(op) {
      const errs = [];
      if (!op.hasAttr('new_shape')) errs.push('reshape missing new_shape');
      if (op.numOperands !== 1) errs.push('reshape expects 1 operand');
      if (errs.length === 0) {
        const inp = op.getOperand(0).type;
        const newShape = op.getAttr<readonly number[]>('new_shape')!;
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
      if (!inShape) return null;
      const newShape = op.getAttr<readonly number[]>('new_shape')!;
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
    fold(constValues) { return constValues[0]; }
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
      const perm = (attrs.get ? attrs.get('permutation') : (attrs as unknown as OpAttrRecord).permutation) as readonly number[];
      if (!perm) return null;
      const newShape = perm.map(i => inp.shape[i]);
      return [new TensorType(newShape, inp.dtype)];
    },
    propagateSymbolicShapes(op, shapes) {
      const inShape = shapes.get(op.getOperand(0));
      if (!inShape) return null;
      const perm = op.getAttr<readonly number[]>('permutation')!;
      if (!perm) return null;
      return [perm.map(i => inShape[i])];
    },
    getCanonicalizationPatterns() { return [new pat.FoldTrivialTranspose()]; },
    fold(constValues, attrs, constOps) {
      const data = constValues[0];
      const dataArr = data as { length: number; [i: number]: AttrValue };
      if (data == null || typeof data === 'number' || typeof dataArr.length !== 'number') return undefined;
      const perm = (attrs.get ? attrs.get('permutation') : (attrs as unknown as OpAttrRecord).permutation) as readonly number[];
      if (!perm) return undefined;
      const inType = constOps && constOps[0] ? constOps[0].getResult(0).type : null;
      const inShape = inType instanceof TensorType ? inType.shape : null;
      if (!inShape || inShape.length !== perm.length) return undefined;
      const inStrides = new Array(inShape.length);
      let acc = 1;
      for (let d = inShape.length - 1; d >= 0; d--) { inStrides[d] = acc; acc *= inShape[d] as number; }
      const outShape = perm.map(i => inShape[i]);
      const n = dataArr.length;
      const out = new Array(n);
      const idx = new Array(perm.length).fill(0);
      for (let i = 0; i < n; i++) {
        let src = 0;
        for (let d = 0; d < perm.length; d++) src += idx[d] * inStrides[perm[d]];
        out[i] = dataArr[src];
        for (let d = perm.length - 1; d >= 0; d--) {
          if (++idx[d] < (outShape[d] as number)) break;
          idx[d] = 0;
        }
      }
      return out;
    },
    verify(op) {
      const errs = [];
      if (!op.hasAttr('permutation')) { errs.push('transpose missing permutation'); return errs; }
      if (op.numOperands !== 1) { errs.push('transpose expects 1 operand'); return errs; }
      const perm = op.getAttr<readonly number[]>('permutation')!;
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
      const starts = (attrs.get ? attrs.get('starts') : (attrs as unknown as OpAttrRecord).starts) as readonly number[];
      const limits = (attrs.get ? attrs.get('limits') : (attrs as unknown as OpAttrRecord).limits) as readonly number[];
      const strides = ((attrs.get ? attrs.get('strides') : (attrs as unknown as OpAttrRecord).strides) as readonly number[]) || starts.map(() => 1);
      const shape = [];
      for (let i = 0; i < starts.length; i++) {
        shape.push(Math.ceil((limits[i] - starts[i]) / strides[i]));
      }
      return [new TensorType(shape, inp.dtype)];
    },
    getCanonicalizationPatterns() { return [new pat.FoldTrivialSlice()]; }
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
      const dim = (attrs.get ? attrs.get('dimension') : (attrs as unknown as OpAttrRecord).dimension) as number;
      if (dim === undefined) return null;
      const shape = [...first.shape];
      for (let i = 1; i < operandTypes.length; i++) {
        const t = operandTypes[i];
        if (!(t instanceof TensorType) || t.dtype !== first.dtype) return null;
        if (t.rank !== first.rank) return null;
        if (shape[dim] === DYNAMIC || t.shape[dim] === DYNAMIC) {
          shape[dim] = DYNAMIC;
        } else {
          shape[dim] = (shape[dim] as number) + (t.shape[dim] as number);
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
      const low = (attrs.get ? attrs.get('low') : (attrs as unknown as OpAttrRecord).low) as readonly number[];
      const high = (attrs.get ? attrs.get('high') : (attrs as unknown as OpAttrRecord).high) as readonly number[];
      const interior = ((attrs.get ? attrs.get('interior') : (attrs as unknown as OpAttrRecord).interior) as readonly number[]) || low.map(() => 0);
      const shape = [];
      for (let i = 0; i < inp.rank; i++) {
        if (inp.shape[i] === DYNAMIC) {
          shape.push(DYNAMIC);
        } else {
          shape.push(low[i] + (inp.shape[i] as number) + ((inp.shape[i] as number) - 1) * interior[i] + high[i]);
        }
      }
      return [new TensorType(shape, inp.dtype)];
    },
    getCanonicalizationPatterns() { return [new pat.FoldTrivialPad()]; }
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
      const offsetDims = (attrs.get ? attrs.get('offset_dims') : (attrs as unknown as OpAttrRecord).offset_dims) as readonly number[];
      const collapsedDims = new Set((attrs.get ? attrs.get('collapsed_slice_dims') : (attrs as unknown as OpAttrRecord).collapsed_slice_dims) as readonly number[]);
      const sliceSizes = (attrs.get ? attrs.get('slice_sizes') : (attrs as unknown as OpAttrRecord).slice_sizes) as readonly number[];
      const indexVectorDim = (attrs.get ? attrs.get('index_vector_dim') : (attrs as unknown as OpAttrRecord).index_vector_dim) as number;
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
    sideEffects: 2,
    inferResultTypes(operandTypes) {
      if (operandTypes.length < 1) return null;
      const inp = operandTypes[0];
      if (!(inp instanceof TensorType)) return null;
      return [new TensorType(inp.shape, inp.dtype)];
    }
  }));

  registry.register(new OpDef({
    name: 'split',
    numOperands: 1,
    numResults: -1,
    attrs: [
      { name: 'dimension', type: 'number', required: true },
      { name: 'split_sizes', type: 'array', required: true }
    ],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length < 1) return null;
      const inp = operandTypes[0];
      if (!(inp instanceof TensorType)) return null;
      const dim = (attrs.get ? attrs.get('dimension') : (attrs as unknown as OpAttrRecord).dimension) as number;
      const sizes = (attrs.get ? attrs.get('split_sizes') : (attrs as unknown as OpAttrRecord).split_sizes) as readonly number[];
      if (dim === undefined || !sizes) return null;
      return sizes.map(s => {
        const shape = [...inp.shape];
        shape[dim] = s;
        return new TensorType(shape, inp.dtype);
      });
    }
  }));
}
