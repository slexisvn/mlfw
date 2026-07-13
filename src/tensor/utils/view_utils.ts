import type { Shape } from './shape_utils.js';

type ViewResult = { sizes: number[]; strides: number[] };
type OffsetViewResult = ViewResult & { offsetDelta: number };

export function computeTranspose(sizes: Shape, strides: Shape, dim0: number, dim1: number): ViewResult {
  const ndim = sizes.length;
  const d0 = dim0 < 0 ? ndim + dim0 : dim0;
  const d1 = dim1 < 0 ? ndim + dim1 : dim1;
  const newSizes = [...sizes];
  const newStrides = [...strides];
  newSizes[d0] = sizes[d1]!;
  newSizes[d1] = sizes[d0]!;
  newStrides[d0] = strides[d1]!;
  newStrides[d1] = strides[d0]!;
  return { sizes: newSizes, strides: newStrides };
}

export function computePermute(sizes: Shape, strides: Shape, dims: Shape): ViewResult {
  const ndim = sizes.length;
  const newSizes = new Array(ndim);
  const newStrides = new Array(ndim);
  for (let i = 0; i < ndim; i++) {
    const dim = dims[i]!;
    const d = dim < 0 ? ndim + dim : dim;
    newSizes[i] = sizes[d]!;
    newStrides[i] = strides[d]!;
  }
  return { sizes: newSizes, strides: newStrides };
}

export function computeExpand(sizes: Shape, strides: Shape, targetShape: Shape): ViewResult {
  const srcNdim = sizes.length;
  const dstNdim = targetShape.length;
  const dimDiff = dstNdim - srcNdim;
  const newStrides = new Array(dstNdim);
  const newSizes = new Array(dstNdim);

  for (let i = 0; i < dstNdim; i++) {
    const srcIdx = i - dimDiff;
    if (srcIdx < 0) {
      newSizes[i] = targetShape[i]!;
      newStrides[i] = 0;
    } else if (sizes[srcIdx] === 1 && targetShape[i] !== 1) {
      newSizes[i] = targetShape[i]!;
      newStrides[i] = 0;
    } else if (sizes[srcIdx] === targetShape[i] || targetShape[i] === -1) {
      newSizes[i] = sizes[srcIdx]!;
      newStrides[i] = strides[srcIdx]!;
    } else {
      throw new Error(`Cannot expand size ${sizes[srcIdx]} to ${targetShape[i]} at dim ${i}`);
    }
  }

  return { sizes: newSizes, strides: newStrides };
}

export function computeSlice(sizes: Shape, strides: Shape, dim: number, start: number | null | undefined, end: number | null | undefined, step: number | null | undefined): OffsetViewResult {
  const ndim = sizes.length;
  const d = dim < 0 ? ndim + dim : dim;
  const dimSize = sizes[d]!;

  let s = start ?? 0;
  let e = end ?? dimSize;
  const st = step ?? 1;

  if (s < 0) s += dimSize;
  if (e < 0) e += dimSize;
  s = Math.max(0, Math.min(s, dimSize));
  e = Math.max(0, Math.min(e, dimSize));

  const newDimSize = Math.max(0, Math.ceil((e - s) / st));
  const offsetDelta = s * strides[d]!;

  const newSizes = [...sizes];
  const newStrides = [...strides];
  newSizes[d] = newDimSize;
  newStrides[d] = strides[d]! * st;

  return { sizes: newSizes, strides: newStrides, offsetDelta };
}

export function computeUnsqueeze(sizes: Shape, strides: Shape, dim: number): ViewResult {
  const ndim = sizes.length;
  const d = dim < 0 ? ndim + 1 + dim : dim;
  const newSizes = [...sizes];
  const newStrides = [...strides];
  const insertStride = d < ndim ? sizes[d]! * strides[d]! : 1;
  newSizes.splice(d, 0, 1);
  newStrides.splice(d, 0, insertStride);
  return { sizes: newSizes, strides: newStrides };
}

export function computeSqueeze(sizes: Shape, strides: Shape, dim?: number | null): ViewResult {
  if (dim !== undefined && dim !== null) {
    const ndim = sizes.length;
    const d = dim < 0 ? ndim + dim : dim;
    if (sizes[d] !== 1) return { sizes: [...sizes], strides: [...strides] };
    const newSizes = [...sizes];
    const newStrides = [...strides];
    newSizes.splice(d, 1);
    newStrides.splice(d, 1);
    return { sizes: newSizes, strides: newStrides };
  }

  const newSizes = [];
  const newStrides = [];
  for (let i = 0; i < sizes.length; i++) {
    if (sizes[i] !== 1) {
      newSizes.push(sizes[i]);
      newStrides.push(strides[i]);
    }
  }
  return { sizes: newSizes, strides: newStrides };
}

export function computeNarrow(sizes: Shape, strides: Shape, dim: number, start: number, length: number): OffsetViewResult {
  const ndim = sizes.length;
  const d = dim < 0 ? ndim + dim : dim;
  const newSizes = [...sizes];
  newSizes[d] = length;
  const offsetDelta = start * strides[d]!;
  return { sizes: newSizes, strides: [...strides], offsetDelta };
}

export function computeSelect(sizes: Shape, strides: Shape, dim: number, index: number): OffsetViewResult {
  const ndim = sizes.length;
  const d = dim < 0 ? ndim + dim : dim;
  const idx = index < 0 ? sizes[d]! + index : index;
  const newSizes = [];
  const newStrides = [];
  for (let i = 0; i < ndim; i++) {
    if (i === d) continue;
    newSizes.push(sizes[i]!);
    newStrides.push(strides[i]!);
  }
  const offsetDelta = idx * strides[d]!;
  return { sizes: newSizes, strides: newStrides, offsetDelta };
}
