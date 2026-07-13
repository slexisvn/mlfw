export type Shape = readonly number[];

export function normalizeAxis(axis: number, rank: number): number {
  return axis < 0 ? rank + axis : axis;
}

export function computeStrides(sizes: Shape): number[] {
  const ndim = sizes.length;
  const strides = new Array(ndim);
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= sizes[i]!;
  }
  return strides;
}

export function computeNumel(sizes: Shape): number {
  let n = 1;
  for (let i = 0; i < sizes.length; i++) n *= sizes[i]!;
  return n;
}

export function isContiguous(sizes: Shape, strides: Shape): boolean {
  const ndim = sizes.length;
  if (ndim === 0) return true;
  let expected = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    const size = sizes[i]!;
    if (size === 0) return true;
    if (size !== 1 && strides[i] !== expected) return false;
    expected *= size;
  }
  return true;
}

export function storageSize(sizes: Shape, strides: Shape, offset: number): number {
  const ndim = sizes.length;
  if (ndim === 0) return offset + 1;
  let maxOffset = offset;
  for (let i = 0; i < ndim; i++) {
    const size = sizes[i]!;
    const stride = strides[i]!;
    if (size === 0) return 0;
    maxOffset += (size - 1) * Math.abs(stride);
  }
  return maxOffset + 1;
}

export function broadcastShapes(a: Shape, b: Shape): number[] | null {
  const aLen = a.length;
  const bLen = b.length;
  const maxLen = Math.max(aLen, bLen);
  const result = new Array(maxLen);
  for (let i = 0; i < maxLen; i++) {
    const da = i < aLen ? a[aLen - 1 - i] : 1;
    const db = i < bLen ? b[bLen - 1 - i] : 1;
    if (da === db) {
      result[maxLen - 1 - i] = da;
    } else if (da === 1) {
      result[maxLen - 1 - i] = db;
    } else if (db === 1) {
      result[maxLen - 1 - i] = da;
    } else {
      return null;
    }
  }
  return result;
}

export function matmulOutputShape(a: Shape, b: Shape): number[] | null {
  const aRank = a.length;
  const bRank = b.length;
  if (aRank === 1 && bRank === 1) return [];
  const aShape = aRank === 1 ? [1, a[0]!] : [...a];
  const bShape = bRank === 1 ? [b[0]!, 1] : [...b];
  const aR = aShape.length;
  const bR = bShape.length;
  const batch = broadcastShapes(aShape.slice(0, aR - 2), bShape.slice(0, bR - 2));
  if (batch === null) return null;
  const out = [...batch, aShape[aR - 2], bShape[bR - 1]];
  const drop = new Set();
  if (bRank === 1) drop.add(out.length - 1);
  if (aRank === 1) drop.add(out.length - 2);
  return out.filter((_, i) => !drop.has(i));
}

export function broadcastShapesMulti(shapes: readonly Shape[]): number[] | null {
  if (shapes.length === 0) return [];
  let result: number[] = [...shapes[0]!];
  for (let i = 1; i < shapes.length; i++) {
    const next = broadcastShapes(result, shapes[i]!);
    if (!next) return null;
    result = next;
  }
  return result;
}

export function broadcastStrides(sizes: Shape, strides: Shape, targetShape: Shape): number[] {
  const srcNdim = sizes.length;
  const dstNdim = targetShape.length;
  const dimDiff = dstNdim - srcNdim;
  const result = new Array(dstNdim);
  for (let i = 0; i < dstNdim; i++) {
    const srcIdx = i - dimDiff;
    if (srcIdx < 0 || sizes[srcIdx] === 1) {
      result[i] = 0;
    } else {
      result[i] = strides[srcIdx]!;
    }
  }
  return result;
}

export function inferReshape(sizes: Shape, strides: Shape, newSizes: Shape): { sizes: number[]; strides: number[]; needsCopy: boolean } | null {
  const numel = computeNumel(sizes);
  let inferDim = -1;
  let knownNumel = 1;
  const resolved = new Array(newSizes.length);

  for (let i = 0; i < newSizes.length; i++) {
    const size = newSizes[i]!;
    if (size === -1) {
      if (inferDim !== -1) return null;
      inferDim = i;
      resolved[i] = -1;
    } else {
      resolved[i] = size;
      knownNumel *= size;
    }
  }

  if (inferDim !== -1) {
    if (knownNumel === 0) return null;
    resolved[inferDim] = (numel / knownNumel) | 0;
    if (resolved[inferDim] * knownNumel !== numel) return null;
  } else if (computeNumel(resolved) !== numel) {
    return null;
  }

  if (isContiguous(sizes, strides)) {
    return { sizes: resolved, strides: computeStrides(resolved), needsCopy: false };
  }

  return { sizes: resolved, strides: computeStrides(resolved), needsCopy: true };
}

export function flatIndex(indices: Shape, strides: Shape, offset: number): number {
  let idx = offset;
  for (let i = 0; i < indices.length; i++) idx += indices[i]! * strides[i]!;
  return idx;
}

export function expandShape(sizes: Shape, targetShape: Shape): number[] | null {
  const srcNdim = sizes.length;
  const dstNdim = targetShape.length;
  if (srcNdim > dstNdim) return null;
  const dimDiff = dstNdim - srcNdim;
  const result = new Array(dstNdim);
  for (let i = 0; i < dstNdim; i++) {
    const srcIdx = i - dimDiff;
    if (srcIdx < 0) {
      result[i] = targetShape[i]!;
    } else if (sizes[srcIdx] === 1 || sizes[srcIdx] === targetShape[i]) {
      result[i] = targetShape[i]!;
    } else if (targetShape[i] === -1) {
      result[i] = sizes[srcIdx]!;
    } else {
      return null;
    }
  }
  return result;
}
