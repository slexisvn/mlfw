type OverlapImpl = {
  sizes(): readonly number[];
  strides(): readonly number[];
  storageOffset(): number;
  storage(): { impl: object } | null;
};

export function hasInternalOverlap(impl: OverlapImpl): boolean {
  const sizes = impl.sizes();
  const strides = impl.strides();
  const ndim = sizes.length;

  if (ndim === 0) return false;

  for (let i = 0; i < ndim; i++) {
    if (sizes[i]! > 1 && strides[i] === 0) return true;
  }

  for (let i = 0; i < ndim; i++) {
    for (let j = i + 1; j < ndim; j++) {
      if (sizes[i]! > 1 && sizes[j]! > 1 &&
          strides[i] !== 0 && strides[j] !== 0) {
        const si = Math.abs(strides[i]!);
        const sj = Math.abs(strides[j]!);
        const smaller = Math.min(si, sj);
        const larger = Math.max(si, sj);
        if (larger % smaller === 0) {
          const ratio = larger / smaller;
          const sizeSmaller = si < sj ? sizes[i]! : sizes[j]!;
          if (ratio < sizeSmaller) return true;
        }
      }
    }
  }

  return false;
}

export function computeStorageRange(impl: OverlapImpl): { start: number; end: number } {
  const sizes = impl.sizes();
  const strides = impl.strides();
  const ndim = sizes.length;
  const offset = impl.storageOffset();

  if (ndim === 0) return { start: offset, end: offset + 1 };

  let minOffset = offset;
  let maxOffset = offset;

  for (let i = 0; i < ndim; i++) {
    if (sizes[i] === 0) return { start: offset, end: offset };
    const s = strides[i]!;
    if (s >= 0) {
      maxOffset += (sizes[i]! - 1) * s;
    } else {
      minOffset += (sizes[i]! - 1) * s;
    }
  }

  return { start: minOffset, end: maxOffset + 1 };
}

export function tensorsOverlap(implA: OverlapImpl, implB: OverlapImpl): boolean {
  const storageA = implA.storage();
  const storageB = implB.storage();

  if (!storageA || !storageB) return false;
  if (storageA.impl !== storageB.impl) return false;

  const rangeA = computeStorageRange(implA);
  const rangeB = computeStorageRange(implB);

  return rangeA.start < rangeB.end && rangeB.start < rangeA.end;
}
