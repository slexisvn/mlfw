export function ramp(shape, stride) {
  let index = 0;
  const fill = (axis) =>
    Array.from({ length: shape[axis] }, () =>
      axis + 1 < shape.length ? fill(axis + 1) : ((index++ * stride) % 11) / 8 - 0.5,
    );
  return shape.length === 0 ? ((index++ * stride) % 11) / 8 - 0.5 : fill(0);
}

export function rowIndex(rows, cols) {
  return Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, () => i));
}

export function colIndex(rows, cols) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, (_, j) => j));
}
