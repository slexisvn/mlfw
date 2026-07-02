export function covHost(a, rows, cols) {
  const mean = new Float64Array(cols);
  for (let j = 0; j < cols; j++) {
    let s = 0;
    for (let i = 0; i < rows; i++) s += a[i * cols + j];
    mean[j] = s / rows;
  }
  const denom = rows > 1 ? rows - 1 : 1;
  const c = new Float64Array(cols * cols);
  for (let i = 0; i < cols; i++) {
    for (let j = i; j < cols; j++) {
      let s = 0;
      for (let r = 0; r < rows; r++) s += (a[r * cols + i] - mean[i]) * (a[r * cols + j] - mean[j]);
      const v = s / denom;
      c[i * cols + j] = v;
      c[j * cols + i] = v;
    }
  }
  return c;
}
