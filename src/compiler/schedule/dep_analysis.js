export function rangesOverlap(ra, rb) {
  if (!ra || !rb || ra.length !== rb.length) return true;
  for (let i = 0; i < ra.length; i++) {
    const a = ra[i], b = rb[i];
    if (a === null || b === null) continue;
    if (a[0] + a[1] <= b[0] || b[0] + b[1] <= a[0]) return false;
  }
  return true;
}
