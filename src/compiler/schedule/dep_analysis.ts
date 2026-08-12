export type Extent = readonly [start: number, size: number];
export type Region = readonly (Extent | null)[];
export type MutableRegion = ([number, number] | null)[];

export function rangesOverlap(ra: Region | null | undefined, rb: Region | null | undefined): boolean {
  if (!ra || !rb || ra.length !== rb.length) return true;
  for (let i = 0; i < ra.length; i++) {
    const a = ra[i], b = rb[i];
    if (a === null || b === null) continue;
    if (a[0] + a[1] <= b[0] || b[0] + b[1] <= a[0]) return false;
  }
  return true;
}

export function rangesContain(outer: Region | null | undefined, inner: Region | null | undefined): boolean {
  if (!outer || !inner || outer.length !== inner.length) return false;
  for (let i = 0; i < outer.length; i++) {
    const o = outer[i], n = inner[i];
    if (o === null || n === null) return false;
    if (n[0] < o[0] || n[0] + n[1] > o[0] + o[1]) return false;
  }
  return true;
}

export function regionHull(regions: Iterable<Region | null | undefined>): MutableRegion | null {
  let hull: MutableRegion | null = null;
  for (const region of regions) {
    if (!region) return null;
    if (hull === null) { hull = region.map((r) => (r ? [r[0], r[1]] as [number, number] : null)); continue; }
    if (hull.length !== region.length) return null;
    for (let i = 0; i < hull.length; i++) {
      const h = hull[i], r = region[i];
      if (h === null || r === null) { hull[i] = null; continue; }
      const lo = Math.min(h[0], r[0]);
      const hi = Math.max(h[0] + h[1], r[0] + r[1]);
      hull[i] = [lo, hi - lo];
    }
  }
  return hull;
}
