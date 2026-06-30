export class GraphCycles {
  constructor(n, edges) {
    this._n = n;
    this._parent = new Int32Array(n);
    this._rank = new Int32Array(n);
    this._nodeAtRank = new Int32Array(n);
    this._out = new Array(n);
    this._in = new Array(n);
    for (let i = 0; i < n; i++) {
      this._parent[i] = i;
      this._rank[i] = i;
      this._nodeAtRank[i] = i;
      this._out[i] = new Set();
      this._in[i] = new Set();
    }
    if (edges) {
      for (const [u, v] of edges) {
        if (u === v) continue;
        this._out[u].add(v);
        this._in[v].add(u);
      }
    }
  }

  find(x) {
    const parent = this._parent;
    let root = x;
    while (parent[root] !== root) root = parent[root];
    while (parent[x] !== root) {
      const next = parent[x];
      parent[x] = root;
      x = next;
    }
    return root;
  }

  wouldCreateCycle(a, b) {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return false;
    const lo = this._rank[ra] < this._rank[rb] ? ra : rb;
    const hi = lo === ra ? rb : ra;
    return this._hasIntermediatePath(lo, hi);
  }

  _hasIntermediatePath(lo, hi) {
    const limit = this._rank[hi];
    const visited = new Set([lo]);
    const stack = [lo];
    while (stack.length > 0) {
      const u = stack.pop();
      for (const raw of this._out[u]) {
        const v = this.find(raw);
        if (v === u || v === lo) continue;
        if (v === hi) {
          if (u !== lo) return true;
          continue;
        }
        if (this._rank[v] >= limit) continue;
        if (visited.has(v)) continue;
        visited.add(v);
        stack.push(v);
      }
    }
    return false;
  }

  merge(a, b) {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return ra;

    const lo = this._rank[ra] < this._rank[rb] ? ra : rb;
    const hi = lo === ra ? rb : ra;
    const loRank = this._rank[lo];
    const hiRank = this._rank[hi];

    const degRa = this._out[ra].size + this._in[ra].size;
    const degRb = this._out[rb].size + this._in[rb].size;
    const s = degRa >= degRb ? ra : rb;
    const t = s === ra ? rb : ra;

    const outS = this._out[s];
    const inS = this._in[s];
    for (const raw of this._in[t]) {
      const p = this.find(raw);
      if (p === s || p === t) continue;
      this._out[p].delete(t);
      this._out[p].add(s);
      inS.add(p);
    }
    for (const raw of this._out[t]) {
      const q = this.find(raw);
      if (q === s || q === t) continue;
      this._in[q].delete(t);
      this._in[q].add(s);
      outS.add(q);
    }
    outS.delete(t);
    inS.delete(t);
    outS.delete(s);
    inS.delete(s);

    this._parent[t] = s;
    this._nodeAtRank[this._rank[t]] = -1;
    this._reorder(loRank, hiRank);
    return s;
  }

  _reorder(loRank, hiRank) {
    const slots = [];
    const nodes = [];
    for (let r = loRank; r <= hiRank; r++) {
      const nd = this._nodeAtRank[r];
      if (nd < 0) continue;
      if (this.find(nd) !== nd) {
        this._nodeAtRank[r] = -1;
        continue;
      }
      slots.push(r);
      nodes.push(nd);
    }
    if (nodes.length <= 1) {
      for (let k = 0; k < nodes.length; k++) {
        this._rank[nodes[k]] = slots[k];
        this._nodeAtRank[slots[k]] = nodes[k];
      }
      return;
    }

    const inSet = new Set(nodes);
    const indeg = new Map();
    for (const nd of nodes) indeg.set(nd, 0);
    for (const nd of nodes) {
      for (const raw of this._out[nd]) {
        const v = this.find(raw);
        if (v !== nd && inSet.has(v)) indeg.set(v, indeg.get(v) + 1);
      }
    }

    const queue = [];
    for (const nd of nodes) {
      if (indeg.get(nd) === 0) queue.push(nd);
    }
    const ordered = [];
    let qi = 0;
    while (qi < queue.length) {
      const u = queue[qi++];
      ordered.push(u);
      for (const raw of this._out[u]) {
        const v = this.find(raw);
        if (v !== u && inSet.has(v)) {
          const d = indeg.get(v) - 1;
          indeg.set(v, d);
          if (d === 0) queue.push(v);
        }
      }
    }

    for (let k = 0; k < ordered.length; k++) {
      const nd = ordered[k];
      const r = slots[k];
      this._rank[nd] = r;
      this._nodeAtRank[r] = nd;
    }
  }
}
