import type { Operation } from '../graph/operation.js';
import type { IRBuilder } from '../graph/builder.js';

export class Pattern {
  name: string;
  benefit: number;
  rootOpName: string | null;
  why: string | null;

  constructor(name: string, benefit = 1, why: string | null = null) {
    this.name = name;
    this.benefit = benefit;
    this.rootOpName = null;
    this.why = why;
  }

  match(op: Operation): boolean { return false; }
  rewrite(op: Operation, builder: IRBuilder): boolean { return false; }
}

export class PatternSet {
  patterns: Pattern[];
  private _byOp: Map<string, Pattern[]>;
  private _generic: Pattern[];
  private _sorted: boolean;

  constructor() {
    this.patterns = [];
    this._byOp = new Map();
    this._generic = [];
    this._sorted = false;
  }

  add(pattern: Pattern): void {
    this.patterns.push(pattern);
    this._sorted = false;
    if (pattern.rootOpName) {
      let list = this._byOp.get(pattern.rootOpName);
      if (!list) {
        list = [];
        this._byOp.set(pattern.rootOpName, list);
      }
      list.push(pattern);
    } else {
      this._generic.push(pattern);
    }
  }

  _ensureSorted(): void {
    if (this._sorted) return;
    const cmp = (a: Pattern, b: Pattern): number => b.benefit - a.benefit;
    for (const [, list] of this._byOp) {
      list.sort(cmp);
    }
    this._generic.sort(cmp);
    this._sorted = true;
  }

  get(): Pattern[] {
    return [...this.patterns].sort((a, b) => b.benefit - a.benefit);
  }

  getForOp(opName: string): Pattern[] {
    this._ensureSorted();
    const specific = this._byOp.get(opName);
    if (!specific) return this._generic;
    if (this._generic.length === 0) return specific;
    const merged = new Array<Pattern>(specific.length + this._generic.length);
    let si = 0, gi = 0, mi = 0;
    while (si < specific.length && gi < this._generic.length) {
      if (specific[si].benefit >= this._generic[gi].benefit) {
        merged[mi++] = specific[si++];
      } else {
        merged[mi++] = this._generic[gi++];
      }
    }
    while (si < specific.length) merged[mi++] = specific[si++];
    while (gi < this._generic.length) merged[mi++] = this._generic[gi++];
    return merged;
  }

  hasPatterns(): boolean {
    return this.patterns.length > 0;
  }
}
