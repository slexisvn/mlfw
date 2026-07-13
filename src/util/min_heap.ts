export class MinHeap<T> {
  private readonly _items: T[];
  private readonly _compare: (a: T, b: T) => number;

  constructor(compare: (a: T, b: T) => number) {
    this._items = [];
    this._compare = compare;
  }

  get size(): number { return this._items.length; }

  peek(): T | null { return this._items.length > 0 ? this._items[0] : null; }

  push(item: T): void {
    const items = this._items;
    items.push(item);
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this._compare(items[i], items[parent]) >= 0) break;
      [items[i], items[parent]] = [items[parent], items[i]];
      i = parent;
    }
  }

  pop(): T | null {
    const items = this._items;
    if (items.length === 0) return null;
    const top = items[0];
    const last = items.pop()!;
    if (items.length > 0) {
      items[0] = last;
      let i = 0;
      const n = items.length;
      for (;;) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        let smallest = i;
        if (l < n && this._compare(items[l], items[smallest]) < 0) smallest = l;
        if (r < n && this._compare(items[r], items[smallest]) < 0) smallest = r;
        if (smallest === i) break;
        [items[i], items[smallest]] = [items[smallest], items[i]];
        i = smallest;
      }
    }
    return top;
  }
}
