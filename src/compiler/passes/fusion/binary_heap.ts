export type HeapEntry<T> = { priority: number; value: T };

export class MaxHeap<T = unknown> {
  private _items: HeapEntry<T>[];

  constructor() {
    this._items = [];
  }

  get size(): number {
    return this._items.length;
  }

  isEmpty(): boolean {
    return this._items.length === 0;
  }

  push(priority: number, value: T): void {
    const items = this._items;
    items.push({ priority, value });
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (items[parent].priority >= items[i].priority) break;
      const tmp = items[parent];
      items[parent] = items[i];
      items[i] = tmp;
      i = parent;
    }
  }

  pop(): T | undefined {
    const items = this._items;
    const n = items.length;
    if (n === 0) return undefined;
    const top = items[0];
    const last = items.pop() as HeapEntry<T>;
    if (n > 1) {
      items[0] = last;
      const size = items.length;
      let i = 0;
      for (;;) {
        const left = i * 2 + 1;
        const right = left + 1;
        let largest = i;
        if (left < size && items[left].priority > items[largest].priority) largest = left;
        if (right < size && items[right].priority > items[largest].priority) largest = right;
        if (largest === i) break;
        const tmp = items[largest];
        items[largest] = items[i];
        items[i] = tmp;
        i = largest;
      }
    }
    return top.value;
  }
}
