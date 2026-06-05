const INLINE_CAPACITY = 6;

export class DimVector {
  constructor(source) {
    this._inline = new Int32Array(INLINE_CAPACITY);
    this._heap = null;
    this._length = 0;

    if (source) {
      if (typeof source === 'number') {
        this._ensureCapacity(source);
        this._length = source;
      } else {
        this._length = source.length;
        if (this._length > INLINE_CAPACITY) {
          this._heap = new Int32Array(this._length);
          for (let i = 0; i < this._length; i++) this._heap[i] = source[i];
        } else {
          for (let i = 0; i < this._length; i++) this._inline[i] = source[i];
        }
      }
    }
  }

  get length() {
    return this._length;
  }

  get(i) {
    return this._heap ? this._heap[i] : this._inline[i];
  }

  set(i, val) {
    if (this._heap) this._heap[i] = val;
    else this._inline[i] = val;
  }

  push(val) {
    const idx = this._length;
    this._ensureCapacity(idx + 1);
    this._length = idx + 1;
    this.set(idx, val);
  }

  pop() {
    if (this._length === 0) return undefined;
    this._length--;
    return this.get(this._length);
  }

  insert(i, val) {
    this._ensureCapacity(this._length + 1);
    const data = this._data();
    for (let j = this._length; j > i; j--) data[j] = data[j - 1];
    data[i] = val;
    this._length++;
  }

  remove(i) {
    const data = this._data();
    const val = data[i];
    for (let j = i; j < this._length - 1; j++) data[j] = data[j + 1];
    this._length--;
    return val;
  }

  toArray() {
    const result = new Array(this._length);
    const data = this._data();
    for (let i = 0; i < this._length; i++) result[i] = data[i];
    return result;
  }

  toInt32Array() {
    const data = this._data();
    return data.slice(0, this._length);
  }

  clone() {
    return new DimVector(this.toArray());
  }

  equals(other) {
    if (this._length !== other._length) return false;
    for (let i = 0; i < this._length; i++) {
      if (this.get(i) !== other.get(i)) return false;
    }
    return true;
  }

  fill(val) {
    const data = this._data();
    for (let i = 0; i < this._length; i++) data[i] = val;
  }

  reverse() {
    const data = this._data();
    let lo = 0, hi = this._length - 1;
    while (lo < hi) {
      const tmp = data[lo];
      data[lo] = data[hi];
      data[hi] = tmp;
      lo++;
      hi--;
    }
  }

  *[Symbol.iterator]() {
    for (let i = 0; i < this._length; i++) yield this.get(i);
  }

  _data() {
    return this._heap || this._inline;
  }

  _ensureCapacity(needed) {
    if (needed <= INLINE_CAPACITY && !this._heap) return;
    if (this._heap && needed <= this._heap.length) return;
    const newCap = Math.max(needed, (this._heap ? this._heap.length : INLINE_CAPACITY) * 2);
    const newData = new Int32Array(newCap);
    const old = this._data();
    for (let i = 0; i < this._length; i++) newData[i] = old[i];
    this._heap = newData;
  }
}
