import { BlockArgument } from './value.js';

export class Block {
  constructor(argTypes = []) {
    this.parentRegion = null;
    this._head = null;
    this._tail = null;
    this._size = 0;
    this.arguments = argTypes.map((type, i) => new BlockArgument(type, this, i));
  }

  get size() { return this._size; }
  get isEmpty() { return this._head === null; }
  get firstOp() { return this._head; }
  get lastOp() { return this._tail; }

  get parentOp() {
    return this.parentRegion ? this.parentRegion.parentOp : null;
  }

  getArgument(index) {
    return this.arguments[index];
  }

  addArgument(type) {
    const arg = new BlockArgument(type, this, this.arguments.length);
    this.arguments.push(arg);
    return arg;
  }

  pushOp(op) {
    op.parentBlock = this;
    op._prev = this._tail;
    op._next = null;
    if (this._tail) {
      this._tail._next = op;
    } else {
      this._head = op;
    }
    this._tail = op;
    this._size++;
  }

  insertBefore(op, ref) {
    op.parentBlock = this;
    op._prev = ref._prev;
    op._next = ref;
    if (ref._prev) {
      ref._prev._next = op;
    } else {
      this._head = op;
    }
    ref._prev = op;
    this._size++;
  }

  insertAfter(op, ref) {
    op.parentBlock = this;
    op._prev = ref;
    op._next = ref._next;
    if (ref._next) {
      ref._next._prev = op;
    } else {
      this._tail = op;
    }
    ref._next = op;
    this._size++;
  }

  removeOp(op) {
    if (op.parentBlock !== this) return;
    if (op._prev) {
      op._prev._next = op._next;
    } else {
      this._head = op._next;
    }
    if (op._next) {
      op._next._prev = op._prev;
    } else {
      this._tail = op._prev;
    }
    op._prev = null;
    op._next = null;
    op.parentBlock = null;
    this._size--;
  }

  *ops() {
    let cur = this._head;
    while (cur) {
      const next = cur._next;
      yield cur;
      cur = next;
    }
  }

  *opsReverse() {
    let cur = this._tail;
    while (cur) {
      const prev = cur._prev;
      yield cur;
      cur = prev;
    }
  }

  opsArray() {
    const result = [];
    let cur = this._head;
    while (cur) {
      result.push(cur);
      cur = cur._next;
    }
    return result;
  }

  [Symbol.iterator]() {
    return this.ops();
  }
}

export class Region {
  constructor(blocks = []) {
    this.parentOp = null;
    this.blocks = [];
    for (let i = 0; i < blocks.length; i++) {
      this.addBlock(blocks[i]);
    }
  }

  get entryBlock() { return this.blocks[0] || null; }
  get isEmpty() { return this.blocks.length === 0; }

  addBlock(block) {
    block.parentRegion = this;
    this.blocks.push(block);
    return block;
  }

  insertBlock(index, block) {
    block.parentRegion = this;
    this.blocks.splice(index, 0, block);
  }

  removeBlock(block) {
    const idx = this.blocks.indexOf(block);
    if (idx !== -1) {
      this.blocks.splice(idx, 1);
      block.parentRegion = null;
    }
  }

  *[Symbol.iterator]() {
    yield* this.blocks;
  }
}
