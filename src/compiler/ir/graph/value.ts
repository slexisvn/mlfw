import type { IRType, Shape } from './types.js';
import type { Operation } from './operation.js';
import type { Block } from './block.js';
import type { GraphFunction } from './function.js';

let _valueIdCounter = 0;

export function resetValueCounter(): void {
  _valueIdCounter = 0;
}

export class UseLink {
  user: Operation;
  operandIndex: number;
  prev: UseLink | null;
  next: UseLink | null;

  constructor(user: Operation, operandIndex: number) {
    this.user = user;
    this.operandIndex = operandIndex;
    this.prev = null;
    this.next = null;
  }
}

export class Value {
  type: IRType;
  declare symbolicShape?: Shape;
  definingOp: Operation | null;
  resultIndex: number;
  id: number;
  private _useHead: UseLink | null;
  private _useTail: UseLink | null;
  private _useCount: number;

  constructor(type: IRType, definingOp: Operation | null = null, resultIndex = 0) {
    this.type = type;
    this.definingOp = definingOp;
    this.resultIndex = resultIndex;
    this.id = _valueIdCounter++;
    this._useHead = null;
    this._useTail = null;
    this._useCount = 0;
  }

  get hasUses(): boolean { return this._useHead !== null; }
  get useCount(): number { return this._useCount; }

  addUse(link: UseLink): void {
    link.prev = this._useTail;
    link.next = null;
    if (this._useTail) {
      this._useTail.next = link;
    } else {
      this._useHead = link;
    }
    this._useTail = link;
    this._useCount++;
  }

  removeUse(link: UseLink): void {
    if (link.prev) {
      link.prev.next = link.next;
    } else {
      this._useHead = link.next;
    }
    if (link.next) {
      link.next.prev = link.prev;
    } else {
      this._useTail = link.prev;
    }
    link.prev = null;
    link.next = null;
    this._useCount--;
  }

  *uses(): Generator<UseLink, void, undefined> {
    let cur = this._useHead;
    while (cur) {
      const next = cur.next;
      yield cur;
      cur = next;
    }
  }

  getUsers(): Operation[] {
    const users: Operation[] = [];
    let cur = this._useHead;
    while (cur) {
      users.push(cur.user);
      cur = cur.next;
    }
    return users;
  }

  replaceAllUsesWith(newValue: Value): void {
    if (this === newValue) return;
    const hadUses = this._useHead !== null;
    let cur = this._useHead;
    while (cur) {
      cur.user.operands[cur.operandIndex] = newValue;
      cur = cur.next;
    }
    if (this._useHead) {
      if (newValue._useTail) {
        newValue._useTail.next = this._useHead;
        this._useHead.prev = newValue._useTail;
      } else {
        newValue._useHead = this._useHead;
      }
      newValue._useTail = this._useTail;
      newValue._useCount += this._useCount;
    }
    this._useHead = null;
    this._useTail = null;
    this._useCount = 0;
    if (hadUses) {
      const fn = this._owningFunction();
      if (fn) fn.bumpVersion();
    }
  }

  _owningFunction(): GraphFunction | null {
    return this.definingOp ? this.definingOp.getParentFunction() : null;
  }

  isBlockArgument(): boolean { return false; }
}

export class BlockArgument extends Value {
  ownerBlock: Block | null;
  argIndex: number;

  constructor(type: IRType, ownerBlock: Block | null, argIndex: number) {
    super(type, null, 0);
    this.ownerBlock = ownerBlock;
    this.argIndex = argIndex;
  }

  override _owningFunction(): GraphFunction | null {
    return this.ownerBlock ? this.ownerBlock._owningFunction() : null;
  }

  override isBlockArgument(): boolean { return true; }
}
