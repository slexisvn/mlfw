let _valueIdCounter = 0;

export function resetValueCounter() {
  _valueIdCounter = 0;
}

export class UseLink {
  constructor(user, operandIndex) {
    this.user = user;
    this.operandIndex = operandIndex;
    this.prev = null;
    this.next = null;
  }
}

export class Value {
  constructor(type, definingOp = null, resultIndex = 0) {
    this.type = type;
    this.definingOp = definingOp;
    this.resultIndex = resultIndex;
    this.id = _valueIdCounter++;
    this._useHead = null;
    this._useTail = null;
    this._useCount = 0;
  }

  get hasUses() { return this._useHead !== null; }
  get useCount() { return this._useCount; }

  addUse(link) {
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

  removeUse(link) {
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

  *uses() {
    let cur = this._useHead;
    while (cur) {
      const next = cur.next;
      yield cur;
      cur = next;
    }
  }

  getUsers() {
    const users = [];
    let cur = this._useHead;
    while (cur) {
      users.push(cur.user);
      cur = cur.next;
    }
    return users;
  }

  replaceAllUsesWith(newValue) {
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

  _owningFunction() {
    return this.definingOp ? this.definingOp.getParentFunction() : null;
  }

  isBlockArgument() { return false; }
}

export class BlockArgument extends Value {
  constructor(type, ownerBlock, argIndex) {
    super(type, null, 0);
    this.ownerBlock = ownerBlock;
    this.argIndex = argIndex;
  }

  _owningFunction() {
    return this.ownerBlock ? this.ownerBlock._owningFunction() : null;
  }

  isBlockArgument() { return true; }
}
