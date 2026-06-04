export class SRef {
  constructor(node, parent = null) {
    this.node = node;
    this.parent = parent;
    this.children = [];
  }

  get type() {
    return this.node.type;
  }

  get isLoop() {
    return this.node.type === 'ForNode';
  }

  get isBlock() {
    return this.node.type === 'BlockNode';
  }

  get isRoot() {
    return this.parent === null;
  }

  ancestors() {
    const result = [];
    let cur = this.parent;
    while (cur) {
      result.push(cur);
      cur = cur.parent;
    }
    return result;
  }

  loopAncestors() {
    const result = [];
    let cur = this.parent;
    while (cur) {
      if (cur.isLoop) result.push(cur);
      cur = cur.parent;
    }
    return result;
  }

  childBlocks() {
    const result = [];
    const stack = [...this.children];
    while (stack.length > 0) {
      const c = stack.pop();
      if (c.isBlock) result.push(c);
      else for (const cc of c.children) stack.push(cc);
    }
    return result;
  }

  childLoops() {
    return this.children.filter(c => c.isLoop);
  }
}

export class SRefTree {
  constructor(primFunc) {
    this._nodeToSRef = new Map();
    this._blockNameToSRef = new Map();
    this._loopSRefs = [];
    this._blockSRefs = [];
    this.root = this._build(primFunc.body, null);
  }

  _build(node, parentSRef) {
    if (!node) return null;

    switch (node.type) {
      case 'ForNode': {
        const sref = new SRef(node, parentSRef);
        this._nodeToSRef.set(node, sref);
        this._loopSRefs.push(sref);
        if (parentSRef) parentSRef.children.push(sref);
        this._build(node.body, sref);
        return sref;
      }
      case 'BlockNode': {
        const sref = new SRef(node, parentSRef);
        this._nodeToSRef.set(node, sref);
        this._blockNameToSRef.set(node.name, sref);
        this._blockSRefs.push(sref);
        if (parentSRef) parentSRef.children.push(sref);
        if (node.initBody) this._build(node.initBody, sref);
        this._build(node.body, sref);
        return sref;
      }
      case 'SeqNode':
        for (const s of node.stmts) this._build(s, parentSRef);
        return null;
      case 'IfThenElseNode':
        this._build(node.thenBody, parentSRef);
        if (node.elseBody) this._build(node.elseBody, parentSRef);
        return null;
      case 'AllocateNode':
        this._build(node.body, parentSRef);
        return null;
      case 'LetStmtNode':
        this._build(node.body, parentSRef);
        return null;
      default:
        return null;
    }
  }

  getSRef(node) {
    return this._nodeToSRef.get(node) || null;
  }

  getBlockSRef(name) {
    return this._blockNameToSRef.get(name) || null;
  }

  allBlocks() {
    return this._blockSRefs;
  }

  allLoops() {
    return this._loopSRefs;
  }

  loopsOf(blockName) {
    const blockSRef = this._blockNameToSRef.get(blockName);
    if (!blockSRef) return [];
    return blockSRef.loopAncestors().reverse();
  }
}
