import { ForKind } from '../ir/tensor/nodes.js';

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

  _build(rootNode, rootParent) {
    const stack = [{ node: rootNode, parentSRef: rootParent }];
    while (stack.length > 0) {
      const { node, parentSRef } = stack.pop();
      if (!node) continue;
      switch (node.type) {
        case 'ForNode': {
          const sref = new SRef(node, parentSRef);
          this._nodeToSRef.set(node, sref);
          this._loopSRefs.push(sref);
          if (parentSRef) parentSRef.children.push(sref);
          stack.push({ node: node.body, parentSRef: sref });
          break;
        }
        case 'BlockNode': {
          const sref = new SRef(node, parentSRef);
          this._nodeToSRef.set(node, sref);
          this._blockNameToSRef.set(node.name, sref);
          this._blockSRefs.push(sref);
          if (parentSRef) parentSRef.children.push(sref);
          stack.push({ node: node.body, parentSRef: sref });
          if (node.initBody) stack.push({ node: node.initBody, parentSRef: sref });
          break;
        }
        case 'SeqNode':
          for (let i = node.stmts.length - 1; i >= 0; i--) stack.push({ node: node.stmts[i], parentSRef });
          break;
        case 'IfThenElseNode':
          if (node.elseBody) stack.push({ node: node.elseBody, parentSRef });
          stack.push({ node: node.thenBody, parentSRef });
          break;
        case 'AllocateNode':
        case 'LetStmtNode':
          stack.push({ node: node.body, parentSRef });
          break;
      }
    }
    return this._nodeToSRef.get(rootNode) || null;
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
    return blockSRef.loopAncestors().filter(s => s.node.kind !== ForKind.RECURRENCE).reverse();
  }

  rebuildFrom(rootNode) {
    this._nodeToSRef.clear();
    this._blockNameToSRef.clear();
    this._loopSRefs.length = 0;
    this._blockSRefs.length = 0;
    this.root = this._build(rootNode, null);
  }

  replaceNode(oldNode, newNode) {
    const oldSRef = this._nodeToSRef.get(oldNode);
    if (!oldSRef) return;

    const newSRef = new SRef(newNode, oldSRef.parent);
    newSRef.children = oldSRef.children;
    for (const child of newSRef.children) child.parent = newSRef;

    if (oldSRef.parent) {
      const idx = oldSRef.parent.children.indexOf(oldSRef);
      if (idx >= 0) oldSRef.parent.children[idx] = newSRef;
    } else {
      this.root = newSRef;
    }

    this._nodeToSRef.delete(oldNode);
    this._nodeToSRef.set(newNode, newSRef);

    if (newNode.type === 'BlockNode') {
      this._blockNameToSRef.delete(oldNode.name);
      this._blockNameToSRef.set(newNode.name, newSRef);
    }
  }
}
