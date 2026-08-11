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
    const stack = [];
    for (let i = this.children.length - 1; i >= 0; i--) stack.push(this.children[i]);
    while (stack.length > 0) {
      const c = stack.pop();
      if (c.isBlock) result.push(c);
      else for (let i = c.children.length - 1; i >= 0; i--) stack.push(c.children[i]);
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
    this._loopSRefs = new Set();
    this._blockSRefs = new Set();
    this.root = this._build(primFunc.body, null);
  }

  _register(sref) {
    this._nodeToSRef.set(sref.node, sref);
    if (sref.isLoop) {
      this._loopSRefs.add(sref);
    } else if (sref.isBlock) {
      this._blockSRefs.add(sref);
      this._blockNameToSRef.set(sref.node.name, sref);
    }
  }

  _unregisterSubtree(sref) {
    const stack = [sref];
    while (stack.length > 0) {
      const s = stack.pop();
      this._nodeToSRef.delete(s.node);
      if (s.isLoop) {
        this._loopSRefs.delete(s);
      } else if (s.isBlock) {
        this._blockSRefs.delete(s);
        if (this._blockNameToSRef.get(s.node.name) === s) this._blockNameToSRef.delete(s.node.name);
      }
      for (const c of s.children) stack.push(c);
    }
  }

  _buildSubtree(rootNode, parent) {
    const top = [];
    const stack = [{ node: rootNode, parentSRef: parent, isTop: true }];
    while (stack.length > 0) {
      const { node, parentSRef, isTop } = stack.pop();
      if (!node) continue;
      switch (node.type) {
        case 'ForNode':
        case 'BlockNode': {
          const sref = new SRef(node, parentSRef);
          this._register(sref);
          if (isTop) top.push(sref);
          else parentSRef.children.push(sref);
          stack.push({ node: node.body, parentSRef: sref, isTop: false });
          if (node.type === 'BlockNode' && node.initBody) {
            stack.push({ node: node.initBody, parentSRef: sref, isTop: false });
          }
          break;
        }
        case 'SeqNode':
          for (let i = node.stmts.length - 1; i >= 0; i--) stack.push({ node: node.stmts[i], parentSRef, isTop });
          break;
        case 'IfThenElseNode':
          if (node.elseBody) stack.push({ node: node.elseBody, parentSRef, isTop });
          stack.push({ node: node.thenBody, parentSRef, isTop });
          break;
        case 'AllocateNode':
        case 'LetStmtNode':
          stack.push({ node: node.body, parentSRef, isTop });
          break;
      }
    }
    return top;
  }

  _build(rootNode, rootParent) {
    const top = this._buildSubtree(rootNode, rootParent);
    if (rootParent) for (const s of top) rootParent.children.push(s);
    return this._nodeToSRef.get(rootNode) || null;
  }

  getSRef(node) {
    return this._nodeToSRef.get(node) || null;
  }

  getBlockSRef(name) {
    return this._blockNameToSRef.get(name) || null;
  }

  allBlocks() {
    return [...this._blockSRefs];
  }

  allLoops() {
    return [...this._loopSRefs];
  }

  loopsOf(blockName) {
    const blockSRef = this._blockNameToSRef.get(blockName);
    if (!blockSRef) return [];
    return blockSRef.loopAncestors().filter(s => s.node.kind !== ForKind.RECURRENCE).reverse();
  }

  rebuildFrom(rootNode) {
    this._nodeToSRef.clear();
    this._blockNameToSRef.clear();
    this._loopSRefs.clear();
    this._blockSRefs.clear();
    this.root = this._build(rootNode, null);
  }

  replaceNode(oldNode, newNode) {
    const oldSRef = this._nodeToSRef.get(oldNode);
    if (!oldSRef) return false;
    const parent = oldSRef.parent;
    const wasRoot = this.root === oldSRef;
    this._unregisterSubtree(oldSRef);
    const top = this._buildSubtree(newNode, parent);
    if (parent) {
      const idx = parent.children.indexOf(oldSRef);
      if (idx >= 0) parent.children.splice(idx, 1, ...top);
      else for (const s of top) parent.children.push(s);
    } else if (wasRoot) {
      this.root = this._nodeToSRef.get(newNode) || null;
    }
    return true;
  }

  removeNode(node) {
    const sref = this._nodeToSRef.get(node);
    if (!sref) return false;
    const parent = sref.parent;
    const wasRoot = this.root === sref;
    this._unregisterSubtree(sref);
    if (parent) {
      const idx = parent.children.indexOf(sref);
      if (idx >= 0) parent.children.splice(idx, 1);
    } else if (wasRoot) {
      this.root = null;
    }
    return true;
  }
}
