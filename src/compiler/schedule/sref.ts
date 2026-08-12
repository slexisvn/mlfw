import { ForKind } from '../ir/tensor/nodes.js';
import type { TirNode, PrimFunc, ForNode, BlockNode, SeqNode, IfThenElseNode, AllocateNode, LetStmtNode } from '../ir/tensor/nodes.js';

type BuildFrame = { node: TirNode | null | undefined; parentSRef: SRef | null; isTop: boolean };

export class SRef {
  node: TirNode;
  parent: SRef | null;
  children: SRef[];

  constructor(node: TirNode, parent: SRef | null = null) {
    this.node = node;
    this.parent = parent;
    this.children = [];
  }

  get type(): string {
    return this.node.type;
  }

  get isLoop(): boolean {
    return this.node.type === 'ForNode';
  }

  get isBlock(): boolean {
    return this.node.type === 'BlockNode';
  }

  get isRoot(): boolean {
    return this.parent === null;
  }

  ancestors(): SRef[] {
    const result: SRef[] = [];
    let cur = this.parent;
    while (cur) {
      result.push(cur);
      cur = cur.parent;
    }
    return result;
  }

  loopAncestors(): SRef[] {
    const result: SRef[] = [];
    let cur = this.parent;
    while (cur) {
      if (cur.isLoop) result.push(cur);
      cur = cur.parent;
    }
    return result;
  }

  childBlocks(): SRef[] {
    const result: SRef[] = [];
    const stack: SRef[] = [];
    for (let i = this.children.length - 1; i >= 0; i--) stack.push(this.children[i]);
    while (stack.length > 0) {
      const c = stack.pop() as SRef;
      if (c.isBlock) result.push(c);
      else for (let i = c.children.length - 1; i >= 0; i--) stack.push(c.children[i]);
    }
    return result;
  }

  childLoops(): SRef[] {
    return this.children.filter(c => c.isLoop);
  }
}

export class SRefTree {
  root: SRef | null;
  private _nodeToSRef: Map<TirNode, SRef>;
  private _blockNameToSRef: Map<string, SRef>;
  private _loopSRefs: Set<SRef>;
  private _blockSRefs: Set<SRef>;

  constructor(primFunc: PrimFunc) {
    this._nodeToSRef = new Map();
    this._blockNameToSRef = new Map();
    this._loopSRefs = new Set();
    this._blockSRefs = new Set();
    this.root = this._build(primFunc.body, null);
  }

  _register(sref: SRef): void {
    this._nodeToSRef.set(sref.node, sref);
    if (sref.isLoop) {
      this._loopSRefs.add(sref);
    } else if (sref.isBlock) {
      this._blockSRefs.add(sref);
      this._blockNameToSRef.set((sref.node as BlockNode).name, sref);
    }
  }

  _unregisterSubtree(sref: SRef): void {
    const stack: SRef[] = [sref];
    while (stack.length > 0) {
      const s = stack.pop() as SRef;
      this._nodeToSRef.delete(s.node);
      if (s.isLoop) {
        this._loopSRefs.delete(s);
      } else if (s.isBlock) {
        this._blockSRefs.delete(s);
        const bn = (s.node as BlockNode).name;
        if (this._blockNameToSRef.get(bn) === s) this._blockNameToSRef.delete(bn);
      }
      for (const c of s.children) stack.push(c);
    }
  }

  _buildSubtree(rootNode: TirNode | null | undefined, parent: SRef | null): SRef[] {
    const top: SRef[] = [];
    const stack: BuildFrame[] = [{ node: rootNode, parentSRef: parent, isTop: true }];
    while (stack.length > 0) {
      const { node, parentSRef, isTop } = stack.pop() as BuildFrame;
      if (!node) continue;
      switch (node.type) {
        case 'ForNode':
        case 'BlockNode': {
          const sref = new SRef(node, parentSRef);
          this._register(sref);
          if (isTop) top.push(sref);
          else (parentSRef as SRef).children.push(sref);
          stack.push({ node: (node as ForNode | BlockNode).body, parentSRef: sref, isTop: false });
          if (node.type === 'BlockNode' && (node as BlockNode).initBody) {
            stack.push({ node: (node as BlockNode).initBody, parentSRef: sref, isTop: false });
          }
          break;
        }
        case 'SeqNode': {
          const seq = node as SeqNode;
          for (let i = seq.stmts.length - 1; i >= 0; i--) stack.push({ node: seq.stmts[i], parentSRef, isTop });
        }
          break;
        case 'IfThenElseNode': {
          const ite = node as IfThenElseNode;
          if (ite.elseBody) stack.push({ node: ite.elseBody, parentSRef, isTop });
          stack.push({ node: ite.thenBody, parentSRef, isTop });
        }
          break;
        case 'AllocateNode':
        case 'LetStmtNode':
          stack.push({ node: (node as AllocateNode | LetStmtNode).body, parentSRef, isTop });
          break;
      }
    }
    return top;
  }

  _build(rootNode: TirNode | null | undefined, rootParent: SRef | null): SRef | null {
    const top = this._buildSubtree(rootNode, rootParent);
    if (rootParent) for (const s of top) rootParent.children.push(s);
    return (rootNode ? this._nodeToSRef.get(rootNode) : null) || null;
  }

  getSRef(node: TirNode): SRef | null {
    return this._nodeToSRef.get(node) || null;
  }

  getBlockSRef(name: string): SRef | null {
    return this._blockNameToSRef.get(name) || null;
  }

  allBlocks(): SRef[] {
    return [...this._blockSRefs];
  }

  allLoops(): SRef[] {
    return [...this._loopSRefs];
  }

  loopsOf(blockName: string): SRef[] {
    const blockSRef = this._blockNameToSRef.get(blockName);
    if (!blockSRef) return [];
    return blockSRef.loopAncestors().filter(s => (s.node as ForNode).kind !== ForKind.RECURRENCE).reverse();
  }

  rebuildFrom(rootNode: TirNode): void {
    this._nodeToSRef.clear();
    this._blockNameToSRef.clear();
    this._loopSRefs.clear();
    this._blockSRefs.clear();
    this.root = this._build(rootNode, null);
  }

  replaceNode(oldNode: TirNode, newNode: TirNode): boolean {
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

  removeNode(node: TirNode): boolean {
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
