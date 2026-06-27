import { walk as irWalk } from '../ir/ir_visitor.js';

export class ScheduleMutator {
  constructor(func) {
    this.func = func;
  }

  replaceNode(oldNode, newNode) {
    if (oldNode._parent) {
      oldNode.replaceWith(newNode);
      return;
    }
    if (this.func.body === oldNode || this.func.body === undefined) {
      this.func.body = newNode;
      if (this.func._setChild) this.func._setChild('body', newNode);
    }
  }

  removeNode(node) {
    const parent = node._parent;
    if (parent && parent.type === 'SeqNode' && Array.isArray(parent.stmts)) {
      const i = parent.stmts.indexOf(node);
      if (i >= 0) {
        parent.stmts.splice(i, 1);
        if (parent._setChildren) parent._setChildren('stmts', parent.stmts);
        return;
      }
    }
    throw new Error('removeNode: node parent is not a SeqNode; cannot remove without duplicating it');
  }

  redirectReads(node, fromBuf, toBuf) {
    if (node) irWalk(node, (n) => { if (n.type === 'BufferLoadNode' && n.buffer === fromBuf) n.buffer = toBuf; });
  }

  redirectBuffer(node, fromBuf, toBuf) {
    if (node) irWalk(node, (n) => { if ((n.type === 'BufferLoadNode' || n.type === 'BufferStoreNode') && n.buffer === fromBuf) n.buffer = toBuf; });
  }
}
