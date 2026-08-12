import { walk as irWalk } from '../ir/ir_visitor.js';
import type { IRNode } from '../ir/ir_visitor.js';
import type { PrimFunc, SeqNode, TirNode, BufferLoadNode, BufferStoreNode } from '../ir/tensor/nodes.js';
import type { Buffer } from '../ir/tensor/buffer.js';

export class ScheduleMutator {
  func: PrimFunc;

  constructor(func: PrimFunc) {
    this.func = func;
  }

  replaceNode(oldNode: TirNode, newNode: TirNode): void {
    if (oldNode._parent) {
      oldNode.replaceWith(newNode);
      return;
    }
    if (this.func.body === oldNode || this.func.body === undefined) {
      this.func.body = newNode;
      if (this.func._setChild) this.func._setChild('body', newNode);
    }
  }

  removeNode(node: TirNode): void {
    const parent = node._parent as SeqNode | null;
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

  redirectReads(node: TirNode | null | undefined, fromBuf: Buffer, toBuf: Buffer): void {
    if (node) irWalk(node, (n: IRNode) => { if (n.type === 'BufferLoadNode' && (n as BufferLoadNode).buffer === fromBuf) (n as BufferLoadNode).buffer = toBuf; });
  }

  redirectBuffer(node: TirNode | null | undefined, fromBuf: Buffer, toBuf: Buffer): void {
    if (node) irWalk(node, (n: IRNode) => { const b = n as BufferLoadNode | BufferStoreNode; if ((n.type === 'BufferLoadNode' || n.type === 'BufferStoreNode') && b.buffer === fromBuf) b.buffer = toBuf; });
  }
}
