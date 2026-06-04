import { PrimFunc, SeqNode, ForNode, BlockNode, BufferStoreNode, BufferLoadNode, MathOpNode, CompareNode, IfThenElseNode, LetStmtNode, AllocateNode, EvaluateNode, CallExternNode, BlockRealizeNode, CastNode, WhileNode } from './nodes.js';

export class TensorVerifier {
  constructor() {
    this.errors = [];
    this.boundVars = new Set();
  }

  verify(func) {
    this.errors = [];
    this.boundVars.clear();

    if (!(func instanceof PrimFunc)) {
      this.errors.push('Expected PrimFunc at root');
      return this.errors;
    }

    for (const param of func.params) {
      this.boundVars.add(param.name);
    }

    this.visit(func.body);
    return this.errors;
  }

  visit(node) {
    if (!node) return;
    const method = 'visit' + node.type;
    if (this[method]) {
      this[method](node);
    } else {
      this.errors.push(`Unrecognized node type: ${node.type}`);
    }
  }

  visitSeqNode(node) {
    for (const stmt of node.stmts) {
      this.visit(stmt);
    }
  }

  visitForNode(node) {
    if (this.boundVars.has(node.loopVar.name)) {
      this.errors.push(`Loop variable ${node.loopVar.name} already bound`);
    }
    this.boundVars.add(node.loopVar.name);
    this.visit(node.body);
    this.boundVars.delete(node.loopVar.name);
  }

  visitBlockNode(node) {
    for (const r of node.iterVars) {
      if (r.iterVar) {
        if (this.boundVars.has(r.iterVar.name)) {
          this.errors.push(`Block variable ${r.iterVar.name} already bound`);
        }
        this.boundVars.add(r.iterVar.name);
      }
    }

    if (node.initBody) this.visit(node.initBody);
    this.visit(node.body);

    for (const r of node.iterVars) {
      if (r.iterVar) this.boundVars.delete(r.iterVar.name);
    }
  }

  visitBlockRealizeNode(node) {
    if (node.binding) this.visit(node.binding);
  }

  visitBufferStoreNode(node) {
    if (!node.buffer) this.errors.push('BufferStore missing buffer');
    if (!node.indices || node.indices.length !== node.buffer.shape.length) {
      this.errors.push(`BufferStore rank mismatch for ${node.buffer ? node.buffer.name : 'unknown'}`);
    }
    if (node.indices) {
      for (const idx of node.indices) this.visit(idx);
    }
    this.visit(node.value);
  }

  visitBufferLoadNode(node) {
    if (!node.buffer) this.errors.push('BufferLoad missing buffer');
    if (!node.indices || node.indices.length !== node.buffer.shape.length) {
      this.errors.push(`BufferLoad rank mismatch for ${node.buffer ? node.buffer.name : 'unknown'}`);
    }
    if (node.indices) {
      for (const idx of node.indices) this.visit(idx);
    }
  }

  visitIfThenElseNode(node) {
    this.visit(node.condition);
    this.visit(node.thenBody);
    if (node.elseBody) this.visit(node.elseBody);
  }

  visitLetStmtNode(node) {
    this.visit(node.value);
    this.boundVars.add(node.variable.name);
    this.visit(node.body);
    this.boundVars.delete(node.variable.name);
  }

  visitAllocateNode(node) {
    if (!node.buffer) this.errors.push('Allocate missing buffer');
    this.visit(node.body);
  }

  visitEvaluateNode(node) {
    this.visit(node.value);
  }

  visitCallExternNode(node) {
    if (!node.externName) this.errors.push('CallExtern missing function name');
    for (const arg of node.args) {
      this.visit(arg);
    }
  }

  visitMathOpNode(node) {
    this.visit(node.a);
    if (node.b) this.visit(node.b);
  }

  visitCompareNode(node) {
    this.visit(node.a);
    this.visit(node.b);
  }

  visitCastNode(node) {
    this.visit(node.expr);
  }

  visitWhileNode(node) {
    if (!node.condVar) this.errors.push('WhileNode missing condition variable');
    this.visit(node.condBody);
    this.visit(node.loopBody);
  }

  visitVariableNode(node) {
    if (!this.boundVars.has(node.name)) {
      this.errors.push(`Unbound variable used: ${node.name}`);
    }
  }

  visitIntImmNode() {}
  visitFloatImmNode() {}
}
