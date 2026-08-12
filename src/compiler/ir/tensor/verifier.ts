import { PrimFunc, SeqNode, ForNode, BlockNode, BufferStoreNode, BufferLoadNode, MathOpNode, CompareNode, IfThenElseNode, LetStmtNode, AllocateNode, EvaluateNode, CallExternNode, BlockRealizeNode, CastNode, WhileNode } from './nodes.js';
import type { TirNode, TensorNode } from './nodes.js';

type UnbindMarker = { type: '_unbind'; name: string };
type VerifyStackItem = TirNode | UnbindMarker;

export class TensorVerifier {
  errors: string[];
  boundVars: Set<string>;

  constructor() {
    this.errors = [];
    this.boundVars = new Set();
  }

  verify(func: unknown): string[] {
    this.errors = [];
    this.boundVars.clear();

    if (!(func instanceof PrimFunc)) {
      this.errors.push('Expected PrimFunc at root');
      return this.errors;
    }

    for (const param of func.params) {
      this.boundVars.add(param.name);
    }
    if (func.shapeParams) {
      for (const sp of func.shapeParams) {
        this.boundVars.add(sp.name);
      }
    }

    this.visit(func.body);
    return this.errors;
  }

  visit(root: TensorNode): void {
    const stack: VerifyStackItem[] = [root as TirNode];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      switch (node.type) {
        case 'SeqNode':
          for (let i = node.stmts.length - 1; i >= 0; i--) stack.push(node.stmts[i]);
          break;
        case 'ForNode':
          if (this.boundVars.has(node.loopVar.name)) {
            this.errors.push(`Loop variable ${node.loopVar.name} already bound`);
          }
          this.boundVars.add(node.loopVar.name);
          stack.push({ type: '_unbind', name: node.loopVar.name });
          stack.push(node.body);
          break;
        case 'BlockNode':
          for (const r of node.iterVars) {
            if (r.iterVar) {
              if (this.boundVars.has(r.iterVar.name)) {
                this.errors.push(`Block variable ${r.iterVar.name} already bound`);
              }
              this.boundVars.add(r.iterVar.name);
            }
          }
          for (let i = node.iterVars.length - 1; i >= 0; i--) {
            if (node.iterVars[i].iterVar) stack.push({ type: '_unbind', name: node.iterVars[i].iterVar.name });
          }
          stack.push(node.body);
          if (node.initBody) stack.push(node.initBody);
          break;
        case 'AllocateNode':
          if (!node.buffer) this.errors.push('Allocate missing buffer');
          stack.push(node.body);
          break;
        case 'LetStmtNode':
          this._visitExpr(node.value);
          this.boundVars.add(node.variable.name);
          stack.push({ type: '_unbind', name: node.variable.name });
          stack.push(node.body);
          break;
        case 'IfThenElseNode':
          this._visitExpr(node.condition);
          if (node.elseBody) stack.push(node.elseBody);
          stack.push(node.thenBody);
          break;
        case 'WhileNode':
          if (!node.condVar) this.errors.push('WhileNode missing condition variable');
          stack.push(node.loopBody);
          stack.push(node.condBody);
          break;
        case 'BufferStoreNode':
          if (!node.buffer) this.errors.push('BufferStore missing buffer');
          if (!node.indices || node.indices.length !== node.buffer.shape.length) {
            this.errors.push(`BufferStore rank mismatch for ${node.buffer ? node.buffer.name : 'unknown'}`);
          }
          if (node.indices) for (const idx of node.indices) this._visitExpr(idx);
          this._visitExpr(node.value);
          break;
        case 'EvaluateNode':
          this._visitExpr(node.value);
          break;
        case '_unbind':
          this.boundVars.delete(node.name);
          break;
        default:
          this._visitExpr(node);
          break;
      }
    }
  }

  _visitExpr(node: TirNode | null | undefined): void {
    if (!node) return;
    switch (node.type) {
      case 'BufferLoadNode':
        if (!node.buffer) this.errors.push('BufferLoad missing buffer');
        if (!node.indices || node.indices.length !== node.buffer.shape.length) {
          this.errors.push(`BufferLoad rank mismatch for ${node.buffer ? node.buffer.name : 'unknown'}`);
        }
        if (node.indices) for (const idx of node.indices) this._visitExpr(idx);
        break;
      case 'MathOpNode':
        this._visitExpr(node.a);
        if (node.b) this._visitExpr(node.b);
        break;
      case 'CompareNode':
        this._visitExpr(node.a);
        this._visitExpr(node.b);
        break;
      case 'CastNode':
        this._visitExpr(node.expr);
        break;
      case 'CallExternNode':
        if (!node.externName) this.errors.push('CallExtern missing function name');
        for (const arg of node.args) this._visitExpr(arg);
        break;
      case 'IfThenElseNode':
        this._visitExpr(node.condition);
        this._visitExpr(node.thenBody);
        if (node.elseBody) this._visitExpr(node.elseBody);
        break;
      case 'VariableNode':
        if (!this.boundVars.has(node.name)) {
          this.errors.push(`Unbound variable used: ${node.name}`);
        }
        break;
      case 'BlockRealizeNode':
        if (node.binding) this._visitExpr(node.binding);
        break;
      case 'IntImmNode':
      case 'FloatImmNode':
        break;
    }
  }
}
