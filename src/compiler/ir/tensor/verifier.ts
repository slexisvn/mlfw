import { PrimFunc } from './nodes.js';
import { formatLocation } from '../location.js';
import type { Location } from '../location.js';
import type { TirNode, TensorNode } from './nodes.js';

type UnbindMarker = { type: '_unbind'; name: string };
type RestoreLocationMarker = { type: '_restoreLocation'; location: Location | null };
type VerifyStackItem = TirNode | UnbindMarker | RestoreLocationMarker;

export class TensorVerifier {
  errors: string[];
  boundVars: Set<string>;
  location: Location | null;

  constructor() {
    this.errors = [];
    this.boundVars = new Set();
    this.location = null;
  }

  fail(message: string): void {
    this.errors.push(this.location === null ? message : `${message} at ${formatLocation(this.location)}`);
  }

  verify(func: unknown): string[] {
    this.errors = [];
    this.boundVars.clear();
    this.location = null;

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
            this.fail(`Loop variable ${node.loopVar.name} already bound`);
          }
          this.boundVars.add(node.loopVar.name);
          stack.push({ type: '_unbind', name: node.loopVar.name });
          stack.push(node.body);
          break;
        case 'BlockNode':
          stack.push({ type: '_restoreLocation', location: this.location });
          if (node.sourceOp) this.location = node.sourceOp.loc;
          for (const r of node.iterVars) {
            if (r.iterVar) {
              if (this.boundVars.has(r.iterVar.name)) {
                this.fail(`Block variable ${r.iterVar.name} already bound`);
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
          if (!node.buffer) this.fail('Allocate missing buffer');
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
          if (!node.condVar) this.fail('WhileNode missing condition variable');
          stack.push(node.loopBody);
          stack.push(node.condBody);
          break;
        case 'BufferStoreNode':
          if (!node.buffer) this.fail('BufferStore missing buffer');
          if (!node.indices || node.indices.length !== node.buffer.shape.length) {
            this.fail(`BufferStore rank mismatch for ${node.buffer ? node.buffer.name : 'unknown'}`);
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
        case '_restoreLocation':
          this.location = node.location;
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
        if (!node.buffer) this.fail('BufferLoad missing buffer');
        if (!node.indices || node.indices.length !== node.buffer.shape.length) {
          this.fail(`BufferLoad rank mismatch for ${node.buffer ? node.buffer.name : 'unknown'}`);
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
        if (!node.externName) this.fail('CallExtern missing function name');
        for (const arg of node.args) this._visitExpr(arg);
        break;
      case 'IfThenElseNode':
        this._visitExpr(node.condition);
        this._visitExpr(node.thenBody);
        if (node.elseBody) this._visitExpr(node.elseBody);
        break;
      case 'VariableNode':
        if (!this.boundVars.has(node.name)) {
          this.fail(`Unbound variable used: ${node.name}`);
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
