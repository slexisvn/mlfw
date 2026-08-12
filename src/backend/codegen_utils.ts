import { SymInt, symVarName } from '../compiler/analysis/sym_int.js';
import type { SymExpr, SymIntOp } from '../compiler/analysis/sym_int.js';
import type { Buffer } from '../compiler/ir/tensor/buffer.js';
import type { AllocateNode, BlockNode, BufferStoreNode, ForNode, IfThenElseNode, LetStmtNode, NodeSlots, PrimFunc, TirNode, VecCopyNode, WhileNode } from '../compiler/ir/tensor/nodes.js';
import type { IRStmtNode, LIRAccumulatorNode, LIRBindingsNode, LIRFlatStoreNode, LIRFunc, LIRThreadBinding } from '../compiler/ir/lir/nodes.js';

export type CodegenDialect = 'c' | 'js' | 'wat' | 'wgsl';
export type CodegenFunc = PrimFunc | LIRFunc;
export type ShapeVarRef = { name: string };
export type ShapeVarFormatter = (v: ShapeVarRef) => string;
export type BufferDecl = { name: string; dtype: string; size: number };
export type DimProductResolver = (buffer: Buffer, dimIdx: number) => string;

export interface StatementVisitor {
  _visitAllocateNode(node: AllocateNode): void;
  _visitForNode(node: ForNode): void;
  _visitBlockNode(node: BlockNode): void;
  _visitIfStmt(node: IfThenElseNode): void;
  _visitLetStmtNode(node: LetStmtNode): void;
  _visitBufferStoreNode(node: BufferStoreNode): void;
  _visitLIRFlatStore(node: LIRFlatStoreNode): void;
  _visitLIRBindings(node: LIRBindingsNode): void;
  _visitLIRAccumulator(node: LIRAccumulatorNode): void;
  _visitWhileNode(node: WhileNode): void;
  _visitVecCopyNode(node: VecCopyNode): void;
  _emitSync(): void;
}

function symOpToString(type: SymIntOp, a: string, b: string | null, dialect: CodegenDialect): string {
  switch (type) {
    case 'add': return `(${a} + ${b})`;
    case 'sub': return `(${a} - ${b})`;
    case 'mul': return `(${a} * ${b})`;
    case 'neg': return `(-${a})`;
    case 'div': return dialect === 'js' ? `((${a} / ${b}) | 0)` : `(${a} / ${b})`;
    case 'mod': return dialect === 'js' ? `((${a} % ${b} + ${b}) % ${b})` : `(${a} % ${b})`;
    case 'ceildiv': return dialect === 'js' ? `(((${a} + ${b} - 1) / ${b}) | 0)` : `((${a} + ${b} - 1) / ${b})`;
    case 'max': return dialect === 'js' ? `Math.max(${a}, ${b})` : `max(${a}, ${b})`;
    case 'min': return dialect === 'js' ? `Math.min(${a}, ${b})` : `min(${a}, ${b})`;
    default: throw new Error(`emitSymInt: unsupported op '${type}'`);
  }
}

export function emitSymInt(expr: SymExpr, formatVar: ShapeVarFormatter, dialect: CodegenDialect = 'c'): string {
  if (typeof expr === 'number') return String(expr);
  if (!(expr instanceof SymInt)) return String(expr);
  if (expr.type === 'var') return formatVar({ name: symVarName(expr.name as string) });
  if (dialect === 'wat') {
    throw new Error('emitSymInt: compound symbolic expressions are not supported on the WASM backend');
  }
  const a = emitSymInt(expr.args[0], formatVar, dialect);
  const b = expr.args.length > 1 ? emitSymInt(expr.args[1], formatVar, dialect) : null;
  return symOpToString(expr.type, a, b, dialect);
}

export function visitStatements(cg: StatementVisitor, start: IRStmtNode): void {
  const stack: IRStmtNode[] = [start];
  while (stack.length > 0) {
    const cur = stack.pop();
    if (!cur) continue;
    switch (cur.type) {
      case 'SeqNode':
        for (let i = cur.stmts.length - 1; i >= 0; i--) stack.push(cur.stmts[i]);
        continue;
      case 'AllocateNode':
        cg._visitAllocateNode(cur);
        stack.push(cur.body);
        continue;
      case 'ForNode': cg._visitForNode(cur); continue;
      case 'BlockNode': cg._visitBlockNode(cur); continue;
      case 'IfThenElseNode': cg._visitIfStmt(cur); continue;
      case 'LetStmtNode': cg._visitLetStmtNode(cur); continue;
      case 'BufferStoreNode': cg._visitBufferStoreNode(cur); continue;
      case 'LIRFlatStoreNode': cg._visitLIRFlatStore(cur); continue;
      case 'LIRBindingsNode': cg._visitLIRBindings(cur); continue;
      case 'LIRAccumulatorNode': cg._visitLIRAccumulator(cur); continue;
      case 'WhileNode': cg._visitWhileNode(cur); continue;
      case 'SyncThreadsNode': cg._emitSync(); continue;
      case 'VecCopyNode': cg._visitVecCopyNode(cur); continue;
      case 'EvaluateNode': continue;
      default: throw new Error(`${cg.constructor.name}: unhandled statement node '${cur.type}'`);
    }
  }
}


export function isZeroFillBody(body: IRStmtNode): boolean {
  let cur: IRStmtNode | null = body;
  while (cur) {
    if (cur.type === 'ForNode' || cur.type === 'BlockNode') { cur = cur.body; continue; }
    if (cur.type === 'BufferStoreNode' || cur.type === 'LIRFlatStoreNode') {
      const val = cur.value;
      if (!val) return false;
      return (val.type === 'FloatImmNode' && val.value === 0) || (val.type === 'IntImmNode' && val.value === 0);
    }
    return false;
  }
  return false;
}

export function resolveShapeParam(primFunc: CodegenFunc | null, buffer: Buffer, dimIdx: number, format: ShapeVarFormatter, label: string, dialect: CodegenDialect = 'c'): string {
  const d = buffer.shape[dimIdx];
  if (d instanceof SymInt) return emitSymInt(d, format, dialect);
  if (primFunc && primFunc.shapeParamMap) {
    const v = primFunc.shapeParamMap.get(`${buffer.name}:${dimIdx}`);
    if (v) return format(v);
  }
  throw new Error(`${label} codegen: missing shape param for ${buffer.name}:${dimIdx}`);
}

export function estimateBufferSize(buffer: Buffer): number {
  let n = 1;
  for (const d of buffer.shape) {
    if (typeof d === 'number' && d > 0) n *= d;
  }
  return n;
}

export function dynamicDimProduct(buffer: Buffer, startDim: number, resolveShapeParam: DimProductResolver): string {
  const parts: string[] = [];
  for (let j = startDim; j < buffer.shape.length; j++) {
    const d = buffer.shape[j];
    if (typeof d === 'number' && d >= 0) parts.push(String(d));
    else parts.push(resolveShapeParam(buffer, j));
  }
  return parts.length === 0 ? '1' : parts.join(' * ');
}

