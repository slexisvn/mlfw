import { cCompareOp } from '../../../util/dtype_map.js';
import { ForKind, IterVarKind } from './nodes.js';
import type { Buffer } from './buffer.js';
import type { TensorNode, PrimFunc, SeqNode, ForNode, BlockNode, BlockRealizeNode, BufferStoreNode, BufferLoadNode, IfThenElseNode, LetStmtNode, AllocateNode, EvaluateNode, WhileNode, VecCopyNode, MathOpNode, CompareNode, CallExternNode, VariableNode, IntImmNode, FloatImmNode, CastNode } from './nodes.js';

type VisitDispatch = Record<string, ((node: TensorNode) => void) | undefined>;

export type PrintableFunc = Readonly<{
  name: string;
  params: readonly VariableNode[];
  body: TensorNode;
  bufferMap: ReadonlyMap<VariableNode, Buffer>;
}>;

export class TensorIRPrinter {
  indent: number;
  out: string[];

  constructor() {
    this.indent = 0;
    this.out = [];
  }

  print(node: TensorNode): string {
    this.out = [];
    this.visit(node);
    return this.out.join('');
  }

  push(str: string): void {
    this.out.push(str);
  }

  newline(): void {
    this.out.push('\n' + '  '.repeat(this.indent));
  }

  open(prefix: string): void {
    this.push(`${prefix}{`);
    this.indent++;
    this.newline();
  }

  close(): void {
    this.indent--;
    this.newline();
    this.push('}');
  }

  scope(prefix: string, body: TensorNode | null | undefined): void {
    this.open(prefix);
    this.visit(body);
    this.close();
  }

  visit(node: TensorNode | null | undefined): void {
    if (!node) return;
    const method = 'visit' + node.type;
    const dispatch = this as unknown as VisitDispatch;
    if (dispatch[method]) {
      (dispatch[method] as (n: TensorNode) => void)(node);
    } else {
      this.push(`[UnknownNode: ${node.type}]`);
    }
  }

  printFunc(keyword: string, node: PrintableFunc): void {
    this.open(`${keyword} ${node.name}(${node.params.map(p => p.name).join(', ')}) `);

    for (const [v, buf] of node.bufferMap) {
      this.push(`${buf.name} = buffer_map(${v.name}, shape=[${buf.shape.join(',')}], dtype=${buf.dtype})`);
      this.newline();
    }

    this.visit(node.body);
    this.close();
  }

  openLoop(loopVar: VariableNode, extent: TensorNode, kind: string, threadTag: string | null): void {
    const annotation = kind === ForKind.SERIAL ? '' : `@${kind} `;
    const tag = threadTag ? `[${threadTag}] ` : '';
    this.push(`for ${loopVar.name} in 0..`);
    this.visit(extent);
    this.open(` ${annotation}${tag}`);
  }

  printBufferRef(name: string, indices: readonly TensorNode[]): void {
    this.push(`${name}[`);
    for (let i = 0; i < indices.length; i++) {
      this.visit(indices[i]);
      if (i < indices.length - 1) this.push(', ');
    }
    this.push(`]`);
  }

  printBinding(realize: BlockRealizeNode): void {
    const kind = realize.kind === IterVarKind.DATA_PAR ? '' : `:${realize.kind}`;
    this.push(`bind ${realize.iterVar.name}${kind} = `);
    this.visit(realize.binding);
  }

  visitPrimFunc(node: PrimFunc): void {
    this.printFunc('prim_func', node);
  }

  visitSeqNode(node: SeqNode): void {
    for (let i = 0; i < node.stmts.length; i++) {
      this.visit(node.stmts[i]);
      if (i < node.stmts.length - 1) this.newline();
    }
  }

  visitForNode(node: ForNode): void {
    this.openLoop(node.loopVar, node.extent, node.kind, node.threadTag);
    this.visit(node.body);
    this.close();
  }

  visitBlockNode(node: BlockNode): void {
    this.open(`block ${node.name} `);

    for (const r of node.iterVars) {
      this.printBinding(r);
      this.newline();
    }

    if (node.reads.length > 0) {
      this.push(`reads([`);
      this.push(node.reads.map(r => `${r.buffer.name}[...]`).join(', '));
      this.push(`])`);
      this.newline();
    }

    if (node.writes.length > 0) {
      this.push(`writes([`);
      this.push(node.writes.map(r => `${r.buffer.name}[...]`).join(', '));
      this.push(`])`);
      this.newline();
    }

    if (node.initBody) {
      this.scope(`init `, node.initBody);
      this.newline();
    }

    this.visit(node.body);
    this.close();
  }

  visitBlockRealizeNode(node: BlockRealizeNode): void {
    this.printBinding(node);
  }

  visitBufferStoreNode(node: BufferStoreNode): void {
    this.printBufferRef(node.buffer.name, node.indices);
    this.push(` = `);
    this.visit(node.value);
  }

  visitBufferLoadNode(node: BufferLoadNode): void {
    this.printBufferRef(node.buffer.name, node.indices);
  }

  visitIfThenElseNode(node: IfThenElseNode): void {
    this.push(`if (`);
    this.visit(node.condition);
    this.scope(`) `, node.thenBody);
    if (node.elseBody) this.scope(` else `, node.elseBody);
  }

  visitLetStmtNode(node: LetStmtNode): void {
    this.push(`let ${node.variable.name} = `);
    this.visit(node.value);
    this.newline();
    this.visit(node.body);
  }

  visitAllocateNode(node: AllocateNode): void {
    this.scope(`allocate ${node.buffer.name}[${node.buffer.shape.join(', ')}] (${node.scope}) `, node.body);
  }

  visitWhileNode(node: WhileNode): void {
    this.open(`while ${node.condVar.name} `);
    this.scope(`cond `, node.condBody);
    this.newline();
    this.visit(node.loopBody);
    this.close();
  }

  visitEvaluateNode(node: EvaluateNode): void {
    this.push(`evaluate `);
    this.visit(node.value);
  }

  visitSyncThreadsNode(): void {
    this.push(`sync_threads()`);
  }

  visitVecCopyNode(node: VecCopyNode): void {
    this.printBufferRef(node.dstBuffer.name, [node.dstIndex]);
    this.push(` = vec_copy<${node.width}>(`);
    this.printBufferRef(node.srcBuffer.name, [node.srcIndex]);
    this.push(`)`);
  }

  visitMathOpNode(node: MathOpNode): void {
    this.push(`(`);
    if (!node.b) this.push(node.op);
    this.visit(node.a);
    if (node.b) {
      this.push(` ${node.op} `);
      this.visit(node.b);
    }
    this.push(`)`);
  }

  visitCompareNode(node: CompareNode): void {
    this.push(`(`);
    this.visit(node.a);
    this.push(` ${cCompareOp(node.direction)} `);
    this.visit(node.b);
    this.push(`)`);
  }

  visitCallExternNode(node: CallExternNode): void {
    this.push(`${node.externName}(`);
    for (let i = 0; i < node.args.length; i++) {
      this.visit(node.args[i]);
      if (i < node.args.length - 1) this.push(', ');
    }
    this.push(`)`);
  }

  visitVariableNode(node: VariableNode): void {
    this.push(node.name);
  }

  visitIntImmNode(node: IntImmNode): void {
    this.push(node.value.toString());
  }

  visitFloatImmNode(node: FloatImmNode): void {
    this.push(node.value.toString());
  }

  visitCastNode(node: CastNode): void {
    this.push(`cast<${node.toDtype}>(`);
    this.visit(node.expr);
    this.push(`)`);
  }
}

export function printTensorIR(node: TensorNode): string {
  return new TensorIRPrinter().print(node);
}
