import { cCompareOp } from '../../../util/dtype_map.js';
import type { TensorNode, PrimFunc, SeqNode, ForNode, BlockNode, BufferStoreNode, BufferLoadNode, IfThenElseNode, LetStmtNode, AllocateNode, EvaluateNode, MathOpNode, CompareNode, CallExternNode, VariableNode, IntImmNode, FloatImmNode, CastNode } from './nodes.js';

type VisitDispatch = Record<string, ((node: TensorNode) => void) | undefined>;

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

  visitPrimFunc(node: PrimFunc): void {
    this.push(`prim_func ${node.name}(${node.params.map(p => p.name).join(', ')}) {`);
    this.indent++;
    this.newline();

    for (const [v, buf] of node.bufferMap) {
      this.push(`${buf.name} = buffer_map(${v.name}, shape=[${buf.shape.join(',')}], dtype=${buf.dtype})`);
      this.newline();
    }

    this.visit(node.body);
    this.indent--;
    this.newline();
    this.push('}');
  }

  visitSeqNode(node: SeqNode): void {
    for (let i = 0; i < node.stmts.length; i++) {
      this.visit(node.stmts[i]);
      if (i < node.stmts.length - 1) this.newline();
    }
  }

  visitForNode(node: ForNode): void {
    const kind = node.kind === 'serial' ? '' : `@${node.kind} `;
    const tag = node.threadTag ? `[${node.threadTag}] ` : '';
    this.push(`for ${node.loopVar.name} in 0..`);
    this.visit(node.extent);
    this.push(` ${kind}${tag}{`);
    this.indent++;
    this.newline();
    this.visit(node.body);
    this.indent--;
    this.newline();
    this.push('}');
  }

  visitBlockNode(node: BlockNode): void {
    this.push(`block ${node.name} {`);
    this.indent++;
    this.newline();

    for (const r of node.iterVars) {
      this.push(`bind ${r.iterVar.name} = `);
      this.visit(r.binding);
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
      this.push(`init {`);
      this.indent++;
      this.newline();
      this.visit(node.initBody);
      this.indent--;
      this.newline();
      this.push(`}`);
      this.newline();
    }

    this.visit(node.body);
    this.indent--;
    this.newline();
    this.push('}');
  }

  visitBufferStoreNode(node: BufferStoreNode): void {
    this.push(`${node.buffer.name}[`);
    for (let i = 0; i < node.indices.length; i++) {
      this.visit(node.indices[i]);
      if (i < node.indices.length - 1) this.push(', ');
    }
    this.push(`] = `);
    this.visit(node.value);
  }

  visitBufferLoadNode(node: BufferLoadNode): void {
    this.push(`${node.buffer.name}[`);
    for (let i = 0; i < node.indices.length; i++) {
      this.visit(node.indices[i]);
      if (i < node.indices.length - 1) this.push(', ');
    }
    this.push(`]`);
  }

  visitIfThenElseNode(node: IfThenElseNode): void {
    this.push(`if (`);
    this.visit(node.condition);
    this.push(`) {`);
    this.indent++;
    this.newline();
    this.visit(node.thenBody);
    this.indent--;
    this.newline();
    this.push('}');
    if (node.elseBody) {
      this.push(` else {`);
      this.indent++;
      this.newline();
      this.visit(node.elseBody);
      this.indent--;
      this.newline();
      this.push('}');
    }
  }

  visitLetStmtNode(node: LetStmtNode): void {
    this.push(`let ${node.variable.name} = `);
    this.visit(node.value);
    this.newline();
    this.visit(node.body);
  }

  visitAllocateNode(node: AllocateNode): void {
    this.push(`allocate ${node.buffer.name}[${node.buffer.shape.join(', ')}] (${node.scope}) {`);
    this.indent++;
    this.newline();
    this.visit(node.body);
    this.indent--;
    this.newline();
    this.push('}');
  }

  visitEvaluateNode(node: EvaluateNode): void {
    this.push(`evaluate `);
    this.visit(node.value);
  }

  visitMathOpNode(node: MathOpNode): void {
    this.push(`(`);
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
