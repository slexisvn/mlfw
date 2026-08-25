import { TensorIRPrinter } from '../tensor/printer.js';
import type { TensorNode } from '../tensor/nodes.js';
import type { Buffer } from '../tensor/buffer.js';
import type { IRStmtNode, LIRFunc, LIRFlatLoadNode, LIRFlatStoreNode, LIRAccumulatorNode, LIRBindingsNode } from './nodes.js';

type FlatRef = Readonly<{ buffer: Buffer; offsetExpr: IRStmtNode }>;

const UNBOUND_ACC = '<acc>';

export class LIRPrinter extends TensorIRPrinter {
  accLocal: string | null;

  constructor() {
    super();
    this.accLocal = null;
  }

  printFlatRef(node: FlatRef): void {
    this.printBufferRef(node.buffer.name, [node.offsetExpr as TensorNode]);
  }

  visitLIRFunc(node: LIRFunc): void {
    this.printFunc('lir_func', node);
  }

  visitLIRFlatLoadNode(node: LIRFlatLoadNode): void {
    this.printFlatRef(node);
  }

  visitLIRFlatStoreNode(node: LIRFlatStoreNode): void {
    this.printFlatRef(node);
    this.push(` = `);
    if (node.value) this.visit(node.value as TensorNode);
    else this.push(this.accLocal || UNBOUND_ACC);
  }

  visitLIRAccumulatorNode(node: LIRAccumulatorNode): void {
    this.open(`accumulator ${node.localName}: ${node.dtype} `);

    this.push(`${node.localName} = `);
    this.visit(node.initLoad);
    this.newline();

    if (node.initBody) {
      this.visit(node.initBody as TensorNode);
      this.newline();
    }

    this.openLoop(node.loopVar, node.extent as TensorNode, node.loopKind, null);
    if (node.prologue) {
      this.visit(node.prologue as TensorNode);
      this.newline();
    }
    this.push(`${node.localName} ${node.op}= `);
    this.visit(node.body as TensorNode);
    this.close();
    this.newline();

    const outer = this.accLocal;
    this.accLocal = node.localName;
    this.visit(node.flushStore);
    this.accLocal = outer;

    this.close();
  }

  visitLIRBindingsNode(node: LIRBindingsNode): void {
    for (const bind of node.bindings) {
      this.push(`bind ${bind.name} = `);
      this.visit(bind.expr as TensorNode);
      this.newline();
    }
    this.visit(node.body as TensorNode);
  }
}

export function printLIR(node: TensorNode): string {
  return new LIRPrinter().print(node);
}
