const STEP_BUDGET = 400_000;
const STORE_BUDGET = 200_000;
const WHILE_BUDGET = 64;
const EXTENT_CAP = 4096;
const FNV_OFFSET = 2166136261;
const FNV_PRIME = 16777619;

export class Budget extends Error {}
export class Unsupported extends Error {}

export type Trace = {
  cells: Map<string, number>;
  buffers: Set<string>;
  orderHash: number;
  stores: number;
  truncated: boolean;
};

type Node = Record<string, unknown> & { type: string };
type Buffer = { name: string; dtype?: string };

const BINARY: Record<string, (a: number, b: number) => number> = {
  '+': (a, b) => a + b,
  '-': (a, b) => a - b,
  '*': (a, b) => a * b,
  '/': (a, b) => (b === 0 ? 0 : a / b),
  '%': (a, b) => (b === 0 ? 0 : ((a % b) + b) % b),
  '//': (a, b) => (b === 0 ? 0 : Math.floor(a / b)),
  tmod: (a, b) => (b === 0 ? 0 : a % b),
  tdiv: (a, b) => (b === 0 ? 0 : Math.trunc(a / b)),
  max: (a, b) => Math.max(a, b),
  min: (a, b) => Math.min(a, b),
  '&&': (a, b) => (a && b ? 1 : 0),
  '||': (a, b) => (a || b ? 1 : 0),
};

const COMPARE: Record<string, (a: number, b: number) => boolean> = {
  eq: (a, b) => a === b,
  ne: (a, b) => a !== b,
  lt: (a, b) => a < b,
  le: (a, b) => a <= b,
  gt: (a, b) => a > b,
  ge: (a, b) => a >= b,
};

const UNARY: Record<string, (a: number) => number> = {
  '-': a => -a,
  '!': a => (a ? 0 : 1),
  '~': a => ~a,
};

const EXTERN: Record<string, (args: number[]) => number> = {
  exp: a => Math.exp(a[0]),
  log: a => Math.log(Math.abs(a[0]) + 1),
  log2: a => Math.log2(Math.abs(a[0]) + 1),
  log10: a => Math.log10(Math.abs(a[0]) + 1),
  exp2: a => Math.pow(2, a[0]),
  sqrt: a => Math.sqrt(Math.abs(a[0])),
  rsqrt: a => 1 / Math.sqrt(Math.abs(a[0]) + 1),
  tanh: a => Math.tanh(a[0]),
  abs: a => Math.abs(a[0]),
  ceil: a => Math.ceil(a[0]),
  floor: a => Math.floor(a[0]),
  round: a => Math.round(a[0]),
  sign: a => Math.sign(a[0]),
  sin: a => Math.sin(a[0]),
  cos: a => Math.cos(a[0]),
  max: a => Math.max(a[0], a[1]),
  min: a => Math.min(a[0], a[1]),
  pow: a => Math.pow(Math.abs(a[0]) + 1, Math.min(a[1], 8)),
  fmod: a => (a[1] === 0 ? 0 : ((a[0] % a[1]) + a[1]) % a[1]),
};

function hash(text: string): number {
  let h = FNV_OFFSET;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, FNV_PRIME);
  }
  return h >>> 0;
}

function seed(key: string): number {
  return 1 + (hash(key) >>> 8) / 2 ** 24;
}

export class Interpreter {
  private readonly env = new Map<string, number>();
  private readonly loops: string[] = [];
  private readonly cells = new Map<string, number>();
  private readonly buffers = new Set<string>();
  private orderHash = FNV_OFFSET;
  private stores = 0;
  private steps = 0;
  private truncated = false;

  run(body: unknown): Trace {
    try {
      this.exec(body as Node);
    } catch (error) {
      if (!(error instanceof Budget)) throw error;
      this.truncated = true;
    }
    return {
      cells: this.cells,
      buffers: this.buffers,
      orderHash: this.orderHash,
      stores: this.stores,
      truncated: this.truncated,
    };
  }

  private tick(): void {
    if (++this.steps > STEP_BUDGET) throw new Budget('step budget');
  }

  private store(cell: string, buffer: string, value: number): void {
    if (++this.stores > STORE_BUDGET) throw new Budget('store budget');
    this.buffers.add(buffer);
    this.cells.set(cell, value);
    this.orderHash = Math.imul(this.orderHash ^ hash(cell), FNV_PRIME) >>> 0;
  }

  private load(cell: string): number {
    const written = this.cells.get(cell);
    return written === undefined ? seed(cell) : written;
  }

  private cellOf(buffer: Buffer, indices: number[]): string {
    return `${buffer.name}[${indices.join(', ')}]`;
  }

  private extentOf(node: Node, what: string): number {
    const value = this.evaluate(node);
    if (!Number.isInteger(value) || value < 0) {
      throw new Unsupported(`${what} is not a constant whole number here — it depends on a shape only known at run time`);
    }
    if (value > EXTENT_CAP) throw new Budget('extent cap');
    return value;
  }

  private exec(node: Node | null | undefined): void {
    if (!node) return;
    this.tick();

    switch (node.type) {
      case 'PrimFunc':
      case 'LIRFunc':
      case 'AllocateNode':
        return this.exec(node.body as Node);

      case 'SeqNode': {
        for (const stmt of node.stmts as Node[]) this.exec(stmt);
        return;
      }

      case 'ForNode': {
        const name = (node.loopVar as { name: string }).name;
        const min = this.extentOf(node.min as Node, 'a loop start');
        const extent = this.extentOf(node.extent as Node, 'a loop extent');
        this.loops.push(name);
        for (let i = 0; i < extent; i++) {
          this.env.set(name, min + i);
          this.exec(node.body as Node);
        }
        this.loops.pop();
        this.env.delete(name);
        return;
      }

      case 'BlockNode': {
        for (const bind of node.iterVars as Node[]) {
          const iterVar = bind.iterVar as { name: string } | null;
          if (iterVar && bind.binding) this.env.set(iterVar.name, this.evaluate(bind.binding as Node));
        }
        if (node.initBody) {
          const innermost = this.loops[this.loops.length - 1];
          if (innermost === undefined || this.env.get(innermost) === 0) this.exec(node.initBody as Node);
        }
        return this.exec(node.body as Node);
      }

      case 'LetStmtNode': {
        this.env.set((node.variable as { name: string }).name, this.evaluate(node.value as Node));
        return this.exec(node.body as Node);
      }

      case 'IfThenElseNode': {
        if (this.evaluate(node.condition as Node)) this.exec(node.thenBody as Node);
        else this.exec(node.elseBody as Node);
        return;
      }

      case 'WhileNode': {
        const guard = node.condVar as Buffer & { name: string };
        for (let i = 0; i < WHILE_BUDGET; i++) {
          this.exec(node.condBody as Node);
          if (!this.load(`${guard.name}[0]`)) return;
          this.exec(node.loopBody as Node);
        }
        this.truncated = true;
        return;
      }

      case 'BufferStoreNode': {
        const buffer = node.buffer as Buffer;
        const indices = (node.indices as Node[]).map(index => this.evaluate(index));
        this.store(this.cellOf(buffer, indices), buffer.name, this.evaluate(node.value as Node));
        return;
      }

      case 'LIRFlatStoreNode': {
        const buffer = node.buffer as Buffer;
        const offset = this.evaluate(node.offsetExpr as Node);
        this.store(`${buffer.name}[${offset}]`, buffer.name, this.evaluate(node.value as Node));
        return;
      }

      case 'LIRBindingsNode': {
        for (const bind of node.bindings as { name: string; expr: Node }[]) {
          this.env.set(bind.name, this.evaluate(bind.expr));
        }
        return this.exec(node.body as Node);
      }

      case 'LIRAccumulatorNode':
        return this.accumulate(node);

      case 'VecCopyNode': {
        const dst = node.dstBuffer as Buffer;
        const src = node.srcBuffer as Buffer;
        const dstIndex = this.evaluate(node.dstIndex as Node);
        const srcIndex = this.evaluate(node.srcIndex as Node);
        for (let i = 0; i < (node.width as number); i++) {
          this.store(`${dst.name}[${dstIndex + i}]`, dst.name, this.load(`${src.name}[${srcIndex + i}]`));
        }
        return;
      }

      case 'EvaluateNode':
        this.evaluate(node.value as Node);
        return;

      case 'SyncThreadsNode':
        return;

      default:
        throw new Unsupported(`no interpreter rule for the statement '${node.type}'`);
    }
  }

  private accumulate(node: Node): void {
    const local = node.localName as string;
    const combine = BINARY[(node.op as string) || '+'];
    if (!combine) throw new Unsupported(`no interpreter rule for the accumulator '${String(node.op)}'`);

    this.exec(node.initBody as Node);
    this.env.set(local, this.evaluate(node.initLoad as Node));

    const name = (node.loopVar as { name: string }).name;
    const extent = this.extentOf(node.extent as Node, 'an accumulator extent');
    this.loops.push(name);
    for (let i = 0; i < extent; i++) {
      this.env.set(name, i);
      this.exec(node.prologue as Node);
      this.env.set(local, combine(this.env.get(local) as number, this.evaluate(node.body as Node)));
    }
    this.loops.pop();
    this.env.delete(name);

    const flush = node.flushStore as Node;
    const buffer = flush.buffer as Buffer;
    const offset = this.evaluate(flush.offsetExpr as Node);
    this.store(`${buffer.name}[${offset}]`, buffer.name, this.env.get(local) as number);
    this.env.delete(local);
  }

  private evaluate(node: Node | null | undefined): number {
    if (!node) return 0;
    this.tick();

    switch (node.type) {
      case 'IntImmNode':
      case 'FloatImmNode':
        return node.value as number;

      case 'VariableNode': {
        const name = node.name as string;
        const bound = this.env.get(name);
        return bound === undefined ? seed(`var:${name}`) : bound;
      }

      case 'MathOpNode': {
        const op = node.op as string;
        if (node.b === null || node.b === undefined) {
          const unary = UNARY[op];
          if (!unary) throw new Unsupported(`no interpreter rule for the unary operator '${op}'`);
          return unary(this.evaluate(node.a as Node));
        }
        const binary = BINARY[op];
        if (!binary) throw new Unsupported(`no interpreter rule for the operator '${op}'`);
        return binary(this.evaluate(node.a as Node), this.evaluate(node.b as Node));
      }

      case 'CompareNode': {
        const compare = COMPARE[node.direction as string];
        if (!compare) throw new Unsupported(`no interpreter rule for the comparison '${String(node.direction)}'`);
        return compare(this.evaluate(node.a as Node), this.evaluate(node.b as Node)) ? 1 : 0;
      }

      case 'CastNode': {
        const inner = this.evaluate(node.expr as Node);
        const to = node.toDtype as string;
        if (to === 'bool') return inner ? 1 : 0;
        return to.startsWith('int') || to.startsWith('uint') ? Math.trunc(inner) : inner;
      }

      case 'IfThenElseNode':
        return this.evaluate(node.condition as Node)
          ? this.evaluate(node.thenBody as Node)
          : this.evaluate(node.elseBody as Node);

      case 'BufferLoadNode': {
        const buffer = node.buffer as Buffer;
        const indices = (node.indices as Node[]).map(index => this.evaluate(index));
        return this.load(this.cellOf(buffer, indices));
      }

      case 'LIRFlatLoadNode': {
        const buffer = node.buffer as Buffer;
        return this.load(`${buffer.name}[${this.evaluate(node.offsetExpr as Node)}]`);
      }

      case 'CallExternNode': {
        const args = (node.args as Node[]).map(arg => this.evaluate(arg));
        const name = node.externName as string;
        const known = EXTERN[name];
        return known ? known(args) : seed(`${name}(${args.join(',')})`);
      }

      default:
        throw new Unsupported(`no interpreter rule for the expression '${node.type}'`);
    }
  }
}

export function interpret(body: unknown): Trace {
  return new Interpreter().run(body);
}
