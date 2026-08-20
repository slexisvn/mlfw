import { floorDiv, floorMod } from '../../util/divmod.js';

const SYM_INT_METHOD_NAMES: Readonly<Record<string, string>> = {
  ceildiv: 'ceilDiv'
};

export const SYM_VAR_PREFIX = '_sym_';

export function symVarName(name: string): string {
  return SYM_VAR_PREFIX + name;
}

export type SymIntOp = 'var' | 'add' | 'sub' | 'mul' | 'div' | 'mod' | 'max' | 'min' | 'neg' | 'ceildiv';
export type SymExpr = number | SymInt;
type BinarySymOp = (a: SymExpr, b: SymExpr) => SymExpr;
type UnarySymOp = (a: SymExpr) => SymExpr;

function mulFactors(expr: SymExpr, out: SymExpr[]): SymExpr[] {
  if (expr instanceof SymInt && expr.type === 'mul') {
    mulFactors(expr.args[0], out);
    mulFactors(expr.args[1], out);
    return out;
  }
  out.push(expr);
  return out;
}

function splitProduct(expr: SymExpr): { constant: number; symbols: SymExpr[] } {
  let constant = 1;
  const symbols: SymExpr[] = [];
  for (const f of mulFactors(expr, [])) {
    if (typeof f === 'number') constant *= f;
    else symbols.push(f);
  }
  return { constant, symbols };
}

function rebuildProduct(constant: number, symbols: readonly SymExpr[]): SymExpr {
  let out: SymExpr = constant;
  for (const s of symbols) out = SymInt.mul(out, s);
  return out;
}

function cancelProductDiv(a: SymExpr, b: SymExpr): SymExpr {
  const num = splitProduct(a);
  const den = splitProduct(b);
  let cancelled = false;

  for (let i = den.symbols.length - 1; i >= 0; i--) {
    const at = num.symbols.findIndex(s => SymInt.equals(s, den.symbols[i]));
    if (at < 0) continue;
    num.symbols.splice(at, 1);
    den.symbols.splice(i, 1);
    cancelled = true;
  }
  if (den.constant !== 1 && num.constant % den.constant === 0) {
    num.constant /= den.constant;
    den.constant = 1;
    cancelled = true;
  }
  if (!cancelled) return new SymInt('div', null, [a, b]);

  const numerator = rebuildProduct(num.constant, num.symbols);
  if (den.constant === 1 && den.symbols.length === 0) return numerator;
  return new SymInt('div', null, [numerator, rebuildProduct(den.constant, den.symbols)]);
}

export class SymInt {
  type: SymIntOp;
  name: string | null;
  args: SymExpr[];

  constructor(type: SymIntOp, name: string | null = null, args: SymExpr[] = []) {
    this.type = type;
    this.name = name;
    this.args = args;
  }

  static var(name: string): SymInt {
    return new SymInt('var', name);
  }

  static const(value: number): number {
    return value;
  }

  static add(a: SymExpr, b: SymExpr): SymExpr {
    if (typeof a === 'number' && typeof b === 'number') return a + b;
    if (a === 0) return b;
    if (b === 0) return a;
    if (SymInt.equals(a, b)) return SymInt.mul(2, a);
    return new SymInt('add', null, [a, b]);
  }

  static sub(a: SymExpr, b: SymExpr): SymExpr {
    if (typeof a === 'number' && typeof b === 'number') return a - b;
    if (b === 0) return a;
    if (SymInt.equals(a, b)) return 0;
    return new SymInt('sub', null, [a, b]);
  }

  static neg(a: SymExpr): SymExpr {
    if (typeof a === 'number') return -a;
    if (a instanceof SymInt && a.type === 'neg') return a.args[0];
    return new SymInt('neg', null, [a]);
  }

  static mul(a: SymExpr, b: SymExpr): SymExpr {
    if (typeof a === 'number' && typeof b === 'number') return a * b;
    if (a === 0 || b === 0) return 0;
    if (a === 1) return b;
    if (b === 1) return a;
    return new SymInt('mul', null, [a, b]);
  }

  static div(a: SymExpr, b: SymExpr): SymExpr {
    if (b === 0) throw new Error('SymInt.div: division by zero');
    if (typeof a === 'number' && typeof b === 'number') return floorDiv(a, b);
    if (a === 0) return 0;
    if (b === 1) return a;
    if (SymInt.equals(a, b)) return 1;
    return cancelProductDiv(a, b);
  }

  static mod(a: SymExpr, b: SymExpr): SymExpr {
    if (b === 0) throw new Error('SymInt.mod: modulo by zero');
    if (typeof a === 'number' && typeof b === 'number') return floorMod(a, b);
    if (a === 0) return 0;
    if (b === 1) return 0;
    if (SymInt.equals(a, b)) return 0;
    return new SymInt('mod', null, [a, b]);
  }

  static max(a: SymExpr, b: SymExpr): SymExpr {
    if (typeof a === 'number' && typeof b === 'number') return Math.max(a, b);
    if (a === b || SymInt.equals(a, b)) return a;
    return new SymInt('max', null, [a, b]);
  }

  static min(a: SymExpr, b: SymExpr): SymExpr {
    if (typeof a === 'number' && typeof b === 'number') return Math.min(a, b);
    if (a === b || SymInt.equals(a, b)) return a;
    return new SymInt('min', null, [a, b]);
  }

  static ceilDiv(a: SymExpr, b: SymExpr): SymExpr {
    if (typeof a === 'number' && typeof b === 'number') return Math.ceil(a / b);
    return new SymInt('ceildiv', null, [a, b]);
  }

  static equals(a: SymExpr, b: SymExpr): boolean {
    if (a === b) return true;
    if (typeof a === 'number' || typeof b === 'number') return false;
    if (!(a instanceof SymInt) || !(b instanceof SymInt)) return false;
    if (a.type !== b.type) return false;
    if (a.type === 'var') return a.name === b.name;
    if (a.args.length !== b.args.length) return false;

    if (a.type === 'add' || a.type === 'mul' || a.type === 'max' || a.type === 'min') {
      const match1 = SymInt.equals(a.args[0], b.args[0]) && SymInt.equals(a.args[1], b.args[1]);
      const match2 = SymInt.equals(a.args[0], b.args[1]) && SymInt.equals(a.args[1], b.args[0]);
      return match1 || match2;
    }

    for (let i = 0; i < a.args.length; i++) {
      if (!SymInt.equals(a.args[i], b.args[i])) return false;
    }
    return true;
  }

  static substitute(expr: SymExpr, varName: string, value: SymExpr): SymExpr {
    if (typeof expr === 'number') return expr;
    if (!(expr instanceof SymInt)) return expr;
    if (expr.type === 'var') {
      return expr.name === varName ? value : expr;
    }
    const newArgs = expr.args.map(a => SymInt.substitute(a, varName, value));
    const ops = SymInt as unknown as Record<string, BinarySymOp | UnarySymOp | undefined>;
    const op = ops[SYM_INT_METHOD_NAMES[expr.type] || expr.type];
    if (op && newArgs.length === 2) return (op as BinarySymOp)(newArgs[0], newArgs[1]);
    if (op && newArgs.length === 1) return (op as UnarySymOp)(newArgs[0]);
    return new SymInt(expr.type, expr.name, newArgs);
  }

  static evaluate(expr: SymExpr, env: ReadonlyMap<string, number>): number {
    if (typeof expr === 'number') return expr;
    if (!(expr instanceof SymInt)) return expr;
    if (expr.type === 'var') {
      if (env.has(expr.name as string)) return env.get(expr.name as string) as number;
      throw new Error(`Unbound symbolic variable: ${expr.name}`);
    }
    const args = expr.args.map(a => SymInt.evaluate(a, env));
    switch (expr.type) {
      case 'add': return args[0] + args[1];
      case 'sub': return args[0] - args[1];
      case 'mul': return args[0] * args[1];
      case 'div': return Math.floor(args[0] / args[1]);
      case 'mod': return ((args[0] % args[1]) + args[1]) % args[1];
      case 'max': return Math.max(args[0], args[1]);
      case 'min': return Math.min(args[0], args[1]);
      case 'neg': return -args[0];
      case 'ceildiv': return Math.ceil(args[0] / args[1]);
      default: throw new Error(`Unknown SymInt op: ${expr.type}`);
    }
  }

  static freeVars(expr: SymExpr, result: Set<string> = new Set()): Set<string> {
    if (typeof expr === 'number') return result;
    if (!(expr instanceof SymInt)) return result;
    if (expr.type === 'var') {
      result.add(expr.name as string);
      return result;
    }
    for (const a of expr.args) SymInt.freeVars(a, result);
    return result;
  }

  static isConst(expr: SymExpr): boolean {
    return typeof expr === 'number';
  }

  static toConst(expr: SymExpr): number | null {
    if (typeof expr === 'number') return expr;
    return null;
  }

  toString(): string {
    if (this.type === 'var') return this.name as string;
    if (this.type === 'add') return `(${this.args[0]} + ${this.args[1]})`;
    if (this.type === 'sub') return `(${this.args[0]} - ${this.args[1]})`;
    if (this.type === 'mul') return `(${this.args[0]} * ${this.args[1]})`;
    if (this.type === 'div') return `(${this.args[0]} / ${this.args[1]})`;
    if (this.type === 'mod') return `(${this.args[0]} % ${this.args[1]})`;
    if (this.type === 'max') return `max(${this.args[0]}, ${this.args[1]})`;
    if (this.type === 'min') return `min(${this.args[0]}, ${this.args[1]})`;
    if (this.type === 'neg') return `(-${this.args[0]})`;
    if (this.type === 'ceildiv') return `ceildiv(${this.args[0]}, ${this.args[1]})`;
    return 'unknown';
  }
}
