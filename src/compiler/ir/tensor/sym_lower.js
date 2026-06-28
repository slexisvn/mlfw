import { SymInt } from '../../analysis/sym_int.js';
import { IntImmNode, MathOpNode } from './nodes.js';

function symOpToNode(type, a, b) {
  switch (type) {
    case 'add': return new MathOpNode('+', a, b);
    case 'sub': return new MathOpNode('-', a, b);
    case 'mul': return new MathOpNode('*', a, b);
    case 'div': return new MathOpNode('//', a, b);
    case 'mod': return new MathOpNode('%', a, b);
    case 'neg': return new MathOpNode('-', a);
    case 'ceildiv':
      return new MathOpNode('//', new MathOpNode('-', new MathOpNode('+', a, b), new IntImmNode(1)), b);
    default:
      throw new Error(`symIntToNode: unsupported op '${type}' in extent/index context`);
  }
}

export function symIntToNode(sym, varNode) {
  if (typeof sym === 'number') return new IntImmNode(sym);
  if (!(sym instanceof SymInt)) return new IntImmNode(sym);
  if (sym.type === 'var') return varNode(sym.name);
  const a = symIntToNode(sym.args[0], varNode);
  const b = sym.args.length > 1 ? symIntToNode(sym.args[1], varNode) : null;
  return symOpToNode(sym.type, a, b);
}
