import { SymInt } from '../../analysis/sym_int.js';
import { IntImmNode, MathOpNode, CallExternNode } from './nodes.js';
import type { TirNode } from './nodes.js';
import type { Dim } from '../graph/types.js';

export type SymVarResolver = (name: string) => TirNode;

function symOpToNode(type: string, a: TirNode, b: TirNode | null): TirNode {
  switch (type) {
    case 'add': return new MathOpNode('+', a, b);
    case 'sub': return new MathOpNode('-', a, b);
    case 'mul': return new MathOpNode('*', a, b);
    case 'div': return new MathOpNode('//', a, b);
    case 'mod': return new MathOpNode('%', a, b);
    case 'neg': return new MathOpNode('-', a);
    case 'ceildiv':
      return new MathOpNode('//', new MathOpNode('-', new MathOpNode('+', a, b), new IntImmNode(1)), b);
    case 'max':
    case 'min':
      return new CallExternNode(type, [a, b as TirNode], 'int32');
    default:
      throw new Error(`symIntToNode: unsupported op '${type}' in extent/index context`);
  }
}

export function symIntToNode(sym: Dim, varNode: SymVarResolver): TirNode {
  if (typeof sym === 'number') return new IntImmNode(sym);
  if (!(sym instanceof SymInt)) return new IntImmNode(sym as number);
  if (sym.type === 'var') return varNode(sym.name as string);
  const a: TirNode = symIntToNode(sym.args[0] as Dim, varNode);
  const b: TirNode | null = sym.args.length > 1 ? symIntToNode(sym.args[1] as Dim, varNode) : null;
  return symOpToNode(sym.type as string, a, b);
}
