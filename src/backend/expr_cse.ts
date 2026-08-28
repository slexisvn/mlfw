import { irChildNodes } from '../compiler/ir/ir_visitor.js';
import { isDtypeInt } from '../util/dtype_map.js';
import { normalizeDtype } from '../compiler/ir/lir/nodes.js';
import type { IRNode } from '../compiler/ir/ir_visitor.js';
import type { IRStmtNode } from '../compiler/ir/lir/nodes.js';

const INT_MATH_OPS = new Set(['+', '-', '*', '//', '%', 'tdiv', 'tmod']);
const DIVIDING_OPS = new Set(['//', '%', 'tdiv', 'tmod']);

export type CseClass = Readonly<{ id: number; node: IRStmtNode }>;
export type CsePlan = Readonly<{ ids: ReadonlyMap<IRStmtNode, number>; hoisted: readonly CseClass[] }>;

type ClassInfo = { node: IRStmtNode; size: number; pure: boolean; refs: number };

function structuralTag(node: IRStmtNode): string | null {
  switch (node.type) {
    case 'IntImmNode': return `I:${node.value}`;
    case 'FloatImmNode': return `F:${node.value}`;
    case 'VariableNode': return `V:${node.name}:${node.dtype}`;
    case 'MathOpNode': return `M:${node.op}`;
    case 'CompareNode': return `C:${node.direction}`;
    case 'CastNode': return `T:${node.fromDtype}>${node.toDtype}`;
    case 'CallExternNode': return `E:${node.externName}:${node.dtype}`;
    case 'BufferLoadNode': return `B:${node.buffer.name}`;
    case 'LIRFlatLoadNode': return `L:${node.buffer.name}`;
    case 'IfThenElseNode': return 'S';
    default: return null;
  }
}

function isSelfPure(node: IRStmtNode): boolean {
  if (node.type === 'IntImmNode') return true;
  if (node.type === 'VariableNode') return isDtypeInt(normalizeDtype(node.dtype || ''));
  if (node.type !== 'MathOpNode') return false;
  if (!INT_MATH_OPS.has(node.op)) return false;
  if (!DIVIDING_OPS.has(node.op)) return true;
  return !!node.b && node.b.type === 'IntImmNode' && node.b.value !== 0;
}

export function planCommonSubexprs(root: IRStmtNode, minSize: number): CsePlan {
  const ids = new Map<IRStmtNode, number>();
  const byKey = new Map<string, number>();
  const classes: ClassInfo[] = [];
  let opaque = 0;

  const visit = (node: IRStmtNode): number => {
    const memo = ids.get(node);
    if (memo !== undefined) return memo;

    const childIds: number[] = [];
    let size = 1;
    let childrenPure = true;
    for (const child of irChildNodes(node as IRNode)) {
      const childId = visit(child as IRStmtNode);
      childIds.push(childId);
      size += classes[childId].size;
      childrenPure = childrenPure && classes[childId].pure;
    }

    const tag = structuralTag(node);
    const key = tag === null ? `O:${opaque++}` : `${tag}|${childIds.join(',')}`;
    let id = byKey.get(key);
    if (id === undefined) {
      id = classes.length;
      byKey.set(key, id);
      classes.push({ node, size, pure: tag !== null && childrenPure && isSelfPure(node), refs: 0 });
    }
    ids.set(node, id);
    for (const childId of childIds) classes[childId].refs++;
    return id;
  };

  visit(root);

  const hoisted: CseClass[] = [];
  for (let id = 0; id < classes.length; id++) {
    const info = classes[id];
    if (info.pure && info.refs >= 2 && info.size >= minSize) hoisted.push({ id, node: info.node });
  }
  return { ids, hoisted };
}
