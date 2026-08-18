import { TensorType } from '../ir/graph/types.js';
import type { GraphFunction } from '../ir/graph/function.js';
import type { GraphModule } from '../ir/graph/module.js';
import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';
import type { ArgIndexBound } from '../../util/index_bounds.js';

export type { ArgIndexBound };

const INDEXED_TABLE_OPS: ReadonlyMap<string, { table: number; indices: number }> = new Map([
  ['embedding', { table: 0, indices: 1 }],
  ['gather', { table: 0, indices: 1 }],
]);

const VALUE_PRESERVING_RESHAPES: ReadonlySet<string> = new Set(['reshape', 'transpose', 'reverse', 'broadcast_in_dim']);

function traceToArg(value: Value, argIndex: ReadonlyMap<Value, number>): number | undefined {
  let v: Value | null = value;
  while (v) {
    const direct = argIndex.get(v);
    if (direct !== undefined) return direct;
    const def: Operation | null = v.definingOp;
    if (!def || !VALUE_PRESERVING_RESHAPES.has(def.opName)) return undefined;
    v = def.getOperand(0);
  }
  return undefined;
}

function indexedDim(op: Operation): number | null {
  if (op.opName === 'embedding') return 0;
  const map = op.getAttr('start_index_map');
  return Array.isArray(map) && map.length === 1 && Number.isInteger(map[0]) ? map[0] as number : null;
}

function tableExtent(table: Value, dim: number): number {
  const t = table.type;
  if (!(t instanceof TensorType) || dim < 0 || dim >= t.rank) return 0;
  const extent = t.shape[dim];
  return typeof extent === 'number' ? extent : 0;
}

export function collectArgIndexBounds(func: GraphFunction): ArgIndexBound[] {
  const argIndex = new Map<Value, number>();
  const args = func.args;
  for (let i = 0; i < args.length; i++) argIndex.set(args[i] as unknown as Value, i);

  const seen = new Set<string>();
  const bounds: ArgIndexBound[] = [];
  for (const op of func.ops()) {
    const spec = INDEXED_TABLE_OPS.get(op.opName);
    if (!spec) continue;
    const idx = traceToArg(op.getOperand(spec.indices), argIndex);
    if (idx === undefined) continue;
    const dim = indexedDim(op);
    if (dim === null) continue;
    const limit = tableExtent(op.getOperand(spec.table), dim);
    if (limit <= 0) continue;
    const key = idx + ':' + limit + ':' + op.opName;
    if (seen.has(key)) continue;
    seen.add(key);
    bounds.push({ argIndex: idx, limit, opName: op.opName });
  }
  return bounds;
}

export function userArgIndexBounds(module: GraphModule, numUserInputs: number): ArgIndexBound[] {
  const func = module.functions().next().value as GraphFunction | undefined;
  if (!func) return [];
  return collectArgIndexBounds(func).filter(b => b.argIndex < numUserInputs);
}
