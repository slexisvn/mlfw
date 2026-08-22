import { TensorType } from '../ir/graph/types.js';
import { isConstantOp } from '../ir/graph/op_traits.js';
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

const OFFSET_OPS: ReadonlySet<string> = new Set(['add', 'sub']);

function scalarIntConstant(value: Value): number | null {
  const def = value.definingOp;
  if (!def) return null;
  if (VALUE_PRESERVING_RESHAPES.has(def.opName)) return scalarIntConstant(def.getOperand(0));
  const raw = isConstantOp(def.opName) ? def.getAttr('value') : undefined;
  return typeof raw === 'number' && Number.isInteger(raw) ? raw : null;
}

type TracedArg = { argIndex: number; offset: number };

function traceToArg(value: Value, argIndex: ReadonlyMap<Value, number>): TracedArg | undefined {
  let v: Value | null = value;
  let offset = 0;
  while (v) {
    const direct = argIndex.get(v);
    if (direct !== undefined) return { argIndex: direct, offset };
    const def: Operation | null = v.definingOp;
    if (!def) return undefined;
    if (VALUE_PRESERVING_RESHAPES.has(def.opName)) {
      v = def.getOperand(0);
      continue;
    }
    if (OFFSET_OPS.has(def.opName) && def.numOperands === 2) {
      const rhs = scalarIntConstant(def.getOperand(1));
      if (rhs !== null) {
        offset += def.opName === 'add' ? rhs : -rhs;
        v = def.getOperand(0);
        continue;
      }
      const lhs = def.opName === 'add' ? scalarIntConstant(def.getOperand(0)) : null;
      if (lhs !== null) {
        offset += lhs;
        v = def.getOperand(1);
        continue;
      }
    }
    return undefined;
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
    const traced = traceToArg(op.getOperand(spec.indices), argIndex);
    if (traced === undefined) continue;
    const dim = indexedDim(op);
    if (dim === null) continue;
    const limit = tableExtent(op.getOperand(spec.table), dim);
    if (limit <= 0) continue;
    const lo = -traced.offset;
    const hi = limit - traced.offset;
    const key = traced.argIndex + ':' + lo + ':' + hi + ':' + op.opName;
    if (seen.has(key)) continue;
    seen.add(key);
    bounds.push({ argIndex: traced.argIndex, lo, hi, opName: op.opName });
  }
  return bounds;
}

export function userArgIndexBounds(module: GraphModule, numUserInputs: number): ArgIndexBound[] {
  const func = module.functions().next().value as GraphFunction | undefined;
  if (!func) return [];
  return collectArgIndexBounds(func).filter(b => b.argIndex < numUserInputs);
}
