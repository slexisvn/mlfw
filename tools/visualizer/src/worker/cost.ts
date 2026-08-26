import { TensorType } from 'mlfw/compiler/ir/graph/types.js';
import { dtypeBytes } from 'mlfw/util/dtype_map.js';
import type { GraphFunction } from 'mlfw/compiler/ir/graph/function.js';
import type { Operation } from 'mlfw/compiler/ir/graph/operation.js';
import type { Value } from 'mlfw/compiler/ir/graph/value.js';

export type Cost = { bytes: number; flops: number };

const FUSION_OPS = new Set(['fusion', 'fused_dot_epilogue']);
const NO_COST = new Set(['constant', 'return', 'yield']);
const MULTIPLY_ADD = 2;

function numel(type: TensorType): number | null {
  let total = 1;
  for (const dim of type.shape) {
    if (typeof dim !== 'number' || dim < 0) return null;
    total *= dim;
  }
  return total;
}

function bytesOf(value: Value): number {
  const type = value.type;
  if (!(type instanceof TensorType)) return 0;
  const count = numel(type);
  return count === null ? 0 : count * dtypeBytes(type.dtype);
}

function contractedElements(op: Operation): number | null {
  const lhs = op.getOperand(0).type;
  const result = op.getResult(0).type;
  if (!(lhs instanceof TensorType) || !(result instanceof TensorType)) return null;

  const contracting = op.getAttr<readonly number[]>('lhs_contracting') ?? [];
  let inner = 1;
  for (const axis of contracting) {
    const dim = lhs.shape[axis];
    if (typeof dim !== 'number' || dim < 0) return null;
    inner *= dim;
  }

  const out = numel(result);
  return out === null ? null : out * inner * MULTIPLY_ADD;
}

function flopsOf(op: Operation): number {
  if (NO_COST.has(op.opName)) return 0;

  if (op.opName === 'dot') return contractedElements(op) ?? 0;

  if (op.opName === 'conv') {
    const input = op.getOperand(0).type;
    const kernel = op.getOperand(1).type;
    const result = op.getResult(0).type;
    if (!(input instanceof TensorType) || !(kernel instanceof TensorType) || !(result instanceof TensorType)) return 0;
    const out = numel(result);
    const window = numel(kernel);
    const channels = numel(input);
    if (out === null || window === null || channels === null || out === 0) return 0;
    return out * (window / Math.max(result.shape[1] as number, 1)) * MULTIPLY_ADD;
  }

  if (op.opName === 'reduce') {
    const input = op.getOperand(0).type;
    if (!(input instanceof TensorType)) return 0;
    return numel(input) ?? 0;
  }

  const result = op.numResults > 0 ? op.getResult(0).type : null;
  if (!(result instanceof TensorType)) return 0;
  return numel(result) ?? 0;
}

function innerFlops(op: Operation): number {
  let total = 0;
  for (const region of op.regions) {
    for (const block of region.blocks) {
      for (const inner of block.ops()) total += innerFlops(inner) + flopsOf(inner);
    }
  }
  return total;
}

export function functionCost(func: GraphFunction): Cost {
  let bytes = 0;
  let flops = 0;

  for (const block of func.body) {
    for (const op of block.ops()) {
      if (NO_COST.has(op.opName)) continue;

      for (const operand of op.operands) bytes += bytesOf(operand);
      for (const result of op.results) bytes += bytesOf(result);

      flops += FUSION_OPS.has(op.opName) || op.regions.length > 0 ? innerFlops(op) : flopsOf(op);
    }
  }

  return { bytes, flops };
}

export function sumCosts(costs: Iterable<Cost>): Cost {
  let bytes = 0;
  let flops = 0;
  for (const cost of costs) {
    bytes += cost.bytes;
    flops += cost.flops;
  }
  return { bytes, flops };
}
