import { registry } from '../../ir/graph/ops.js';
import { canInlineFuse } from '../lowering/graph_to_tensor.js';

export function getYieldOp(block) {
  let last = null;
  for (const op of block.ops()) last = op;
  return last && last.opName === 'yield' ? last : null;
}

export function countInnerOps(fusionOp) {
  let count = 0;
  const block = fusionOp.regions[0]?.entryBlock;
  if (!block) return 0;
  for (const op of block.ops()) {
    if (op.opName !== 'yield') count++;
  }
  return count;
}

export function countReductions(fusionOp) {
  let count = 0;
  const block = fusionOp.regions[0]?.entryBlock;
  if (!block) return 0;
  for (const op of block.ops()) {
    const def = registry.get(op.opName);
    if (def && def.isReduction) count++;
  }
  return count;
}

export function allInnerOpsFusable(fusionOp) {
  const block = fusionOp.regions[0]?.entryBlock;
  if (!block) return false;
  for (const op of block.ops()) {
    if (op.opName === 'yield') continue;
    if (!canInlineFuse(op.opName)) return false;
  }
  return true;
}

export function blockPositionIndex(block) {
  const idx = new Map();
  let i = 0;
  for (let cur = block.firstOp; cur; cur = cur._next) idx.set(cur, i++);
  return idx;
}

export function makeComesBefore(block) {
  const idx = blockPositionIndex(block);
  return (a, b) => {
    const pa = idx.get(a), pb = idx.get(b);
    return pa !== undefined && pb !== undefined && pa < pb;
  };
}

export function remapOperands(op, valueMap) {
  const mapped = new Array(op.numOperands);
  for (let i = 0; i < op.numOperands; i++) {
    const orig = op.getOperand(i);
    mapped[i] = valueMap.get(orig) || orig;
  }
  return mapped;
}
