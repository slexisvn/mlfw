import { registry } from '../../ir/graph/ops.js';
import { canInlineFuse } from '../lowering/graph_to_tensor.js';
import { Operation } from '../../ir/graph/operation.js';
import { Block, Region } from '../../ir/graph/block.js';
import { topoSortByOperands } from '../../ir/graph/graph_algorithms.js';
import { fuseLocations } from '../../ir/location.js';
import type { Value } from '../../ir/graph/value.js';
import type { FusionGroup } from './fusion_groups.js';

export type OpOrderPredicate = (a: Operation, b: Operation) => boolean;

export function getYieldOp(block: Block): Operation | null {
  let last: Operation | null = null;
  for (const op of block.ops()) last = op;
  return last && last.opName === 'yield' ? last : null;
}

export function fusionSubject(fusionOp: Operation): string {
  const region = fusionOp.regions[0];
  if (!region) return fusionOp.opName;
  const names: string[] = [];
  for (const block of region.blocks) {
    for (const op of block.ops()) {
      if (op.opName !== 'yield') names.push(op.opName);
    }
  }
  return names.length > 0 ? names.join('+') : fusionOp.opName;
}

export function countInnerOps(fusionOp: Operation): number {
  let count = 0;
  const block = fusionOp.regions[0]?.entryBlock;
  if (!block) return 0;
  for (const op of block.ops()) {
    if (op.opName !== 'yield') count++;
  }
  return count;
}

export function countReductions(fusionOp: Operation): number {
  let count = 0;
  const block = fusionOp.regions[0]?.entryBlock;
  if (!block) return 0;
  for (const op of block.ops()) {
    const def = registry.get(op.opName);
    if (def && def.isReduction) count++;
  }
  return count;
}

export function allInnerOpsFusable(fusionOp: Operation): boolean {
  const block = fusionOp.regions[0]?.entryBlock;
  if (!block) return false;
  for (const op of block.ops()) {
    if (op.opName === 'yield') continue;
    if (!canInlineFuse(op.opName)) return false;
  }
  return true;
}

export function blockPositionIndex(block: Block): Map<Operation, number> {
  const idx = new Map<Operation, number>();
  let i = 0;
  for (let cur = block.firstOp; cur; cur = cur._next) idx.set(cur, i++);
  return idx;
}

export function makeComesBefore(block: Block): OpOrderPredicate {
  const idx = blockPositionIndex(block);
  return (a: Operation, b: Operation): boolean => {
    const pa = idx.get(a), pb = idx.get(b);
    return pa !== undefined && pb !== undefined && pa < pb;
  };
}

export function remapOperands(op: Operation, valueMap: ReadonlyMap<Value, Value>): Value[] {
  const mapped: Value[] = new Array(op.numOperands);
  for (let i = 0; i < op.numOperands; i++) {
    const orig = op.getOperand(i);
    mapped[i] = valueMap.get(orig) || orig;
  }
  return mapped;
}

export function materializeFusionGroup(group: FusionGroup, fallbackKind: string): Operation | null {
  const sortedOps = topoSortByOperands(group.ops, (op: Operation) => group.hasOp(op), 'null');
  if (sortedOps === null || sortedOps.length === 0) return null;

  group.invalidateIO();
  const inputValues = group.getInputValues();
  const outputValues = group.getOutputValues();

  const inputTypes = inputValues.map(v => v.type);
  const outputTypes = outputValues.map(v => v.type);

  const bodyRegion = new Region();
  const bodyBlock = new Block(inputTypes);
  bodyRegion.addBlock(bodyBlock);

  const valueMap = new Map<Value, Value>();
  for (let i = 0; i < inputValues.length; i++) {
    valueMap.set(inputValues[i], bodyBlock.arguments[i]);
  }

  for (const op of sortedOps) {
    bodyBlock.pushOp(op.clone(valueMap));
  }

  const yieldValues = outputValues.map(v => {
    const mapped = valueMap.get(v);
    if (mapped === undefined) {
      throw new Error('Fusion materialization: output value not found in valueMap');
    }
    return mapped;
  });
  bodyBlock.pushOp(new Operation('yield', yieldValues, []));

  const fusionOp = new Operation(
    'fusion',
    inputValues,
    outputTypes,
    { fusion_kind: group.kind || fallbackKind },
    [bodyRegion]
  );
  fusionOp.loc = fuseLocations(sortedOps.map(op => op.loc));

  const block = sortedOps[0].parentBlock;
  if (!block) return null;

  const comesBefore = makeComesBefore(block);
  let insertAfter: Operation | null = null;
  for (const val of inputValues) {
    const producer = val.definingOp;
    if (!producer || group.hasOp(producer)) continue;
    if (!insertAfter || !comesBefore(producer, insertAfter)) {
      insertAfter = producer;
    }
  }

  if (insertAfter && insertAfter.parentBlock === block) {
    block.insertAfter(fusionOp, insertAfter);
  } else {
    block.insertBefore(fusionOp, sortedOps[0]);
  }

  for (let i = 0; i < outputValues.length; i++) {
    outputValues[i].replaceAllUsesWith(fusionOp.getResult(i));
  }

  for (const op of sortedOps) {
    op.dropAllOperands();
    if (op.parentBlock) op.parentBlock.removeOp(op);
  }

  return fusionOp;
}
