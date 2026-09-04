import { FunctionPass, PassResult } from '../pass.js';
import { OpAttrKey } from '../../ir/graph/op_registry.js';
import { registry } from '../../ir/graph/ops.js';
import { isConstantOp } from '../../ir/graph/op_traits.js';
import { capturedValues } from '../../ir/graph/graph_algorithms.js';
import { explainer } from '../explain.js';
import { TraceLevel } from '../../support/trace.js';
import type { Block, Region } from '../../ir/graph/block.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { PassResultValue, PassTarget } from '../pass.js';

type OperandSite = { op: Operation; index: number };

function captureSites(region: Region, wanted: ReadonlySet<Value>): Map<Value, OperandSite[]> {
  const sites = new Map<Value, OperandSite[]>();
  const walk = (block: Block): void => {
    for (const inner of block.ops()) {
      for (let i = 0; i < inner.numOperands; i++) {
        const operand = inner.getOperand(i);
        if (!wanted.has(operand)) continue;
        const found = sites.get(operand);
        if (found) found.push({ op: inner, index: i });
        else sites.set(operand, [{ op: inner, index: i }]);
      }
      for (const nested of inner.regions) for (const block0 of nested.blocks) walk(block0);
    }
  };
  for (const block of region.blocks) walk(block);
  return sites;
}

function rebind(sites: readonly OperandSite[] | undefined, to: Value): void {
  for (const site of sites ?? []) site.op.replaceOperand(site.index, to);
}

function prepend(block: Block, op: Operation): void {
  if (block.firstOp) block.insertBefore(op, block.firstOp);
  else block.pushOp(op);
}

export class IsolateRegionsPass extends FunctionPass {
  constructor() {
    super('IsolateRegionsPass');
  }

  override run(func: PassTarget): PassResultValue {
    const roots: Operation[] = [];
    for (const op of (func as GraphFunction).opsRecursive()) {
      if (op.regions.length > 0) roots.push(op);
    }

    const explain = explainer(this.trace, this.name);
    let sunk = 0;
    let lifted = 0;

    for (const op of roots.reverse()) {
      const def = registry.get(op.opName);
      if (!def || !def.getAttr<boolean>(OpAttrKey.ISOLATED_REGIONS)) continue;

      const free = capturedValues(op);
      if (free.length === 0) continue;
      const wanted = new Set(free);
      const sitesByRegion = op.regions.map((region) => captureSites(region, wanted));

      for (const value of free) {
        const producer = value.definingOp;
        if (producer && isConstantOp(producer.opName)) {
          for (let r = 0; r < op.regions.length; r++) {
            const sites = sitesByRegion[r].get(value);
            if (!sites) continue;
            const clone = producer.clone(new Map());
            prepend(op.regions[r].entryBlock as Block, clone);
            rebind(sites, clone.getResult(value.resultIndex));
          }
          sunk++;
          continue;
        }
        const args = op.regions.map((region) => (region.entryBlock as Block).addArgument(value.type));
        op.appendOperand(value);
        if (op.hasAttr('num_consts')) {
          op.setAttr('num_consts', (op.getAttr<number>('num_consts') as number) + 1);
        }
        for (let r = 0; r < op.regions.length; r++) rebind(sitesByRegion[r].get(value), args[r]);
        lifted++;
      }

      if (explain) {
        explain(op.opName, 'body captures made explicit',
          'a body that reads the block around it cannot be given to a dialect whose regions are isolated, and leaves a captured gradient with no operand to land on',
          { constantsSunk: sunk, operandsAdded: lifted });
      }
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        constantsSunk: sunk, operandsAdded: lifted, level: TraceLevel.DEBUG,
      });
    }
    return sunk + lifted > 0 ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
