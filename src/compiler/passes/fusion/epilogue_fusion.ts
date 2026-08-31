import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { opsLocation } from '../../ir/graph/op_location.js';
import { registry } from '../../ir/graph/ops.js';
import { isBroadcastOp, isConstantOp } from '../../ir/graph/op_traits.js';

import { TraceLevel } from '../../support/trace.js';
import { explainer } from '../explain.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Block } from '../../ir/graph/block.js';
import type { Value } from '../../ir/graph/value.js';
import type { AttrValue, TensorType } from '../../ir/graph/types.js';
import type { PassResultValue, PassTarget } from '../pass.js';
import type { FusionAwareTarget } from '../../support/config_types.js';

export type EpilogueFusionConfig = { maxEpilogueOps?: number; target?: Partial<FusionAwareTarget> | null };
type EpilogueTagFn = (op: Operation, chainSet: ReadonlySet<Operation>) => string;
type ChainAnalysis = { chain: Operation[]; chainSet: Set<Operation>; tags: string[]; lastOp: Operation; extras: Value[] };
type PrologueAnalysis = { lhsCast: string | null; rhsCast: string | null; lhsInput: Value | null; rhsInput: Value | null; removed: Set<Operation> };

function isPassthrough(op: Operation): boolean {
  return isBroadcastOp(op.opName) || isConstantOp(op.opName);
}

function isEpilogueCandidate(op: Operation): boolean {
  if (isPassthrough(op)) return true;
  const def = registry.get(op.opName);
  return def !== null && def.isElementwise;
}

const EPILOGUE_TAG_TABLE = new Map<string, EpilogueTagFn>([
  ['add',     (op, chainSet) => chainSet.has(resolveOtherOperand(op, chainSet) as Operation) ? 'residual_add' : 'bias'],
  ['sub',     () => 'bias'],
  ['mul',     () => 'scale'],
  ['maximum', () => 'relu'],
  ['clamp',   () => 'clamp'],
  ['neg',     () => 'neg'],
  ['exp',     () => 'exp'],
  ['tanh',    () => 'tanh'],
  ['sqrt',    () => 'sqrt'],
  ['abs',     () => 'abs'],
  ['log',     () => 'log'],
]);

function resolveOtherOperand(op: Operation, chainSet: ReadonlySet<Operation>): Operation | null {
  const op0Def = op.getOperand(0).definingOp;
  if (op0Def && chainSet.has(op0Def) && !isPassthrough(op0Def)) {
    return op.getOperand(1).definingOp;
  }
  return op0Def;
}

for (const [opName, tagFn] of EPILOGUE_TAG_TABLE) {
  if (registry.has(opName)) registry.registerOpAttr(opName, 'epilogueTag', tagFn);
}

function comesBefore(opA: Operation, opB: Operation): boolean {
  if (!opA.parentBlock || opA.parentBlock !== opB.parentBlock) return false;
  let cur = opA.parentBlock.firstOp;
  while (cur) {
    if (cur === opA) return true;
    if (cur === opB) return false;
    cur = cur._next;
  }
  return false;
}

function hasEscapingUse(removedSet: ReadonlySet<Operation>, lastOp: Operation): boolean {
  for (const op of removedSet) {
    if (op === lastOp) continue;
    for (let j = 0; j < op.numResults; j++) {
      for (const use of op.getResult(j).uses()) {
        if (!removedSet.has(use.user)) return true;
      }
    }
  }
  return false;
}

function classifyTag(op: Operation, chainSet: ReadonlySet<Operation>): string {
  const def = registry.get(op.opName);
  const fn = def && def.getAttr<EpilogueTagFn>('epilogueTag');
  if (fn) return fn(op, chainSet);
  return 'activation';
}

function collectChainAndAnalyze(dotOp: Operation): ChainAnalysis {
  const chain: Operation[] = [];
  const chainSet = new Set<Operation>();
  const visited = new Set<Operation>();

  function absorb(op: Operation): void {
    if (visited.has(op) || op === dotOp) return;
    visited.add(op);
    for (let i = 0; i < op.numOperands; i++) {
      const defOp = op.getOperand(i).definingOp;
      if (defOp && defOp !== dotOp && !visited.has(defOp) && isPassthrough(defOp)) {
        absorb(defOp);
      }
    }
    chain.push(op);
    chainSet.add(op);
  }

  const worklist: Operation[] = [];
  const dotResult = dotOp.getResult(0);
  for (const use of dotResult.uses()) {
    if (isEpilogueCandidate(use.user)) worklist.push(use.user);
  }

  while (worklist.length > 0) {
    const op = worklist.pop() as Operation;
    if (visited.has(op)) continue;
    absorb(op);
    for (let i = 0; i < op.numResults; i++) {
      for (const use of op.getResult(i).uses()) {
        if (!visited.has(use.user) && isEpilogueCandidate(use.user)) {
          worklist.push(use.user);
        }
      }
    }
  }

  const tags: string[] = [];
  for (const op of chain) {
    if (isPassthrough(op)) continue;
    tags.push(classifyTag(op, chainSet));
  }

  let lastOp = chain[chain.length - 1];
  for (let i = chain.length - 1; i >= 0; i--) {
    const op = chain[i];
    for (let j = 0; j < op.numResults; j++) {
      for (const use of op.getResult(j).uses()) {
        if (!chainSet.has(use.user)) {
          lastOp = op;
          i = -1;
          break;
        }
      }
      if (i < 0) break;
    }
  }

  const dotOperands = new Set<Value>();
  for (let i = 0; i < dotOp.numOperands; i++) dotOperands.add(dotOp.getOperand(i));

  const extras: Value[] = [];
  const seenVals = new Set<Value>();
  for (const op of chain) {
    for (let i = 0; i < op.numOperands; i++) {
      const val = op.getOperand(i);
      if (seenVals.has(val)) continue;
      seenVals.add(val);
      const defOp = val.definingOp;
      if (defOp === dotOp) continue;
      if (defOp && chainSet.has(defOp)) continue;
      if (dotOperands.has(val)) continue;
      extras.push(val);
    }
  }

  return { chain, chainSet, tags, lastOp, extras };
}

function collectPrologue(dotOp: Operation): PrologueAnalysis {
  let lhsCast: string | null = null, rhsCast: string | null = null, lhsInput: Value | null = null, rhsInput: Value | null = null;
  const removed = new Set<Operation>();
  for (let i = 0; i < 2; i++) {
    const operand = dotOp.getOperand(i);
    const def = operand.definingOp;
    if (!def || def.opName !== 'convert') continue;
    let externalUse = false;
    for (const use of operand.uses()) {
      if (use.user !== dotOp) { externalUse = true; break; }
    }
    if (externalUse) continue;
    const targetDtype = def.getAttr<string>('target_dtype') || (def.getResult(0).type as TensorType).dtype;
    if (i === 0) { lhsCast = targetDtype; lhsInput = def.getOperand(0); }
    else { rhsCast = targetDtype; rhsInput = def.getOperand(0); }
    removed.add(def);
  }
  return { lhsCast, rhsCast, lhsInput, rhsInput, removed };
}

export class EpilogueFusionPass extends FunctionPass {
  maxEpilogueOps: number;
  target: Partial<FusionAwareTarget> | null;

  constructor(config: EpilogueFusionConfig = {}) {
    super('EpilogueFusionPass');
    this.maxEpilogueOps = config.maxEpilogueOps || 16;
    this.target = config.target || null;
  }

  override run(func: PassTarget): PassResultValue {
    if (this.target && !this.target.enableEpilogueFusion) {
      return PassResult.UNCHANGED;
    }

    const explain = explainer(this.trace, this.name);
    let changed = false;

    const dots: Operation[] = [];
    for (const op of (func as GraphFunction).ops()) {
      const def = registry.get(op.opName);
      if (def && def.isOutEWiseFusable) dots.push(op);
    }

    for (const dotOp of dots) {
      const analysis = collectChainAndAnalyze(dotOp);
      if (analysis.chain.length > this.maxEpilogueOps) {
        if (explain) {
          explain(dotOp.opName, 'tail left unfused',
            `the elementwise chain behind it is ${analysis.chain.length} ops long, past the ${this.maxEpilogueOps}-op budget for one epilogue`,
            { chainLength: analysis.chain.length, maxEpilogueOps: this.maxEpilogueOps });
        }
        continue;
      }

      const prologue = collectPrologue(dotOp);
      const hasEpilogue = analysis.chain.length > 0;
      const hasPrologue = prologue.lhsCast !== null || prologue.rhsCast !== null;
      if (!hasEpilogue && !hasPrologue) continue;

      const { chain, tags, extras } = analysis;

      let extrasConsumed = 0;
      for (const tag of tags) {
        if (tag === 'bias' || tag === 'residual_add' || tag === 'scale') extrasConsumed++;
        else if (tag === 'clamp') extrasConsumed += 2;
      }
      if (extrasConsumed !== extras.length) continue;

      const lhsOperand = prologue.lhsInput || dotOp.getOperand(0);
      const rhsOperand = prologue.rhsInput || dotOp.getOperand(1);
      const allInputs = [lhsOperand, rhsOperand, ...extras];
      const lastOp = hasEpilogue ? analysis.lastOp : dotOp;
      const outputType = lastOp.getResult(0).type;

      const attrs = new Map<string, AttrValue>(dotOp.attributes);
      attrs.set('epilogue_ops', chain.map(o => o.opName));
      attrs.set('epilogue_tags', tags);
      attrs.set('num_dot_operands', 2);
      attrs.set('num_extra_inputs', extras.length);
      if (prologue.lhsCast) attrs.set('lhs_prologue_cast', prologue.lhsCast);
      if (prologue.rhsCast) attrs.set('rhs_prologue_cast', prologue.rhsCast);

      const fusedOp = new Operation(
        'fused_dot_epilogue',
        allInputs,
        [outputType],
        attrs
      );

      const block = dotOp.parentBlock;
      if (!block) continue;

      const removedSet = new Set(chain);
      removedSet.add(dotOp);
      for (const r of prologue.removed) removedSet.add(r);
      fusedOp.loc = opsLocation(removedSet);

      if (hasEscapingUse(removedSet, lastOp)) {
        if (explain) {
          explain(dotOp.opName, 'tail left unfused',
            'an op in the chain is read from outside it, so folding the chain away would delete a value someone else still needs');
        }
        continue;
      }

      let insertAfter: Operation | null = null;
      for (const val of allInputs) {
        const producer = val.definingOp;
        if (!producer || removedSet.has(producer)) continue;
        if (!insertAfter || !comesBefore(producer, insertAfter)) {
          insertAfter = producer;
        }
      }

      if (insertAfter && insertAfter.parentBlock === block) {
        block.insertAfter(fusedOp, insertAfter);
      } else {
        block.insertBefore(fusedOp, dotOp);
      }
      lastOp.getResult(0).replaceAllUsesWith(fusedOp.getResult(0));

      for (let i = chain.length - 1; i >= 0; i--) {
        chain[i].dropAllOperands();
        if (chain[i].parentBlock) (chain[i].parentBlock as Block).removeOp(chain[i]);
      }
      dotOp.dropAllOperands();
      if (dotOp.parentBlock) dotOp.parentBlock.removeOp(dotOp);
      for (const r of prologue.removed) {
        r.dropAllOperands();
        if (r.parentBlock) r.parentBlock.removeOp(r);
      }

      if (explain) {
        explain(dotOp.opName, `absorbed ${chain.length} elementwise ops into its tail`,
          'the chain reads the dot output once and elementwise, so it can run on each tile while that tile is still in registers',
          { epilogueOps: chain.map(o => o.opName).join('+') });
      }
      changed = true;
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        dotsFound: dots.length, changed,
        level: TraceLevel.DEBUG,
      });
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
