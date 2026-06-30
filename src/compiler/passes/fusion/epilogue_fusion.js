import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { registry } from '../../ir/graph/ops.js';
import { isBroadcastOp, isConstantOp } from '../../ir/graph/op_traits.js';

import { TraceLevel } from '../../pipeline/trace.js';

function isPassthrough(op) {
  return isBroadcastOp(op.opName) || isConstantOp(op.opName);
}

function isEpilogueCandidate(op) {
  if (isPassthrough(op)) return true;
  const def = registry.get(op.opName);
  return def !== null && def.isElementwise;
}

const EPILOGUE_TAG_TABLE = new Map([
  ['add',     (op, chainSet) => chainSet.has(resolveOtherOperand(op, chainSet)) ? 'residual_add' : 'bias'],
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

function resolveOtherOperand(op, chainSet) {
  const op0Def = op.getOperand(0).definingOp;
  if (op0Def && chainSet.has(op0Def) && !isPassthrough(op0Def)) {
    return op.getOperand(1).definingOp;
  }
  return op0Def;
}

for (const [opName, tagFn] of EPILOGUE_TAG_TABLE) {
  if (registry.has(opName)) registry.registerOpAttr(opName, 'epilogueTag', tagFn);
}

function comesBefore(opA, opB) {
  if (!opA.parentBlock || opA.parentBlock !== opB.parentBlock) return false;
  let cur = opA.parentBlock.firstOp;
  while (cur) {
    if (cur === opA) return true;
    if (cur === opB) return false;
    cur = cur._next;
  }
  return false;
}

function hasEscapingUse(removedSet, lastOp) {
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

function classifyTag(op, chainSet) {
  const def = registry.get(op.opName);
  const fn = def && def.getAttr('epilogueTag');
  if (fn) return fn(op, chainSet);
  return 'activation';
}

function collectChainAndAnalyze(dotOp) {
  const chain = [];
  const chainSet = new Set();
  const visited = new Set();

  function absorb(op) {
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

  const worklist = [];
  const dotResult = dotOp.getResult(0);
  for (const use of dotResult.uses()) {
    if (isEpilogueCandidate(use.user)) worklist.push(use.user);
  }

  while (worklist.length > 0) {
    const op = worklist.pop();
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

  const tags = [];
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

  const dotOperands = new Set();
  for (let i = 0; i < dotOp.numOperands; i++) dotOperands.add(dotOp.getOperand(i));

  const extras = [];
  const seenVals = new Set();
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

function collectPrologue(dotOp) {
  let lhsCast = null, rhsCast = null, lhsInput = null, rhsInput = null;
  const removed = new Set();
  for (let i = 0; i < 2; i++) {
    const operand = dotOp.getOperand(i);
    const def = operand.definingOp;
    if (!def || def.opName !== 'convert') continue;
    let externalUse = false;
    for (const use of operand.uses()) {
      if (use.user !== dotOp) { externalUse = true; break; }
    }
    if (externalUse) continue;
    const targetDtype = def.getAttr('target_dtype') || def.getResult(0).type.dtype;
    if (i === 0) { lhsCast = targetDtype; lhsInput = def.getOperand(0); }
    else { rhsCast = targetDtype; rhsInput = def.getOperand(0); }
    removed.add(def);
  }
  return { lhsCast, rhsCast, lhsInput, rhsInput, removed };
}

export class EpilogueFusionPass extends FunctionPass {
  constructor(config = {}) {
    super('EpilogueFusionPass');
    this.maxEpilogueOps = config.maxEpilogueOps || 16;
    this.target = config.target || null;
  }

  run(func) {
    if (this.target && !this.target.enableEpilogueFusion) {
      return PassResult.UNCHANGED;
    }

    let changed = false;

    const dots = [];
    for (const op of func.ops()) {
      const def = registry.get(op.opName);
      if (def && def.isOutEWiseFusable) dots.push(op);
    }

    for (const dotOp of dots) {
      const analysis = collectChainAndAnalyze(dotOp);
      if (analysis.chain.length > this.maxEpilogueOps) continue;

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

      const attrs = new Map(dotOp.attributes);
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

      if (hasEscapingUse(removedSet, lastOp)) continue;

      let insertAfter = null;
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
        if (chain[i].parentBlock) chain[i].parentBlock.removeOp(chain[i]);
      }
      dotOp.dropAllOperands();
      if (dotOp.parentBlock) dotOp.parentBlock.removeOp(dotOp);
      for (const r of prologue.removed) {
        r.dropAllOperands();
        if (r.parentBlock) r.parentBlock.removeOp(r);
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
