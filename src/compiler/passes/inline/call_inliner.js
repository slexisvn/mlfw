import { ModulePass, PassResult } from '../pass.js';
import { TraceLevel } from '../../pipeline/trace.js';

function calleeOf(op) {
  return op.opName === 'call' ? op.getAttr('callee') : null;
}

function callsIn(func) {
  const calls = [];
  for (const op of func.opsRecursive()) {
    if (calleeOf(op) !== null) calls.push(op);
  }
  return calls;
}

function terminatorResults(func) {
  const block = func.entryBlock;
  const last = block.lastOp;
  if (!last || last.opName !== 'return') {
    throw new Error(`CallInliner: '${func.name}' does not end in a return`);
  }
  return last.operands;
}

function callGraphOrder(module) {
  const callees = new Map();
  for (const func of module) callees.set(func.name, new Set());
  for (const func of module) {
    for (const op of callsIn(func)) {
      const name = calleeOf(op);
      if (!module.hasFunction(name)) {
        throw new Error(`CallInliner: '${func.name}' calls unknown function '${name}'`);
      }
      callees.get(func.name).add(name);
    }
  }

  const state = new Map();
  const order = [];
  const visit = (name, stack) => {
    const mark = state.get(name);
    if (mark === 'done') return;
    if (mark === 'active') {
      throw new Error(`CallInliner: recursive call cycle ${[...stack, name].join(' -> ')}`);
    }
    state.set(name, 'active');
    for (const next of callees.get(name)) visit(next, [...stack, name]);
    state.set(name, 'done');
    order.push(name);
  };
  for (const func of module) visit(func.name, []);
  return order;
}

function inlineCall(op, callee) {
  const block = op.parentBlock;
  const valueMap = new Map();
  const args = callee.entryBlock.arguments;
  for (let i = 0; i < args.length; i++) valueMap.set(args[i], op.getOperand(i));

  for (const inner of callee.entryBlock) {
    if (inner.opName === 'return') continue;
    block.insertBefore(inner.clone(valueMap), op);
  }

  const returned = terminatorResults(callee);
  for (let i = 0; i < op.numResults; i++) {
    const produced = valueMap.get(returned[i]) || returned[i];
    op.getResult(i).replaceAllUsesWith(produced);
  }
  block.removeOp(op);
}

export class CallInlinerPass extends ModulePass {
  constructor() {
    super('CallInlinerPass');
  }

  run(module) {
    const order = callGraphOrder(module);
    let inlined = 0;
    for (const name of order) {
      const func = module.getFunction(name);
      for (const op of callsIn(func)) {
        inlineCall(op, module.getFunction(calleeOf(op)));
        inlined++;
      }
      if (inlined > 0) func.bumpVersion();
    }
    if (inlined === 0) return PassResult.UNCHANGED;

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({ type: 'pass_detail', passName: this.name, inlinedCalls: inlined, level: TraceLevel.DEBUG });
    }
    return PassResult.CHANGED;
  }
}
