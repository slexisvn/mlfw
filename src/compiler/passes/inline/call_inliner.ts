import { ModulePass, PassResult } from '../pass.js';
import { TraceLevel } from '../../support/trace.js';
import { explainer } from '../explain.js';
import type { GraphModule } from '../../ir/graph/module.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { Block } from '../../ir/graph/block.js';
import type { PassResultValue, PassTarget } from '../pass.js';

function calleeOf(op: Operation): string | null {
  return op.opName === 'call' ? op.getAttr<string>('callee') ?? null : null;
}

function callsIn(func: GraphFunction): Operation[] {
  const calls: Operation[] = [];
  for (const op of func.opsRecursive()) {
    if (calleeOf(op) !== null) calls.push(op);
  }
  return calls;
}

function terminatorResults(func: GraphFunction): readonly Value[] {
  const block = func.entryBlock as Block;
  const last = block.lastOp;
  if (!last || last.opName !== 'return') {
    throw new Error(`CallInliner: '${func.name}' does not end in a return`);
  }
  return last.operands;
}

function callGraphOrder(module: GraphModule): string[] {
  const callees = new Map<string, Set<string>>();
  for (const func of module) callees.set(func.name, new Set());
  for (const func of module) {
    for (const op of callsIn(func)) {
      const name = calleeOf(op) as string;
      if (!module.hasFunction(name)) {
        throw new Error(`CallInliner: '${func.name}' calls unknown function '${name}'`);
      }
      (callees.get(func.name) as Set<string>).add(name);
    }
  }

  const state = new Map<string, string>();
  const order: string[] = [];
  const visit = (name: string, stack: readonly string[]): void => {
    const mark = state.get(name);
    if (mark === 'done') return;
    if (mark === 'active') {
      throw new Error(`CallInliner: recursive call cycle ${[...stack, name].join(' -> ')}`);
    }
    state.set(name, 'active');
    for (const next of callees.get(name) as Set<string>) visit(next, [...stack, name]);
    state.set(name, 'done');
    order.push(name);
  };
  for (const func of module) visit(func.name, []);
  return order;
}

function inlineCall(op: Operation, callee: GraphFunction): void {
  const block = op.parentBlock as Block;
  const valueMap = new Map<Value, Value>();
  const args = (callee.entryBlock as Block).arguments;
  for (let i = 0; i < args.length; i++) valueMap.set(args[i], op.getOperand(i));

  for (const inner of callee.entryBlock as Block) {
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

  override run(module: PassTarget): PassResultValue {
    const mod = module as GraphModule;
    const order = callGraphOrder(mod);
    const explain = explainer(this.trace, this.name);
    let inlined = 0;
    for (const name of order) {
      const func = mod.getFunction(name) as GraphFunction;
      for (const op of callsIn(func)) {
        const callee = calleeOf(op) as string;
        inlineCall(op, mod.getFunction(callee) as GraphFunction);
        inlined++;
        if (explain) {
          explain(callee, `pasted into ${name}`,
            'one flat graph lets fusion and dead-code elimination cross what used to be a call boundary');
        }
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
