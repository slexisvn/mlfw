import { ModulePass, PassResult } from './pass.js';
import { Schedule } from '../schedule/schedule.js';
import { SchedulePolicy } from '../schedule/rules.js';

export class SchedulePlanningPass extends ModulePass {
  constructor(target, options = {}) {
    super('SchedulePlanning');
    this.target = target;
    this.policy = options.policy || new SchedulePolicy(target);
    this.precomputedTraces = options.precomputedTraces || null;
  }

  run(module, analysisManager) {
    if (!module.tensorFuncs && !module._tensorFuncs) {
      return PassResult.UNCHANGED;
    }

    const funcs = module.tensorFuncs || module._tensorFuncs || [];
    let changed = false;

    for (const func of funcs) {
      if (this.precomputedTraces && this.precomputedTraces.has(func.name)) {
        const trace = this.precomputedTraces.get(func.name);
        const sch = new Schedule(func);
        trace.replay(sch);
        changed = true;
        continue;
      }

      const sch = new Schedule(func);
      const applied = this.policy.applyToAllBlocks(sch);

      if (applied.size > 0) {
        changed = true;
      }

      const errors = sch.verify();
      if (errors.length > 0) {
        throw new Error(`Schedule validation failed for ${func.name}: ${errors.join('; ')}`);
      }
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
