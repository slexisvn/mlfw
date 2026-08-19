import type { GraphModule } from '../ir/graph/module.js';
import type { GraphFunction } from '../ir/graph/function.js';
import { AnalysisManager } from '../analysis/analysis_manager.js';
import type { AnalysisCtor } from '../analysis/analysis_manager.js';
import type { TraceLog } from '../pipeline/trace.js';

export type PassResultValue = (typeof PassResult)[keyof typeof PassResult];
export type PassTarget = GraphModule | GraphFunction;
export type AnalysisRef = AnalysisCtor | string;
export type PassContextOpts = Readonly<{
  optLevel?: number;
  disabledPasses?: Iterable<string> | ReadonlySet<string>;
  requiredPasses?: Iterable<string> | ReadonlySet<string>;
  config?: ReadonlyMap<string, unknown> | Record<string, unknown>;
}>;

export const PassResult = Object.freeze({
  UNCHANGED: 0,
  CHANGED: 1,
  FAILED: 2
});

export class Pass {
  name: string;
  preservedAnalyses: Set<AnalysisRef>;
  invalidatedAnalyses: Set<AnalysisRef>;
  requiredAnalyses: AnalysisCtor[];
  optLevel: number;
  trace: TraceLog | null;
  private _ownAnalyses: AnalysisManager | null;

  constructor(name: string) {
    this.name = name;
    this.preservedAnalyses = new Set();
    this.invalidatedAnalyses = new Set();
    this.requiredAnalyses = [];
    this.optLevel = 0;
    this.trace = null;
    this._ownAnalyses = null;
  }

  analyses(analysisManager?: AnalysisManager): AnalysisManager {
    if (analysisManager) return analysisManager;
    if (!this._ownAnalyses) this._ownAnalyses = new AnalysisManager();
    return this._ownAnalyses;
  }

  getAnalysis<TResult>(AnalysisClass: AnalysisCtor<TResult>, func: GraphFunction, analysisManager?: AnalysisManager): TResult {
    return this.analyses(analysisManager).getAnalysis(AnalysisClass, func);
  }

  run(target: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    throw new Error('Not implemented');
  }
}

export class PassContext {
  optLevel: number;
  disabledPasses: ReadonlySet<string>;
  requiredPasses: ReadonlySet<string>;
  config: ReadonlyMap<string, unknown>;

  constructor({ optLevel = Infinity, disabledPasses = [], requiredPasses = [], config = {} }: PassContextOpts = {}) {
    this.optLevel = optLevel;
    this.disabledPasses = disabledPasses instanceof Set ? disabledPasses : new Set(disabledPasses);
    this.requiredPasses = requiredPasses instanceof Set ? requiredPasses : new Set(requiredPasses);
    this.config = config instanceof Map ? config : new Map(Object.entries(config as Record<string, unknown>));
  }

  shouldRun(pass: Pass): boolean {
    if (this.disabledPasses.has(pass.name)) return false;
    if (this.requiredPasses.has(pass.name)) return true;
    if ((pass.optLevel || 0) > this.optLevel) return false;
    return true;
  }
}

export class FunctionPass extends Pass {
  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    throw new Error('Not implemented');
  }
}

export class ModulePass extends Pass {
  override run(module: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    throw new Error('Not implemented');
  }
}
