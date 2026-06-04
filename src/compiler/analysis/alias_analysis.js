import { registry } from '../ir/graph/ops.js';
import { OpTrait } from '../ir/graph/op_registry.js';
import { UseDefAnalysis } from './use_def.js';

export const AliasResult = Object.freeze({
  NoAlias: 0,
  MayAlias: 1,
  MustAlias: 2
});

export class AliasAnalysisResult {
  constructor(bases) {
    this.bases = bases;
  }

  alias(a, b) {
    if (a === b) return AliasResult.MustAlias;

    const basesA = this.bases.get(a) || new Set([a]);
    const basesB = this.bases.get(b) || new Set([b]);

    for (const baseA of basesA) {
      if (basesB.has(baseA)) return AliasResult.MayAlias;
    }

    return AliasResult.NoAlias;
  }

  mayAlias(a, b) {
    return this.alias(a, b) !== AliasResult.NoAlias;
  }
}

export class AliasAnalysis {
  static get name() { return 'alias'; }
  static get depKey() { return 'alias'; }
  static get dependencies() { return [UseDefAnalysis]; }

  static compute(func, deps = {}) {
    const useDef = deps.useDef || UseDefAnalysis.compute(func);
    const bases = new Map();

    const addBase = (val, baseVal) => {
      let set = bases.get(val);
      if (!set) {
        set = new Set();
        bases.set(val, set);
      }
      set.add(baseVal);
    };

    for (const arg of func.args) {
      addBase(arg, arg);
    }

    for (const op of func.ops()) {
      for (let i = 0; i < op.numResults; i++) {
        addBase(op.getResult(i), op.getResult(i));
      }
    }

    for (const op of useDef.topologicalOrder) {
      const def = registry.get(op.opName);
      if (def && def.hasTrait && def.hasTrait(OpTrait.VIEW)) {
        const operand = op.getOperand(0);
        const result = op.getResult(0);

        const operandBases = bases.get(operand) || new Set();
        const resultBases = bases.get(result) || new Set();

        for (const b of operandBases) {
          resultBases.add(b);
        }
      }
    }

    return new AliasAnalysisResult(bases);
  }
}
