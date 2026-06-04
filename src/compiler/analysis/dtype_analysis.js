import { registry } from '../ir/graph/ops.js';

export class DtypeResult {
  constructor(dtypes, inconsistencies) {
    this.dtypes = dtypes;
    this.inconsistencies = inconsistencies;
  }

  get isValid() {
    return this.inconsistencies.length === 0;
  }

  dtypeOf(value) {
    return this.dtypes.get(value) || null;
  }
}

export class DtypeAnalysis {
  static get name() { return 'dtype'; }
  static get depKey() { return 'dtype'; }
  static get dependencies() { return []; }

  static compute(func) {
    const dtypes = new Map();
    const inconsistencies = [];

    for (const arg of func.args) {
      dtypes.set(arg, arg.type.dtype || null);
    }

    for (const op of func.ops()) {
      for (let i = 0; i < op.numResults; i++) {
        const t = op.getResult(i).type;
        dtypes.set(op.getResult(i), t.dtype || null);
      }

      const def = registry.get(op.opName);
      if (def && def.verify) {
        const errs = def.verify(op);
        for (const err of errs) {
          inconsistencies.push(err);
        }
      }
    }

    return new DtypeResult(dtypes, inconsistencies);
  }
}
