import { TensorType, isFloatType } from '../ir/graph/types.js';

export class SensitivityResult {
  constructor(scores) {
    this.scores = scores;
  }

  getScore(op) {
    return this.scores.get(op) || 0;
  }

  isSensitive(op, threshold) {
    return this.getScore(op) < threshold;
  }
}

export class SensitivityAnalysis {
  static compute(func, calibrationResult, scheme, numBits = 8) {
    const scores = new Map();
    const maxQuant = (1 << (numBits - 1)) - 1;

    for (const op of func.ops()) {
      if (op.opName === 'return' || op.opName === 'yield') continue;

      let hasFloatOutput = false;
      for (let i = 0; i < op.numResults; i++) {
        const t = op.getResult(i).type;
        if (t instanceof TensorType && isFloatType(t.dtype)) {
          hasFloatOutput = true;
          break;
        }
      }
      if (!hasFloatOutput) {
        scores.set(op, Infinity);
        continue;
      }

      let sqnr = Infinity;
      for (let i = 0; i < op.numResults; i++) {
        const val = op.getResult(i);
        const range = calibrationResult ? calibrationResult.getRange(val) : null;
        if (!range) continue;

        const absMax = Math.max(Math.abs(range.min), Math.abs(range.max));
        if (absMax === 0) {
          sqnr = Infinity;
          continue;
        }

        const stepSize = absMax / maxQuant;
        const quantNoisePower = (stepSize * stepSize) / 12;
        const signalPower = (absMax * absMax) / 3;
        const ratio = signalPower / quantNoisePower;
        const db = 10 * Math.log10(ratio);

        if (db < sqnr) sqnr = db;
      }

      scores.set(op, sqnr);
    }

    return new SensitivityResult(scores);
  }
}
