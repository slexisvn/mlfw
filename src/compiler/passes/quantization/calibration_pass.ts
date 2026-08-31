import { ModulePass, PassResult } from '../pass.js';
import { collectCalibration } from '../../analysis/calibrate_exec.js';
import { TraceLevel } from '../../support/trace.js';
import type { CalibrationTarget, CompileFn } from '../../analysis/calibrate_exec.js';
import type { QuantizationConfig } from './quantization_pass.js';
import type { GraphModule } from '../../ir/graph/module.js';
import type { PassResultValue, PassTarget } from '../pass.js';

export type CalibrationPassOpts = Readonly<{
  quantConfig: QuantizationConfig;
  batches: readonly unknown[];
  mode?: string;
  target: CalibrationTarget;
  compileFn?: CompileFn;
}>;

export class CalibrationPass extends ModulePass {
  private _quantConfig: QuantizationConfig;
  private _batches: readonly unknown[];
  private _mode: string;
  private _target: CalibrationTarget;
  private _compileFn: CompileFn | undefined;

  constructor(opts: CalibrationPassOpts) {
    super('CalibrationPass');
    this._quantConfig = opts.quantConfig;
    this._batches = opts.batches;
    this._mode = opts.mode || 'minmax';
    this._target = opts.target;
    this._compileFn = opts.compileFn;
  }

  override run(module: PassTarget): PassResultValue {
    if (this._quantConfig.calibration) return PassResult.UNCHANGED;
    const graphModule = module as GraphModule;
    const entry = graphModule.functionNames()[0];
    const func = entry ? graphModule.getFunction(entry) : null;
    if (!func) return PassResult.UNCHANGED;

    const result = collectCalibration(func, this._target, this._batches, {
      mode: this._mode,
      quantizableOps: this._quantConfig.quantizableOps,
      compileFn: this._compileFn,
    });
    this._quantConfig.calibration = result;

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      let observed = 0;
      for (const value of result.values()) if (result.hasData(value)) observed++;
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        funcName: func.name, batches: this._batches.length, observedValues: observed,
        level: TraceLevel.DEBUG,
      });
    }

    return PassResult.UNCHANGED;
  }
}
