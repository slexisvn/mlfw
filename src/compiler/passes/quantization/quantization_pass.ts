import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { derivedFrom } from '../../ir/graph/op_location.js';
import { TensorType, ScalarType, isFloatType, scalarBytes } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';
import { QuantizationScheme, QuantizationParams, DEFAULT_EXCLUDE_OPS, DEFAULT_QUANTIZABLE_OPS } from '../../ir/graph/quantization_types.js';
import type { QuantizationSchemeValue } from '../../ir/graph/quantization_types.js';
import { TraceLevel } from '../../support/trace.js';
import { explainer } from '../explain.js';
import { isTerminatorOp } from '../../ir/graph/op_traits.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Value } from '../../ir/graph/value.js';
import type { AttrValue, ScalarDType, TensorType as TensorTypeT } from '../../ir/graph/types.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';
import type { CompileTarget } from '../../support/config_types.js';

type QuantParams = InstanceType<typeof QuantizationParams>;
export type CalibrationSource = {
  hasData(value: Value): boolean;
  getQuantParams(value: Value, scheme: QuantizationSchemeValue, dtype: ScalarDType): QuantParams | null;
};
type QuantAttrs = Record<string, AttrValue>;
type SensitivityResult = { isSensitive(op: Operation, threshold: number): boolean };

export type QuantizationConfigOpts = {
  scheme?: QuantizationSchemeValue;
  calibration?: CalibrationSource | null;
  targetDtype?: ScalarDType;
  excludeOps?: ReadonlySet<string>;
  quantizableOps?: ReadonlySet<string>;
  sensitivityThreshold?: number;
  sensitivityResult?: SensitivityResult | null;
  weightOnly?: boolean;
  target?: CompileTarget | null;
};


const NATIVE_QUANTIZED_VARIANTS = new Map([
  ['dot', 'quantized_dot'],
  ['conv', 'quantized_conv']
]);

for (const [opName, variant] of NATIVE_QUANTIZED_VARIANTS) {
  if (registry.has(opName)) registry.registerOpAttr(opName, 'quantizedVariant', variant);
}

export class QuantizationConfig {
  scheme: QuantizationSchemeValue;
  calibration: CalibrationSource | null;
  targetDtype: ScalarDType;
  excludeOps: ReadonlySet<string>;
  quantizableOps: ReadonlySet<string>;
  sensitivityThreshold: number;
  sensitivityResult: SensitivityResult | null;
  weightOnly: boolean;
  target: CompileTarget | null;

  constructor(opts: QuantizationConfigOpts = {}) {
    this.scheme = opts.scheme || QuantizationScheme.PER_TENSOR_SYMMETRIC;
    this.calibration = opts.calibration || null;
    this.targetDtype = opts.targetDtype || ScalarType.I8;
    this.excludeOps = opts.excludeOps || DEFAULT_EXCLUDE_OPS;
    this.quantizableOps = opts.quantizableOps || DEFAULT_QUANTIZABLE_OPS;
    this.sensitivityThreshold = opts.sensitivityThreshold || 0;
    this.sensitivityResult = opts.sensitivityResult || null;
    this.weightOnly = opts.weightOnly || false;
    this.target = opts.target || null;
  }
}

export class QuantizationPass extends FunctionPass {
  config: QuantizationConfig;
  private _paramsByValue!: Map<Value, QuantParams>;

  constructor(config: QuantizationConfigOpts | QuantizationConfig = {}) {
    super('QuantizationPass');
    this.requiredAnalyses = [UseDefAnalysis];
    this.config = config instanceof QuantizationConfig ? config : new QuantizationConfig(config);
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const graphFunc = func as GraphFunction;
    const useDef = this.getAnalysis(UseDefAnalysis, graphFunc, analysisManager);

    const topo = useDef.topologicalOrder;
    const quantizedValues = new Set<Value>();
    this._paramsByValue = new Map();
    const cfg = this.config;
    const explain = explainer(this.trace, this.name);
    let changed = false;

    if (cfg.target && !cfg.target.supportsInt8) return PassResult.UNCHANGED;

    for (let i = 0; i < topo.length; i++) {
      const op = topo[i];
      if (isTerminatorOp(op.opName)) continue;
      if (cfg.excludeOps.has(op.opName) || !cfg.quantizableOps.has(op.opName)) {
        if (explain) {
          explain(op.opName, 'left in float',
            'no int8 form of this op is declared, so its operands are dequantized back before it runs');
        }
        changed = this._dequantizeOperands(op, quantizedValues, cfg) || changed;
        continue;
      }

      if (cfg.sensitivityResult && cfg.sensitivityThreshold > 0
          && cfg.sensitivityResult.isSensitive(op, cfg.sensitivityThreshold)) {
        if (explain) {
          explain(op.opName, 'left in float',
            'calibration measured this op above the sensitivity threshold, so int8 would move its output too far',
            { sensitivityThreshold: cfg.sensitivityThreshold });
        }
        changed = this._dequantizeOperands(op, quantizedValues, cfg) || changed;
        continue;
      }

      if (cfg.weightOnly && !hasConstantOperand(op)) {
        if (explain) {
          explain(op.opName, 'left in float',
            'weight-only quantization was asked for and no operand of this op is a compile-time constant');
        }
        changed = this._dequantizeOperands(op, quantizedValues, cfg) || changed;
        continue;
      }

      const opDef = registry.get(op.opName);
      const nativeVariant = opDef ? opDef.getAttr<string>('quantizedVariant') : null;
      if (nativeVariant && allOperandsCanQuantize(op, quantizedValues, cfg)) {
        if (cfg.scheme === QuantizationScheme.PER_CHANNEL && this._canPerChannelDot(op, quantizedValues)) {
          if (explain) {
            explain(op.opName, `replaced with ${nativeVariant}, one scale per channel`,
              'every operand is already integer and a per-channel scale keeps more of the range than one scale for the whole tensor');
          }
          changed = this._replacePerChannelDot(op, cfg) || changed;
        } else {
          if (explain) {
            explain(op.opName, `replaced with ${nativeVariant}`,
              'every operand is already integer, so the op runs in int8 without a round trip through float');
          }
          changed = this._replaceWithNativeQuantized(op, nativeVariant, quantizedValues, cfg) || changed;
        }
        continue;
      }

      if (explain) {
        explain(op.opName, 'wrapped in a dequantize/quantize pair',
          'an int8 form exists but at least one operand is still float, so the boundary is paid here');
      }
      changed = this._insertDequantQuantBoundary(op, quantizedValues, cfg) || changed;
    }

    const retOp = graphFunc.getReturnOp();
    if (retOp) changed = this._dequantizeOperands(retOp, quantizedValues, cfg) || changed;

    if (this.trace && this.trace.level >= TraceLevel.DEBUG && changed) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        opsProcessed: topo.length, changed,
        level: TraceLevel.DEBUG,
      });
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _getQuantParams(value: Value, cfg: QuantizationConfig): QuantParams | null {
    const numBits = scalarBytes(cfg.targetDtype) * 8;
    if (cfg.calibration && cfg.calibration.hasData(value)) {
      return cfg.calibration.getQuantParams(value, cfg.scheme, cfg.targetDtype);
    }
    const defOp = value.definingOp;
    if (defOp && defOp.opName === 'constant') {
      const val = defOp.getAttr<number | readonly number[]>('value') as number | ArrayLike<number>;
      if (typeof val === 'number') return QuantizationParams.fromRange(-Math.abs(val) || -1, Math.abs(val) || 1, cfg.scheme, cfg.targetDtype, numBits);
      if (val && typeof (val as ArrayLike<number>).length === 'number') return QuantizationParams.fromConstantArray(val as ArrayLike<number>, cfg.scheme, cfg.targetDtype, numBits);
    }
    if (value.type instanceof TensorType && isFloatType((value.type as TensorTypeT).dtype as ScalarDType)) {
      return QuantizationParams.defaultForActivation(cfg.scheme, cfg.targetDtype, numBits);
    }
    return null;
  }

  _resolveQuantParams(val: Value, cfg: QuantizationConfig): QuantParams | null {
    const tracked = this._paramsByValue.get(val);
    if (tracked) return tracked;
    const defOp = val.definingOp;
    if (defOp && defOp.opName === 'quantize') {
      return new QuantizationParams({
        scheme: defOp.getAttr<QuantizationSchemeValue>('scheme') || cfg.scheme,
        scale: defOp.getAttr<number>('scale') as number,
        zeroPoint: defOp.getAttr<number>('zero_point') || 0,
        dtype: defOp.getAttr<ScalarDType>('target_dtype') || cfg.targetDtype,
      });
    }
    return this._getQuantParams(val, cfg);
  }

  _insertQuantizeAfter(op: Operation, resultIdx: number, cfg: QuantizationConfig): Value | null {
    const val = op.getResult(resultIdx);
    const qp = this._getQuantParams(val, cfg);
    if (!qp) return null;

    const resultType = new TensorType((val.type as TensorTypeT).shape, cfg.targetDtype);
    const quantOp = new Operation('quantize', [val], [resultType], {
      scale: qp.getScalarScale(),
      zero_point: qp.getScalarZeroPoint(),
      scheme: cfg.scheme,
      target_dtype: cfg.targetDtype,
      ...(qp.axis !== null ? { axis: qp.axis } : {})
    });

    derivedFrom(quantOp, op);
    if (op.parentBlock) {
      op.parentBlock.insertAfter(quantOp, op);
    }

    const result = quantOp.getResult(0);
    this._paramsByValue.set(result, qp);
    return result;
  }

  _insertDequantBefore(consumer: Operation, operandIdx: number, val: Value, cfg: QuantizationConfig): boolean {
    const qp = this._resolveQuantParams(val, cfg);
    if (!qp) return false;

    const outputDtype = ScalarType.F32;
    const resultType = new TensorType((val.type as TensorTypeT).shape, outputDtype);
    const dequantOp = new Operation('dequantize', [val], [resultType], {
      scale: qp.getScalarScale(),
      zero_point: qp.getScalarZeroPoint(),
      scheme: cfg.scheme,
      target_dtype: outputDtype
    });

    derivedFrom(dequantOp, val.definingOp || consumer);
    if (consumer.parentBlock) {
      consumer.parentBlock.insertBefore(dequantOp, consumer);
      consumer.replaceOperand(operandIdx, dequantOp.getResult(0));
    }
    return true;
  }

  _dequantizeOperands(op: Operation, quantizedValues: Set<Value>, cfg: QuantizationConfig): boolean {
    let changed = false;
    for (let i = 0; i < op.numOperands; i++) {
      const val = op.getOperand(i);
      if (quantizedValues.has(val)) {
        changed = this._insertDequantBefore(op, i, val, cfg) || changed;
      }
    }
    return changed;
  }

  _feedsExcludedOp(val: Value, cfg: QuantizationConfig): boolean {
    for (const use of val.uses()) {
      if (cfg.excludeOps.has(use.user.opName)) return true;
    }
    return false;
  }

  _insertDequantQuantBoundary(op: Operation, quantizedValues: Set<Value>, cfg: QuantizationConfig): boolean {
    let changed = this._dequantizeOperands(op, quantizedValues, cfg);

    for (let r = 0; r < op.numResults; r++) {
      const val = op.getResult(r);
      if (!(val.type instanceof TensorType) || !isFloatType((val.type as TensorTypeT).dtype)) continue;
      if (this._feedsExcludedOp(val, cfg)) continue;
      const qResult = this._insertQuantizeAfter(op, r, cfg);
      if (qResult) {
        const uses = [...val.uses()];
        for (const use of uses) {
          if (use.user !== qResult.definingOp) {
            use.user.replaceOperand(use.operandIndex, qResult);
          }
        }
        quantizedValues.add(qResult);
        changed = true;
      }
    }

    return changed;
  }

  _replaceWithNativeQuantized(op: Operation, nativeOpName: string, quantizedValues: Set<Value>, cfg: QuantizationConfig): boolean {
    const quantizedInputs: Value[] = [];
    const attrs: QuantAttrs = {};

    for (const [key, val] of op.attributes || []) {
      attrs[key] = val;
    }

    for (let i = 0; i < op.numOperands; i++) {
      const operand = op.getOperand(i);
      if (quantizedValues.has(operand)) {
        quantizedInputs.push(operand);
        const qp = this._resolveQuantParams(operand, cfg);
        if (qp) {
          const prefix = i === 0 ? 'lhs' : 'rhs';
          if (nativeOpName === 'quantized_dot') {
            attrs[`${prefix}_scale`] = qp.getScalarScale();
            attrs[`${prefix}_zero_point`] = qp.getScalarZeroPoint();
          } else {
            const label = i === 0 ? 'input' : 'kernel';
            attrs[`${label}_scale`] = qp.getScalarScale();
            attrs[`${label}_zero_point`] = qp.getScalarZeroPoint();
          }
        }
        continue;
      }

      const qp = this._getQuantParams(operand, cfg);
      if (!qp) return false;

      const quantType = new TensorType((operand.type as TensorTypeT).shape, cfg.targetDtype);
      const quantOp = new Operation('quantize', [operand], [quantType], {
        scale: qp.getScalarScale(),
        zero_point: qp.getScalarZeroPoint(),
        scheme: cfg.scheme,
        target_dtype: cfg.targetDtype
      });

      if (op.parentBlock) {
        op.parentBlock.insertBefore(quantOp, op);
      }
      quantizedInputs.push(quantOp.getResult(0));

      const prefix = i === 0 ? (nativeOpName === 'quantized_dot' ? 'lhs' : 'input')
                              : (nativeOpName === 'quantized_dot' ? 'rhs' : 'kernel');
      attrs[`${prefix}_scale`] = qp.getScalarScale();
      attrs[`${prefix}_zero_point`] = qp.getScalarZeroPoint();
    }

    const lhsScale = (attrs.lhs_scale || attrs.input_scale || 1) as number;
    const rhsScale = (attrs.rhs_scale || attrs.kernel_scale || 1) as number;
    attrs.output_scale = lhsScale * rhsScale;
    attrs.output_zero_point = 0;

    const resultType = new TensorType((op.getResult(0).type as TensorTypeT).shape, ScalarType.I32);
    const quantizedOp = new Operation(nativeOpName, quantizedInputs, [resultType], attrs);

    if (op.parentBlock) {
      op.parentBlock.insertBefore(quantizedOp, op);

      const dequantType = new TensorType((op.getResult(0).type as TensorTypeT).shape, (op.getResult(0).type as TensorTypeT).dtype);
      const outScale = (attrs.output_scale || 1) as number;
      const dequantOp = new Operation('dequantize', [quantizedOp.getResult(0)], [dequantType], {
        scale: outScale,
        zero_point: attrs.output_zero_point || 0,
        scheme: cfg.scheme,
        target_dtype: (op.getResult(0).type as TensorTypeT).dtype
      });
      op.parentBlock.insertBefore(dequantOp, op);
      op.replaceAllResultsWith([dequantOp.getResult(0)]);
      op.erase();
    }

    return true;
  }

  _canPerChannelDot(op: Operation, quantizedValues: ReadonlySet<Value>): boolean {
    if (op.opName !== 'dot') return false;
    const lhs = op.getOperand(0);
    const rhs = op.getOperand(1);
    if (quantizedValues.has(lhs) || quantizedValues.has(rhs)) return false;
    if (!(lhs.type instanceof TensorType) || (lhs.type as TensorTypeT).shape.length !== 2) return false;
    if (!(rhs.type instanceof TensorType) || (rhs.type as TensorTypeT).shape.length !== 2) return false;
    const rhsDef = rhs.definingOp;
    if (!rhsDef || rhsDef.opName !== 'constant') return false;
    const data = rhsDef.getAttr<number | readonly number[]>('value');
    if (!data || typeof data === 'number' || typeof (data as ArrayLike<number>).length !== 'number') return false;
    const rhsC = op.getAttr<readonly number[]>('rhs_contracting') || [];
    const lhsC = op.getAttr<readonly number[]>('lhs_contracting') || [];
    if (rhsC.length !== 1 || lhsC.length !== 1) return false;
    if ((op.getAttr<readonly number[]>('rhs_batch') || []).length !== 0) return false;
    if ((op.getAttr<readonly number[]>('lhs_batch') || []).length !== 0) return false;
    return true;
  }

  _activationParams(value: Value, cfg: QuantizationConfig): QuantParams | null {
    const numBits = scalarBytes(cfg.targetDtype) * 8;
    const scheme = QuantizationScheme.PER_TENSOR_SYMMETRIC;
    if (cfg.calibration && cfg.calibration.hasData(value)) {
      return cfg.calibration.getQuantParams(value, scheme, cfg.targetDtype);
    }
    return QuantizationParams.defaultForActivation(scheme, cfg.targetDtype, numBits);
  }

  _replacePerChannelDot(op: Operation, cfg: QuantizationConfig): boolean {
    const lhs = op.getOperand(0);
    const rhs = op.getOperand(1);
    const rhsShape = (rhs.type as TensorTypeT).shape;
    const wData = (rhs.definingOp as Operation).getAttr<readonly number[]>('value') as readonly number[];
    const numBits = scalarBytes(cfg.targetDtype) * 8;

    const rhsContracting = op.getAttr<readonly number[]>('rhs_contracting') as readonly number[];
    const channelAxis = rhsContracting[0] === 0 ? 1 : 0;
    const wp = QuantizationParams.fromConstantArrayPerChannel([...wData], rhsShape as readonly number[], channelAxis, cfg.targetDtype, numBits);
    const wInt8 = wp.quantizeArrayPerChannel([...wData], rhsShape as readonly number[]);

    const aqp = this._activationParams(lhs, cfg) as QuantParams;
    const aScale = aqp.getScalarScale();
    const aZp = aqp.getScalarZeroPoint();

    const block = op.parentBlock;
    if (!block) return false;

    const wConstType = new TensorType(rhsShape, cfg.targetDtype);
    const wConst = new Operation('constant', [], [wConstType], { value: wInt8, tensor_type: wConstType });
    block.insertBefore(wConst, op);

    const aqType = new TensorType((lhs.type as TensorTypeT).shape, cfg.targetDtype);
    const aQuant = new Operation('quantize', [lhs], [aqType], {
      scale: aScale, zero_point: aZp,
      scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC, target_dtype: cfg.targetDtype
    });
    block.insertBefore(aQuant, op);

    const outShape = (op.getResult(0).type as TensorTypeT).shape;
    const attrs: QuantAttrs = {};
    for (const [k, v] of op.attributes || []) attrs[k] = v;
    attrs.lhs_scale = aScale; attrs.lhs_zero_point = aZp;
    attrs.rhs_scale = 1; attrs.rhs_zero_point = 0;
    attrs.output_scale = 1; attrs.output_zero_point = 0;

    const i32Type = new TensorType(outShape, ScalarType.I32);
    const qdot = new Operation('quantized_dot', [aQuant.getResult(0), wConst.getResult(0)], [i32Type], attrs);
    block.insertBefore(qdot, op);

    const f32Type = new TensorType(outShape, ScalarType.F32);
    const conv = new Operation('convert', [qdot.getResult(0)], [f32Type], { target_dtype: ScalarType.F32 });
    block.insertBefore(conv, op);

    const numCh = rhsShape[channelAxis] as number;
    const scaleVec: number[] = new Array(numCh);
    for (let c = 0; c < numCh; c++) scaleVec[c] = aScale * wp.getScaleForChannel(c);

    const lhsC = new Set<number>(op.getAttr<readonly number[]>('lhs_contracting') || []);
    let lhsSpatial = 0;
    for (let i = 0; i < (lhs.type as TensorTypeT).shape.length; i++) if (!lhsC.has(i)) lhsSpatial++;
    const outChannelAxis = lhsSpatial;

    const svType = new TensorType([numCh], ScalarType.F32);
    const sv = new Operation('constant', [], [svType], { value: scaleVec, tensor_type: svType });
    block.insertBefore(sv, op);

    const bc = new Operation('broadcast_in_dim', [sv.getResult(0)], [f32Type], {
      broadcast_dimensions: [outChannelAxis], result_shape: outShape
    });
    block.insertBefore(bc, op);

    const mul = new Operation('mul', [conv.getResult(0), bc.getResult(0)], [f32Type], {});
    block.insertBefore(mul, op);

    op.replaceAllResultsWith([mul.getResult(0)]);
    op.erase();
    return true;
  }
}

function hasConstantOperand(op: Operation): boolean {
  for (let i = 0; i < op.numOperands; i++) {
    const def = op.getOperand(i).definingOp;
    if (def && def.opName === 'constant') return true;
  }
  return false;
}

function allOperandsCanQuantize(op: Operation, quantizedValues: ReadonlySet<Value>, cfg: QuantizationConfig): boolean {
  for (let i = 0; i < op.numOperands; i++) {
    const operand = op.getOperand(i);
    if (quantizedValues.has(operand)) continue;
    if (!(operand.type instanceof TensorType) || !isFloatType((operand.type as TensorTypeT).dtype)) return false;
  }
  return op.numOperands > 0;
}
