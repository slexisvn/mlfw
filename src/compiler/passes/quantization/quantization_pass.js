import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { TensorType, ScalarType, isFloatType, scalarBytes } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import { OpTrait } from '../../ir/graph/op_registry.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';
import { QuantizationScheme, QuantizationParams } from '../../ir/graph/quantization_types.js';
import { TraceLevel } from '../../pipeline/trace.js';

const DEFAULT_EXCLUDE_OPS = new Set(['softmax', 'sqrt', 'div', 'rsqrt', 'log', 'exp', 'tanh']);
const DEFAULT_QUANTIZABLE_OPS = new Set(['dot', 'conv', 'add', 'mul', 'sub']);

const NATIVE_QUANTIZED_VARIANTS = new Map([
  ['dot', 'quantized_dot'],
  ['conv', 'quantized_conv']
]);

for (const [opName, variant] of NATIVE_QUANTIZED_VARIANTS) {
  if (registry.has(opName)) registry.registerOpAttr(opName, 'quantizedVariant', variant);
}

export class QuantizationConfig {
  constructor(opts = {}) {
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
  constructor(config = {}) {
    super('QuantizationPass');
    this.requiredAnalyses = [UseDefAnalysis];
    this.config = config instanceof QuantizationConfig ? config : new QuantizationConfig(config);
  }

  run(func, analysisManager) {
    const useDef = analysisManager
      ? analysisManager.getAnalysis(UseDefAnalysis, func)
      : UseDefAnalysis.compute(func);

    const topo = useDef.topologicalOrder;
    const quantizedValues = new Set();
    const cfg = this.config;
    let changed = false;

    if (cfg.target && !cfg.target.supportsInt8) return PassResult.UNCHANGED;

    for (let i = 0; i < topo.length; i++) {
      const op = topo[i];
      if (op.opName === 'return' || op.opName === 'yield') continue;
      if (cfg.excludeOps.has(op.opName) || !cfg.quantizableOps.has(op.opName)) {
        for (let o = 0; o < op.numOperands; o++) {
          if (quantizedValues.has(op.getOperand(o))) {
            changed = this._insertDequantBefore(op, o, op.getOperand(o), cfg) || changed;
          }
        }
        continue;
      }

      if (cfg.sensitivityResult && cfg.sensitivityThreshold > 0) {
        if (cfg.sensitivityResult.isSensitive(op, cfg.sensitivityThreshold)) continue;
      }

      if (cfg.weightOnly && !hasConstantOperand(op)) continue;

      const opDef = registry.get(op.opName);
      const nativeVariant = opDef ? opDef.getAttr('quantizedVariant') : null;
      if (nativeVariant && allOperandsCanQuantize(op, quantizedValues, cfg)) {
        if (cfg.scheme === QuantizationScheme.PER_CHANNEL && this._canPerChannelDot(op, quantizedValues)) {
          changed = this._replacePerChannelDot(op, cfg) || changed;
        } else {
          changed = this._replaceWithNativeQuantized(op, nativeVariant, quantizedValues, cfg) || changed;
        }
        continue;
      }

      const def = registry.get(op.opName);
      const isEW = def && def.hasTrait(OpTrait.ELEMENTWISE);
      if (isEW && allOperandsQuantized(op, quantizedValues)) {
        for (let r = 0; r < op.numResults; r++) {
          quantizedValues.add(op.getResult(r));
        }
        continue;
      }

      changed = this._insertDequantQuantBoundary(op, quantizedValues, cfg) || changed;
    }

    const retOp = func.getReturnOp();
    if (retOp) {
      for (let i = 0; i < retOp.numOperands; i++) {
        const val = retOp.getOperand(i);
        if (quantizedValues.has(val)) {
          changed = this._insertDequantBefore(retOp, i, val, cfg) || changed;
        }
      }
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG && changed) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        opsProcessed: topo.length, changed,
        level: TraceLevel.DEBUG,
      });
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _getQuantParams(value, cfg) {
    const numBits = scalarBytes(cfg.targetDtype) * 8;
    if (cfg.calibration && cfg.calibration.hasData(value)) {
      return cfg.calibration.getQuantParams(value, cfg.scheme, cfg.targetDtype);
    }
    const defOp = value.definingOp;
    if (defOp && defOp.opName === 'constant') {
      const val = defOp.getAttr('value');
      if (typeof val === 'number') return QuantizationParams.fromRange(-Math.abs(val) || -1, Math.abs(val) || 1, cfg.scheme, cfg.targetDtype, numBits);
      if (val && typeof val.length === 'number') return QuantizationParams.fromConstantArray(val, cfg.scheme, cfg.targetDtype, numBits);
    }
    if (value.type instanceof TensorType && isFloatType(value.type.dtype)) {
      return QuantizationParams.defaultForActivation(cfg.scheme, cfg.targetDtype, numBits);
    }
    return null;
  }

  _insertQuantizeAfter(op, resultIdx, cfg) {
    const val = op.getResult(resultIdx);
    const qp = this._getQuantParams(val, cfg);
    if (!qp) return null;

    const resultType = new TensorType(val.type.shape, cfg.targetDtype);
    const quantOp = new Operation('quantize', [val], [resultType], {
      scale: qp.getScalarScale(),
      zero_point: qp.getScalarZeroPoint(),
      scheme: cfg.scheme,
      target_dtype: cfg.targetDtype,
      ...(qp.axis !== null ? { axis: qp.axis } : {})
    });

    if (op.parentBlock) {
      op.parentBlock.insertAfter(quantOp, op);
    }

    return quantOp.getResult(0);
  }

  _insertDequantBefore(consumer, operandIdx, val, cfg) {
    const qp = this._getQuantParams(val, cfg);
    if (!qp) return false;

    const outputDtype = ScalarType.F32;
    const resultType = new TensorType(val.type.shape, outputDtype);
    const dequantOp = new Operation('dequantize', [val], [resultType], {
      scale: qp.getScalarScale(),
      zero_point: qp.getScalarZeroPoint(),
      scheme: cfg.scheme,
      target_dtype: outputDtype
    });

    if (consumer.parentBlock) {
      consumer.parentBlock.insertBefore(dequantOp, consumer);
      consumer.replaceOperand(operandIdx, dequantOp.getResult(0));
    }
    return true;
  }

  _insertDequantQuantBoundary(op, quantizedValues, cfg) {
    let changed = false;

    for (let i = 0; i < op.numOperands; i++) {
      const val = op.getOperand(i);
      if (quantizedValues.has(val)) {
        changed = this._insertDequantBefore(op, i, val, cfg) || changed;
      }
    }

    for (let r = 0; r < op.numResults; r++) {
      const val = op.getResult(r);
      if (!(val.type instanceof TensorType) || !isFloatType(val.type.dtype)) continue;
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

  _replaceWithNativeQuantized(op, nativeOpName, quantizedValues, cfg) {
    const quantizedInputs = [];
    const attrs = {};

    for (const [key, val] of op.attributes || []) {
      attrs[key] = val;
    }

    for (let i = 0; i < op.numOperands; i++) {
      const operand = op.getOperand(i);
      if (quantizedValues.has(operand)) {
        quantizedInputs.push(operand);
        const qp = this._getQuantParams(operand, cfg);
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

      const quantType = new TensorType(operand.type.shape, cfg.targetDtype);
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

    const lhsScale = attrs.lhs_scale || attrs.input_scale || 1;
    const rhsScale = attrs.rhs_scale || attrs.kernel_scale || 1;
    attrs.output_scale = lhsScale * rhsScale;
    attrs.output_zero_point = 0;

    const resultType = new TensorType(op.getResult(0).type.shape, ScalarType.I32);
    const quantizedOp = new Operation(nativeOpName, quantizedInputs, [resultType], attrs);

    if (op.parentBlock) {
      op.parentBlock.insertBefore(quantizedOp, op);

      const dequantType = new TensorType(op.getResult(0).type.shape, op.getResult(0).type.dtype);
      const outScale = attrs.output_scale || 1;
      const dequantOp = new Operation('dequantize', [quantizedOp.getResult(0)], [dequantType], {
        scale: outScale,
        zero_point: attrs.output_zero_point || 0,
        scheme: cfg.scheme,
        target_dtype: op.getResult(0).type.dtype
      });
      op.parentBlock.insertBefore(dequantOp, op);
      op.replaceAllResultsWith([dequantOp.getResult(0)]);
      op.erase();
    }

    return true;
  }

  _canPerChannelDot(op, quantizedValues) {
    if (op.opName !== 'dot') return false;
    const lhs = op.getOperand(0);
    const rhs = op.getOperand(1);
    if (quantizedValues.has(lhs) || quantizedValues.has(rhs)) return false;
    if (!(lhs.type instanceof TensorType) || lhs.type.shape.length !== 2) return false;
    if (!(rhs.type instanceof TensorType) || rhs.type.shape.length !== 2) return false;
    const rhsDef = rhs.definingOp;
    if (!rhsDef || rhsDef.opName !== 'constant') return false;
    const data = rhsDef.getAttr('value');
    if (!data || typeof data === 'number' || typeof data.length !== 'number') return false;
    const rhsC = op.getAttr('rhs_contracting') || [];
    const lhsC = op.getAttr('lhs_contracting') || [];
    if (rhsC.length !== 1 || lhsC.length !== 1) return false;
    if ((op.getAttr('rhs_batch') || []).length !== 0) return false;
    if ((op.getAttr('lhs_batch') || []).length !== 0) return false;
    return true;
  }

  _activationParams(value, cfg) {
    const numBits = scalarBytes(cfg.targetDtype) * 8;
    const scheme = QuantizationScheme.PER_TENSOR_SYMMETRIC;
    if (cfg.calibration && cfg.calibration.hasData(value)) {
      return cfg.calibration.getQuantParams(value, scheme, cfg.targetDtype);
    }
    return QuantizationParams.defaultForActivation(scheme, cfg.targetDtype, numBits);
  }

  _replacePerChannelDot(op, cfg) {
    const lhs = op.getOperand(0);
    const rhs = op.getOperand(1);
    const rhsShape = rhs.type.shape;
    const wData = rhs.definingOp.getAttr('value');
    const numBits = scalarBytes(cfg.targetDtype) * 8;

    const rhsContracting = op.getAttr('rhs_contracting');
    const channelAxis = rhsContracting[0] === 0 ? 1 : 0;
    const wp = QuantizationParams.fromConstantArrayPerChannel([...wData], rhsShape, channelAxis, cfg.targetDtype, numBits);
    const wInt8 = wp.quantizeArrayPerChannel([...wData], rhsShape);

    const aqp = this._activationParams(lhs, cfg);
    const aScale = aqp.getScalarScale();
    const aZp = aqp.getScalarZeroPoint();

    const block = op.parentBlock;
    if (!block) return false;

    const wConstType = new TensorType(rhsShape, cfg.targetDtype);
    const wConst = new Operation('constant', [], [wConstType], { value: wInt8, tensor_type: wConstType });
    block.insertBefore(wConst, op);

    const aqType = new TensorType(lhs.type.shape, cfg.targetDtype);
    const aQuant = new Operation('quantize', [lhs], [aqType], {
      scale: aScale, zero_point: aZp,
      scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC, target_dtype: cfg.targetDtype
    });
    block.insertBefore(aQuant, op);

    const outShape = op.getResult(0).type.shape;
    const attrs = {};
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

    const numCh = rhsShape[channelAxis];
    const scaleVec = new Array(numCh);
    for (let c = 0; c < numCh; c++) scaleVec[c] = aScale * wp.getScaleForChannel(c);

    const lhsC = new Set(op.getAttr('lhs_contracting') || []);
    let lhsSpatial = 0;
    for (let i = 0; i < lhs.type.shape.length; i++) if (!lhsC.has(i)) lhsSpatial++;
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

function hasConstantOperand(op) {
  for (let i = 0; i < op.numOperands; i++) {
    const def = op.getOperand(i).definingOp;
    if (def && def.opName === 'constant') return true;
  }
  return false;
}

function allOperandsQuantized(op, quantizedValues) {
  for (let i = 0; i < op.numOperands; i++) {
    if (!quantizedValues.has(op.getOperand(i))) return false;
  }
  return op.numOperands > 0;
}

function allOperandsCanQuantize(op, quantizedValues, cfg) {
  for (let i = 0; i < op.numOperands; i++) {
    const operand = op.getOperand(i);
    if (quantizedValues.has(operand)) continue;
    if (!(operand.type instanceof TensorType) || !isFloatType(operand.type.dtype)) return false;
  }
  return op.numOperands > 0;
}
