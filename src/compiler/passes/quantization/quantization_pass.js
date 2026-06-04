import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { TensorType, ScalarType, isFloatType } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import { OpTrait } from '../../ir/graph/op_registry.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';
import { QuantizationScheme, QuantizationParams } from '../../ir/graph/quantization_types.js';

const DEFAULT_EXCLUDE_OPS = new Set(['softmax', 'sqrt', 'div', 'rsqrt', 'log', 'exp', 'tanh']);
const DEFAULT_QUANTIZABLE_OPS = new Set(['dot', 'conv', 'add', 'mul', 'sub']);

const NATIVE_QUANTIZED_VARIANTS = new Map([
  ['dot', 'quantized_dot'],
  ['conv', 'quantized_conv']
]);

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
      if (cfg.excludeOps.has(op.opName)) continue;
      if (!cfg.quantizableOps.has(op.opName)) continue;

      if (cfg.sensitivityResult && cfg.sensitivityThreshold > 0) {
        if (cfg.sensitivityResult.isSensitive(op, cfg.sensitivityThreshold)) continue;
      }

      if (cfg.weightOnly && !hasConstantOperand(op)) continue;

      const nativeVariant = NATIVE_QUANTIZED_VARIANTS.get(op.opName);
      if (nativeVariant && allOperandsCanQuantize(op, quantizedValues, cfg)) {
        changed = this._replaceWithNativeQuantized(op, nativeVariant, quantizedValues, cfg) || changed;
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

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _getQuantParams(value, cfg) {
    if (cfg.calibration && cfg.calibration.hasData(value)) {
      return cfg.calibration.getQuantParams(value, cfg.scheme, cfg.targetDtype);
    }
    return null;
  }

  _getConstantQuantParams(op, operandIdx, cfg) {
    const operand = op.getOperand(operandIdx);
    const defOp = operand.definingOp;
    if (!defOp || defOp.opName !== 'constant') return null;
    const val = defOp.getAttr('value');
    if (typeof val !== 'number') return null;
    return QuantizationParams.fromRange(-Math.abs(val), Math.abs(val), cfg.scheme, cfg.targetDtype);
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
        const users = [...val.getUsers()];
        for (const use of users) {
          if (use.owner && use.owner !== qResult.definingOp) {
            use.owner.replaceOperand(use.operandIndex, qResult);
            quantizedValues.add(qResult);
          }
        }
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

    const outputQP = this._getQuantParams(op.getResult(0), cfg);
    if (outputQP) {
      attrs.output_scale = outputQP.getScalarScale();
      attrs.output_zero_point = outputQP.getScalarZeroPoint();
    } else {
      attrs.output_scale = 1;
      attrs.output_zero_point = 0;
    }

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
