export class RematPolicy {
  constructor(opts = {}) {
    this._maxRematDepth = opts.maxRematDepth || 1;
    this._sizeThreshold = opts.sizeThreshold || 1024 * 1024;
    this._alwaysRemat = new Set(opts.alwaysRemat || [
      'neg', 'abs', 'sign', 'floor', 'ceil',
      'exp', 'log', 'sqrt', 'rsqrt',
      'sin', 'cos', 'tanh',
    ]);
    this._neverRemat = new Set(opts.neverRemat || [
      'matmul', 'dot', 'conv', 'reduce',
      'custom_call', 'pool2d',
    ]);
  }

  shouldRematerialize(op) {
    if (this._alwaysRemat.has(op.opName)) return true;
    if (this._neverRemat.has(op.opName)) return false;

    const resultType = op.numResults > 0 ? op.getResult(0).type : null;
    if (!resultType || !resultType.shape) return false;

    const numel = resultType.numel();
    if (numel > this._sizeThreshold) return false;

    return this._isElementwise(op);
  }

  _isElementwise(op) {
    if (op.numOperands === 0 || op.numResults === 0) return false;
    const resultShape = op.getResult(0).type.shape;
    for (let i = 0; i < op.numOperands; i++) {
      const operandType = op.getOperand(i).type;
      if (!operandType || !operandType.shape) return false;
      if (operandType.shape.length !== resultShape.length) return false;
      for (let d = 0; d < resultShape.length; d++) {
        if (operandType.shape[d] !== resultShape[d]) return false;
      }
    }
    return true;
  }
}
