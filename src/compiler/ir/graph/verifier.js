import { TensorType, TupleType, TokenType, typeToString, DYNAMIC } from './types.js';
import { Value, BlockArgument } from './value.js';
import { registry } from './ops.js';

export class VerificationError {
  constructor(message, op = null, func = null) {
    this.message = message;
    this.op = op;
    this.func = func;
  }

  toString() {
    let loc = '';
    if (this.func) loc += `[${this.func.name}] `;
    if (this.op) loc += `op '${this.op.opName}' (id=${this.op.id}): `;
    return loc + this.message;
  }
}

export function verifyModule(module) {
  const errors = [];
  if (module.functionCount === 0) {
    errors.push(new VerificationError('Module has no functions'));
  }
  for (const func of module) {
    verifyFunction(func, errors);
  }
  return errors;
}

export function verifyFunction(func, errors = []) {
  if (!func.entryBlock) {
    errors.push(new VerificationError('Function has no entry block', null, func));
    return errors;
  }

  if (func.entryBlock.arguments.length !== func.inputTypes.length) {
    errors.push(new VerificationError(
      `Entry block has ${func.entryBlock.arguments.length} args, expected ${func.inputTypes.length}`,
      null, func
    ));
  }

  for (let i = 0; i < func.entryBlock.arguments.length; i++) {
    const arg = func.entryBlock.arguments[i];
    const expected = func.inputTypes[i];
    if (expected && !arg.type.equals(expected)) {
      errors.push(new VerificationError(
        `Block arg ${i} type ${typeToString(arg.type)} != expected ${typeToString(expected)}`,
        null, func
      ));
    }
  }

  const definedValues = new Set();
  for (const arg of func.entryBlock.arguments) {
    definedValues.add(arg);
  }

  for (const block of func.body) {
    verifyBlock(block, func, definedValues, errors);
  }

  const ret = func.getReturnOp();
  if (!ret) {
    errors.push(new VerificationError('Missing return op', null, func));
  } else {
    if (ret.numOperands !== func.outputTypes.length) {
      errors.push(new VerificationError(
        `Return has ${ret.numOperands} operands, function declares ${func.outputTypes.length} outputs`,
        ret, func
      ));
    }
    for (let i = 0; i < Math.min(ret.numOperands, func.outputTypes.length); i++) {
      const actual = ret.getOperand(i).type;
      const expected = func.outputTypes[i];
      if (expected instanceof TensorType && actual instanceof TensorType) {
        if (actual.dtype !== expected.dtype) {
          errors.push(new VerificationError(
            `Return operand ${i} dtype ${actual.dtype} != expected ${expected.dtype}`,
            ret, func
          ));
        }
        if (!actual.shapeCompatible(expected)) {
          errors.push(new VerificationError(
            `Return operand ${i} shape incompatible: ${typeToString(actual)} vs ${typeToString(expected)}`,
            ret, func
          ));
        }
      }
    }
  }

  return errors;
}

function verifyBlock(block, func, definedValues, errors) {
  for (const op of block) {
    verifyOperation(op, func, definedValues, errors);
    for (let i = 0; i < op.numResults; i++) {
      definedValues.add(op.getResult(i));
    }
  }

  if (block.size > 0) {
    const last = block.lastOp;
    if (block.parentRegion && block.parentRegion.parentOp) {
      const parentOpName = block.parentRegion.parentOp.opName;
      if (parentOpName === 'fusion' || parentOpName === 'if' || parentOpName === 'while' || parentOpName === 'reduce') {
        if (!last.isTerminator()) {
          errors.push(new VerificationError(
            `Block in ${parentOpName} region must end with terminator, got '${last.opName}'`,
            last, func
          ));
        }
      }
    }
  }
}

function verifyOperation(op, func, definedValues, errors) {
  for (let i = 0; i < op.numOperands; i++) {
    const operand = op.getOperand(i);
    if (!operand) {
      errors.push(new VerificationError(`Operand ${i} is null`, op, func));
      continue;
    }
    if (!(operand instanceof Value)) {
      errors.push(new VerificationError(`Operand ${i} is not a Value`, op, func));
      continue;
    }
  }

  for (let i = 0; i < op.numResults; i++) {
    const result = op.getResult(i);
    if (!result) {
      errors.push(new VerificationError(`Result ${i} is null`, op, func));
    }
    if (result && result.definingOp !== op) {
      errors.push(new VerificationError(`Result ${i} definingOp mismatch`, op, func));
    }
  }

  const opDef = registry.get(op.opName);
  if (!opDef) {
    if (!['return', 'yield'].includes(op.opName) && !registry.has(op.opName)) {
      errors.push(new VerificationError(`Unknown op '${op.opName}'`, op, func));
    }
    return;
  }

  if (opDef.numOperands >= 0 && op.numOperands !== opDef.numOperands) {
    errors.push(new VerificationError(
      `'${op.opName}' expects ${opDef.numOperands} operands, got ${op.numOperands}`,
      op, func
    ));
  }

  if (opDef.numResults >= 0 && op.numResults !== opDef.numResults) {
    errors.push(new VerificationError(
      `'${op.opName}' expects ${opDef.numResults} results, got ${op.numResults}`,
      op, func
    ));
  }

  for (const attrDef of opDef.attrs) {
    if (attrDef.required && !op.hasAttr(attrDef.name)) {
      errors.push(new VerificationError(
        `'${op.opName}' missing required attribute '${attrDef.name}'`,
        op, func
      ));
    }
  }

  if (opDef.hasRegions && opDef.numRegions > 0) {
    if (op.numRegions !== opDef.numRegions) {
      errors.push(new VerificationError(
        `'${op.opName}' expects ${opDef.numRegions} regions, got ${op.numRegions}`,
        op, func
      ));
    }
  }

  if (opDef.verify) {
    const opErrors = opDef.verify(op);
    if (opErrors) {
      for (const msg of opErrors) {
        errors.push(new VerificationError(msg, op, func));
      }
    }
  }

  if (opDef.inferResultTypes && op.numResults > 0) {
    const operandTypes = [];
    for (let i = 0; i < op.numOperands; i++) {
      operandTypes.push(op.getOperand(i).type);
    }
    const inferred = opDef.inferResultTypes(operandTypes, op.attributes, op.results.map(r => r.type));
    if (inferred) {
      for (let i = 0; i < Math.min(inferred.length, op.numResults); i++) {
        const actual = op.getResult(i).type;
        const expected = inferred[i];
        if (expected instanceof TensorType && actual instanceof TensorType) {
          if (actual.dtype !== expected.dtype) {
            errors.push(new VerificationError(
              `Result ${i} dtype ${actual.dtype} != inferred ${expected.dtype}`,
              op, func
            ));
          }
        }
      }
    }
  }

  for (const region of op.regions) {
    const regionDefinedValues = new Set(definedValues);
    for (const block of region) {
      for (const arg of block.arguments) {
        regionDefinedValues.add(arg);
      }
      verifyBlock(block, func, regionDefinedValues, errors);
    }
  }
}
