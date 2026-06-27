import { TensorType, typeToString } from './types.js';
import { Value } from './value.js';
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
    collectScopeDefs(block, definedValues);
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

function collectScopeDefs(block, scope) {
  for (const arg of block.arguments) scope.add(arg);
  for (const op of block) {
    for (let i = 0; i < op.numResults; i++) {
      const result = op.getResult(i);
      if (result) scope.add(result);
    }
  }
}

function detectCycles(block, func, errors) {
  const inBlock = new Set();
  for (const op of block) inBlock.add(op);

  const VISITING = 1;
  const DONE = 2;
  const state = new Map();
  const reported = new Set();

  for (const root of block) {
    if (state.get(root) !== undefined) continue;
    const stack = [{ op: root, i: 0 }];
    state.set(root, VISITING);
    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const op = frame.op;
      if (frame.i < op.numOperands) {
        const operand = op.getOperand(frame.i);
        frame.i++;
        const def = operand && operand.definingOp;
        if (!def || !inBlock.has(def)) continue;
        const seen = state.get(def);
        if (seen === VISITING) {
          if (!reported.has(def)) {
            reported.add(def);
            errors.push(new VerificationError('participates in a value dependency cycle', def, func));
          }
        } else if (seen === undefined) {
          state.set(def, VISITING);
          stack.push({ op: def, i: 0 });
        }
        continue;
      }
      state.set(op, DONE);
      stack.pop();
    }
  }
}

function verifyBlock(block, func, definedValues, errors) {
  detectCycles(block, func, errors);

  for (const op of block) {
    verifyOperation(op, func, definedValues, errors);
  }

  if (block.size > 0) {
    const last = block.lastOp;
    if (block.parentRegion && block.parentRegion.parentOp) {
      const parentOpName = block.parentRegion.parentOp.opName;
      const parentDef = registry.get(parentOpName);
      if (parentDef && parentDef.hasRegions) {
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
    if (!definedValues.has(operand)) {
      errors.push(new VerificationError(`Operand ${i} used before definition`, op, func));
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
          if (!actual.shapeCompatible(expected)) {
            errors.push(new VerificationError(
              `Result ${i} shape ${typeToString(actual)} incompatible with inferred ${typeToString(expected)}`,
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
      collectScopeDefs(block, regionDefinedValues);
    }
    for (const block of region) {
      verifyBlock(block, func, regionDefinedValues, errors);
    }
  }
}
