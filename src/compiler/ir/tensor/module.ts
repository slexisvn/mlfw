import { verifyIR, IRLevel } from '../verify.js';
import { IRModule } from '../module_base.js';
import type { PrimFunc } from './nodes.js';

export class TirModule extends IRModule<PrimFunc> {
  replaceFunction(name: string, primFunc: PrimFunc): PrimFunc {
    if (!this._functions.has(name)) {
      throw new Error(`TirModule '${this.name}' has no function '${name}' to replace`);
    }
    if (primFunc.name !== name) this._functions.delete(name);
    this._functions.set(primFunc.name, primFunc);
    primFunc._module = this;
    this._version++;
    return primFunc;
  }

  verify(): string[] {
    const errors: string[] = [];
    if (this._functions.size === 0) errors.push('Module has no functions');
    for (const [name, func] of this._functions) {
      if (func.name !== name) errors.push(`Function registered as '${name}' but named '${func.name}'`);
      for (const message of verifyIR(IRLevel.TIR, func)) errors.push(`${func.name}: ${message}`);
    }
    return errors;
  }
}
