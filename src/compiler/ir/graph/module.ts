import { cloneGraphFunction } from './function.js';
import { IRModule } from '../module_base.js';
import type { GraphFunction } from './function.js';

export class GraphModule extends IRModule<GraphFunction> {
  restoreFrom(snapshot: GraphModule): void {
    this._functions.clear();
    for (const func of snapshot) {
      this._functions.set(func.name, func);
      func._module = this;
    }
    this._version++;
  }

  verify(): string[] {
    const errors: string[] = [];
    if (this._functions.size === 0) {
      errors.push('Module has no functions');
    }
    for (const func of this._functions.values()) {
      const funcErrors = func.verify();
      for (let i = 0; i < funcErrors.length; i++) {
        errors.push(`${func.name}: ${funcErrors[i]}`);
      }
    }
    return errors;
  }
}

export function cloneGraphModule(module: GraphModule): GraphModule {
  const clone = new GraphModule(module.name);
  for (const func of module) {
    clone.addFunction(cloneGraphFunction(func));
  }
  clone._version = module._version;
  return clone;
}
