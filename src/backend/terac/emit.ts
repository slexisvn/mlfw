import { printModule } from '../../compiler/ir/graph/printer.js';
import { ExplicitBroadcastPass } from '../../compiler/passes/normalize/explicit_broadcast.js';
import { IsolateRegionsPass } from '../../compiler/passes/normalize/isolate_regions.js';
import { MaterializeShapesPass } from '../../compiler/passes/normalize/materialize_shapes.js';

import type { GraphModule } from '../../compiler/ir/graph/module.js';
import type { TeracEntry } from './runtime_module.js';

export function normalizeForTera(module: GraphModule): GraphModule {
  for (const func of module) {
    new ExplicitBroadcastPass().run(func);
    new IsolateRegionsPass().run(func);
    new MaterializeShapesPass().run(func);
  }
  return module;
}

export function teraEntries(module: GraphModule): TeracEntry[] {
  return [...module].map((func) => ({
    name: func.name,
    inputs: func.inputTypes.length,
    outputs: func.outputTypes.length,
  }));
}

export function emitTeraModule(module: GraphModule): string {
  return printModule(normalizeForTera(module)) + '\n';
}
