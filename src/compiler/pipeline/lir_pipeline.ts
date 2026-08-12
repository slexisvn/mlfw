import { FlatIndexSimplifyPass } from '../passes/simplify/flat_index_simplify.js';
import { lirPassesForPhase } from './lir_pass_registry.js';
import type { LirFuncPass } from '../passes/lir_pass.js';
import type { CompilerConfig } from './pipeline_types.js';

export function buildLirPipeline(config: CompilerConfig): LirFuncPass[] {
  const passes: LirFuncPass[] = [];

  for (const p of lirPassesForPhase('pre', config)) passes.push(p);

  passes.push(new FlatIndexSimplifyPass());

  for (const p of lirPassesForPhase('post', config)) passes.push(p);

  return passes;
}
