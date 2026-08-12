import type { LIRFunc } from '../ir/lir/nodes.js';
import type { TraceLog } from '../pipeline/trace.js';
import type { CompilerConfig } from '../pipeline/pipeline_types.js';

export type LirPassCtx = {
  trace: TraceLog;
  config?: CompilerConfig;
  [key: string]: unknown;
};

export class LirFuncPass {
  name: string;
  phase: string;
  trace: TraceLog | null;

  constructor(name: string, phase: string | null = null) {
    this.name = name;
    this.phase = phase || name;
    this.trace = null;
  }

  begin(ctx: LirPassCtx): void {}

  run(func: LIRFunc, ctx: LirPassCtx): LIRFunc | void {
    throw new Error('LirFuncPass.run not implemented');
  }

  end(ctx: LirPassCtx): void {}
}
