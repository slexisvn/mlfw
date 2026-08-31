import type { PrimFunc } from '../ir/tensor/nodes.js';
import type { TirModule } from '../ir/tensor/module.js';
import type { TraceLog } from '../support/trace.js';
import type { CompilerConfig } from '../support/config_types.js';

export type TirPassCtx = {
  trace: TraceLog;
  config?: CompilerConfig;
  [key: string]: unknown;
};

export class PrimFuncPass {
  name: string;
  phase: string;
  snapshotPoint: string | null;
  trace: TraceLog | null;

  constructor(name: string, phase: string | null = null) {
    this.name = name;
    this.phase = phase || name;
    this.snapshotPoint = null;
    this.trace = null;
  }

  begin(ctx: TirPassCtx): void {}

  run(primFunc: PrimFunc, ctx: TirPassCtx): PrimFunc | void {
    throw new Error('PrimFuncPass.run not implemented');
  }

  end(ctx: TirPassCtx): void {}
}

export class TirModulePass {
  name: string;
  phase: string;
  snapshotPoint: string | null;
  trace: TraceLog | null;

  constructor(name: string, phase: string | null = null) {
    this.name = name;
    this.phase = phase || name;
    this.snapshotPoint = null;
    this.trace = null;
  }

  begin(ctx: TirPassCtx): void {}

  runModule(module: TirModule, ctx: TirPassCtx): void {
    throw new Error('TirModulePass.runModule not implemented');
  }

  end(ctx: TirPassCtx): void {}
}
