export class PrimFuncPass {
  constructor(name, phase = null) {
    this.name = name;
    this.phase = phase || name;
    this.snapshotPoint = null;
    this.trace = null;
  }

  begin(ctx) {}

  run(primFunc, ctx) {
    throw new Error('PrimFuncPass.run not implemented');
  }

  end(ctx) {}
}

export class TirModulePass {
  constructor(name, phase = null) {
    this.name = name;
    this.phase = phase || name;
    this.snapshotPoint = null;
    this.trace = null;
  }

  begin(ctx) {}

  runModule(module, ctx) {
    throw new Error('TirModulePass.runModule not implemented');
  }

  end(ctx) {}
}
