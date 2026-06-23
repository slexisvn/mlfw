import { FunctionPass, PassResult } from '../pass.js';

const _rewrites = [];

export function registerBackendOpRewrite(rewrite) {
  _rewrites.push(rewrite);
  return rewrite;
}

export function listBackendOpRewrites() {
  return [..._rewrites];
}

export class BackendOpRewritePass extends FunctionPass {
  constructor(config = {}) {
    super(config.name || 'BackendOpRewritePass');
    this.config = config;
    this.rewrites = config.rewrites || _rewrites;
  }

  run(func) {
    let changed = false;
    for (const op of [...func.ops()]) {
      const block = op.parentBlock;
      if (!block) continue;
      for (const rw of this.rewrites) {
        if (rw.match(op, this.config)) {
          rw.build(op, block, this.config);
          changed = true;
          break;
        }
      }
    }
    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
