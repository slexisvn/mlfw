import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Block } from '../../ir/graph/block.js';
import type { PassResultValue, PassTarget } from '../pass.js';

export type BackendRewriteOpts = { name?: string; rewrites?: BackendOpRewrite[]; [key: string]: unknown };
export type BackendOpRewrite = {
  name: string;
  match(op: Operation, config?: BackendRewriteOpts): boolean;
  build(op: Operation, block: Block, config?: BackendRewriteOpts): void;
};

import { FunctionPass, PassResult } from '../pass.js';

const _rewrites: BackendOpRewrite[] = [];

export function registerBackendOpRewrite(rewrite: BackendOpRewrite): BackendOpRewrite {
  _rewrites.push(rewrite);
  return rewrite;
}

export function listBackendOpRewrites(): BackendOpRewrite[] {
  return [..._rewrites];
}

export class BackendOpRewritePass extends FunctionPass {
  config: BackendRewriteOpts;
  rewrites: BackendOpRewrite[];

  constructor(config: BackendRewriteOpts = {}) {
    super(config.name || 'BackendOpRewritePass');
    this.config = config;
    this.rewrites = config.rewrites || _rewrites;
  }

  override run(func: PassTarget): PassResultValue {
    let changed = false;
    for (const op of [...(func as GraphFunction).ops()]) {
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
