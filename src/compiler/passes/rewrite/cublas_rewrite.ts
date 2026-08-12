import { Operation } from '../../ir/graph/operation.js';
import { cublasDotInfo } from '../partition/cublas_split.js';
import { registerBackendOpRewrite, BackendOpRewritePass } from './backend_rewrite.js';
import type { Block } from '../../ir/graph/block.js';
import type { BackendRewriteOpts } from './backend_rewrite.js';

const CUBLAS_DOT_REWRITE = registerBackendOpRewrite({
  name: 'dot->cublas_gemm',
  match: (op: Operation) => op.opName === 'dot' && op.numOperands === 2 && cublasDotInfo(op) !== null,
  build: (op: Operation, block: Block) => {
    const resultTypes = op.results.map(r => r.type);
    const gemm = new Operation('cublas_gemm', [op.getOperand(0), op.getOperand(1)], resultTypes, new Map(op.attributes), null);
    block.insertBefore(gemm, op);
    op.replaceAllResultsWith(gemm.results);
    op.erase();
  },
});

export class CublasRewritePass extends BackendOpRewritePass {
  constructor(config: Partial<BackendRewriteOpts> = {}) {
    super({ ...config, name: 'CublasRewritePass', rewrites: [CUBLAS_DOT_REWRITE] });
  }
}
