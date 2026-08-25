import { PrimFuncPass } from '../tir_pass.js';
import { SeqNode } from '../../ir/tensor/nodes.js';
import { FuncAttr } from '../../ir/func_attrs.js';
import { CONSTANT_BLOCK_HINT, numberedBlockName } from '../../ir/tensor/block_name.js';
import { expandConstantStores } from './lowering_registry.js';
import type { PrimFunc, TirNode, VariableNode } from '../../ir/tensor/nodes.js';
import type { Buffer } from '../../ir/tensor/buffer.js';
import type { ConstBuffer } from './lowering_registry.js';

export class LegalizeConstBuffersPass extends PrimFuncPass {
  constructor() {
    super('LegalizeConstBuffersPass', 'constBufferLegalization');
  }

  override run(pf: PrimFunc): void {
    const constBuffers = pf.getAttr<ConstBuffer[]>(FuncAttr.CONST_BUFFERS);
    if (!constBuffers || constBuffers.length === 0) return;

    const bound = new Set<Buffer>(constBuffers.map(cb => cb.buffer));
    const dropped = new Set<VariableNode>();
    for (const [v, buf] of pf.bufferMap) {
      if (bound.has(buf)) dropped.add(v);
    }
    for (const v of dropped) pf.bufferMap.delete(v);
    pf.params = pf.params.filter(p => !dropped.has(p));

    const stores: TirNode[] = constBuffers.map((cb, i) =>
      expandConstantStores(cb.buffer, cb.data, numberedBlockName(CONSTANT_BLOCK_HINT, i)));
    const body = pf.body;
    const stmts = body && body.type === 'SeqNode' ? [...stores, ...(body as SeqNode).stmts] : [...stores, body];
    const newBody = new SeqNode(stmts);
    pf.body = newBody;
    pf._setChild('body', newBody);

    pf.removeAttr(FuncAttr.CONST_BUFFERS);
  }
}
