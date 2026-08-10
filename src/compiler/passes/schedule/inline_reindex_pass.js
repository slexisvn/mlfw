import { PrimFuncPass } from '../tir_pass.js';
import { Schedule } from '../../schedule/schedule.js';
import { invalidateClassifyCache, primFuncHasRecurrence } from '../../schedule/rules.js';
import { irChildNodes } from '../../ir/ir_visitor.js';
import { FuncAttr } from '../../ir/func_attrs.js';

function analyzeFunc(root) {
  const blocks = [];
  const loadCount = new Map();
  const storeWriters = new Map();
  const stack = [{ node: root, block: null }];
  while (stack.length > 0) {
    const { node, block } = stack.pop();
    if (node.type === 'BufferLoadNode' && node.buffer) {
      loadCount.set(node.buffer.name, (loadCount.get(node.buffer.name) || 0) + 1);
      if (block) block.reads.add(node.buffer.name);
    } else if (node.type === 'BufferStoreNode' && node.buffer && block) {
      block.writes.add(node.buffer.name);
      let writers = storeWriters.get(node.buffer.name);
      if (!writers) { writers = new Set(); storeWriters.set(node.buffer.name, writers); }
      writers.add(block.name);
    }
    let childBlock = block;
    if (node.type === 'BlockNode') {
      childBlock = { name: node.name, writes: new Set(), reads: new Set(), hasInit: node.initBody != null };
      blocks.push(childBlock);
    }
    for (const child of irChildNodes(node)) stack.push({ node: child, block: childBlock });
  }
  return { blocks, loadCount, storeWriters };
}

export class InlineReindexPass extends PrimFuncPass {
  constructor(config) {
    super('InlineReindexPass', 'scheduling');
    this.config = config;
    this.target = config.target;
    this.snapshotPoint = 'afterInlineReindex';
  }

  run(pf, ctx) {
    if (!this.target.isGPU() || this.target.isWebGPU()) return;
    if (pf.hasAttr(FuncAttr.CUBLAS_INFO) || pf.hasAttr(FuncAttr.TENSOR_INTRIN)) return;
    if (primFuncHasRecurrence(pf)) return;
    const sCfg = this.config.scheduling;
    if (!(sCfg.enabled || sCfg.gpuTiling || sCfg.autotune)) return;

    const storage = new Set();
    for (const [, buf] of pf.bufferMap) storage.add(buf.name);

    const { blocks, loadCount, storeWriters } = analyzeFunc(pf.body);
    const sch = new Schedule(pf);
    let inlined = false;
    for (const b of blocks) {
      if (b.hasInit || b.writes.size === 0) continue;
      const writes = [...b.writes];
      if (!writes.every((w) => !storage.has(w) && storeWriters.get(w).size === 1)) continue;
      if (!writes.some((w) => (loadCount.get(w) || 0) > 0)) continue;
      if (![...b.reads].every((r) => (storeWriters.get(r) ? storeWriters.get(r).size : 0) <= 1)) continue;
      try {
        sch.computeInlineBlock(b.name);
        inlined = true;
      } catch (_) {}
    }

    if (inlined) invalidateClassifyCache(pf);
  }
}
