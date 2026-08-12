import { PrimFuncPass } from '../tir_pass.js';
import { Schedule } from '../../schedule/schedule.js';
import { invalidateClassifyCache, primFuncHasRecurrence } from '../../schedule/rules.js';
import { irChildNodes } from '../../ir/ir_visitor.js';
import { FuncAttr } from '../../ir/func_attrs.js';
import type { BlockNode, BufferLoadNode, BufferStoreNode, PrimFunc, TirNode } from '../../ir/tensor/nodes.js';
import type { IRNode } from '../../ir/ir_visitor.js';
import type { TirPassCtx } from '../tir_pass.js';
import type { CompilerConfig, CompileTarget } from '../../pipeline/pipeline_types.js';

type BlockSummary = { name: string; writes: Set<string>; reads: Set<string>; hasInit: boolean };
type FuncAnalysis = { blocks: BlockSummary[]; loadCount: Map<string, number>; storeWriters: Map<string, Set<string>> };
type WalkEntry = { node: TirNode; block: BlockSummary | null };

function analyzeFunc(root: TirNode): FuncAnalysis {
  const blocks: BlockSummary[] = [];
  const loadCount = new Map<string, number>();
  const storeWriters = new Map<string, Set<string>>();
  const stack: WalkEntry[] = [{ node: root, block: null }];
  while (stack.length > 0) {
    const { node, block } = stack.pop() as WalkEntry;
    const buf = (node as BufferLoadNode | BufferStoreNode).buffer;
    if (node.type === 'BufferLoadNode' && buf) {
      loadCount.set(buf.name, (loadCount.get(buf.name) || 0) + 1);
      if (block) block.reads.add(buf.name);
    } else if (node.type === 'BufferStoreNode' && buf && block) {
      block.writes.add(buf.name);
      let writers = storeWriters.get(buf.name);
      if (!writers) { writers = new Set<string>(); storeWriters.set(buf.name, writers); }
      writers.add(block.name);
    }
    let childBlock = block;
    if (node.type === 'BlockNode') {
      const b = node as BlockNode;
      childBlock = { name: b.name, writes: new Set<string>(), reads: new Set<string>(), hasInit: b.initBody != null };
      blocks.push(childBlock);
    }
    for (const child of irChildNodes(node as unknown as IRNode)) stack.push({ node: child as unknown as TirNode, block: childBlock });
  }
  return { blocks, loadCount, storeWriters };
}

export class InlineReindexPass extends PrimFuncPass {
  config: CompilerConfig;
  target: CompileTarget;

  constructor(config: CompilerConfig) {
    super('InlineReindexPass', 'scheduling');
    this.config = config;
    this.target = config.target;
    this.snapshotPoint = 'afterInlineReindex';
  }

  override run(pf: PrimFunc, ctx: TirPassCtx): void {
    if (!this.target.isGPU() || this.target.isWebGPU()) return;
    if (pf.hasAttr(FuncAttr.EXTERNAL_CODEGEN) || pf.hasAttr(FuncAttr.TENSOR_INTRIN)) return;
    if (primFuncHasRecurrence(pf)) return;
    const sCfg = this.config.scheduling as Record<string, boolean>;
    if (!(sCfg.enabled || sCfg.gpuTiling || sCfg.autotune)) return;

    const storage = new Set<string>();
    for (const [, buf] of pf.bufferMap) storage.add(buf.name);

    const { blocks, loadCount, storeWriters } = analyzeFunc(pf.body);
    const sch = new Schedule(pf);
    let inlined = false;
    for (const b of blocks) {
      if (b.hasInit || b.writes.size === 0) continue;
      const writes = [...b.writes];
      if (!writes.every((w) => !storage.has(w) && (storeWriters.get(w) as Set<string>).size === 1)) continue;
      if (!writes.some((w) => (loadCount.get(w) || 0) > 0)) continue;
      if (![...b.reads].every((r) => (storeWriters.get(r) ? (storeWriters.get(r) as Set<string>).size : 0) <= 1)) continue;
      try {
        sch.computeInlineBlock(b.name);
        inlined = true;
      } catch (_) {}
    }

    if (inlined) invalidateClassifyCache(pf);
  }
}
