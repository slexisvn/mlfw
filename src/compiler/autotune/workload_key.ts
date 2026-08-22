import type { TirNode, PrimFunc, BlockNode, SeqNode, ForNode, LetStmtNode, IfThenElseNode, BufferLoadNode, BufferStoreNode, MathOpNode, CompareNode, CastNode, CallExternNode } from '../ir/tensor/nodes.js';
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';

type NodeSlots = Record<string, TirNode | TirNode[] | undefined>;
type BlockMap = Map<string, BlockNode> & { __readersByBuffer?: Map<string, BlockNode[]> };

export function buildBlockMap(root: TirNode | null | undefined): BlockMap {
  const map: BlockMap = new Map<string, BlockNode>();
  const stack: (TirNode | null | undefined)[] = [root];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n) continue;
    if (n.type === 'BlockNode') map.set((n as BlockNode).name, n as BlockNode);
    const slots = n as unknown as NodeSlots;
    if (slots.body) stack.push(slots.body as TirNode);
    if (slots.stmts) for (const st of slots.stmts as TirNode[]) stack.push(st);
    if (slots.thenBody) stack.push(slots.thenBody as TirNode);
    if (slots.elseBody) stack.push(slots.elseBody as TirNode);
    if (slots.initBody) stack.push(slots.initBody as TirNode);
  }
  return map;
}

export function computeWorkloadKey(primFunc: PrimFunc, blockName: string, target: ScheduleTarget, blockMap: BlockMap | null = null, numericMode = 'n1'): string {
  const map = blockMap || buildBlockMap(primFunc.body);
  const block = map.get(blockName) || null;
  const parts: string[] = [];

  if (block) {
    const bufShapes: string[] = [];
    for (const r of block.reads) bufShapes.push(`${r.buffer.shape.join('x')}:${r.buffer.dtype}`);
    for (const w of block.writes) bufShapes.push(`${w.buffer.shape.join('x')}:${w.buffer.dtype}`);
    parts.push(bufShapes.join(','));

    const ops: string[] = [];
    collectBlockOps(block.body, ops);
    if (block.initBody) collectBlockOps(block.initBody, ops);
    parts.push(ops.join(';'));

    if (block.writes.length === 1) {
      const outName = block.writes[0].buffer.name;
      const consumers: string[] = [];
      for (const other of readersByBuffer(map).get(outName) || []) {
        if (other === block) continue;
        const cops: string[] = [];
        collectBlockOps(other.body, cops);
        if (other.initBody) collectBlockOps(other.initBody, cops);
        consumers.push(cops.join(';'));
      }
      if (consumers.length > 0) {
        consumers.sort();
        parts.push(`consumers:${consumers.join('|')}`);
      }
    }
  }

  parts.push(target.name);
  parts.push(target.kind as string);
  parts.push(numericMode);

  return fnv1a(parts.join('|'));
}

function readersByBuffer(map: BlockMap): Map<string, BlockNode[]> {
  if (map.__readersByBuffer) return map.__readersByBuffer;
  const idx = new Map<string, BlockNode[]>();
  for (const blk of map.values()) {
    if (!blk.reads) continue;
    const names = new Set<string>();
    for (const r of blk.reads) if (r.buffer) names.add(r.buffer.name);
    for (const name of names) {
      let arr = idx.get(name);
      if (!arr) { arr = []; idx.set(name, arr); }
      arr.push(blk);
    }
  }
  map.__readersByBuffer = idx;
  return idx;
}

function collectBlockOps(node: TirNode | null | undefined, ops: string[]): void {
  if (!node || typeof node !== 'object') return;
  switch (node.type) {
    case 'BufferStoreNode': {
      const st = node as BufferStoreNode;
      ops.push('store');
      if (st.indices) for (const idx of st.indices) collectBlockOps(idx, ops);
      collectBlockOps(st.value, ops);
      return;
    }
    case 'BufferLoadNode': {
      const ld = node as BufferLoadNode;
      ops.push(`load:${ld.buffer.shape.join('x')}:${ld.buffer.dtype}`);
      if (ld.indices) for (const idx of ld.indices) collectBlockOps(idx, ops);
      return;
    }
    case 'MathOpNode': {
      const m = node as MathOpNode;
      ops.push(`math:${m.op}`);
      collectBlockOps(m.a, ops);
      if (m.b) collectBlockOps(m.b, ops);
      return;
    }
    case 'CallExternNode': {
      const c = node as CallExternNode;
      ops.push(`call:${c.externName}`);
      for (const arg of c.args) collectBlockOps(arg, ops);
      return;
    }
    case 'CompareNode': {
      const cn = node as CompareNode;
      ops.push(`cmp:${cn.direction}`);
      collectBlockOps(cn.a, ops);
      collectBlockOps(cn.b, ops);
      return;
    }
    case 'CastNode': {
      const ct = node as CastNode;
      ops.push(`cast:${ct.fromDtype}->${ct.toDtype}`);
      collectBlockOps(ct.expr, ops);
      return;
    }
    case 'IfThenElseNode': {
      const ite = node as IfThenElseNode;
      collectBlockOps(ite.condition, ops);
      collectBlockOps(ite.thenBody, ops);
      if (ite.elseBody) collectBlockOps(ite.elseBody, ops);
      return;
    }
    case 'SeqNode':
      for (const st2 of (node as SeqNode).stmts) collectBlockOps(st2, ops);
      return;
    case 'ForNode':
      collectBlockOps((node as ForNode).body, ops);
      return;
    case 'BlockNode': {
      const b = node as BlockNode;
      if (b.initBody) collectBlockOps(b.initBody, ops);
      collectBlockOps(b.body, ops);
      return;
    }
    case 'LetStmtNode': {
      const ls = node as LetStmtNode;
      collectBlockOps(ls.value, ops);
      collectBlockOps(ls.body, ops);
      return;
    }
    default:
      return;
  }
}

function fnv1a(str: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}
