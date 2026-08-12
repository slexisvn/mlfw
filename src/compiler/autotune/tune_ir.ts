import { PrimFunc, ForNode } from '../ir/tensor/nodes.js';
import { cloneIRShared } from '../ir/clone_ir.js';
import type { CloneableIRNode, CloneRecurse } from '../ir/clone_ir.js';
import type { TirNode, BlockNode, ForNode as ForNodeType } from '../ir/tensor/nodes.js';
import type { Buffer } from '../ir/tensor/buffer.js';

type CloneSlots = CloneableIRNode & { _setChild?: (k: string, c: unknown) => void };

function deepCloneIR(node: unknown): TirNode {
  return cloneIRShared(node, deepCloneIR as CloneRecurse, (n: CloneableIRNode, copy: CloneableIRNode, rec: CloneRecurse) => {
    const src = n as CloneSlots;
    const dst = copy as CloneSlots;
    switch (n.type) {
      case 'PrimFunc':
        dst.name = n.name;
        dst.params = n.params;
        dst.body = rec(src.body);
        dst.bufferMap = new Map(src.bufferMap as Map<unknown, unknown>);
        dst.shapeParams = n.shapeParams;
        dst.shapeParamMap = src.shapeParamMap instanceof Map ? new Map(src.shapeParamMap) : src.shapeParamMap;
        (dst._setChild as (k: string, c: unknown) => void)('body', dst.body);
        return copy;
      case 'AllocateNode':
        dst.buffer = n.buffer;
        dst.scope = n.scope;
        dst.body = rec(src.body);
        (dst._setChild as (k: string, c: unknown) => void)('body', dst.body);
        return copy;
      case 'LetStmtNode':
        dst.variable = n.variable;
        dst.value = rec(src.value);
        dst.body = rec(src.body);
        (dst._setChild as (k: string, c: unknown) => void)('body', dst.body);
        return copy;
      case 'WhileNode':
        dst.condVar = n.condVar;
        dst.condBody = rec(src.condBody);
        dst.loopBody = rec(src.loopBody);
        (dst._setChild as (k: string, c: unknown) => void)('condBody', dst.condBody);
        (dst._setChild as (k: string, c: unknown) => void)('loopBody', dst.loopBody);
        return copy;
      default:
        for (const key of Object.keys(src)) {
          if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
          const val = src[key];
          if (val instanceof Map) copy[key] = new Map(val);
          else if (Array.isArray(val)) copy[key] = val.map(rec);
          else if (typeof val === 'object' && val !== null && (val as CloneableIRNode).type) copy[key] = rec(val);
          else copy[key] = val;
        }
        return copy;
    }
  }) as TirNode;
}

export function clonePrimFunc(primFunc: PrimFunc): PrimFunc {
  return deepCloneIR(primFunc) as PrimFunc;
}

export function cloneTensorIR(node: TirNode): TirNode {
  return deepCloneIR(node);
}

function cloneBlockSubtree(block: TirNode): TirNode {
  return deepCloneIR(block);
}

export function extractBlockMini(primFunc: PrimFunc, blockName: string, blockMap: ReadonlyMap<string, BlockNode>): PrimFunc | null {
  const block = blockMap.get(blockName);
  if (!block) return null;

  const path: ForNodeType[] = [];
  let cur: TirNode | null = block._parent as TirNode | null;
  while (cur && cur !== primFunc) {
    if (cur.type === 'ForNode') path.push(cur as ForNodeType);
    cur = cur._parent as TirNode | null;
  }
  path.reverse();

  let body: TirNode = cloneBlockSubtree(block);

  for (let i = path.length - 1; i >= 0; i--) {
    const loop = path[i];
    const wrapper = new ForNode(
      loop.loopVar,
      cloneBlockSubtree(loop.min),
      cloneBlockSubtree(loop.extent),
      loop.kind,
      body,
      loop.threadTag
    );
    wrapper._setChild('body', body);
    body = wrapper;
  }

  const bufs = new Map<string, Buffer>();
  for (const r of block.reads) bufs.set(r.buffer.name, r.buffer);
  for (const w of block.writes) bufs.set(w.buffer.name, w.buffer);

  const params: PrimFunc['params'] = [];
  for (const p of primFunc.params) {
    if (bufs.has(p.name)) params.push(p);
  }

  return new PrimFunc('__tune_' + blockName, params, body, bufs as never, []);
}
