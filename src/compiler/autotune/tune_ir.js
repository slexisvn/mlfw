import { PrimFunc, ForNode } from '../ir/tensor/nodes.js';
import { cloneIRShared } from '../ir/clone_ir.js';

function deepCloneIR(node) {
  return cloneIRShared(node, deepCloneIR, (n, copy, rec) => {
    switch (n.type) {
      case 'PrimFunc':
        copy.name = n.name;
        copy.params = n.params;
        copy.body = rec(n.body);
        copy.bufferMap = new Map(n.bufferMap);
        copy.shapeParams = n.shapeParams;
        copy.shapeParamMap = n.shapeParamMap instanceof Map ? new Map(n.shapeParamMap) : n.shapeParamMap;
        copy._setChild('body', copy.body);
        return copy;
      case 'AllocateNode':
        copy.buffer = n.buffer;
        copy.scope = n.scope;
        copy.body = rec(n.body);
        copy._setChild('body', copy.body);
        return copy;
      case 'LetStmtNode':
        copy.variable = n.variable;
        copy.value = rec(n.value);
        copy.body = rec(n.body);
        copy._setChild('body', copy.body);
        return copy;
      case 'WhileNode':
        copy.condVar = n.condVar;
        copy.condBody = rec(n.condBody);
        copy.loopBody = rec(n.loopBody);
        copy._setChild('condBody', copy.condBody);
        copy._setChild('loopBody', copy.loopBody);
        return copy;
      default:
        for (const key of Object.keys(n)) {
          if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
          const val = n[key];
          if (val instanceof Map) copy[key] = new Map(val);
          else if (Array.isArray(val)) copy[key] = val.map(rec);
          else if (typeof val === 'object' && val !== null && val.type) copy[key] = rec(val);
          else copy[key] = val;
        }
        return copy;
    }
  });
}

export function clonePrimFunc(primFunc) {
  return deepCloneIR(primFunc);
}

export function cloneTensorIR(node) {
  return deepCloneIR(node);
}

function cloneBlockSubtree(block) {
  return deepCloneIR(block);
}

export function extractBlockMini(primFunc, blockName, blockMap) {
  const block = blockMap.get(blockName);
  if (!block) return null;

  const path = [];
  let cur = block._parent;
  while (cur && cur !== primFunc) {
    if (cur.type === 'ForNode') path.push(cur);
    cur = cur._parent;
  }
  path.reverse();

  let body = cloneBlockSubtree(block);

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

  const bufs = new Map();
  for (const r of block.reads) bufs.set(r.buffer.name, r.buffer);
  for (const w of block.writes) bufs.set(w.buffer.name, w.buffer);

  const params = [];
  for (const p of primFunc.params) {
    if (bufs.has(p.name)) params.push(p);
  }

  return new PrimFunc('__tune_' + blockName, params, body, bufs, []);
}
