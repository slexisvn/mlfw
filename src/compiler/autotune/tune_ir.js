import { PrimFunc, ForNode } from '../ir/tensor/nodes.js';

export function clonePrimFunc(primFunc) {
  const cloneNode = (node) => {
    if (!node || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(cloneNode);
    const copy = Object.create(Object.getPrototypeOf(node));
    copy.type = node.type;
    copy._parent = null;
    copy._parentKey = null;
    copy._parentIdx = -1;
    switch (node.type) {
      case 'PrimFunc':
        copy.name = node.name;
        copy.params = node.params;
        copy.body = cloneNode(node.body);
        copy.bufferMap = new Map(node.bufferMap);
        copy.shapeParams = node.shapeParams;
        copy._setChild('body', copy.body);
        break;
      case 'ForNode':
        copy.loopVar = node.loopVar;
        copy.min = cloneNode(node.min);
        copy.extent = cloneNode(node.extent);
        copy.kind = node.kind;
        copy.body = cloneNode(node.body);
        copy.threadTag = node.threadTag;
        copy._setChild('body', copy.body);
        break;
      case 'BlockNode':
        copy.name = node.name;
        copy.iterVars = node.iterVars.map(cloneNode);
        copy.reads = node.reads;
        copy.writes = node.writes;
        copy.body = cloneNode(node.body);
        copy.initBody = cloneNode(node.initBody);
        copy._setChild('body', copy.body);
        copy._setChild('initBody', copy.initBody);
        break;
      case 'SeqNode':
        copy.stmts = node.stmts.map(cloneNode);
        copy._setChildren('stmts', copy.stmts);
        break;
      case 'AllocateNode':
        copy.buffer = node.buffer;
        copy.scope = node.scope;
        copy.body = cloneNode(node.body);
        copy._setChild('body', copy.body);
        break;
      case 'LetStmtNode':
        copy.variable = node.variable;
        copy.value = cloneNode(node.value);
        copy.body = cloneNode(node.body);
        copy._setChild('body', copy.body);
        break;
      case 'IfThenElseNode':
        copy.condition = cloneNode(node.condition);
        copy.thenBody = cloneNode(node.thenBody);
        copy.elseBody = cloneNode(node.elseBody);
        copy._setChild('thenBody', copy.thenBody);
        copy._setChild('elseBody', copy.elseBody);
        break;
      case 'WhileNode':
        copy.condVar = node.condVar;
        copy.condBody = cloneNode(node.condBody);
        copy.loopBody = cloneNode(node.loopBody);
        copy._setChild('condBody', copy.condBody);
        copy._setChild('loopBody', copy.loopBody);
        break;
      default:
        for (const key of Object.keys(node)) {
          if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
          const val = node[key];
          if (val instanceof Map) copy[key] = new Map(val);
          else if (Array.isArray(val)) copy[key] = val.map(cloneNode);
          else if (typeof val === 'object' && val !== null && val.type) copy[key] = cloneNode(val);
          else copy[key] = val;
        }
        break;
    }
    return copy;
  };
  return cloneNode(primFunc);
}

export function cloneBlockSubtree(block) {
  const cloneNode = (node) => {
    if (!node || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(cloneNode);
    const copy = Object.create(Object.getPrototypeOf(node));
    copy.type = node.type;
    copy._parent = null;
    copy._parentKey = null;
    copy._parentIdx = -1;
    for (const key of Object.keys(node)) {
      if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
      const val = node[key];
      if (val instanceof Map) copy[key] = new Map(val);
      else if (Array.isArray(val)) copy[key] = val.map(cloneNode);
      else if (typeof val === 'object' && val !== null && val.type) copy[key] = cloneNode(val);
      else copy[key] = val;
    }
    if (copy._setChild) {
      if (copy.body) copy._setChild('body', copy.body);
      if (copy.initBody) copy._setChild('initBody', copy.initBody);
      if (copy.thenBody) copy._setChild('thenBody', copy.thenBody);
      if (copy.elseBody) copy._setChild('elseBody', copy.elseBody);
    }
    if (copy._setChildren && copy.stmts) copy._setChildren('stmts', copy.stmts);
    return copy;
  };
  return cloneNode(block);
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
