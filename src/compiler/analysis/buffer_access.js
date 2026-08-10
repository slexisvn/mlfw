import { walk } from '../ir/ir_visitor.js';
import { exactCoverRange } from './iter_map.js';

export const AccessKind = Object.freeze({ READ: 'read', WRITE: 'write' });

export class BufferAccess {
  constructor(buffer, kind, position, conditional, regions, selfReferential) {
    this.buffer = buffer;
    this.kind = kind;
    this.position = position;
    this.conditional = conditional;
    this.regions = regions;
    this.selfReferential = selfReferential;
  }

  get isWrite() { return this.kind === AccessKind.WRITE; }
  get isRead() { return this.kind === AccessKind.READ; }
}

function constValue(node) {
  return node && node.type === 'IntImmNode' && Number.isInteger(node.value) ? node.value : null;
}

function indexRange(expr, loopRanges) {
  return expr ? exactCoverRange(expr, loopRanges) : null;
}

function loadsBuffer(expr, buffer) {
  let found = false;
  walk(expr, (n) => {
    if (n.type === 'BufferLoadNode' && n.buffer === buffer) { found = true; return false; }
  });
  return found;
}

function storeIsSelfReferential(node) {
  if (loadsBuffer(node.value, node.buffer)) return true;
  for (const idx of node.indices) {
    if (loadsBuffer(idx, node.buffer)) return true;
  }
  return false;
}

export function collectBufferAccesses(root) {
  const order = [];
  const byBuffer = new Map();
  const loopRanges = new Map();
  const scopes = [];
  let conditionalDepth = 0;
  let position = 0;

  const bind = (name, range) => {
    const frame = scopes[scopes.length - 1];
    frame.push([name, loopRanges.get(name)]);
    if (range === null) loopRanges.delete(name);
    else loopRanges.set(name, range);
  };

  const openScope = () => scopes.push([]);
  const closeScope = () => {
    for (const [name, prev] of scopes.pop()) {
      if (prev === undefined) loopRanges.delete(name);
      else loopRanges.set(name, prev);
    }
  };

  const record = (buffer, kind, indices, selfReferential) => {
    const regions = indices.map((idx) => indexRange(idx, loopRanges));
    const access = new BufferAccess(buffer, kind, position++, conditionalDepth > 0, regions, selfReferential);
    order.push(access);
    let list = byBuffer.get(buffer);
    if (!list) { list = []; byBuffer.set(buffer, list); }
    list.push(access);
    return access;
  };

  walk(root, {
    pre(node) {
      switch (node.type) {
        case 'ForNode': {
          openScope();
          const min = constValue(node.min);
          const extent = constValue(node.extent);
          if (node.loopVar) {
            bind(node.loopVar.name, min === null || extent === null ? null : [min, extent]);
          }
          break;
        }
        case 'BlockNode': {
          openScope();
          for (const iv of node.iterVars || []) {
            if (iv && iv.iterVar) bind(iv.iterVar.name, indexRange(iv.binding, loopRanges));
          }
          break;
        }
        case 'IfThenElseNode':
        case 'WhileNode':
          conditionalDepth++;
          break;
        case 'BufferStoreNode':
          record(node.buffer, AccessKind.WRITE, node.indices, storeIsSelfReferential(node));
          break;
        case 'BufferLoadNode':
          record(node.buffer, AccessKind.READ, node.indices, false);
          break;
      }
    },
    post(node) {
      switch (node.type) {
        case 'ForNode':
        case 'BlockNode':
          closeScope();
          break;
        case 'IfThenElseNode':
        case 'WhileNode':
          conditionalDepth--;
          break;
      }
    },
  });

  return { order, byBuffer };
}

export function coversWholeBuffer(access) {
  const shape = access.buffer.shape;
  if (access.regions.length !== shape.length) return false;
  for (let d = 0; d < shape.length; d++) {
    const extent = shape[d];
    if (typeof extent !== 'number') return false;
    const range = access.regions[d];
    if (!range || range[0] !== 0 || range[1] !== extent) return false;
  }
  return true;
}
