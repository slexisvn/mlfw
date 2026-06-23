import { LIRMetadata, normalizeDtype, isWasmNativeOp } from './nodes.js';
import { wasmBytes } from '../../../backend/dtype_map.js';
import { ForKind } from '../tensor/nodes.js';

export function scanMetadata(primFunc, target) {
  const meta = new LIRMetadata();

  for (const [, buf] of primFunc.bufferMap) {
    meta.paramBuffers.add(buf.name);
  }

  walkTree(primFunc.body, meta, target);
  computeMemoryLayout(primFunc, meta, target);
  detectZeroBuffers(primFunc.body, meta);

  return meta;
}

function walkTree(root, meta, target) {
  const stack = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node || typeof node !== 'object') continue;

    switch (node.type) {
      case 'ForNode':
        if (node.loopVar) {
          meta.locals.set(node.loopVar.name, 'i32');
        }
        if (node.kind === ForKind.THREAD_BINDING && node.threadTag) {
          const extent = node.extent && node.extent.type === 'IntImmNode' ? node.extent.value : 0;
          const isDynamic = !node.extent || node.extent.type !== 'IntImmNode';
          const entry = { varName: node.loopVar.name, extent, isDynamic, extentNode: node.extent };
          if (!meta.threadBindings.has(node.threadTag)) {
            meta.threadBindings.set(node.threadTag, [entry]);
          } else {
            meta.threadBindings.get(node.threadTag).push(entry);
          }
        }
        break;

      case 'LetStmtNode':
        if (node.variable) {
          meta.locals.set(node.variable.name, normalizeDtype(node.variable.dtype));
        }
        break;

      case 'BufferStoreNode':
      case 'BufferLoadNode':
        if (node.buffer) {
          meta.usedBuffers.set(node.buffer.name, node.buffer);
        }
        break;

      case 'AllocateNode':
        if (node.buffer) {
          meta.allocatedBuffers.add(node.buffer.name);
        }
        if (node.scope === 'shared' && node.buffer) {
          meta.sharedBuffers.push(node.buffer);
        }
        break;

      case 'CallExternNode':
        if (node.externName && !isWasmNativeOp(node.externName)) {
          meta.externCalls.set(node.externName, {
            argCount: node.args ? node.args.length : 0,
            dtype: node.dtype || 'f32',
          });
        }
        break;

      case 'BlockNode':
        if (node.reads) {
          for (const r of node.reads) {
            if (r.buffer) meta.usedBuffers.set(r.buffer.name, r.buffer);
          }
        }
        if (node.writes) {
          for (const w of node.writes) {
            if (w.buffer) meta.usedBuffers.set(w.buffer.name, w.buffer);
          }
        }
        if (node.iterVars) {
          for (const iv of node.iterVars) {
            if (iv.iterVar) {
              meta.locals.set(iv.iterVar.name, normalizeDtype(iv.iterVar.dtype));
            }
            if (iv.binding && typeof iv.binding === 'object' && iv.binding.type) {
              stack.push(iv.binding);
            }
          }
        }
        break;
    }

    if (node.body) stack.push(node.body);
    if (node.value && typeof node.value === 'object' && node.value.type) stack.push(node.value);
    if (node.stmts) for (const s of node.stmts) stack.push(s);
    if (node.thenBody) stack.push(node.thenBody);
    if (node.elseBody) stack.push(node.elseBody);
    if (node.initBody) stack.push(node.initBody);
    if (node.condBody) stack.push(node.condBody);
    if (node.loopBody) stack.push(node.loopBody);
    if (node.condition && typeof node.condition === 'object' && node.condition.type) stack.push(node.condition);
    if (node.a && typeof node.a === 'object' && node.a.type) stack.push(node.a);
    if (node.b && typeof node.b === 'object' && node.b.type) stack.push(node.b);
    if (node.expr && typeof node.expr === 'object' && node.expr.type) stack.push(node.expr);
    if (node.args) {
      for (const a of node.args) {
        if (typeof a === 'object' && a !== null && a.type) stack.push(a);
      }
    }
    if (node.indices) {
      for (const idx of node.indices) {
        if (typeof idx === 'object' && idx !== null && idx.type) stack.push(idx);
      }
    }
  }
}

const DYNAMIC_BUFFER_SLAB_BYTES = 65536;

function computeMemoryLayout(primFunc, meta, target) {
  const align = meta.memoryLayout.alignment;
  let offset = 0;

  const bufBytes = (buf) => {
    const isDynamic = buf.shape.some(d => typeof d !== 'number' || d < 0);
    const numel = buf.numel();
    if (!isDynamic && numel >= 0) return numel * wasmBytes(buf.dtype);
    let staticLowerBound = 1;
    for (const d of buf.shape) staticLowerBound *= (typeof d === 'number' && d > 0) ? d : 1;
    return Math.max(DYNAMIC_BUFFER_SLAB_BYTES, staticLowerBound * wasmBytes(buf.dtype));
  };

  for (const [, buf] of primFunc.bufferMap) {
    offset = Math.ceil(offset / align) * align;
    meta.memoryLayout.bufferOffsets.set(buf.name, offset);
    offset += bufBytes(buf);
  }

  for (const [name, buf] of meta.usedBuffers) {
    if (meta.memoryLayout.bufferOffsets.has(name)) continue;
    offset = Math.ceil(offset / align) * align;
    meta.memoryLayout.bufferOffsets.set(name, offset);
    offset += bufBytes(buf);
  }

  meta.memoryLayout.totalBytes = offset;
}

function detectZeroBuffers(root, meta) {
  const bufferWrites = new Map();
  const stack = [root];

  while (stack.length > 0) {
    const node = stack.pop();
    if (!node || typeof node !== 'object') continue;

    if (node.type === 'BufferStoreNode' && node.buffer) {
      const name = node.buffer.name;
      if (!meta.paramBuffers.has(name)) {
        if (!bufferWrites.has(name)) bufferWrites.set(name, []);
        bufferWrites.get(name).push(node.value);
      }
    }

    if (node.body) stack.push(node.body);
    if (node.value && typeof node.value === 'object' && node.value.type) stack.push(node.value);
    if (node.stmts) for (const s of node.stmts) stack.push(s);
    if (node.thenBody) stack.push(node.thenBody);
    if (node.elseBody) stack.push(node.elseBody);
    if (node.initBody) stack.push(node.initBody);
    if (node.condBody) stack.push(node.condBody);
    if (node.loopBody) stack.push(node.loopBody);
  }

  for (const [name, writes] of bufferWrites) {
    const allZero = writes.every(v =>
      (v && v.type === 'IntImmNode' && v.value === 0) ||
      (v && v.type === 'FloatImmNode' && v.value === 0)
    );
    if (allZero) meta.zeroBuffers.add(name);

    if (writes.length === 1 && writes[0]) {
      const w = writes[0];
      if (w.type === 'IntImmNode' || w.type === 'FloatImmNode') {
        meta.constantBuffers.set(name, w.value);
      }
    }
  }
}
