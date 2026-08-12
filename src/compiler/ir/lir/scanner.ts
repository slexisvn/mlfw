import { LIRMetadata, normalizeDtype, isWasmNativeOp } from './nodes.js';
import { dtypeBytes } from '../../../util/dtype_map.js';
import { ForKind } from '../tensor/nodes.js';
import { collect as irCollect } from '../ir_visitor.js';
import type { IRNode } from '../ir_visitor.js';
import type { PrimFunc, TirNode, BufferStoreNode, IntImmNode, FloatImmNode } from '../tensor/nodes.js';
import type { LIRMetadata as LIRMetadataType } from './nodes.js';
import type { Buffer } from '../tensor/buffer.js';
export type TargetLike = { isGPU?: () => boolean; kind?: string; name?: string } | null | undefined;

export function scanMetadata(primFunc: PrimFunc, target: TargetLike): LIRMetadataType {
  const meta = new LIRMetadata();

  for (const [, buf] of primFunc.bufferMap) {
    meta.paramBuffers.add(buf.name);
  }

  walkTree(primFunc.body, meta, target);
  computeMemoryLayout(primFunc, meta, target);
  detectZeroBuffers(primFunc.body, meta);

  return meta;
}

function walkTree(root: TirNode, meta: LIRMetadataType, target: TargetLike): void {
  const stack: (TirNode | null | undefined)[] = [root];
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
            (meta.threadBindings.get(node.threadTag) as unknown[]).push(entry);
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

    const slots = node as unknown as Record<string, TirNode | TirNode[] | undefined>;
    if (slots.body) stack.push(slots.body as TirNode);
    if (slots.value && typeof slots.value === 'object' && (slots.value as TirNode).type) stack.push(slots.value as TirNode);
    if (slots.stmts) for (const st of slots.stmts as TirNode[]) stack.push(st);
    if (slots.thenBody) stack.push(slots.thenBody as TirNode);
    if (slots.elseBody) stack.push(slots.elseBody as TirNode);
    if (slots.initBody) stack.push(slots.initBody as TirNode);
    if (slots.condBody) stack.push(slots.condBody as TirNode);
    if (slots.loopBody) stack.push(slots.loopBody as TirNode);
    if (slots.condition && typeof slots.condition === 'object' && (slots.condition as TirNode).type) stack.push(slots.condition as TirNode);
    if (slots.a && typeof slots.a === 'object' && (slots.a as TirNode).type) stack.push(slots.a as TirNode);
    if (slots.b && typeof slots.b === 'object' && (slots.b as TirNode).type) stack.push(slots.b as TirNode);
    if (slots.expr && typeof slots.expr === 'object' && (slots.expr as TirNode).type) stack.push(slots.expr as TirNode);
    if (slots.args) {
      for (const a of slots.args as TirNode[]) {
        if (typeof a === 'object' && a !== null && a.type) stack.push(a);
      }
    }
    if (slots.indices) {
      for (const idx of slots.indices as TirNode[]) {
        if (typeof idx === 'object' && idx !== null && idx.type) stack.push(idx);
      }
    }
  }
}

const DYNAMIC_BUFFER_SLAB_BYTES = 65536;

function computeMemoryLayout(primFunc: PrimFunc, meta: LIRMetadataType, target: TargetLike): void {
  const align = meta.memoryLayout.alignment;
  let offset = 0;

  const bufBytes = (buf: Buffer): number => {
    const isDynamic = buf.shape.some(d => typeof d !== 'number' || d < 0);
    const numel = buf.numel();
    if (!isDynamic && numel >= 0) return numel * dtypeBytes(buf.dtype);
    let staticLowerBound = 1;
    for (const d of buf.shape) staticLowerBound *= (typeof d === 'number' && d > 0) ? d : 1;
    return Math.max(DYNAMIC_BUFFER_SLAB_BYTES, staticLowerBound * dtypeBytes(buf.dtype));
  };

  for (const [, buf] of primFunc.bufferMap) {
    offset = Math.ceil(offset / align) * align;
    meta.memoryLayout.bufferOffsets.set(buf.name, offset);
    offset += bufBytes(buf);
  }

  for (const [name, buf] of meta.usedBuffers as ReadonlyMap<string, Buffer>) {
    if (meta.memoryLayout.bufferOffsets.has(name)) continue;
    offset = Math.ceil(offset / align) * align;
    meta.memoryLayout.bufferOffsets.set(name, offset);
    offset += bufBytes(buf);
  }

  meta.memoryLayout.totalBytes = offset;
}

function detectZeroBuffers(root: TirNode, meta: LIRMetadataType): void {
  const bufferWrites = new Map<string, TirNode[]>();

  for (const node of irCollect(root, (n: IRNode) => n.type === 'BufferStoreNode' && !!n.buffer && !meta.paramBuffers.has(n.buffer.name)) as BufferStoreNode[]) {
    const name = node.buffer.name;
    if (!bufferWrites.has(name)) bufferWrites.set(name, []);
    (bufferWrites.get(name) as TirNode[]).push(node.value);
  }

  for (const [name, writes] of bufferWrites) {
    const allZero = writes.every((v: TirNode | null | undefined) =>
      (v && v.type === 'IntImmNode' && (v as IntImmNode).value === 0) ||
      (v && v.type === 'FloatImmNode' && (v as FloatImmNode).value === 0)
    );
    if (allZero) meta.zeroBuffers.add(name);

    if (writes.length === 1 && writes[0]) {
      const w = writes[0];
      if (w.type === 'IntImmNode' || w.type === 'FloatImmNode') {
        meta.constantBuffers.set(name, (w as IntImmNode | FloatImmNode).value);
      }
    }
  }
}
