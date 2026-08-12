import { ForKind } from '../ir/tensor/nodes.js';
import { walk, walkScoped } from '../ir/ir_visitor.js';
import { parseThreadAxis, maxBindingExtent } from './thread_binding.js';
import type { Buffer } from '../ir/tensor/buffer.js';
import type { IRNode } from '../ir/ir_visitor.js';
import type { PrimFunc } from '../ir/tensor/nodes.js';
import type { LIRFunc, LIRThreadBinding } from '../ir/lir/nodes.js';

export type GpuAccessKind = 'store' | 'load';

export type GpuAccess = Readonly<{
  kind: GpuAccessKind;
  bindingSig: string;
  narrow: boolean;
  underBlockBinding: boolean;
  multiplicity: number;
  seqLoopIndexed: boolean;
  literalValue: boolean;
}>;

export type GpuBufferProfile = {
  name: string;
  buffer: Buffer;
  isStorage: boolean;
  isShared: boolean;
  isThreadPrivate: boolean;
  numel: number;
  stores: GpuAccess[];
  loads: GpuAccess[];
};

export type GpuAccessProfile = ReadonlyMap<string, GpuBufferProfile>;

export type GpuLaunchDiagnosis = Readonly<{ reason: string; buffers: readonly string[] }>;

export const GpuRaceReason = Object.freeze({
  CROSS_BLOCK_RAW: 'cross-block read-after-write on a global buffer',
  THREAD_SHARED_INTERMEDIATE: 'kernel-local buffer read by a thread that did not write it',
  MULTI_EXTENT_BLOCK_BINDING: 'one block axis bound at two different extents',
  RECURRENCE_EXCEEDS_WORKGROUP: 'sequential recurrence does not fit a single workgroup',
  EXTENT_MISMATCH: 'buffer written and read under different thread multiplicities',
});

export type GpuProfileOpts = Readonly<{
  sharedBuffers?: Iterable<Buffer>;
  threadBindings?: ReadonlyMap<string, readonly LIRThreadBinding[]>;
}>;

type RaceScope = Readonly<{
  bindings: readonly string[];
  narrow: boolean;
  underThread: boolean;
  underBlock: boolean;
  multiplicity: number;
  seqVars: readonly string[];
}>;

type IndexedNode = { buffer?: Buffer; indices?: readonly IRNode[]; offsetExpr?: IRNode | null; value?: IRNode | null };

const ROOT_SCOPE: RaceScope = {
  bindings: [],
  narrow: false,
  underThread: false,
  underBlock: false,
  multiplicity: 1,
  seqVars: [],
};

const STORE_TYPES = new Set(['BufferStoreNode', 'LIRFlatStoreNode']);
const LOAD_TYPES = new Set(['BufferLoadNode', 'LIRFlatLoadNode']);
const SEQ_LOOP_TYPES = new Set(['WhileNode', 'LIRAccumulatorNode']);

function collectVarNames(node: IRNode | null | undefined, out: Set<string>): void {
  if (!node) return;
  walk(node, (n) => { if (n.type === 'VariableNode') out.add((n as { name: string }).name); });
}

function dependsOn(expr: IRNode | null | undefined, vars: readonly string[]): boolean {
  if (!expr || vars.length === 0) return false;
  const names = new Set<string>();
  collectVarNames(expr, names);
  for (const v of vars) if (names.has(v)) return true;
  return false;
}

function staticExtent(node: { extent?: IRNode | null }): number {
  const e = node.extent as { type?: string; value?: number } | null | undefined;
  return e && e.type === 'IntImmNode' ? (e.value as number) : 0;
}

function isLiteral(node: IRNode | null | undefined): boolean {
  return !!node && (node.type === 'FloatImmNode' || node.type === 'IntImmNode');
}

function accessIndices(node: IndexedNode): readonly IRNode[] {
  if (node.indices) return node.indices;
  return node.offsetExpr ? [node.offsetExpr] : [];
}

function bufferNumel(buffer: Buffer): number {
  return typeof buffer.numel === 'function' ? buffer.numel() : -1;
}

export function profileGpuAccesses(func: PrimFunc | LIRFunc, opts: GpuProfileOpts = {}): GpuAccessProfile {
  const storage = new Set<string>();
  for (const [, buf] of func.bufferMap) storage.add(buf.name);
  const shared = new Set<string>();
  for (const buf of opts.sharedBuffers || []) shared.add(buf.name);
  const threadBindings = opts.threadBindings || new Map<string, readonly LIRThreadBinding[]>();

  const profiles = new Map<string, GpuBufferProfile>();
  const threadPrivate = new Set<string>();

  const entryFor = (buffer: Buffer): GpuBufferProfile => {
    let p = profiles.get(buffer.name);
    if (!p) {
      p = {
        name: buffer.name,
        buffer,
        isStorage: storage.has(buffer.name),
        isShared: shared.has(buffer.name),
        isThreadPrivate: false,
        numel: bufferNumel(buffer),
        stores: [],
        loads: [],
      };
      profiles.set(buffer.name, p);
    }
    return p;
  };

  const record = (kind: GpuAccessKind, node: IndexedNode, scope: RaceScope): void => {
    const buffer = node.buffer;
    if (!buffer) return;
    const access: GpuAccess = {
      kind,
      bindingSig: [...scope.bindings].sort().join(','),
      narrow: scope.narrow,
      underBlockBinding: scope.underBlock,
      multiplicity: scope.multiplicity,
      seqLoopIndexed: accessIndices(node).some((idx) => dependsOn(idx, scope.seqVars)),
      literalValue: kind === 'store' && isLiteral(node.value),
    };
    const entry = entryFor(buffer);
    if (kind === 'store') entry.stores.push(access);
    else entry.loads.push(access);
  };

  walkScoped<RaceScope>(func.body as IRNode, ROOT_SCOPE, (node, scope) => {
    let next = scope;

    if (node.type === 'ForNode') {
      const forNode = node as unknown as { kind: string; threadTag: string | null; loopVar?: { name: string } };
      if (forNode.kind === ForKind.THREAD_BINDING && forNode.threadTag) {
        const tag = forNode.threadTag;
        const extent = staticExtent(node as unknown as { extent?: IRNode | null });
        const maxExtent = maxBindingExtent(threadBindings, tag);
        const axis = parseThreadAxis(tag);
        next = {
          bindings: [...next.bindings, `${tag}:${extent}`],
          narrow: next.narrow || (extent > 0 && maxExtent > 0 && extent < maxExtent),
          underThread: next.underThread || (axis !== null && axis.space === 'thread'),
          underBlock: next.underBlock || (axis !== null && axis.space === 'block'),
          multiplicity: extent > 0 ? next.multiplicity * extent : next.multiplicity,
          seqVars: next.seqVars,
        };
      } else if (forNode.loopVar) {
        next = { ...next, seqVars: [...next.seqVars, forNode.loopVar.name] };
      }
    } else if (SEQ_LOOP_TYPES.has(node.type)) {
      const loopVar = (node as unknown as { loopVar?: { name: string } }).loopVar;
      if (loopVar) next = { ...next, seqVars: [...next.seqVars, loopVar.name] };
    }

    if (node.type === 'LIRBindingsNode') {
      const bindings = (node as unknown as { bindings?: readonly { name: string; expr: IRNode }[] }).bindings || [];
      let seqVars = next.seqVars;
      for (const b of bindings) {
        if (!seqVars.includes(b.name) && dependsOn(b.expr, seqVars)) seqVars = [...seqVars, b.name];
      }
      if (seqVars !== next.seqVars) next = { ...next, seqVars };
    } else if (node.type === 'BlockNode') {
      const iterVars = (node as unknown as { iterVars?: readonly { iterVar?: { name: string }; binding?: IRNode }[] }).iterVars || [];
      let seqVars = next.seqVars;
      for (const iv of iterVars) {
        if (iv.iterVar && iv.binding && !seqVars.includes(iv.iterVar.name) && dependsOn(iv.binding, seqVars)) {
          seqVars = [...seqVars, iv.iterVar.name];
        }
      }
      if (seqVars !== next.seqVars) next = { ...next, seqVars };
    } else if (node.type === 'LetStmtNode') {
      const let_ = node as unknown as { variable?: { name: string }; value?: IRNode };
      if (let_.variable && !next.seqVars.includes(let_.variable.name) && dependsOn(let_.value, next.seqVars)) {
        next = { ...next, seqVars: [...next.seqVars, let_.variable.name] };
      }
    } else if (node.type === 'AllocateNode') {
      const buffer = (node as unknown as { buffer?: Buffer }).buffer;
      if (buffer) {
        entryFor(buffer);
        if (next.underThread) threadPrivate.add(buffer.name);
      }
    }

    if (STORE_TYPES.has(node.type)) record('store', node as unknown as IndexedNode, next);
    else if (LOAD_TYPES.has(node.type)) record('load', node as unknown as IndexedNode, next);

    return next;
  });

  for (const name of threadPrivate) {
    const p = profiles.get(name);
    if (p) p.isThreadPrivate = true;
  }
  return profiles;
}

export type LaunchGeometry = Readonly<{ blockThreads: number; gridThreads: number }>;

export function launchGeometry(func: PrimFunc | LIRFunc): LaunchGeometry {
  const blockDim = [1, 1, 1];
  const gridDim = [1, 1, 1];
  walk(func.body as IRNode, (node) => {
    if (node.type !== 'ForNode') return;
    const forNode = node as unknown as { kind: string; threadTag: string | null };
    if (forNode.kind !== ForKind.THREAD_BINDING || !forNode.threadTag) return;
    const axis = parseThreadAxis(forNode.threadTag);
    if (!axis) return;
    const extent = staticExtent(node as unknown as { extent?: IRNode | null });
    if (extent <= 0) return;
    const dims = axis.space === 'thread' ? blockDim : gridDim;
    dims[axis.axis] = Math.max(dims[axis.axis], extent);
  });
  return {
    blockThreads: blockDim[0] * blockDim[1] * blockDim[2],
    gridThreads: gridDim[0] * gridDim[1] * gridDim[2],
  };
}

export function crossBlockRAWBuffers(profile: GpuAccessProfile): Set<string> {
  const result = new Set<string>();
  for (const [name, p] of profile) {
    if (!p.isStorage || p.stores.length === 0 || p.loads.length === 0) continue;
    const sigs = new Set<string>();
    for (const a of p.stores) sigs.add(a.bindingSig);
    for (const a of p.loads) sigs.add(a.bindingSig);
    if (sigs.size > 1) result.add(name);
  }
  return result;
}

export function threadSharedIntermediates(profile: GpuAccessProfile): Set<string> {
  const result = new Set<string>();
  for (const [name, p] of profile) {
    if (p.isStorage || p.isShared || p.isThreadPrivate) continue;
    if (p.stores.length === 0 || p.loads.length === 0) continue;
    if (p.numel !== 1) { result.add(name); continue; }
    if (p.stores.some((a) => a.narrow)) result.add(name);
  }
  return result;
}

export function loopCarriedIntermediates(profile: GpuAccessProfile): Set<string> {
  const result = new Set<string>();
  for (const [name, p] of profile) {
    if (p.isStorage) continue;
    if (p.loads.some((a) => a.seqLoopIndexed)) result.add(name);
  }
  return result;
}

export function extentMismatchBuffers(profile: GpuAccessProfile): Set<string> {
  const result = new Set<string>();
  for (const [name, p] of profile) {
    if (p.isStorage || p.stores.length === 0) continue;
    if (!p.stores.some((a) => !a.literalValue)) continue;
    const stored = new Set(p.stores.map((a) => a.multiplicity));
    if (p.loads.some((a) => !stored.has(a.multiplicity))) result.add(name);
  }
  return result;
}

export function storedUnderBlockBinding(profile: GpuAccessProfile, names: ReadonlySet<string>): boolean {
  for (const name of names) {
    const p = profile.get(name);
    if (p && p.stores.some((a) => a.underBlockBinding)) return true;
  }
  return false;
}

export function hasMultiExtentBlockBinding(threadBindings: ReadonlyMap<string, readonly LIRThreadBinding[]>): { blockSpace: boolean; threadSpace: boolean } {
  let blockSpace = false;
  let threadSpace = false;
  for (const [tag, entries] of threadBindings) {
    const extents = new Set<number>();
    for (const e of entries) if (e.extent > 0) extents.add(e.extent);
    if (extents.size <= 1) continue;
    const axis = parseThreadAxis(tag);
    if (axis && axis.space === 'block') blockSpace = true;
    else threadSpace = true;
  }
  return { blockSpace, threadSpace };
}
