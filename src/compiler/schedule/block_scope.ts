import { AccessKind } from '../analysis/buffer_access.js';
import { DepKind } from '../analysis/dependence.js';
import { rangesOverlap, rangesContain, regionHull } from './dep_analysis.js';
import type { MutableRegion, Region } from './dep_analysis.js';
import type { DepKindValue } from '../analysis/dependence.js';
import type { BufferAccess, BufferAccessResult, AccessRegion } from '../analysis/buffer_access.js';
import type { Buffer } from '../ir/tensor/buffer.js';
import type { BlockNode } from '../ir/tensor/nodes.js';
import type { SRef } from './sref.js';

export type AccessUnit = {
  member: ScopeMember;
  read: MutableRegion | null;
  write: MutableRegion | null;
  position: number;
};

export type RecordDepFn = (src: AccessUnit, dst: AccessUnit, kind: DepKindValue, buffer: Buffer) => void;

export class BlockDependency {
  src: SRef;
  dst: SRef;
  kind: DepKindValue;
  buffer: Buffer;

  constructor(src: SRef, dst: SRef, kind: DepKindValue, buffer: Buffer) {
    this.src = src;
    this.dst = dst;
    this.kind = kind;
    this.buffer = buffer;
  }
}

export class BlockInfo {
  sref: SRef;
  affineBinding: boolean;
  regionCover: boolean;

  constructor(sref: SRef, affineBinding: boolean, regionCover: boolean) {
    this.sref = sref;
    this.affineBinding = affineBinding;
    this.regionCover = regionCover;
  }
}

class ScopeMember {
  sref: SRef;
  position: number;
  reads: Map<Buffer, AccessRegion[][]>;
  writes: Map<Buffer, AccessRegion[][]>;

  constructor(sref: SRef) {
    this.sref = sref;
    this.position = Infinity;
    this.reads = new Map();
    this.writes = new Map();
  }

  add(access: BufferAccess): void {
    if (access.position < this.position) this.position = access.position;
    const target = access.kind === AccessKind.WRITE ? this.writes : this.reads;
    let regions = target.get(access.buffer);
    if (!regions) { regions = []; target.set(access.buffer, regions); }
    regions.push(access.regions);
  }

  hull(target: ReadonlyMap<Buffer, AccessRegion[][]>, buffer: Buffer): MutableRegion | null {
    const regions = target.get(buffer);
    return regions ? regionHull(regions as unknown as Region[]) : null;
  }
}

export class BlockScope {
  root: SRef | null;
  members: Map<SRef, ScopeMember>;
  opaqueAccesses: BufferAccess[];
  deps: BlockDependency[];
  stagePipeline: boolean;
  _bySrc: Map<SRef, BlockDependency[]>;
  _byDst: Map<SRef, BlockDependency[]>;
  _writers: Map<Buffer, SRef[]>;
  _blockInfo: Map<SRef, BlockInfo>;

  constructor(root: SRef | null) {
    this.root = root;
    this.members = new Map();
    this.opaqueAccesses = [];
    this.deps = [];
    this._bySrc = new Map();
    this._byDst = new Map();
    this._writers = new Map();
    this._blockInfo = new Map();
    this.stagePipeline = false;
  }

  get children(): SRef[] {
    return [...this.members.keys()];
  }

  memberOf(sref: SRef): ScopeMember | null {
    return this.members.get(sref) || null;
  }

  depsBySrc(sref: SRef): BlockDependency[] {
    return this._bySrc.get(sref) || [];
  }

  depsByDst(sref: SRef): BlockDependency[] {
    return this._byDst.get(sref) || [];
  }

  writersOf(buffer: Buffer): SRef[] {
    return this._writers.get(buffer) || [];
  }

  producersOf(sref: SRef): SRef[] {
    return this.depsByDst(sref).filter((d) => d.kind === DepKind.RAW).map((d) => d.src);
  }

  consumersOf(sref: SRef): SRef[] {
    return this.depsBySrc(sref).filter((d) => d.kind === DepKind.RAW).map((d) => d.dst);
  }

  blockInfo(sref: SRef): BlockInfo | null {
    return this._blockInfo.get(sref) || null;
  }

  _record(dep: BlockDependency): void {
    this.deps.push(dep);
    let src = this._bySrc.get(dep.src);
    if (!src) { src = []; this._bySrc.set(dep.src, src); }
    src.push(dep);
    let dst = this._byDst.get(dep.dst);
    if (!dst) { dst = []; this._byDst.set(dep.dst, dst); }
    dst.push(dep);
  }
}

function scopeRootOf(sref: SRef): SRef | null {
  let cur = sref.parent;
  while (cur) {
    if (cur.isBlock) return cur;
    cur = cur.parent;
  }
  return null;
}

function blockChain(sref: SRef): SRef[] {
  const chain: SRef[] = [];
  let cur: SRef | null = sref;
  while (cur) {
    if (cur.isBlock) chain.push(cur);
    cur = cur.parent;
  }
  return chain;
}

export function linkAccessUnits(units: readonly AccessUnit[], buffer: Buffer, record: RecordDepFn): void {
  for (const writer of units) {
    if (!writer.write) continue;
    for (const other of units) {
      if (other === writer) continue;
      if (other.write && other.position < writer.position) continue;
      const [src, dst] = writer.position <= other.position ? [writer, other] : [other, writer];
      if (src.write && dst.read && rangesOverlap(src.write, dst.read)) record(src, dst, DepKind.RAW, buffer);
      if (src.write && dst.write && rangesOverlap(src.write, dst.write)) record(src, dst, DepKind.WAW, buffer);
      if (src.read && dst.write && rangesOverlap(src.read, dst.write)) record(src, dst, DepKind.WAR, buffer);
    }
  }
}

function linkBuffer(scope: BlockScope, buffer: Buffer, touchers: readonly Omit<AccessUnit, 'position'>[]): void {
  const units: AccessUnit[] = touchers.map((t) => ({ ...t, position: t.member.position }));
  linkAccessUnits(units, buffer, (src, dst, kind) => {
    scope._record(new BlockDependency(src.member.sref, dst.member.sref, kind, buffer));
  });
}

function computeRegionCover(scope: BlockScope, member: ScopeMember, touchersOf: ReadonlyMap<Buffer, Omit<AccessUnit, 'position'>[]>): boolean {
  for (const [buffer] of member.reads) {
    const producers = (touchersOf.get(buffer) as Omit<AccessUnit, 'position'>[]).filter((t) => t.write && t.member !== member);
    if (producers.length === 0) continue;
    const covered = regionHull(producers.map((t) => t.write as MutableRegion));
    if (!rangesContain(covered, member.hull(member.reads, buffer))) return false;
  }
  return true;
}

export function buildBlockScopes(blockSRefs: Iterable<SRef>, accessInfo: BufferAccessResult): Map<SRef | null, BlockScope> {
  const srefOfBlock = new Map<BlockNode, SRef>();
  for (const sref of blockSRefs) srefOfBlock.set(sref.node as BlockNode, sref);

  const scopes = new Map<SRef | null, BlockScope>();
  const scopeFor = (root: SRef | null): BlockScope => {
    let scope = scopes.get(root);
    if (!scope) { scope = new BlockScope(root); scopes.set(root, scope); }
    return scope;
  };
  scopeFor(null);
  for (const sref of blockSRefs) scopeFor(scopeRootOf(sref));

  for (const access of accessInfo.order) {
    const innermost = access.block ? srefOfBlock.get(access.block) : null;
    if (!innermost) {
      scopeFor(null).opaqueAccesses.push(access);
      continue;
    }
    for (const sref of blockChain(innermost)) {
      const scope = scopeFor(scopeRootOf(sref));
      let member = scope.members.get(sref);
      if (!member) { member = new ScopeMember(sref); scope.members.set(sref, member); }
      member.add(access);
    }
  }

  for (const scope of scopes.values()) {
    const members = [...scope.members.values()].sort((a, b) => a.position - b.position);
    scope.members = new Map(members.map((m) => [m.sref, m]));

    const indexed = new Map<Buffer, Map<ScopeMember, Omit<AccessUnit, 'position'>>>();
    const toucher = (buffer: Buffer, member: ScopeMember): Omit<AccessUnit, 'position'> => {
      let byMember = indexed.get(buffer);
      if (!byMember) { byMember = new Map(); indexed.set(buffer, byMember); }
      let entry = byMember.get(member);
      if (!entry) { entry = { member, read: null, write: null }; byMember.set(member, entry); }
      return entry;
    };
    for (const member of members) {
      for (const [buffer] of member.reads) toucher(buffer, member).read = member.hull(member.reads, buffer);
      for (const [buffer] of member.writes) {
        toucher(buffer, member).write = member.hull(member.writes, buffer);
        let writers = scope._writers.get(buffer);
        if (!writers) { writers = []; scope._writers.set(buffer, writers); }
        writers.push(member.sref);
      }
    }
    const touchersOf = new Map<Buffer, Omit<AccessUnit, 'position'>[]>();
    for (const [buffer, byMember] of indexed) touchersOf.set(buffer, [...byMember.values()]);
    for (const [buffer, touchers] of touchersOf) linkBuffer(scope, buffer, touchers);

    let pipeline = scope.opaqueAccesses.length === 0;
    for (const member of members) {
      const info = accessInfo.byBlock.get(member.sref.node as BlockNode);
      const regionCover = computeRegionCover(scope, member, touchersOf);
      scope._blockInfo.set(member.sref, new BlockInfo(member.sref, info ? info.affineBinding : false, regionCover));
      if (!regionCover) pipeline = false;
    }
    if (scope.deps.some((d) => d.kind === DepKind.WAR)) pipeline = false;
    scope.stagePipeline = pipeline;
  }

  return scopes;
}

export function scopeRootSRef(sref: SRef): SRef | null {
  return scopeRootOf(sref);
}
