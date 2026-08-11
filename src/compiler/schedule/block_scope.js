import { AccessKind } from '../analysis/buffer_access.js';
import { DepKind } from '../analysis/dependence.js';
import { rangesOverlap, rangesContain, regionHull } from './dep_analysis.js';

export class BlockDependency {
  constructor(src, dst, kind, buffer) {
    this.src = src;
    this.dst = dst;
    this.kind = kind;
    this.buffer = buffer;
  }
}

export class BlockInfo {
  constructor(sref, affineBinding, regionCover) {
    this.sref = sref;
    this.affineBinding = affineBinding;
    this.regionCover = regionCover;
  }
}

class ScopeMember {
  constructor(sref) {
    this.sref = sref;
    this.position = Infinity;
    this.reads = new Map();
    this.writes = new Map();
  }

  add(access) {
    if (access.position < this.position) this.position = access.position;
    const target = access.kind === AccessKind.WRITE ? this.writes : this.reads;
    let regions = target.get(access.buffer);
    if (!regions) { regions = []; target.set(access.buffer, regions); }
    regions.push(access.regions);
  }

  hull(target, buffer) {
    const regions = target.get(buffer);
    return regions ? regionHull(regions) : null;
  }
}

export class BlockScope {
  constructor(root) {
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

  get children() {
    return [...this.members.keys()];
  }

  memberOf(sref) {
    return this.members.get(sref) || null;
  }

  depsBySrc(sref) {
    return this._bySrc.get(sref) || [];
  }

  depsByDst(sref) {
    return this._byDst.get(sref) || [];
  }

  writersOf(buffer) {
    return this._writers.get(buffer) || [];
  }

  producersOf(sref) {
    return this.depsByDst(sref).filter((d) => d.kind === DepKind.RAW).map((d) => d.src);
  }

  consumersOf(sref) {
    return this.depsBySrc(sref).filter((d) => d.kind === DepKind.RAW).map((d) => d.dst);
  }

  blockInfo(sref) {
    return this._blockInfo.get(sref) || null;
  }

  _record(dep) {
    this.deps.push(dep);
    let src = this._bySrc.get(dep.src);
    if (!src) { src = []; this._bySrc.set(dep.src, src); }
    src.push(dep);
    let dst = this._byDst.get(dep.dst);
    if (!dst) { dst = []; this._byDst.set(dep.dst, dst); }
    dst.push(dep);
  }
}

function scopeRootOf(sref) {
  let cur = sref.parent;
  while (cur) {
    if (cur.isBlock) return cur;
    cur = cur.parent;
  }
  return null;
}

function blockChain(sref) {
  const chain = [];
  let cur = sref;
  while (cur) {
    if (cur.isBlock) chain.push(cur);
    cur = cur.parent;
  }
  return chain;
}

export function linkAccessUnits(units, buffer, record) {
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

function linkBuffer(scope, buffer, touchers) {
  const units = touchers.map((t) => ({ ...t, position: t.member.position }));
  linkAccessUnits(units, buffer, (src, dst, kind) => {
    scope._record(new BlockDependency(src.member.sref, dst.member.sref, kind, buffer));
  });
}

function computeRegionCover(scope, member, touchersOf) {
  for (const [buffer] of member.reads) {
    const producers = touchersOf.get(buffer).filter((t) => t.write && t.member !== member);
    if (producers.length === 0) continue;
    const covered = regionHull(producers.map((t) => t.write));
    if (!rangesContain(covered, member.hull(member.reads, buffer))) return false;
  }
  return true;
}

export function buildBlockScopes(blockSRefs, accessInfo) {
  const srefOfBlock = new Map();
  for (const sref of blockSRefs) srefOfBlock.set(sref.node, sref);

  const scopes = new Map();
  const scopeFor = (root) => {
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

    const indexed = new Map();
    const toucher = (buffer, member) => {
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
    const touchersOf = new Map();
    for (const [buffer, byMember] of indexed) touchersOf.set(buffer, [...byMember.values()]);
    for (const [buffer, touchers] of touchersOf) linkBuffer(scope, buffer, touchers);

    let pipeline = scope.opaqueAccesses.length === 0;
    for (const member of members) {
      const info = accessInfo.byBlock.get(member.sref.node);
      const regionCover = computeRegionCover(scope, member, touchersOf);
      scope._blockInfo.set(member.sref, new BlockInfo(member.sref, info ? info.affineBinding : false, regionCover));
      if (!regionCover) pipeline = false;
    }
    if (scope.deps.some((d) => d.kind === DepKind.WAR)) pipeline = false;
    scope.stagePipeline = pipeline;
  }

  return scopes;
}

export function scopeRootSRef(sref) {
  return scopeRootOf(sref);
}
