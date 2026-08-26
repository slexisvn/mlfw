import type { Dag, DagNode, NestNode } from '../protocol.js';
import type { Box, Layout } from './dag.js';
import { nestOps } from './nest.js';

export type Change = 'kept' | 'added' | 'removed' | 'rewritten' | 'lowered' | 'emitted';

export type Plan = {
  changes: Map<string, Change>;
  links: Map<string, string>;
  counts: Record<Change, number>;
};

export type Placement = {
  box: Box;
  x: number;
  y: number;
  width: number;
  height: number;
  opacity: number;
  scale: number;
  change: Change;
};

export type Frame = {
  placements: Map<string, Placement>;
  width: number;
  height: number;
};

const REMOVE_END = 0.3;
const MOVE_END = 0.76;
const DRIFT_DISTANCE = 160;

function flatten(nodes: readonly DagNode[], into: Map<number, DagNode> = new Map()): Map<number, DagNode> {
  for (const node of nodes) {
    into.set(node.id, node);
    for (const region of node.regions) flatten(region, into);
  }
  return into;
}

function signatureOf(node: DagNode): string {
  return `${node.opName}(${node.operands.join(',')}):${node.resultTypes.join(',')}`;
}

export function linkRewrites(before: Dag | null, after: Dag | null): Map<string, string> {
  const links = new Map<string, string>();
  if (!before || !after) return links;

  const beforeNodes = flatten(before.nodes);
  const afterNodes = flatten(after.nodes);
  const bySignature = new Map<string, number[]>();

  for (const [id, node] of beforeNodes) {
    if (afterNodes.has(id)) continue;
    const signature = signatureOf(node);
    const bucket = bySignature.get(signature);
    if (bucket) bucket.push(id);
    else bySignature.set(signature, [id]);
  }

  for (const [id, node] of afterNodes) {
    if (beforeNodes.has(id)) continue;
    const bucket = bySignature.get(signatureOf(node));
    if (!bucket || bucket.length === 0) continue;
    links.set(`op${bucket.shift() as number}`, `op${id}`);
  }

  return links;
}

export function linkLowering(dag: Dag | null, nest: NestNode | null): Map<string, string> {
  const links = new Map<string, string>();
  if (!dag || !nest) return links;

  const blocksByOp = new Map<string, string[]>();
  for (const [id, op] of nestOps(nest)) {
    const bucket = blocksByOp.get(op);
    if (bucket) bucket.push(id);
    else blocksByOp.set(op, [id]);
  }

  for (const node of flatten(dag.nodes).values()) {
    const bucket = blocksByOp.get(node.opName);
    if (!bucket || bucket.length === 0) continue;
    links.set(`op${node.id}`, bucket.shift() as string);
  }

  return links;
}

const NAMED_NEST_ID = /^(?:for|acc):(.+)$/;

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function linkSourceLines(nest: NestNode | null, source: NestNode | null): Map<string, string> {
  const links = new Map<string, string>();
  if (!nest || !source) return links;

  const names: { id: string; name: string }[] = [];
  const collectNames = (node: NestNode): void => {
    const match = NAMED_NEST_ID.exec(node.id);
    if (match) names.push({ id: node.id, name: match[1] });
    for (const child of node.children) collectNames(child);
  };
  collectNames(nest);

  const lines: NestNode[] = [];
  const collectLines = (node: NestNode): void => {
    if (node.kind === 'line') lines.push(node);
    for (const child of node.children) collectLines(child);
  };
  collectLines(source);

  const taken = new Set<string>();
  for (const { id, name } of names) {
    const pattern = new RegExp(`\\b${escapeRegExp(name)}\\b`);
    const hit = lines.find(line => !taken.has(line.id) && pattern.test(line.label));
    if (!hit) continue;
    taken.add(hit.id);
    links.set(id, hit.id);
  }

  return links;
}

export function planTransition(before: Layout, after: Layout, links: Map<string, string>, linkChange: Change): Plan {
  const changes = new Map<string, Change>();
  const counts: Record<Change, number> = { kept: 0, added: 0, removed: 0, rewritten: 0, lowered: 0, emitted: 0 };
  const beforeIds = new Set(before.boxes.map(box => box.id));
  const afterIds = new Set(after.boxes.map(box => box.id));
  const linkedTargets = new Set(links.values());
  const scaffolding = new Set(
    [...before.boxes, ...after.boxes].filter(box => box.kind === 'port').map(box => box.id),
  );

  for (const id of beforeIds) {
    if (afterIds.has(id)) changes.set(id, 'kept');
    else if (links.has(id)) changes.set(id, linkChange);
    else changes.set(id, 'removed');
  }

  for (const id of afterIds) {
    if (beforeIds.has(id)) continue;
    changes.set(id, linkedTargets.has(id) ? linkChange : 'added');
  }

  for (const [id, change] of changes) {
    if (change === linkChange && links.has(id)) continue;
    if (scaffolding.has(id)) continue;
    counts[change]++;
  }

  return { changes, links, counts };
}

function easeInOut(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function stage(t: number, start: number, end: number): number {
  if (t <= start) return 0;
  if (t >= end) return 1;
  return (t - start) / (end - start);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function frameAt(before: Layout, after: Layout, plan: Plan, t: number): Frame {
  const placements = new Map<string, Placement>();
  const beforeBoxes = new Map(before.boxes.map(box => [box.id, box]));
  const afterBoxes = new Map(after.boxes.map(box => [box.id, box]));

  const move = easeInOut(stage(t, REMOVE_END, MOVE_END));
  const fadeOut = 1 - stage(t, 0, REMOVE_END);
  const fadeIn = stage(t, MOVE_END, 1);
  const linkSources = new Set(plan.links.keys());

  const glide = (id: string, from: Box, to: Box, change: Change): void => {
    placements.set(id, {
      box: move < 0.5 ? from : to,
      x: lerp(from.x, to.x, move),
      y: lerp(from.y, to.y, move),
      width: lerp(from.width, to.width, move),
      height: lerp(from.height, to.height, move),
      opacity: 1,
      scale: 1,
      change,
    });
  };

  for (const [id, box] of beforeBoxes) {
    if (afterBoxes.has(id) || linkSources.has(id)) continue;
    if (fadeOut <= 0) continue;
    placements.set(id, {
      box, x: box.x, y: box.y, width: box.width, height: box.height,
      opacity: fadeOut, scale: lerp(1, 0.62, 1 - fadeOut), change: plan.changes.get(id) ?? 'removed',
    });
  }

  for (const [sourceId, targetId] of plan.links) {
    const from = beforeBoxes.get(sourceId);
    const to = afterBoxes.get(targetId);
    if (!from || !to) continue;
    glide(targetId, from, to, plan.changes.get(targetId) ?? 'rewritten');
  }

  for (const [id, box] of afterBoxes) {
    if (placements.has(id)) continue;
    const from = beforeBoxes.get(id);
    const change = plan.changes.get(id) ?? 'kept';

    if (!from) {
      if (fadeIn <= 0) continue;
      placements.set(id, {
        box, x: box.x, y: box.y, width: box.width, height: box.height,
        opacity: fadeIn, scale: lerp(0.72, 1, fadeIn), change,
      });
      continue;
    }

    glide(id, from, box, change);
  }

  return {
    placements,
    width: lerp(before.width, after.width, move),
    height: lerp(before.height, after.height, move),
  };
}

export function driftScore(before: Layout, after: Layout): number {
  const beforeBoxes = new Map(before.boxes.map(box => [box.id, box]));
  let moved = 0;
  let shared = 0;

  for (const box of after.boxes) {
    const from = beforeBoxes.get(box.id);
    if (!from) continue;
    shared++;
    if (Math.hypot(box.x - from.x, box.y - from.y) > DRIFT_DISTANCE) moved++;
  }

  return shared === 0 ? 0 : moved / shared;
}
