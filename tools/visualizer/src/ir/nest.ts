import type { NestNode } from '../protocol.js';
import type { Box, Layout } from './dag.js';

const BOX_KIND: Partial<Record<NestNode['kind'], Box['kind']>> = {
  block: 'nest-block',
  func: 'nest-func',
  source: 'source',
  line: 'line',
};

const HEADER = 22;
const PAD_X = 10;
const PAD_BOTTOM = 10;
const GAP = 5;
const LEAF_HEIGHT = 24;
const CHAR_WIDTH = 6.6;
const MIN_WIDTH = 130;
const MAX_WIDTH = 620;

function labelWidth(node: NestNode): number {
  const text = node.detail ? `${node.label}   ${node.detail}` : node.label;
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, Math.round(text.length * CHAR_WIDTH) + PAD_X * 2));
}

const measured = new WeakMap<NestNode, { width: number; height: number }>();

function measure(node: NestNode): { width: number; height: number } {
  const cached = measured.get(node);
  if (cached) return cached;

  const size = computeSize(node);
  measured.set(node, size);
  return size;
}

function computeSize(node: NestNode): { width: number; height: number } {
  if (node.children.length === 0) {
    return { width: labelWidth(node), height: LEAF_HEIGHT };
  }

  let inner = 0;
  let widest = 0;
  for (const child of node.children) {
    const size = measure(child);
    inner += size.height + GAP;
    widest = Math.max(widest, size.width);
  }

  return {
    width: Math.max(labelWidth(node), widest + PAD_X * 2),
    height: HEADER + inner - GAP + PAD_BOTTOM,
  };
}

function place(node: NestNode, x: number, y: number, width: number, depth: number, boxes: Box[]): number {
  const size = measure(node);
  const height = size.height;

  boxes.push({
    id: node.id,
    opId: null,
    kind: BOX_KIND[node.kind] ?? 'nest',
    label: node.label,
    detail: node.detail,
    x,
    y,
    width,
    height,
    depth,
  });

  let cursor = y + HEADER;
  for (const child of node.children) {
    const childSize = measure(child);
    const childWidth = node.children.length === 1 && child.children.length > 0
      ? width - PAD_X * 2
      : Math.min(childSize.width, width - PAD_X * 2);
    cursor += place(child, x + PAD_X, cursor, Math.max(childWidth, MIN_WIDTH / 2), depth + 1, boxes) + GAP;
  }

  return height;
}

export function layoutNest(root: NestNode): Layout {
  const boxes: Box[] = [];
  const size = measure(root);
  place(root, 0, 0, size.width, 0, boxes);
  return { width: size.width, height: size.height, boxes, edges: [] };
}

export function nestOps(root: NestNode): Map<string, string> {
  const byId = new Map<string, string>();
  const walk = (node: NestNode): void => {
    if (node.op) byId.set(node.id, node.op);
    for (const child of node.children) walk(child);
  };
  walk(root);
  return byId;
}
