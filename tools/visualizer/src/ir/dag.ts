import type { Dag, DagNode, DagValue } from '../protocol.js';

type ElkEngine = { layout(graph: unknown): Promise<unknown> };
type ElkModule = (new () => ElkEngine) & { default?: new () => ElkEngine };

export type Box = {
  id: string;
  opId: number | null;
  kind: 'op' | 'region' | 'arg' | 'output' | 'nest' | 'nest-block' | 'nest-func' | 'source' | 'line';
  label: string;
  detail: string;
  note: string;
  x: number;
  y: number;
  width: number;
  height: number;
  depth: number;
};

export type Edge = {
  id: string;
  from: string;
  to: string;
  label: string;
};

export type Layout = {
  width: number;
  height: number;
  boxes: Box[];
  edges: Edge[];
};

let engine: Promise<ElkEngine> | null = null;

function elk(): Promise<ElkEngine> {
  if (!engine) {
    engine = import('elkjs/lib/elk.bundled.js').then(module => {
      const Elk = (module.default ?? module) as unknown as ElkModule;
      return new (Elk.default ?? Elk)();
    });
  }
  return engine;
}

const NODE_HEIGHT = 30;
const PILL_HEIGHT = 22;
const CHAR_WIDTH = 7.1;
const NODE_PADDING = 26;
const MIN_NODE_WIDTH = 74;

const LAYOUT_OPTIONS: Record<string, string> = {
  'elk.algorithm': 'layered',
  'elk.direction': 'DOWN',
  'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
  'elk.layered.spacing.nodeNodeBetweenLayers': '34',
  'elk.spacing.nodeNode': '16',
  'elk.spacing.edgeNode': '14',
  'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
  'elk.layered.crossingMinimization.semiInteractive': 'true',
  'elk.padding': '[top=26,left=14,bottom=14,right=14]',
};

type ElkNode = {
  id: string;
  width?: number;
  height?: number;
  x?: number;
  y?: number;
  children?: ElkNode[];
  edges?: unknown[];
  layoutOptions?: Record<string, string>;
};

function widthFor(label: string, min = MIN_NODE_WIDTH): number {
  return Math.max(min, Math.round(label.length * CHAR_WIDTH) + NODE_PADDING);
}

export type NoteLookup = (opId: number) => string;

const NO_NOTE: NoteLookup = () => '';

function shortType(type: string): string {
  return type.replace(/^tensor</, '').replace(/>$/, '');
}

export function nodeId(opId: number): string {
  return `op${opId}`;
}

export function argId(valueId: number): string {
  return `arg${valueId}`;
}

export function buildLayoutRequest(dag: Dag, note: NoteLookup = NO_NOTE): { graph: ElkNode; meta: Map<string, Box> } {
  const meta = new Map<string, Box>();
  const producers = new Map<number, number>();
  const valueById = new Map<number, DagValue>();

  for (const value of [...dag.args, ...dag.values]) {
    valueById.set(value.id, value);
    if (value.producer !== null) producers.set(value.id, value.producer);
  }

  const argNodes: ElkNode[] = dag.args.map(arg => {
    const label = `${arg.name}: ${shortType(arg.type)}`;
    meta.set(argId(arg.id), {
      id: argId(arg.id), opId: null, kind: 'arg', label, detail: arg.type, note: '',
      x: 0, y: 0, width: widthFor(label, 60), height: PILL_HEIGHT, depth: 0,
    });
    return { id: argId(arg.id), width: widthFor(label, 60), height: PILL_HEIGHT };
  });

  const edges: { id: string; sources: string[]; targets: string[]; label: string }[] = [];
  const seenEdges = new Set<string>();

  const addEdge = (from: string, to: string, label: string): void => {
    const id = `${from}->${to}`;
    if (seenEdges.has(id)) return;
    seenEdges.add(id);
    edges.push({ id, sources: [from], targets: [to], label });
  };

  const convert = (node: DagNode, depth: number, enclosing: string | null): ElkNode => {
    const id = nodeId(node.id);
    const hasRegions = node.regions.some(region => region.length > 0);
    const detail = node.resultTypes.map(shortType).join(', ');
    const label = node.opName;

    for (const operand of node.operands) {
      const producer = producers.get(operand);
      const value = valueById.get(operand);
      const edgeLabel = value ? value.name : '';
      if (producer !== undefined) addEdge(nodeId(producer), id, edgeLabel);
      else if (valueById.has(operand) && dag.args.some(a => a.id === operand)) addEdge(argId(operand), id, edgeLabel);
      else if (enclosing) addEdge(enclosing, id, edgeLabel);
    }

    const badge = note(node.id);
    const boxWidth = widthFor(badge ? `${label}  ${badge}` : label);

    if (!hasRegions) {
      meta.set(id, {
        id, opId: node.id, kind: 'op', label, detail, note: badge,
        x: 0, y: 0, width: boxWidth, height: NODE_HEIGHT, depth,
      });
      return { id, width: boxWidth, height: NODE_HEIGHT };
    }

    const children: ElkNode[] = [];
    for (const region of node.regions) {
      for (const inner of region) children.push(convert(inner, depth + 1, id));
    }

    meta.set(id, {
      id, opId: node.id, kind: 'region', label, detail, note: badge,
      x: 0, y: 0, width: 0, height: 0, depth,
    });

    return { id, children, layoutOptions: LAYOUT_OPTIONS };
  };

  const opNodes = [...dag.nodes]
    .sort((a, b) => a.id - b.id)
    .map(node => convert(node, 0, null));

  const outputs: ElkNode[] = [];
  dag.returns.forEach((valueId, index) => {
    const value = valueById.get(valueId);
    const label = value ? `return ${value.name}` : 'return';
    const id = `out${index}`;
    meta.set(id, {
      id, opId: null, kind: 'output', label, detail: value ? value.type : '', note: '',
      x: 0, y: 0, width: widthFor(label, 70), height: PILL_HEIGHT, depth: 0,
    });
    outputs.push({ id, width: widthFor(label, 70), height: PILL_HEIGHT });
    const producer = producers.get(valueId);
    if (producer !== undefined) addEdge(nodeId(producer), id, '');
  });

  return {
    graph: {
      id: 'root',
      layoutOptions: LAYOUT_OPTIONS,
      children: [...argNodes, ...opNodes, ...outputs],
      edges,
    },
    meta,
  };
}

const cache = new WeakMap<Dag, Promise<Layout>>();
let queue: Promise<unknown> = Promise.resolve();

export function layoutDag(dag: Dag, note: NoteLookup = NO_NOTE): Promise<Layout> {
  const cached = cache.get(dag);
  if (cached) return cached;

  const settled = () => runLayout(dag, note);
  const pending = queue.then(settled, settled);
  queue = pending.then(() => undefined, () => undefined);
  cache.set(dag, pending);
  return pending;
}

async function runLayout(dag: Dag, note: NoteLookup): Promise<Layout> {
  const { graph, meta } = buildLayoutRequest(dag, note);
  const laid = await (await elk()).layout(graph as never) as ElkNode & { width?: number; height?: number };

  const boxes: Box[] = [];
  const place = (node: ElkNode, offsetX: number, offsetY: number): void => {
    const info = meta.get(node.id);
    const x = offsetX + (node.x ?? 0);
    const y = offsetY + (node.y ?? 0);
    if (info) {
      boxes.push({ ...info, x, y, width: node.width ?? info.width, height: node.height ?? info.height });
    }
    for (const child of node.children ?? []) place(child, x, y);
  };

  for (const child of laid.children ?? []) place(child, 0, 0);

  const edges: Edge[] = (graph.edges as { id: string; sources: string[]; targets: string[]; label: string }[])
    .map(edge => ({ id: edge.id, from: edge.sources[0], to: edge.targets[0], label: edge.label }));

  return {
    width: laid.width ?? 0,
    height: laid.height ?? 0,
    boxes,
    edges,
  };
}
