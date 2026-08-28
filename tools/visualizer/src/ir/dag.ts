import type { Dag, DagNode, DagValue } from '../protocol.js';

type ElkEngine = { layout(graph: unknown): Promise<unknown> };
type ElkModule = (new () => ElkEngine) & { default?: new () => ElkEngine };

export type Box = {
  id: string;
  opId: number | null;
  kind: 'op' | 'region' | 'arg' | 'output' | 'port' | 'nest' | 'nest-block' | 'nest-func' | 'source' | 'line';
  label: string;
  detail: string;
  line: number | null;
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
const PORT_SIZE = 8;
const ROOT_ID = 'root';
const YIELD_OP = 'yield';

const REGION_OPTIONS: Record<string, string> = {
  'elk.algorithm': 'layered',
  'elk.direction': 'DOWN',
  'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
  'elk.layered.spacing.nodeNodeBetweenLayers': '34',
  'elk.spacing.nodeNode': '16',
  'elk.spacing.edgeNode': '14',
  'elk.layered.crossingMinimization.semiInteractive': 'true',
  'elk.padding': '[top=26,left=14,bottom=14,right=14]',
  'elk.portConstraints': 'FIXED_SIDE',
  'elk.layered.nodePlacement.strategy': 'LINEAR_SEGMENTS',
};

const ROOT_OPTIONS: Record<string, string> = {
  ...REGION_OPTIONS,
  'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
};

type ElkEdge = { id: string; sources: string[]; targets: string[]; label: string };

type ElkPort = {
  id: string;
  width: number;
  height: number;
  x?: number;
  y?: number;
  layoutOptions: Record<string, string>;
};

type ElkNode = {
  id: string;
  width?: number;
  height?: number;
  x?: number;
  y?: number;
  children?: ElkNode[];
  ports?: ElkPort[];
  edges?: ElkEdge[];
  layoutOptions?: Record<string, string>;
};

function widthFor(label: string, min = MIN_NODE_WIDTH): number {
  return Math.max(min, Math.round(label.length * CHAR_WIDTH) + NODE_PADDING);
}

function shortType(type: string): string {
  return type.replace(/^tensor</, '').replace(/>$/, '');
}

export function nodeId(opId: number): string {
  return `op${opId}`;
}

export function argId(valueId: number): string {
  return `arg${valueId}`;
}

export function inPortId(valueId: number): string {
  return `pin${valueId}`;
}

export function outPortId(valueId: number): string {
  return `pout${valueId}`;
}

export function buildLayoutRequest(dag: Dag): { graph: ElkNode; meta: Map<string, Box>; edges: Edge[] } {
  const meta = new Map<string, Box>();
  const producers = new Map<number, number>();
  const valueById = new Map<number, DagValue>();

  for (const value of [...dag.args, ...dag.values]) {
    valueById.set(value.id, value);
    if (value.producer !== null) producers.set(value.id, value.producer);
  }

  const funcArgs = new Set(dag.args.map(arg => arg.id));
  const typeOf = (valueId: number): string => valueById.get(valueId)?.type ?? '';
  const nameOf = (valueId: number): string => valueById.get(valueId)?.name ?? '';

  const inPorts = new Set<number>();
  const outPorts = new Set<number>();
  const portsOf = new Map<string, ElkPort[]>();

  const addPort = (owner: string, portId: string, valueId: number, side: string, depth: number): void => {
    const port: ElkPort = { id: portId, width: PORT_SIZE, height: PORT_SIZE, layoutOptions: { 'elk.port.side': side } };
    const existing = portsOf.get(owner);
    if (existing) existing.push(port);
    else portsOf.set(owner, [port]);
    meta.set(portId, {
      id: portId, opId: null, kind: 'port', label: nameOf(valueId), detail: typeOf(valueId), line: null,
      x: 0, y: 0, width: PORT_SIZE, height: PORT_SIZE, depth,
    });
  };

  const registerPorts = (nodes: readonly DagNode[], depth: number): void => {
    for (const node of nodes) {
      if (node.regions.some(region => region.length > 0)) {
        const owner = nodeId(node.id);
        for (const valueId of node.regionArgs.flat()) {
          inPorts.add(valueId);
          addPort(owner, inPortId(valueId), valueId, 'NORTH', depth + 1);
        }
        for (const valueId of node.results) {
          outPorts.add(valueId);
          addPort(owner, outPortId(valueId), valueId, 'SOUTH', depth + 1);
        }
      }
      for (const region of node.regions) registerPorts(region, depth + 1);
    }
  };

  registerPorts(dag.nodes, 0);

  const sourceOf = (valueId: number): string | null => {
    if (inPorts.has(valueId)) return inPortId(valueId);
    if (outPorts.has(valueId)) return outPortId(valueId);
    const producer = producers.get(valueId);
    if (producer !== undefined) return nodeId(producer);
    return funcArgs.has(valueId) ? argId(valueId) : null;
  };

  const byOwner = new Map<string, ElkEdge[]>();
  const seenEdges = new Set<string>();
  const edges: Edge[] = [];

  const addEdge = (owner: string, from: string, to: string, label: string): void => {
    const id = `${from}->${to}`;
    if (seenEdges.has(id)) return;
    seenEdges.add(id);
    const existing = byOwner.get(owner);
    const elkEdge: ElkEdge = { id, sources: [from], targets: [to], label };
    if (existing) existing.push(elkEdge);
    else byOwner.set(owner, [elkEdge]);
    edges.push({ id, from, to, label });
  };

  const argNodes: ElkNode[] = dag.args.map(arg => {
    const label = `${arg.name}: ${shortType(arg.type)}`;
    meta.set(argId(arg.id), {
      id: argId(arg.id), opId: null, kind: 'arg', label, detail: arg.type, line: null,
      x: 0, y: 0, width: widthFor(label, 60), height: PILL_HEIGHT, depth: 0,
    });
    return { id: argId(arg.id), width: widthFor(label, 60), height: PILL_HEIGHT };
  });

  const convert = (node: DagNode, depth: number, container: string): ElkNode => {
    const id = nodeId(node.id);
    const hasRegions = node.regions.some(region => region.length > 0);
    const detail = node.resultTypes.map(shortType).join(', ');
    const label = node.opName;
    const line = node.lines.length > 0 ? node.lines[0] : null;
    const boxWidth = widthFor(label);

    const blockArgs = node.regionArgs.flat();
    const feedsPorts = hasRegions
      && blockArgs.length === node.operands.length
      && blockArgs.every((blockArg, index) => typeOf(blockArg) === typeOf(node.operands[index]));

    node.operands.forEach((operand, index) => {
      const from = sourceOf(operand);
      if (from === null) return;
      addEdge(container, from, feedsPorts ? inPortId(blockArgs[index]) : id, nameOf(operand));
    });

    if (!hasRegions) {
      meta.set(id, {
        id, opId: node.id, kind: 'op', label, detail, line,
        x: 0, y: 0, width: boxWidth, height: NODE_HEIGHT, depth,
      });
      return { id, width: boxWidth, height: NODE_HEIGHT };
    }

    const children: ElkNode[] = [];
    for (const region of node.regions) {
      for (const inner of region) children.push(convert(inner, depth + 1, id));
    }

    const yielded = node.regions.flat().filter(inner => inner.opName === YIELD_OP);
    const yieldOp = yielded.length === 1 ? yielded[0] : null;
    if (yieldOp
      && yieldOp.operands.length === node.results.length
      && yieldOp.operands.every((operand, index) => typeOf(operand) === node.resultTypes[index])) {
      node.results.forEach((result, index) => {
        addEdge(id, nodeId(yieldOp.id), outPortId(result), nameOf(yieldOp.operands[index]));
      });
    }

    meta.set(id, {
      id, opId: node.id, kind: 'region', label, detail, line,
      x: 0, y: 0, width: 0, height: 0, depth,
    });

    return { id, children, ports: portsOf.get(id), edges: byOwner.get(id), layoutOptions: REGION_OPTIONS };
  };

  const opNodes = [...dag.nodes]
    .sort((a, b) => a.id - b.id)
    .map(node => convert(node, 0, ROOT_ID));

  const outputs: ElkNode[] = [];
  dag.returns.forEach((valueId, index) => {
    const value = valueById.get(valueId);
    const label = value ? `return ${value.name}` : 'return';
    const id = `out${index}`;
    meta.set(id, {
      id, opId: null, kind: 'output', label, detail: value ? value.type : '', line: null,
      x: 0, y: 0, width: widthFor(label, 70), height: PILL_HEIGHT, depth: 0,
    });
    outputs.push({ id, width: widthFor(label, 70), height: PILL_HEIGHT });
    const from = sourceOf(valueId);
    if (from !== null) addEdge(ROOT_ID, from, id, '');
  });

  return {
    graph: {
      id: ROOT_ID,
      layoutOptions: ROOT_OPTIONS,
      children: [...argNodes, ...opNodes, ...outputs],
      edges: byOwner.get(ROOT_ID) ?? [],
    },
    meta,
    edges,
  };
}

const cache = new WeakMap<Dag, Promise<Layout>>();
let queue: Promise<unknown> = Promise.resolve();

export function layoutDag(dag: Dag): Promise<Layout> {
  const cached = cache.get(dag);
  if (cached) return cached;

  const settled = () => runLayout(dag);
  const pending = queue.then(settled, settled);
  queue = pending.then(() => undefined, () => undefined);
  cache.set(dag, pending);
  return pending;
}

async function runLayout(dag: Dag): Promise<Layout> {
  const { graph, meta, edges } = buildLayoutRequest(dag);
  const laid = await (await elk()).layout(graph as never) as ElkNode & { width?: number; height?: number };

  const boxes: Box[] = [];
  const place = (node: ElkNode, offsetX: number, offsetY: number): void => {
    const info = meta.get(node.id);
    const x = offsetX + (node.x ?? 0);
    const y = offsetY + (node.y ?? 0);
    if (info) {
      boxes.push({ ...info, x, y, width: node.width ?? info.width, height: node.height ?? info.height });
    }
    for (const port of node.ports ?? []) {
      const portInfo = meta.get(port.id);
      if (portInfo) boxes.push({ ...portInfo, x: x + (port.x ?? 0), y: y + (port.y ?? 0) });
    }
    for (const child of node.children ?? []) place(child, x, y);
  };

  for (const child of laid.children ?? []) place(child, 0, 0);

  return {
    width: laid.width ?? 0,
    height: laid.height ?? 0,
    boxes,
    edges,
  };
}
