import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent, ReactElement } from 'react';
import { layoutDag } from '../ir/dag.js';
import { layoutNest } from '../ir/nest.js';
import { driftScore, frameAt, linkLowering, linkRewrites, linkSourceLines, planTransition } from '../ir/transition.js';
import { actions, useStore } from '../store.js';
import { opLabel } from '../catalog/naming.js';
import { usePanZoom } from './pan_zoom.js';
import { useElementSize } from './use_element_size.js';
import type { Box, Layout } from '../ir/dag.js';
import type { Change, Frame, Placement, Plan } from '../ir/transition.js';
import type { CompileStep, Dag, NestNode, Snapshot } from '../protocol.js';

const EMPTY_LAYOUT: Layout = { width: 0, height: 0, boxes: [], edges: [] };

type Side = { dag: Dag | null; nest: NestNode | null };

function sideOf(snapshot: Snapshot, index: number): Side {
  return { dag: snapshot.dags[index] ?? null, nest: snapshot.nests[index] ?? snapshot.nests[0] ?? null };
}

async function layoutSide(side: Side): Promise<Layout> {
  if (side.dag) return layoutDag(side.dag);
  if (side.nest) return layoutNest(side.nest);
  return EMPTY_LAYOUT;
}

function linksBetween(from: Side, to: Side): { links: Map<string, string>; change: Change } {
  if (from.dag && to.dag) return { links: linkRewrites(from.dag, to.dag), change: 'rewritten' };
  if (from.dag && to.nest) return { links: linkLowering(from.dag, to.nest), change: 'lowered' };
  if (from.nest && to.nest && to.nest.kind === 'source') {
    return { links: linkSourceLines(from.nest, to.nest), change: 'emitted' };
  }
  return { links: new Map(), change: 'rewritten' };
}

const BASE_DURATION_MS = 900;
const DRIFT_LIMIT = 0.55;
const PADDING = 14;
const STALL_GRACE_MS = 1200;
const TALL_RATIO = 1.35;
const ZOOM_STEP = 1.4;
const DEFAULT_VIEWPORT = { width: 800, height: 600 };

type Prepared = { key: string; before: Layout; after: Layout; plan: Plan; drift: number };

const OP_KINDS = new Set(['op', 'region', 'output']);

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? '' : 's'}`;
}

function shapeOf(layout: Layout, fallback: number): string {
  let ops = 0;
  let inputs = 0;
  for (const box of layout.boxes) {
    if (OP_KINDS.has(box.kind)) ops++;
    else if (box.kind === 'arg') inputs++;
  }
  if (ops === 0 && inputs === 0) return plural(fallback, 'box');
  return [plural(ops, 'op'), inputs > 0 ? plural(inputs, 'input') : ''].filter(Boolean).join(' · ');
}

export function GraphView({ step }: { step: CompileStep }) {
  const speed = useStore(s => s.speed);
  const [dagIndex, setDagIndex] = useState(0);
  const [prepared, setPrepared] = useState<Prepared | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [replayToken, setReplayToken] = useState(0);
  const [failure, setFailure] = useState<string | null>(null);
  const [wholeGraph, setWholeGraph] = useState(false);
  const { size: viewport, ref: measureSvg } = useElementSize<SVGSVGElement>(DEFAULT_VIEWPORT);
  const raf = useRef(0);
  const played = useRef<string | null>(null);
  const { view, panning, reset, zoomBy, dragged, ref: attachSvg, surface } = usePanZoom();

  const dags = step.after.dags.length > 0 ? step.after.dags : step.before.dags;
  const index = Math.min(dagIndex, Math.max(dags.length - 1, 0));
  const before = sideOf(step.before, index);
  const after = sideOf(step.after, index);
  const empty = !before.dag && !before.nest && !after.dag && !after.nest;

  const measure = useCallback((node: SVGSVGElement | null) => {
    attachSvg(node);
    measureSvg(node);
  }, [attachSvg, measureSvg]);

  useEffect(() => {
    if (empty) { setPrepared(null); return; }
    let cancelled = false;
    setFailure(null);

    void (async () => {
      const b = await layoutSide(before);
      const a = await layoutSide(after);
      if (cancelled) return;
      const { links, change } = linksBetween(before, after);
      setPrepared({
        key: `${step.index}:${index}`,
        before: b,
        after: a,
        plan: planTransition(b, a, links, change),
        drift: driftScore(b, a),
      });
    })().catch((error: unknown) => {
      if (!cancelled) setFailure(error instanceof Error ? error.message : String(error));
    });

    return () => { cancelled = true; };
  }, [step, index, empty]);

  useEffect(() => {
    if (!prepared) return;
    const { before: from, after: to, plan, drift } = prepared;
    const key = `${prepared.key}:${replayToken}`;
    const stepChanged = played.current !== null && played.current !== key;
    played.current = key;
    const instant = speed === 0 || drift > DRIFT_LIMIT || !stepChanged;

    if (instant) {
      setFrame(frameAt(from, to, plan, 1));
      return;
    }

    const duration = BASE_DURATION_MS / speed;
    const started = performance.now();
    setFrame(frameAt(from, to, plan, 0));

    const guard = setTimeout(() => setFrame(frameAt(from, to, plan, 1)), duration + STALL_GRACE_MS);

    const tick = (now: number) => {
      const t = Math.min(1, (now - started) / duration);
      setFrame(frameAt(from, to, plan, t));
      if (t < 1) raf.current = requestAnimationFrame(tick);
      else clearTimeout(guard);
    };

    raf.current = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf.current);
      clearTimeout(guard);
    };
  }, [prepared, speed, replayToken]);

  const geometry = useMemo(() => {
    if (!prepared) return { viewBox: '0 0 100 100', cropped: false };
    const width = Math.max(prepared.before.width, prepared.after.width) + PADDING * 2;
    const height = Math.max(prepared.before.height, prepared.after.height) + PADDING * 2;
    const windowHeight = width * (viewport.height / viewport.width);
    const cropped = !wholeGraph && height > windowHeight * TALL_RATIO;
    const shown = cropped ? windowHeight : height;
    return { viewBox: `${-PADDING} ${-PADDING} ${width} ${shown}`, cropped };
  }, [prepared, viewport, wholeGraph]);

  if (empty) {
    return <div className="pane-empty">This step has no structure to draw — the IR tab shows what it changed.</div>;
  }

  if (failure) return <div className="pane-empty">graph layout failed: {failure}</div>;
  if (!prepared || !frame) return <div className="pane-empty">laying out the graph…</div>;

  const edges = collectEdges(prepared, frame);
  const zoom = Math.round(view.k * 100);
  const linked = prepared.after.boxes.some(box => box.line !== null);

  const ordered = [...frame.placements.values()]
    .sort((a, b) => a.box.depth - b.box.depth || rank(a.box.kind) - rank(b.box.kind));
  const frames = ordered.filter(placement => CONTAINER_KINDS.has(placement.box.kind));
  const leaves = ordered.filter(placement => !CONTAINER_KINDS.has(placement.box.kind));

  const drawNode = (placement: Placement): ReactElement => {
    const line = placement.box.line;
    const hover = (event: ReactPointerEvent<SVGGElement>, shown: number | null): void => {
      if (line !== null && event.pointerType === 'mouse') actions.focusSource(shown);
    };
    return (
      <g
        key={placement.box.id}
        className={[
          'node',
          placement.box.kind,
          placement.change,
          line === null ? '' : 'traceable',
        ].filter(Boolean).join(' ')}
        style={{ opacity: placement.opacity }}
        transform={transformFor(placement)}
        onPointerEnter={event => hover(event, line)}
        onPointerLeave={event => hover(event, null)}
        onClick={() => { if (line !== null && !dragged()) actions.focusSource(line); }}
      >
        <rect
          width={placement.width}
          height={placement.height}
          rx={placement.box.kind === 'op' ? 6 : 9}
        />
        {showsLabel(placement) && <text x={10} y={headerY(placement)}>{placement.box.label}</text>}
        {showsDetail(placement) && (
          <text className="detail" x={placement.width - 10} y={headerY(placement)} textAnchor="end">
            {placement.box.detail}
          </text>
        )}
        <title>{tooltipFor(placement.box, line)}</title>
      </g>
    );
  };

  return (
    <div className="graph">
      <div className="graph-bar">
        <Legend plan={prepared.plan} layout={prepared.after} isInput={step.kind === 'input'} />
        {dags.length > 1 && (
          <select value={index} aria-label="function" onChange={e => setDagIndex(Number(e.target.value))}>
            {dags.map((dag, i) => <option key={dag.func} value={i}>{dag.func}</option>)}
          </select>
        )}
        {prepared.drift > DRIFT_LIMIT && <span className="drift">layout moved too far to animate</span>}
        <div className="graph-tools">
          {linked && <span className="graph-hint">point at a box to light up its line</span>}
          <button onClick={() => zoomBy(1 / ZOOM_STEP)} aria-label="zoom out" title="zoom out">−</button>
          <span className="zoom">{zoom}%</span>
          <button onClick={() => zoomBy(ZOOM_STEP)} aria-label="zoom in" title="zoom in">+</button>
          <button
            className={wholeGraph ? 'active' : ''}
            onClick={() => { setWholeGraph(v => !v); reset(); }}
            title={wholeGraph ? 'go back to a readable size' : 'shrink until the whole graph fits'}
          >
            {wholeGraph ? 'readable' : 'whole graph'}
          </button>
          <button onClick={reset} title="back to the starting view">reset</button>
          <button
            onClick={() => setReplayToken(token => token + 1)}
            disabled={step.kind === 'input'}
            title="replay the transition into this step"
          >
            replay
          </button>
        </div>
      </div>

      <svg
        ref={measure}
        className={panning ? 'graph-svg panning' : 'graph-svg'}
        viewBox={geometry.viewBox}
        preserveAspectRatio="xMidYMin meet"
        {...surface}
      >
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="arrowhead" />
          </marker>
        </defs>

        <g transform={`translate(${view.tx} ${view.ty}) scale(${view.k})`}>
          <g className="frames">{frames.map(drawNode)}</g>

          <g className="edges">
            {edges.map(edge => (
              <path key={edge.id} d={edge.d} className="edge" style={{ opacity: edge.opacity }} markerEnd="url(#arrow)" />
            ))}
          </g>

          <g className="nodes">{leaves.map(drawNode)}</g>
        </g>
      </svg>
    </div>
  );
}

function tooltipFor(box: Box, line: number | null): string {
  const parts: string[] = [box.kind === 'op' ? `${box.label} — ${opLabel(box.label)}` : box.label];
  if (box.detail) parts.push(box.detail);
  if (line !== null) parts.push(`from line ${line} of your code`);
  return parts.join('\n');
}

const CONTAINER_KINDS = new Set(['region', 'nest', 'nest-block', 'nest-func', 'source', 'line']);

function rank(kind: string): number {
  if (kind === 'region' || kind === 'nest-func' || kind === 'source') return 0;
  if (CONTAINER_KINDS.has(kind)) return 1;
  return 2;
}

const DETAIL_KINDS = new Set(['op', 'output', 'region', 'nest', 'nest-block', 'nest-func', 'source']);
const LABEL_CHAR = 6.7;
const DETAIL_CHAR = 6.1;
const TEXT_GAP = 18;
const MIN_LABEL_HEIGHT = 16;

function showsLabel(placement: { box: Box; height: number }): boolean {
  return placement.box.label !== '' && placement.height >= MIN_LABEL_HEIGHT;
}

function showsDetail(placement: { box: Box; width: number }): boolean {
  const { box } = placement;
  if (!box.detail || !DETAIL_KINDS.has(box.kind)) return false;
  const needed = box.label.length * LABEL_CHAR + box.detail.length * DETAIL_CHAR + TEXT_GAP + 20;
  return needed <= placement.width;
}

function headerY(placement: { box: Box; height: number }): number {
  const stacked = placement.box.kind === 'region' || CONTAINER_KINDS.has(placement.box.kind);
  return stacked && placement.height > 30 ? 15 : placement.height / 2 + 4;
}

function transformFor(placement: { x: number; y: number; width: number; height: number; scale: number }): string {
  if (placement.scale === 1) return `translate(${placement.x} ${placement.y})`;
  const cx = placement.width / 2;
  const cy = placement.height / 2;
  return `translate(${placement.x} ${placement.y}) translate(${cx} ${cy}) scale(${placement.scale}) translate(${-cx} ${-cy})`;
}

type DrawnEdge = { id: string; d: string; opacity: number };

function collectEdges(prepared: Prepared, frame: Frame): DrawnEdge[] {
  const seen = new Set<string>();
  const drawn: DrawnEdge[] = [];

  for (const edge of [...prepared.after.edges, ...prepared.before.edges]) {
    if (seen.has(edge.id)) continue;
    seen.add(edge.id);

    const from = frame.placements.get(edge.from);
    const to = frame.placements.get(edge.to);
    if (!from || !to) continue;

    const x1 = from.x + from.width / 2;
    const y1 = from.y + from.height;
    const x2 = to.x + to.width / 2;
    const y2 = to.y;
    const mid = (y1 + y2) / 2;

    drawn.push({
      id: edge.id,
      d: `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2 - 2}`,
      opacity: Math.min(from.opacity, to.opacity) * 0.75,
    });
  }

  return drawn;
}

const LEGEND: { change: Change; label: string }[] = [
  { change: 'kept', label: 'kept' },
  { change: 'added', label: 'added' },
  { change: 'removed', label: 'removed' },
  { change: 'rewritten', label: 'rewritten' },
  { change: 'lowered', label: 'linked to its loops' },
  { change: 'emitted', label: 'became code' },
];

function quiet(plan: Plan): boolean {
  return plan.counts.added === 0 && plan.counts.removed === 0 && plan.counts.rewritten === 0
    && plan.counts.lowered === 0 && plan.counts.emitted === 0 && plan.links.size === 0;
}

function Legend({ plan, layout, isInput }: { plan: Plan; layout: Layout; isInput: boolean }) {
  if (quiet(plan)) {
    return (
      <div className="legend">
        <span className="chip quiet">{shapeOf(layout, plan.counts.kept)}</span>
        <span className="chip quiet">{isInput ? 'the starting point' : 'this step drew the same shape'}</span>
      </div>
    );
  }

  return (
    <div className="legend">
      {LEGEND.map(entry => (
        plan.counts[entry.change] > 0 && (
          <span key={entry.change} className={`chip ${entry.change}`}>
            {plan.counts[entry.change]} {entry.label}
          </span>
        )
      ))}
    </div>
  );
}
