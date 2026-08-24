import { useCallback, useRef, useState } from 'react';
import type React from 'react';

export type View = { k: number; tx: number; ty: number };

const RESET_VIEW: View = { k: 1, tx: 0, ty: 0 };
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 6;
const WHEEL_SENSITIVITY = 0.0016;

type Point = { x: number; y: number };
type Anchor = { mid: Point; spread: number };
type Surface = {
  onPointerDown: (event: React.PointerEvent<SVGSVGElement>) => void;
  onPointerMove: (event: React.PointerEvent<SVGSVGElement>) => void;
  onPointerUp: (event: React.PointerEvent<SVGSVGElement>) => void;
  onPointerCancel: (event: React.PointerEvent<SVGSVGElement>) => void;
  onLostPointerCapture: (event: React.PointerEvent<SVGSVGElement>) => void;
};

export type PanZoom = {
  view: View;
  panning: boolean;
  reset: () => void;
  ref: (node: SVGSVGElement | null) => void;
  surface: Surface;
};

function userSpace(svg: SVGSVGElement, clientX: number, clientY: number): Point {
  const matrix = svg.getScreenCTM();
  if (!matrix) return { x: clientX, y: clientY };
  const point = svg.createSVGPoint();
  point.x = clientX;
  point.y = clientY;
  const mapped = point.matrixTransform(matrix.inverse());
  return { x: mapped.x, y: mapped.y };
}

function anchorOf(points: Map<number, Point>): Anchor | null {
  const list = [...points.values()];
  if (list.length === 0) return null;
  const sum = list.reduce((total, p) => ({ x: total.x + p.x, y: total.y + p.y }), { x: 0, y: 0 });
  return {
    mid: { x: sum.x / list.length, y: sum.y / list.length },
    spread: list.length > 1 ? Math.hypot(list[0].x - list[1].x, list[0].y - list[1].y) : 0,
  };
}

function zoomed(view: View, factor: number, at: Point): View {
  const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, view.k * factor));
  const ratio = k / view.k;
  return { k, tx: at.x - (at.x - view.tx) * ratio, ty: at.y - (at.y - view.ty) * ratio };
}

export function usePanZoom(): PanZoom {
  const [view, setView] = useState<View>(RESET_VIEW);
  const [panning, setPanning] = useState(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const points = useRef(new Map<number, Point>());
  const anchor = useRef<Anchor | null>(null);

  const onWheel = useCallback((event: WheelEvent) => {
    const svg = svgRef.current;
    if (!svg) return;
    event.preventDefault();
    const at = userSpace(svg, event.clientX, event.clientY);
    setView(current => zoomed(current, Math.exp(-event.deltaY * WHEEL_SENSITIVITY), at));
  }, []);

  const onTouchMove = useCallback((event: TouchEvent) => {
    if (points.current.size > 0) event.preventDefault();
  }, []);

  const ref = useCallback((node: SVGSVGElement | null) => {
    const previous = svgRef.current;
    if (previous) {
      previous.removeEventListener('wheel', onWheel);
      previous.removeEventListener('touchmove', onTouchMove);
    }
    svgRef.current = node;
    if (node) {
      node.addEventListener('wheel', onWheel, { passive: false });
      node.addEventListener('touchmove', onTouchMove, { passive: false });
    }
  }, [onWheel, onTouchMove]);

  const track = (event: React.PointerEvent<SVGSVGElement>): void => {
    const svg = svgRef.current;
    if (!svg || event.button !== 0) return;
    svg.setPointerCapture(event.pointerId);
    points.current.set(event.pointerId, userSpace(svg, event.clientX, event.clientY));
    anchor.current = anchorOf(points.current);
    setPanning(true);
  };

  const drag = (event: React.PointerEvent<SVGSVGElement>): void => {
    const svg = svgRef.current;
    const from = anchor.current;
    if (!svg || !from || !points.current.has(event.pointerId)) return;
    points.current.set(event.pointerId, userSpace(svg, event.clientX, event.clientY));
    const to = anchorOf(points.current);
    if (!to) return;
    anchor.current = to;
    setView(current => {
      const pinched = from.spread > 0 && to.spread > 0
        ? zoomed(current, to.spread / from.spread, to.mid)
        : current;
      return { ...pinched, tx: pinched.tx + to.mid.x - from.mid.x, ty: pinched.ty + to.mid.y - from.mid.y };
    });
  };

  const release = (event: React.PointerEvent<SVGSVGElement>): void => {
    if (!points.current.delete(event.pointerId)) return;
    anchor.current = anchorOf(points.current);
    if (points.current.size === 0) setPanning(false);
  };

  return {
    view,
    panning,
    reset: () => setView(RESET_VIEW),
    ref,
    surface: {
      onPointerDown: track,
      onPointerMove: drag,
      onPointerUp: release,
      onPointerCancel: release,
      onLostPointerCapture: release,
    },
  };
}
