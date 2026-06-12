import { panDomain, zoomDomain } from './zoom.js';

export function createZoomInteraction(host, getContext, onChange, onReset) {
  let dragging = false;
  let last = null;

  const wheel = event => {
    const context = activeContext(event);
    if (!context) return;
    event.preventDefault();
    const point = svgPoint(event, context);
    const factor = Math.exp(event.deltaY * 0.0015);
    onChange({
      x: zoomDomain(context.domains.x, context.x.invert(point.x), factor, context.bounds.x),
      y: zoomDomain(context.domains.y, context.y.invert(point.y), factor, context.bounds.y),
    });
  };

  const pointerDown = event => {
    if (event.button !== 0 || !activeContext(event)) return;
    dragging = true;
    last = { x: event.clientX, y: event.clientY };
    host.classList.add('chart-panning');
    host.setPointerCapture?.(event.pointerId);
  };

  const pointerMove = event => {
    if (!dragging || !last) return;
    const context = getContext();
    if (!context?.enabled) return;
    const dx = event.clientX - last.x;
    const dy = event.clientY - last.y;
    last = { x: event.clientX, y: event.clientY };
    const rect = context.svg.getBoundingClientRect();
    const xSpan = context.domains.x[1] - context.domains.x[0];
    const ySpan = context.domains.y[1] - context.domains.y[0];
    onChange({
      x: panDomain(context.domains.x, -dx / rect.width * xSpan, context.bounds.x),
      y: panDomain(context.domains.y, dy / rect.height * ySpan, context.bounds.y),
    });
  };

  const pointerUp = event => {
    dragging = false;
    last = null;
    host.classList.remove('chart-panning');
    host.releasePointerCapture?.(event.pointerId);
  };

  const doubleClick = event => {
    if (!activeContext(event)) return;
    event.preventDefault();
    onReset();
  };

  host.addEventListener('wheel', wheel, { passive: false });
  host.addEventListener('pointerdown', pointerDown);
  host.addEventListener('pointermove', pointerMove);
  host.addEventListener('pointerup', pointerUp);
  host.addEventListener('pointercancel', pointerUp);
  host.addEventListener('dblclick', doubleClick);

  return () => {
    host.removeEventListener('wheel', wheel);
    host.removeEventListener('pointerdown', pointerDown);
    host.removeEventListener('pointermove', pointerMove);
    host.removeEventListener('pointerup', pointerUp);
    host.removeEventListener('pointercancel', pointerUp);
    host.removeEventListener('dblclick', doubleClick);
  };

  function activeContext(event) {
    const context = getContext();
    if (!context?.enabled) return null;
    const point = svgPoint(event, context);
    return point.x >= context.layout.left && point.x <= context.layout.right
      && point.y >= context.layout.top && point.y <= context.layout.bottom
      ? context
      : null;
  }
}

function svgPoint(event, context) {
  const rect = context.svg.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / rect.width * context.layout.width,
    y: (event.clientY - rect.top) / rect.height * context.layout.height,
  };
}
