import { renderAxes } from './axis.js';
import { renderLegend } from './legend.js';
import { createZoomInteraction } from './interaction.js';
import { colorAt } from './palette.js';
import { getRenderer, registerRenderer } from './registry.js';
import { createScale } from './scales.js';
import { isChartSpec } from './spec.js';
import { svgElement, svgText } from './svg.js';
import { createTooltip } from './tooltip.js';
import { renderBar, renderHistogram } from './renderers/bar.js';
import { renderLine } from './renderers/line.js';
import { renderScatter } from './renderers/scatter.js';
import { renderArea } from './renderers/area.js';
import { renderHexbin } from './renderers/hexbin.js';
import { renderRegression } from './renderers/regression.js';
import { renderPayloadChart } from './payload_renderers.js';
import { domainsEqual } from './zoom.js';

registerRenderer('line', renderLine);
registerRenderer('bar', renderBar);
registerRenderer('scatter', renderScatter);
registerRenderer('histogram', renderHistogram);
registerRenderer('area', renderArea);
registerRenderer('density', renderLine);
registerRenderer('ecdf', renderLine);
registerRenderer('hexbin', renderHexbin);
registerRenderer('regression', renderRegression);

export function renderChart(host, spec) {
  if (!isChartSpec(spec)) throw new Error('renderChart expects a ChartSpec');
  if (spec.payload != null) return renderPayloadChart(host, spec);
  const hidden = new Set();
  const initialSeries = layoutSeries(spec.series, spec);
  const allSpecPoints = initialSeries.flatMap(series => series.points);
  const xValues = allSpecPoints.map(point => point.x);
  const yValues = allSpecPoints.flatMap(point => [point.y, point.y0, point.y1].filter(value => value != null));
  const baseX = createScale(xValues, 0, 1, { padding: spec.type === 'bar' ? 0 : 0.03 });
  const baseY = createScale(yValues, 1, 0, { zero: ['bar', 'histogram', 'area', 'density'].includes(spec.type), padding: 0.08 });
  const zoomEnabled = spec.options.zoom && ['line', 'scatter', 'histogram', 'area', 'density', 'hexbin', 'regression', 'ecdf'].includes(spec.type) && baseX.type === 'linear' && baseY.type === 'linear';
  const bounds = zoomEnabled ? { x: baseX.domain, y: baseY.domain } : null;
  let domains = zoomEnabled ? { x: [...bounds.x], y: [...bounds.y] } : null;
  let interactionContext = null;
  let observer = null;
  let tooltip = null;
  let frame = 0;

  const drawSoon = () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(draw);
  };

  const draw = () => {
    tooltip?.remove();
    host.innerHTML = '';
    host.className = 'chart-view';
    host.classList.toggle('chart-zoom-enabled', zoomEnabled);
    const width = spec.options.width ?? Math.max(320, host.clientWidth || 720);
    const height = spec.options.height;
    const layout = { width, height, left: 58, right: width - 20, top: spec.options.title ? 42 : 20, bottom: height - 48 };
    const selected = spec.series.map((series, index) => ({ ...series, index, color: colorAt(index) })).filter(series => !hidden.has(series.index));
    const visible = layoutSeries(selected, spec);
    const svg = svgElement('svg', { class: 'chart-svg', viewBox: `0 0 ${width} ${height}`, role: 'img' });
    if (spec.options.title) svg.append(svgText(spec.options.title, { class: 'chart-title', x: layout.left, y: 24 }));
    const x = createScale(xValues, layout.left, layout.right, { padding: spec.type === 'bar' ? 0 : 0.03, domain: domains?.x });
    const y = createScale(yValues, layout.bottom, layout.top, { zero: ['bar', 'histogram', 'area', 'density'].includes(spec.type), padding: 0.08, domain: domains?.y });
    renderAxes(svg, layout, x, y, { x: spec.options.xLabel, y: spec.options.yLabel });
    if (spec.options.width != null) svg.style.width = `${width}px`;
    tooltip = createTooltip(host);
    const clipId = `chart-clip-${nextChartId++}`;
    const defs = svgElement('defs');
    const clip = svgElement('clipPath', { id: clipId });
    clip.append(svgElement('rect', { x: layout.left, y: layout.top, width: layout.right - layout.left, height: layout.bottom - layout.top }));
    defs.append(clip);
    svg.append(defs);
    const marks = svgElement('g', { class: 'chart-marks', 'clip-path': `url(#${clipId})` });
    const renderer = getRenderer(spec.type);
    const visibleSeriesCount = Math.max(1, visible.length);
    const maxSeriesPoints = Math.max(1, ...visible.map(series => series.points.length));
    visible.forEach((series, visibleIndex) => {
      const group = svgElement('g', { class: `chart-series chart-series-${series.index}` });
      const groupWidth = x.type === 'category' ? x.step * 0.78 : Math.max(4, (layout.right - layout.left) / maxSeriesPoints) * 0.78;
      const stacked = spec.options.mode === 'stacked';
      const seriesOffset = visibleSeriesCount === 1 || stacked ? 0 : (visibleIndex - (visibleSeriesCount - 1) / 2) * (groupWidth / visibleSeriesCount);
      renderer(group, series, { x, y, layout, tooltip, visibleSeriesCount, maxSeriesPoints, seriesOffset, stacked });
      marks.append(group);
    });
    svg.append(marks);
    host.append(svg);
    interactionContext = {
      enabled: zoomEnabled,
      bounds,
      domains,
      layout,
      svg,
      x,
      y,
    };
    if (zoomEnabled) renderZoomControls(host, isZoomed(), resetZoom);
    if (spec.options.legend && spec.series.length > 1) {
      renderLegend(host, spec.series.map((series, index) => ({ ...series, color: colorAt(index) })), hidden, drawSoon);
    }
  };

  const changeZoom = next => {
    domains = next;
    drawSoon();
  };
  const resetZoom = () => {
    domains = zoomEnabled ? { x: [...bounds.x], y: [...bounds.y] } : null;
    drawSoon();
  };
  const interactionCleanup = createZoomInteraction(host, () => interactionContext, changeZoom, resetZoom);
  draw();
  if (typeof ResizeObserver !== 'undefined' && spec.options.width == null) {
    observer = new ResizeObserver(drawSoon);
    observer.observe(host);
  }
  return () => {
    cancelAnimationFrame(frame);
    observer?.disconnect();
    interactionCleanup();
    tooltip?.remove();
  };

  function isZoomed() {
    return zoomEnabled && (!domainsEqual(domains.x, bounds.x) || !domainsEqual(domains.y, bounds.y));
  }
}

export function layoutSeries(series, spec) {
  if (spec.options.mode !== 'stacked') return series;
  const positive = new Map();
  const negative = new Map();
  return series.map(item => ({
    ...item,
    points: item.points.map(point => {
      const key = String(point.x);
      const map = point.y >= 0 ? positive : negative;
      const y0 = map.get(key) ?? 0;
      const y1 = y0 + point.y;
      map.set(key, y1);
      return { ...point, y0, y1 };
    }),
  }));
}

let nextChartId = 1;

function renderZoomControls(host, zoomed, reset) {
  const controls = document.createElement('div');
  controls.className = 'chart-zoom-controls';
  const hint = document.createElement('span');
  hint.textContent = 'Wheel zoom · drag pan';
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = 'Reset zoom';
  button.disabled = !zoomed;
  button.addEventListener('click', reset);
  controls.append(hint, button);
  host.append(controls);
}
