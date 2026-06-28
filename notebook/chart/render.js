import { renderAxes, renderRightAxis } from './axis.js';
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
import { renderBubble } from './renderers/bubble.js';
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
registerRenderer('bubble', renderBubble);

export function renderChart(host, spec) {
  if (!isChartSpec(spec)) throw new Error('renderChart expects a ChartSpec');
  if (spec.type === 'figure') return renderFigure(host, spec);
  if (spec.payload != null) return renderPayloadChart(host, spec);
  const hidden = new Set();
  const initialSeries = layoutSeries(spec.series, spec);
  const allSpecPoints = initialSeries.flatMap(series => series.points);
  const xValues = allSpecPoints.map(point => point.x);
  const yValues = allSpecPoints.flatMap(point => [point.y, point.y0, point.y1].filter(value => value != null));
  const baseX = createScale(xValues, 0, 1, { padding: spec.type === 'bar' ? 0 : 0.03 });
  const baseY = createScale(yValues, 1, 0, { zero: ['bar', 'histogram', 'area', 'density'].includes(spec.type), padding: 0.08 });
  const zoomEnabled = spec.options.zoom && ['line', 'scatter', 'histogram', 'area', 'density', 'hexbin', 'regression', 'ecdf', 'bubble'].includes(spec.type) && baseX.type === 'linear' && baseY.type === 'linear';
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
    const height = chartHeight(width, spec.options.height);
    const compact = width < 520;
    host.classList.toggle('chart-compact', compact);
    const layout = { width, height, left: compact ? 44 : 58, right: width - 16, top: spec.options.title ? 42 : 20, bottom: height - (compact ? 42 : 48) };
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
      renderer(group, series, { x, y, layout, tooltip, visibleSeriesCount, maxSeriesPoints, seriesOffset, stacked, allPoints: visible.flatMap(item => item.points) });
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

const FIGURE_ZERO_MARKS = new Set(['bar', 'area', 'histogram']);

export function renderFigure(host, spec) {
  if (Array.isArray(spec.panels)) return renderFacet(host, spec);
  const items = flattenFigureItems(spec.layers);
  const hidden = new Set();
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
    const width = spec.options.width ?? Math.max(320, host.clientWidth || 720);
    const height = chartHeight(width, spec.options.height);
    const compact = width < 520;
    host.classList.toggle('chart-compact', compact);
    const visible = items.filter(item => !hidden.has(item.index));
    const hasRight = visible.some(item => item.axis === 'right');
    const layout = {
      width,
      height,
      frameRight: width,
      left: compact ? 44 : 58,
      right: width - (hasRight ? (compact ? 44 : 58) : 16),
      top: spec.options.title ? 42 : 20,
      bottom: height - (compact ? 42 : 48),
    };
    const svg = svgElement('svg', { class: 'chart-svg', viewBox: `0 0 ${width} ${height}`, role: 'img' });
    if (spec.options.title) svg.append(svgText(spec.options.title, { class: 'chart-title', x: layout.left, y: 24 }));
    if (spec.options.width != null) svg.style.width = `${width}px`;
    tooltip = createTooltip(host);
    const scales = figureScales(visible, layout);
    renderAxes(svg, layout, scales.x, scales.y, { x: spec.options.xLabel, y: spec.options.yLabel });
    if (scales.yRight) renderRightAxis(svg, layout, scales.yRight, spec.options.y2Label);
    paintFigurePanel(svg, layout, visible, scales, tooltip);
    host.append(svg);
    if (spec.options.legend && items.length > 1) {
      renderLegend(host, items.map(item => ({ name: item.series.name, color: item.color })), hidden, drawSoon);
    }
  };

  draw();
  if (typeof ResizeObserver !== 'undefined' && spec.options.width == null) {
    observer = new ResizeObserver(drawSoon);
    observer.observe(host);
  }
  return () => {
    cancelAnimationFrame(frame);
    observer?.disconnect();
    tooltip?.remove();
  };
}

function renderFacet(host, spec) {
  const panels = spec.panels.map(panel => ({ label: panel.label, items: flattenFigureItems(panel.layers) }));
  const globalItems = panels.flatMap(panel => panel.items);
  const hidden = new Set();
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
    const width = spec.options.width ?? Math.max(320, host.clientWidth || 720);
    const compact = width < 520;
    host.classList.toggle('chart-compact', compact);
    const minPanelWidth = compact ? 150 : 240;
    const autoCols = Math.max(1, Math.floor(width / minPanelWidth));
    const cols = Math.max(1, Math.min(panels.length, spec.facet.cols ?? autoCols));
    const rows = Math.ceil(panels.length / cols);
    const cellWidth = width / cols;
    const cellHeight = spec.options.height ? spec.options.height / rows : chartHeight(cellWidth, null);
    const headOffset = spec.options.title ? 34 : 6;
    const height = headOffset + rows * cellHeight;
    const svg = svgElement('svg', { class: 'chart-svg', viewBox: `0 0 ${width} ${height}`, role: 'img' });
    if (spec.options.title) svg.append(svgText(spec.options.title, { class: 'chart-title', x: 12, y: 24 }));
    if (spec.options.width != null) svg.style.width = `${width}px`;
    tooltip = createTooltip(host);
    const visibleGlobal = globalItems.filter(item => !hidden.has(item.index));
    panels.forEach((panel, panelIndex) => {
      const col = panelIndex % cols;
      const row = Math.floor(panelIndex / cols);
      const originX = col * cellWidth;
      const originY = headOffset + row * cellHeight;
      const visible = panel.items.filter(item => !hidden.has(item.index));
      const hasRight = visible.some(item => item.axis === 'right');
      const layout = {
        width,
        height,
        frameRight: originX + cellWidth,
        left: originX + (compact ? 40 : 50),
        right: originX + cellWidth - (hasRight ? (compact ? 38 : 48) : 12),
        top: originY + 24,
        bottom: originY + cellHeight - 34,
      };
      const cell = svgElement('g', { class: 'chart-facet-panel' });
      cell.append(svgText(panel.label, { class: 'chart-facet-label', x: layout.left, y: originY + 16 }));
      const scales = figureScales(visibleGlobal, layout);
      renderAxes(cell, layout, scales.x, scales.y, { x: col === 0 ? spec.options.xLabel : null, y: col === 0 ? spec.options.yLabel : null });
      if (scales.yRight) renderRightAxis(cell, layout, scales.yRight, col === cols - 1 ? spec.options.y2Label : null);
      paintFigurePanel(cell, layout, visible, scales, tooltip);
      svg.append(cell);
    });
    host.append(svg);
    const legendItems = panels[0]?.items ?? [];
    if (spec.options.legend && legendItems.length > 1) {
      renderLegend(host, legendItems.map(item => ({ name: item.series.name, color: item.color })), hidden, drawSoon);
    }
  };

  draw();
  if (typeof ResizeObserver !== 'undefined' && spec.options.width == null) {
    observer = new ResizeObserver(drawSoon);
    observer.observe(host);
  }
  return () => {
    cancelAnimationFrame(frame);
    observer?.disconnect();
    tooltip?.remove();
  };
}

function figureScales(domainItems, layout) {
  const hasBar = domainItems.some(item => item.mark === 'bar');
  const xValues = domainItems.flatMap(item => item.series.points.map(point => point.x));
  const leftValues = figureAxisValues(domainItems.filter(item => item.axis === 'left'));
  const rightValues = figureAxisValues(domainItems.filter(item => item.axis === 'right'));
  const x = createScale(xValues, layout.left, layout.right, { padding: hasBar ? 0 : 0.03 });
  const y = createScale(leftValues, layout.bottom, layout.top, { zero: domainItems.some(item => item.axis === 'left' && FIGURE_ZERO_MARKS.has(item.mark)), padding: 0.08 });
  const yRight = rightValues.length
    ? createScale(rightValues, layout.bottom, layout.top, { zero: domainItems.some(item => item.axis === 'right' && FIGURE_ZERO_MARKS.has(item.mark)), padding: 0.08 })
    : null;
  return { x, y, yRight };
}

function paintFigurePanel(svg, layout, visible, scales, tooltip) {
  const clipId = `chart-clip-${nextChartId++}`;
  const defs = svgElement('defs');
  const clip = svgElement('clipPath', { id: clipId });
  clip.append(svgElement('rect', { x: layout.left, y: layout.top, width: Math.max(0, layout.right - layout.left), height: Math.max(0, layout.bottom - layout.top) }));
  defs.append(clip);
  svg.append(defs);
  const marks = svgElement('g', { class: 'chart-marks', 'clip-path': `url(#${clipId})` });
  const barItems = visible.filter(item => item.mark === 'bar');
  const barCount = Math.max(1, barItems.length);
  const maxBarPoints = Math.max(1, ...barItems.map(item => item.series.points.length));
  const allPoints = visible.flatMap(item => item.series.points);
  visible.forEach(item => {
    const group = svgElement('g', { class: `chart-series chart-series-${item.index}` });
    const stacked = item.mode === 'stacked';
    let seriesOffset = 0;
    if (item.mark === 'bar' && !stacked && barCount > 1) {
      const groupWidth = scales.x.type === 'category' ? scales.x.step * 0.78 : Math.max(4, (layout.right - layout.left) / maxBarPoints) * 0.78;
      seriesOffset = (barItems.indexOf(item) - (barCount - 1) / 2) * (groupWidth / barCount);
    }
    getRenderer(item.mark)(group, { ...item.series, color: item.color, index: item.index }, {
      x: scales.x,
      y: item.axis === 'right' ? scales.yRight : scales.y,
      layout,
      tooltip,
      visibleSeriesCount: barCount,
      maxSeriesPoints: maxBarPoints,
      seriesOffset,
      stacked,
      allPoints,
    });
    marks.append(group);
  });
  svg.append(marks);
}

function flattenFigureItems(layers) {
  const items = [];
  let index = 0;
  for (const layer of layers) {
    for (const series of layer.series) {
      items.push({ mark: layer.mark, axis: layer.axis, mode: layer.mode, series, color: colorAt(index), index });
      index++;
    }
  }
  return items;
}

function figureAxisValues(items) {
  return items.flatMap(item => item.series.points.flatMap(point => [point.y, point.y0, point.y1].filter(value => value != null)));
}

function chartHeight(width, explicit) {
  if (explicit != null) return explicit;
  if (width < 420) return 260;
  if (width < 720) return 310;
  return 360;
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
