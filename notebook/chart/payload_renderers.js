import { colorAt } from './palette.js';
import { createScale } from './scales.js';
import { svgElement, svgText, formatValue } from './svg.js';
import { createTooltip } from './tooltip.js';

export function renderPayloadChart(host, spec) {
  host.innerHTML = '';
  host.className = 'chart-view';
  const width = spec.options.width ?? Math.max(320, host.clientWidth || 720);
  const height = spec.options.height;
  const svg = svgElement('svg', { class: 'chart-svg', viewBox: `0 0 ${width} ${height}`, role: 'img' });
  if (spec.options.title) svg.append(svgText(spec.options.title, { class: 'chart-title', x: 58, y: 24 }));
  if (spec.family === 'matrix') renderMatrix(svg, spec.payload, width, height);
  else renderDistribution(svg, spec.type, spec.payload, width, height, host);
  if (spec.options.width != null) svg.style.width = `${width}px`;
  host.append(svg);
  return () => {};
}

function renderDistribution(svg, type, groups, width, height, host) {
  const layout = { left: 58, right: width - 20, top: 42, bottom: height - 48 };
  const values = groups.flatMap(group => group.summary ? [group.summary.low, group.summary.high, ...group.summary.outliers] : []);
  const y = createScale(values, layout.bottom, layout.top, { padding: 0.1 });
  const x = createScale(groups.map(group => group.name), layout.left, layout.right);
  const tooltip = createTooltip(host);
  renderSimpleAxes(svg, layout, x, y);
  groups.forEach((group, index) => {
    if (!group.summary) return;
    const center = x.scale(group.name);
    const widthValue = Math.min(48, x.step * 0.55);
    const color = colorAt(index);
    if (type === 'violin' && group.density?.points.length) {
      const maxDensity = Math.max(...group.density.points.map(point => point.y));
      const right = group.density.points.map((point, pointIndex) => `${pointIndex ? 'L' : 'M'}${center + point.y / maxDensity * widthValue / 2},${y.scale(point.x)}`).join(' ');
      const left = [...group.density.points].reverse().map(point => `L${center - point.y / maxDensity * widthValue / 2},${y.scale(point.x)}`).join(' ');
      svg.append(svgElement('path', { class: 'chart-violin', d: `${right}${left}Z`, fill: color, stroke: color }));
    }
    const summary = group.summary;
    const rect = svgElement('rect', { class: 'chart-box', x: center - widthValue / 4, y: y.scale(summary.q3), width: widthValue / 2, height: Math.max(1, y.scale(summary.q1) - y.scale(summary.q3)), fill: color });
    const tooltipPoint = { x: group.name, y: summary.median, tooltip: `count: ${group.count}  missing: ${group.missing}  q1: ${formatValue(summary.q1)}  median: ${formatValue(summary.median)}  q3: ${formatValue(summary.q3)}` };
    tooltip.bind(rect, tooltipPoint, { name: group.name });
    svg.append(svgElement('line', { class: 'chart-box-line', x1: center, x2: center, y1: y.scale(summary.low), y2: y.scale(summary.high) }));
    svg.append(rect);
    svg.append(svgElement('line', { class: 'chart-box-median', x1: center - widthValue / 4, x2: center + widthValue / 4, y1: y.scale(summary.median), y2: y.scale(summary.median) }));
    for (const outlier of summary.outliers) svg.append(svgElement('circle', { class: 'chart-box-outlier', cx: center, cy: y.scale(outlier), r: 2.5, fill: color }));
  });
}

function renderMatrix(svg, payload, width, height) {
  const layout = { left: 90, right: width - 20, top: 55, bottom: height - 65 };
  const count = payload.columns.length;
  const cellWidth = (layout.right - layout.left) / count;
  const cellHeight = (layout.bottom - layout.top) / count;
  payload.cells.forEach(cell => {
    const column = payload.columns.indexOf(cell.x);
    const row = payload.columns.indexOf(cell.y);
    const value = Number.isFinite(cell.value) ? cell.value : 0;
    const rect = svgElement('rect', { class: 'chart-correlation-cell', x: layout.left + column * cellWidth, y: layout.top + row * cellHeight, width: cellWidth, height: cellHeight, fill: correlationColor(value) });
    const title = svgElement('title');
    title.textContent = `${cell.y} × ${cell.x}: ${Number.isFinite(cell.value) ? formatValue(cell.value) : 'NaN'} (${payload.method}, n=${cell.count})`;
    rect.append(title);
    svg.append(rect);
    if (cellWidth >= 38 && cellHeight >= 25) svg.append(svgText(Number.isFinite(cell.value) ? value.toFixed(2) : 'NaN', { class: 'chart-correlation-value', x: layout.left + (column + 0.5) * cellWidth, y: layout.top + (row + 0.5) * cellHeight + 4, 'text-anchor': 'middle' }));
  });
  payload.columns.forEach((column, index) => {
    svg.append(svgText(column, { class: 'chart-matrix-label', x: layout.left - 8, y: layout.top + (index + 0.5) * cellHeight + 4, 'text-anchor': 'end' }));
    svg.append(svgText(column, { class: 'chart-matrix-label', transform: `translate(${layout.left + (index + 0.5) * cellWidth} ${layout.bottom + 8}) rotate(-45)`, 'text-anchor': 'end' }));
  });
}

function renderSimpleAxes(svg, layout, x, y) {
  for (const tick of y.ticks) {
    const position = y.scale(tick);
    svg.append(svgElement('line', { class: 'chart-payload-grid', x1: layout.left, x2: layout.right, y1: position, y2: position }));
    svg.append(svgText(formatValue(tick), { class: 'chart-payload-tick', x: layout.left - 8, y: position + 4, 'text-anchor': 'end' }));
  }
  for (const tick of x.ticks) svg.append(svgText(tick, { class: 'chart-payload-tick', x: x.scale(tick), y: layout.bottom + 20, 'text-anchor': 'middle' }));
}

function correlationColor(value) {
  const amount = Math.min(1, Math.abs(value));
  return value < 0 ? `rgba(224,108,117,${0.18 + amount * 0.82})` : `rgba(79,107,237,${0.18 + amount * 0.82})`;
}
