import { svgElement } from '../svg.js';

export function renderLine(root, series, context) {
  const path = series.points.map((point, index) => `${index === 0 ? 'M' : 'L'}${context.x.scale(point.x)},${context.y.scale(point.y)}`).join(' ');
  root.append(svgElement('path', { class: 'chart-line', d: path, stroke: series.color }));
  for (const point of series.points) {
    const circle = svgElement('circle', { class: 'chart-point', cx: context.x.scale(point.x), cy: context.y.scale(point.y), r: 3, fill: series.color });
    context.tooltip.bind(circle, point, series);
    root.append(circle);
  }
}
