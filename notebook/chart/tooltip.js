import { formatValue } from './svg.js';

export function createTooltip(host) {
  const tooltip = document.createElement('div');
  tooltip.className = 'chart-tooltip';
  host.append(tooltip);
  return {
    bind(element, point, series) {
      element.addEventListener('pointerenter', event => show(event, point, series));
      element.addEventListener('pointermove', event => move(event));
      element.addEventListener('pointerleave', hide);
    },
    hide,
    remove() {
      tooltip.remove();
    },
  };

  function show(event, point, series) {
    tooltip.innerHTML = '';
    const name = document.createElement('strong');
    name.textContent = series.name;
    const values = document.createElement('span');
    values.textContent = point.tooltip ?? `x: ${formatValue(point.x)}  y: ${formatValue(point.y)}`;
    tooltip.append(name, values);
    tooltip.classList.add('visible');
    move(event);
  }

  function move(event) {
    const rect = host.getBoundingClientRect();
    tooltip.style.left = `${event.clientX - rect.left + 12}px`;
    tooltip.style.top = `${event.clientY - rect.top + 12}px`;
  }

  function hide() {
    tooltip.classList.remove('visible');
  }
}
