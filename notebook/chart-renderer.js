import { renderChart } from './chart/index.js';
import chartCss from './chart/chart.css';

let styleInjected = false;

function injectStyle() {
  if (styleInjected) return;
  const style = document.createElement('style');
  style.textContent = chartCss;
  document.head.appendChild(style);
  styleInjected = true;
}

export function activate() {
  return {
    renderOutputItem(item, element) {
      injectStyle();
      const host = document.createElement('div');
      host.className = 'tera-chart-output';
      element.replaceChildren(host);
      try {
        renderChart(host, item.json());
      } catch (err) {
        host.textContent = 'chart render error: ' + (err && err.message ? err.message : String(err));
      }
    },
  };
}
