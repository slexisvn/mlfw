export function renderLegend(host, series, hidden, onChange) {
  const legend = document.createElement('div');
  legend.className = 'chart-legend';
  series.forEach((item, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = hidden.has(index) ? 'chart-legend-item hidden' : 'chart-legend-item';
    const swatch = document.createElement('span');
    swatch.className = 'chart-legend-swatch';
    swatch.style.background = item.color;
    const label = document.createElement('span');
    label.textContent = item.name;
    button.append(swatch, label);
    button.addEventListener('click', () => {
      if (hidden.has(index)) hidden.delete(index);
      else hidden.add(index);
      onChange();
    });
    legend.append(button);
  });
  host.append(legend);
}
