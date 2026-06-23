import { describe, expect, it } from 'vitest';
import { createChartApi } from '../../notebook/chart/api.js';
import { adaptHistogram, adaptSeries } from '../../notebook/chart/adapters.js';
import { adaptCorrelation, adaptDistribution, adaptEcdf, adaptHeatmap, adaptHexbin, adaptRegression, prepareSeriesMode } from '../../notebook/chart/advanced_adapters.js';
import { rendererTypes } from '../../notebook/chart/registry.js';
import '../../notebook/chart/render.js';
import { layoutSeries } from '../../notebook/chart/render.js';
import { createScale } from '../../notebook/chart/scales.js';
import { isChartSpec, MAX_POINTS } from '../../notebook/chart/spec.js';
import { clampDomain, domainsEqual, panDomain, zoomDomain } from '../../notebook/chart/zoom.js';
import { boxSummary, correlationMatrix, kde, linearRegression, makeHexbins, pearson, spearman } from '../../notebook/chart/statistics.js';
import { highlightHtml } from '../../notebook/highlight.js';
import { CHART_METHOD_DOCS } from '../../notebook/chart/docs.js';
import { regularHexagonPoints } from '../../notebook/chart/renderers/hexbin.js';
import { TeraRuntime } from '../../src/cli/runtime.js';

describe('notebook chart API', () => {
  it('creates line specs from arrays and multiple 2D series', async () => {
    const chart = createChartApi();
    const line = await chart.line([[1, 10], [2, 20]], { __named: true, x: 0, y: [0, 1], title: 'Data' });
    expect(isChartSpec(line)).toBe(true);
    expect(line.type).toBe('line');
    expect(line.series).toHaveLength(2);
    expect(line.series[1].points[1]).toEqual({ x: 2, y: 20 });
    expect(line.options.title).toBe('Data');
    expect(line.options.zoom).toBe(true);
  });

  it('creates histogram bins and skips null and NaN', async () => {
    const series = await adaptHistogram([1, 2, null, NaN, 4], { bins: 3 });
    expect(series[0].points).toHaveLength(3);
    expect(series[0].points.reduce((sum, point) => sum + point.y, 0)).toBe(3);
  });

  it('rejects non-numeric measures and oversized charts', async () => {
    await expect(adaptSeries([1, 'bad'], {})).rejects.toThrow(/non-numeric/);
    await expect(adaptSeries(new Array(MAX_POINTS + 1).fill(1), {})).rejects.toThrow(/maximum/);
  });

  it('materializes only selected DataFrame columns', async () => {
    const calls = [];
    const frame = {
      count: async () => 2,
      select: (...columns) => {
        calls.push(columns);
        return { collect: async () => [{ epoch: 1, loss: 0.8 }, { epoch: 2, loss: 0.5 }] };
      },
      collect: async () => [],
    };
    const series = await adaptSeries(frame, { x: 'epoch', y: 'loss' });
    expect(calls).toEqual([['epoch', 'loss']]);
    expect(series[0].points).toEqual([{ x: 1, y: 0.8 }, { x: 2, y: 0.5 }]);
  });

  it('groups DataFrame series by categorical color', async () => {
    const rows = [
      { x: 1, y: 2, species: 'a' },
      { x: 2, y: 3, species: 'b' },
      { x: 3, y: 4, species: 'a' },
    ];
    const frame = {
      count: async () => rows.length,
      select: () => ({ collect: async () => rows }),
      collect: async () => rows,
    };
    const series = await adaptSeries(frame, { x: 'x', y: 'y', color: 'species' });
    expect(series.map(item => item.name)).toEqual(['y · a', 'y · b']);
    expect(series[0].points).toHaveLength(2);
  });

  it('registers all v1 renderers and creates numeric and categorical scales', () => {
    expect(rendererTypes().sort()).toEqual(['area', 'bar', 'density', 'ecdf', 'hexbin', 'histogram', 'line', 'regression', 'scatter']);
    expect(createScale([0, 10], 0, 100).scale(5)).toBe(50);
    expect(createScale(['a', 'b'], 0, 100).scale('a')).toBe(25);
  });

  it('supports zoom=false', async () => {
    const chart = createChartApi();
    const line = await chart.line([1, 2, 3], { __named: true, zoom: false });
    expect(line.options.zoom).toBe(false);
  });

  it('zooms around an anchor, pans within bounds, and resets by domain equality', () => {
    const bounds = [0, 100];
    expect(zoomDomain(bounds, 25, 0.5, bounds)).toEqual([12.5, 62.5]);
    expect(panDomain([20, 60], 80, bounds)).toEqual([60, 100]);
    expect(clampDomain([-20, 30], bounds)).toEqual([0, 50]);
    expect(domainsEqual([0, 100], bounds)).toBe(true);
  });

  it('computes Tukey summaries and KDE defaults', () => {
    const summary = boxSummary([1, 2, 3, 4, 100]);
    expect(summary.median).toBe(3);
    expect(summary.outliers).toEqual([100]);
    const density = kde([1, 2, 3, 4]);
    expect(density.bandwidth).toBeGreaterThan(0);
    expect(density.points).toHaveLength(80);
  });

  it('computes Pearson, Spearman, and pairwise correlation counts', () => {
    expect(pearson([1, 2, 3], [2, 4, 6])).toBeCloseTo(1);
    expect(spearman([1, 2, 3], [10, 30, 20])).toBeCloseTo(0.5);
    const cells = correlationMatrix([{ a: 1, b: 2 }, { a: 2, b: null }, { a: 3, b: 6 }], ['a', 'b']);
    expect(cells.find(cell => cell.x === 'b' && cell.y === 'a').count).toBe(2);
  });

  it('assigns numeric points into hexbins', () => {
    const bins = makeHexbins([{ x: 0, y: 0 }, { x: 0.01, y: 0.01 }, { x: 10, y: 10 }], 10);
    expect(bins.reduce((sum, bin) => sum + bin.count, 0)).toBe(3);
    expect(bins.some(bin => bin.count >= 2)).toBe(true);
  });

  it('creates regular screen-space hexagon geometry', () => {
    const vertices = regularHexagonPoints(0, 0, 10).split(' ').map(pair => pair.split(',').map(Number));
    const width = Math.max(...vertices.map(point => point[0])) - Math.min(...vertices.map(point => point[0]));
    const height = Math.max(...vertices.map(point => point[1])) - Math.min(...vertices.map(point => point[1]));
    expect(width).toBeCloseTo(20);
    expect(height).toBeCloseTo(Math.sqrt(3) * 10);
    expect(width / height).toBeCloseTo(2 / Math.sqrt(3));
  });

  it('adapts distribution groups and reports missing values', async () => {
    const rows = [{ value: 1, group: 'a' }, { value: null, group: 'a' }, { value: 4, group: 'b' }];
    const frame = mockFrame(rows, ['value', 'group'], [
      { name: 'value', dataType: 'FLOAT64' },
      { name: 'group', dataType: 'VARCHAR' },
    ]);
    const groups = await adaptDistribution(frame, { x: 'value', color: 'group' }, true);
    expect(groups.map(group => [group.name, group.count, group.missing])).toEqual([['a', 1, 1], ['b', 1, 0]]);
    expect(groups[0].density.points).toHaveLength(80);
  });

  it('computes linear regression fits', () => {
    const fit = linearRegression([{ x: 1, y: 2 }, { x: 2, y: 4 }, { x: 3, y: 6 }]);
    expect(fit.slope).toBeCloseTo(2);
    expect(fit.intercept).toBeCloseTo(0);
    expect(fit.r2).toBeCloseTo(1);
    expect(fit.points).toHaveLength(2);
  });

  it('includes KDE support in violin scale domains', async () => {
    const groups = await adaptDistribution([2, 4, 4, 5, 6, 8, 9], {}, true);
    const summaryValues = groups.flatMap(group => [group.summary.low, group.summary.high, ...group.summary.outliers]);
    const densityValues = groups.flatMap(group => group.density.points.map(point => point.x));
    expect(Math.min(...densityValues)).toBeLessThan(Math.min(...summaryValues));
    expect(Math.max(...densityValues)).toBeGreaterThan(Math.max(...summaryValues));
    const domainValues = groups.flatMap(group => [
      group.summary.low,
      group.summary.high,
      ...group.summary.outliers,
      ...group.density.points.map(point => point.x),
    ]);
    const scale = createScale(domainValues, 240, 42, { padding: 0.1 });
    for (const value of densityValues) {
      expect(scale.scale(value)).toBeGreaterThanOrEqual(42);
      expect(scale.scale(value)).toBeLessThanOrEqual(240);
    }
  });

  it('auto-selects numeric correlation columns', async () => {
    const rows = [{ a: 1, b: 2, name: 'x' }, { a: 2, b: 4, name: 'y' }];
    const frame = mockFrame(rows, ['a', 'b', 'name'], [
      { name: 'a', dataType: 'INT32' },
      { name: 'b', dataType: 'FLOAT64' },
      { name: 'name', dataType: 'VARCHAR' },
    ]);
    const result = await adaptCorrelation(frame, {});
    expect(result.columns).toEqual(['a', 'b']);
    expect(result.cells).toHaveLength(4);
  });

  it('adapts hexbin and validates series modes', async () => {
    const result = await adaptHexbin([[1, 2], [2, 3], [3, 5]], { x: 0, y: 1, bins: 10 });
    expect(result.bins.reduce((sum, bin) => sum + bin.count, 0)).toBe(3);
    const series = [
      { name: 'a', points: [{ x: 'N', y: 2 }] },
      { name: 'b', points: [{ x: 'S', y: 3 }] },
    ];
    const stacked = prepareSeriesMode(series, 'bar', 'stacked');
    expect(stacked.series[0].points).toEqual([{ x: 'N', y: 2 }, { x: 'S', y: 0 }]);
    expect(() => prepareSeriesMode(series, 'area', 'stacked')).toThrow(/matching x values/);
  });

  it('adapts heatmap, regression, and ECDF inputs', async () => {
    const heatmap = await adaptHeatmap([[1, 2], [3, 4]], {});
    expect(heatmap.columns).toEqual(['0', '1']);
    expect(heatmap.rows).toEqual(['0', '1']);
    expect(heatmap.cells).toHaveLength(4);

    const regression = await adaptRegression([[1, 2], [2, 4], [3, 6]], { x: 0, y: 1 });
    expect(regression.series).toHaveLength(2);
    expect(regression.fit.slope).toBeCloseTo(2);

    const ecdf = await adaptEcdf([3, 1, 2], {});
    expect(ecdf[0].points).toEqual([{ x: 1, y: 1 / 3 }, { x: 2, y: 2 / 3 }, { x: 3, y: 1 }]);
  });

  it('creates every advanced chart through the public API', async () => {
    const chart = createChartApi();
    expect((await chart.box([1, 2, 3])).family).toBe('distribution');
    expect((await chart.violin([1, 2, 3])).type).toBe('violin');
    expect((await chart.density([1, 2, 3])).series[0].points).toHaveLength(80);
    expect((await chart.hexbin([[1, 2], [2, 3]], { __named: true, bins: 10 })).type).toBe('hexbin');
    expect((await chart.area([[1, 2], [2, 3]], { __named: true, x: 0, y: 1 })).options.mode).toBe('overlay');
    expect((await chart.heatmap([[1, 2], [3, 4]])).type).toBe('heatmap');
    expect((await chart.regression([[1, 2], [2, 4]], { __named: true, x: 0, y: 1 })).type).toBe('regression');
    expect((await chart.ecdf([3, 1, 2])).type).toBe('ecdf');
  });

  it('stacks positive and negative series independently', () => {
    const series = [
      { name: 'a', points: [{ x: 'N', y: 3 }, { x: 'S', y: -2 }] },
      { name: 'b', points: [{ x: 'N', y: 4 }, { x: 'S', y: -5 }] },
    ];
    const stacked = layoutSeries(series, { options: { mode: 'stacked' } });
    expect(stacked[1].points).toEqual([
      { x: 'N', y: 4, y0: 3, y1: 7 },
      { x: 'S', y: -5, y0: -2, y1: -7 },
    ]);
  });
});

describe('notebook syntax highlight', () => {
  it('keeps chart plain and highlights chart methods', () => {
    const html = highlightHtml('chart.line(data)');
    expect(html).toContain('<span class="tok-ident">chart</span>.<span class="tok-method">line</span>');
    expect(html).not.toContain('tok-builtin">chart');
  });

  it('provides hover descriptions for every chart method', () => {
    const methods = ['line', 'bar', 'scatter', 'histogram', 'area', 'box', 'violin', 'density', 'correlation', 'hexbin', 'heatmap', 'regression', 'ecdf'];
    expect([...CHART_METHOD_DOCS.keys()]).toEqual(methods);
    for (const method of methods) {
      const info = CHART_METHOD_DOCS.get(method);
      expect(info.display).toContain(`chart.${method}(`);
      expect(info.description.length).toBeGreaterThan(30);
    }
  });
});

describe('runtime host extensions', () => {
  it('registers chart only when the notebook host injects it', async () => {
    const runtime = new TeraRuntime({ output: () => {} });
    expect(runtime.getVariable('chart')).toBeUndefined();
    runtime.registerGlobal('chart', createChartApi());
    const result = await runtime.execute('chart.scatter([1, 3, 2], title="Values")');
    expect(result.type).toBe('scatter');
    expect(result.options.title).toBe('Values');
  });

  it('validates registered global names', () => {
    const runtime = new TeraRuntime({ output: () => {} });
    expect(() => runtime.registerGlobal('bad-name', {})).toThrow(/valid identifier/);
  });
});

function mockFrame(rows, columns, fields) {
  return {
    count: async () => rows.length,
    columns: () => columns,
    schema: () => ({ _fields: fields }),
    select: (...selected) => ({ collect: async () => rows.map(row => Object.fromEntries(selected.map(column => [column, row[column]]))) }),
    collect: async () => rows,
  };
}
