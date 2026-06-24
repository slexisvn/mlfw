export const CHART_SPEC = 'tera.notebook.chart';
export const MAX_POINTS = 10000;
export const CHART_TYPES = new Set([
  'line', 'bar', 'scatter', 'histogram', 'area',
  'box', 'violin', 'density', 'correlation', 'hexbin',
  'heatmap', 'regression', 'ecdf', 'bubble', 'funnel', 'waterfall',
]);

export function isChartSpec(value) {
  return value?.kind === CHART_SPEC && CHART_TYPES.has(value.type) && (Array.isArray(value.series) || value.payload != null);
}

export function createPayloadSpec(type, family, payload, options = {}) {
  if (!CHART_TYPES.has(type)) throw new Error(`Unsupported chart type '${type}'`);
  return { kind: CHART_SPEC, type, family, payload, pointCount: payloadPointCount(payload), options: normalizeOptions(options) };
}

export function createSpec(type, series, options = {}) {
  if (!CHART_TYPES.has(type)) throw new Error(`Unsupported chart type '${type}'`);
  const pointCount = series.reduce((sum, item) => sum + item.points.length, 0);
  if (pointCount > MAX_POINTS) {
    throw new Error(`Chart has ${pointCount} points; maximum is ${MAX_POINTS}. Filter or sample the data before charting.`);
  }
  return {
    kind: CHART_SPEC,
    type,
    series,
    pointCount,
    options: normalizeOptions(options),
  };
}

function normalizeOptions(options) {
  const width = finitePositive(options.width, null);
  const height = finitePositive(options.height, null);
  return {
    title: textOrNull(options.title),
    xLabel: textOrNull(options.x_label ?? options.xLabel),
    yLabel: textOrNull(options.y_label ?? options.yLabel),
    width,
    height,
    legend: options.legend !== false,
    zoom: options.zoom !== false,
    mode: options.mode ?? null,
  };
}

function payloadPointCount(payload) {
  if (Array.isArray(payload)) return payload.length;
  if (Array.isArray(payload?.steps)) return payload.steps.length;
  if (Array.isArray(payload?.cells)) return payload.cells.length;
  if (Array.isArray(payload?.bins)) return payload.bins.length;
  return 0;
}

function finitePositive(value, fallback) {
  if (value == null) return fallback;
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) throw new Error('Chart width and height must be positive numbers');
  return number;
}

function textOrNull(value) {
  return value == null ? null : String(value);
}
