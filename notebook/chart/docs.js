export const CHART_METHOD_DOCS = new Map([
  ['line', doc(
    'chart.line(data, x?, y?, color?, title?, x_label?, y_label?, zoom=true)',
    'Draw a line chart for ordered values or trends. Use y=[...] for multiple series and color= to group DataFrame rows.'
  )],
  ['bar', doc(
    'chart.bar(data, x?, y?, color?, mode="grouped", title?)',
    'Compare values across categories. Use mode="stacked" to stack multiple series; aggregate DataFrame rows before charting.'
  )],
  ['scatter', doc(
    'chart.scatter(data, x?, y?, color?, title?, zoom=true)',
    'Plot numeric X/Y observations to inspect relationships, clusters, and outliers. Use color= to split DataFrame groups.'
  )],
  ['histogram', doc(
    'chart.histogram(data, x?, color?, bins=20, title?, zoom=true)',
    'Show the frequency distribution of numeric values. Bins are computed automatically and can be grouped with color=.'
  )],
  ['area', doc(
    'chart.area(data, x?, y?, color?, mode="overlay", title?, zoom=true)',
    'Show trends with the area below each series filled. Use mode="stacked" when aligned series should accumulate.'
  )],
  ['box', doc(
    'chart.box(data, x?, color?, whisker=1.5, title?)',
    'Summarize a numeric distribution with Tukey quartiles, median, whiskers, and outliers. Use color= for grouped boxes.'
  )],
  ['violin', doc(
    'chart.violin(data, x?, color?, bandwidth?, whisker=1.5, title?)',
    'Show a mirrored kernel-density distribution together with median and quartile markers. Use color= to compare groups.'
  )],
  ['density', doc(
    'chart.density(data, x?, color?, bandwidth?, title?, zoom=true)',
    'Estimate and draw a smooth numeric probability density using a Gaussian kernel. Bandwidth defaults to Silverman.'
  )],
  ['correlation', doc(
    'chart.correlation(data, columns?, method="pearson", title?)',
    'Draw a correlation matrix for numeric DataFrame columns. Supports method="pearson" and method="spearman".'
  )],
  ['hexbin', doc(
    'chart.hexbin(data, x?, y?, bins=30, title?, zoom=true)',
    'Aggregate dense numeric X/Y observations into hexagonal bins whose intensity represents the number of points.'
  )],
]);

export function chartMethodOwner(pre, span) {
  let text = '';
  for (const node of pre.childNodes) {
    if (node === span) break;
    text += node.textContent ?? '';
  }
  const match = text.match(/([A-Za-z_]\w*)\.\s*$/);
  return match?.[1] ?? null;
}

function doc(display, description) {
  return { display, kind: 'method of chart', description };
}
