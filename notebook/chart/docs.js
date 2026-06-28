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
  ['heatmap', doc(
    'chart.heatmap(data, x?, y?, value?, title?)',
    'Draw a numeric matrix heatmap. For DataFrame input, provide x, y, and value columns; 2D arrays are supported directly.'
  )],
  ['regression', doc(
    'chart.regression(data, x?, y?, title?, zoom=true)',
    'Plot numeric X/Y observations with a least-squares linear fit and R² tooltip.'
  )],
  ['ecdf', doc(
    'chart.ecdf(data, x?, color?, title?, zoom=true)',
    'Draw an empirical cumulative distribution function for comparing numeric distributions without binning.'
  )],
  ['bubble', doc(
    'chart.bubble(data, x?, y?, size?, color?, title?, zoom=true)',
    'Plot X/Y observations with marker area scaled by a third numeric variable. Useful for spend, revenue, or segment size.'
  )],
  ['funnel', doc(
    'chart.funnel(data, step?, value?, title?)',
    'Show a conversion funnel across ordered stages, including overall and step-to-step retention.'
  )],
  ['waterfall', doc(
    'chart.waterfall(data, step?, value?, title?)',
    'Show how positive and negative contributions accumulate from a starting point to a final total.'
  )],
  ['figure', doc(
    'chart.figure(data, title?).encode(x?, color?).bar(y?).line(y?, axis?).facet(col?)',
    'Compose multiple marks on one coordinate system. Chain .line/.bar/.scatter/.point/.area/.histogram/.regression/.bubble; pass axis="right" for a secondary y-axis, or .facet("column") to split into small-multiple panels.'
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
