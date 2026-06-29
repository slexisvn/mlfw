import {
  backtest as qBacktest, walkForward as qWalkForward, compose,
  momentum, meanReversion, zscore,
  equalWeight, crossSectional, longShortRank, meanVariance,
  sharpe as qSharpe, deflatedSharpe, minTrackRecordLength, probabilityOfBacktestOverfitting,
  riskParity, hierarchicalRiskParity, sampleCovariance,
  parseProduct, checkProduct, priceProduct,
} from '#plugins/quantc';

const DEFAULT_SIGNAL = 'momentum';
const DEFAULT_PORTFOLIO = 'long_short';
const DEFAULT_PATHS = 100000;
const DEFAULT_SEED = 1;

const SIGNAL_RESOLVERS = {
  momentum: named => momentum(named.lookback),
  mean_reversion: named => meanReversion(named.lookback),
  zscore: named => zscore(named.window),
};

const PORTFOLIO_RESOLVERS = {
  equal_weight: () => equalWeight(),
  cross_sectional: () => crossSectional(),
  long_short: named => longShortRank(named.fraction),
};

function resolveSignal(spec, named) {
  if (typeof spec === 'function') return spec;
  const factory = SIGNAL_RESOLVERS[spec];
  if (!factory) throw new Error(`Unknown signal '${spec}'; expected one of ${Object.keys(SIGNAL_RESOLVERS).join(', ')} or a signal handle`);
  return factory(named);
}

function resolvePortfolio(spec, named) {
  if (typeof spec === 'function') return spec;
  const factory = PORTFOLIO_RESOLVERS[spec];
  if (!factory) throw new Error(`Unknown portfolio '${spec}'; expected one of ${Object.keys(PORTFOLIO_RESOLVERS).join(', ')} or a portfolio handle`);
  return factory(named);
}

function backtestConfig(named) {
  const config = {};
  if (named.cost !== undefined) config.cost = named.cost;
  if (named.max_leverage !== undefined) config.maxLeverage = named.max_leverage;
  if (named.start !== undefined) config.start = named.start;
  if (named.periods_per_year !== undefined) config.periodsPerYear = named.periods_per_year;
  return config;
}

function toPlainObject(value) {
  return value instanceof Map ? Object.fromEntries(value) : value;
}

function parseAndCheck(source) {
  if (typeof source !== 'string') throw new Error('expected Quill product source as a string');
  const product = parseProduct(source);
  const errors = checkProduct(product);
  if (errors.length > 0) throw new Error(`Quill type errors:\n${errors.map(e => `  line ${e.line}:${e.col} ${e.message}`).join('\n')}`);
  return product;
}

function buildMarket(named) {
  const market = {};
  if (named.spot !== undefined) market.spot = named.spot;
  if (named.vol !== undefined) market.vol = named.vol;
  if (named.spots !== undefined) market.spots = toPlainObject(named.spots);
  if (named.vols !== undefined) market.vols = toPlainObject(named.vols);
  if (named.params !== undefined) market.params = toPlainObject(named.params);
  if (named.model_params !== undefined) market.modelParams = toPlainObject(named.model_params);
  if (named.correlation !== undefined) market.correlation = named.correlation;
  if (named.curve !== undefined) market.curve = named.curve;
  if (named.rate === undefined) throw new Error('price() requires a rate, e.g. price(rate=0.03, spot=100, vol=0.2)');
  market.rate = named.rate;
  return market;
}

function makeProductHandle(takeNamed, product) {
  return teraRecord({
    name: product.name ?? null,
    price: (...args) => {
      const named = takeNamed(args);
      const market = buildMarket(named);
      const paths = named.paths ?? DEFAULT_PATHS;
      const seed = named.seed ?? DEFAULT_SEED;
      const options = named.greeks !== undefined ? { greeks: named.greeks } : {};
      const result = priceProduct(product, market, paths, seed, options);
      return teraRecord({ price: result.price, standard_error: result.standardError, greeks: teraRecord(result.greeks) });
    },
  });
}

export function installQuantBuiltins(define, ctx) {
  const { takeNamed, DataFrame, createDataFrame, fs } = ctx;

  async function dfToPanel(df, assetColumns) {
    const rows = await df.collect();
    const names = assetColumns ?? numericColumns(df, rows);
    const t = rows.length;
    const n = names.length;
    const panel = new Array(t);
    for (let i = 0; i < t; i += 1) {
      const out = new Array(n);
      for (let j = 0; j < n; j += 1) out[j] = Number(rows[i][names[j]]);
      panel[i] = out;
    }
    return panel;
  }

  function numericColumns(df, rows) {
    const cols = df.columns();
    if (rows.length === 0) return cols;
    return cols.filter(name => typeof rows[0][name] === 'number');
  }

  function resultToTera(result) {
    return teraRecord({
      metrics: teraRecord(result.metrics),
      equity: createDataFrame(result.equity.map(value => ({ equity: value }))),
      weights: result.weights,
      port_returns: createDataFrame(result.portReturns.map(value => ({ port_return: value }))),
    });
  }

  async function toReturnSeries(value) {
    if (value instanceof DataFrame) return (await dfToPanel(value)).map(row => row[0]);
    if (Array.isArray(value)) return value.map(Number);
    throw new Error('expected a returns array or a DataFrame');
  }

  async function toMatrix(value) {
    if (value instanceof DataFrame) return dfToPanel(value);
    if (Array.isArray(value)) return value.map(row => (Array.isArray(row) ? row.map(Number) : [Number(row)]));
    throw new Error('expected a matrix or a DataFrame');
  }

  async function toCovariance(value) {
    if (value instanceof DataFrame) return sampleCovariance(await dfToPanel(value));
    if (Array.isArray(value)) return value.map(row => row.map(Number));
    throw new Error('expected a covariance matrix or a returns DataFrame');
  }

  async function runBacktest(args, walkForward) {
    const named = takeNamed(args);
    const df = args[0];
    if (!(df instanceof DataFrame)) throw new Error(`${walkForward ? 'walk_forward' : 'backtest'}() requires a price DataFrame as the first argument`);
    const prices = await dfToPanel(df, named.asset_columns);
    const signal = resolveSignal(named.signal ?? args[1] ?? DEFAULT_SIGNAL, named);
    const portfolio = resolvePortfolio(named.portfolio ?? DEFAULT_PORTFOLIO, named);
    const config = backtestConfig(named);
    if (!walkForward) return resultToTera(qBacktest(prices, compose(signal, portfolio), config));
    if (named.folds !== undefined) config.folds = named.folds;
    if (named.min_train_fraction !== undefined) config.minTrainFraction = named.min_train_fraction;
    return resultToTera(qWalkForward(prices, () => compose(signal, portfolio), config));
  }

  define('backtest', (...args) => runBacktest(args, false));
  define('walk_forward', (...args) => runBacktest(args, true));

  define('momentum', (...args) => momentum(takeNamed(args).lookback ?? args[0]));
  define('mean_reversion', (...args) => meanReversion(takeNamed(args).lookback ?? args[0]));
  define('zscore', (...args) => zscore(takeNamed(args).window ?? args[0]));
  define('equal_weight', () => equalWeight());
  define('cross_sectional', () => crossSectional());
  define('long_short', (...args) => longShortRank(takeNamed(args).fraction ?? args[0]));

  define('sharpe', async (...args) => {
    const named = takeNamed(args);
    return qSharpe(await toReturnSeries(args[0]), named.periods_per_year ?? args[1]);
  });
  define('deflated_sharpe', async (...args) => {
    const named = takeNamed(args);
    return deflatedSharpe(await toReturnSeries(args[0]), named.trial_sharpes ?? args[1] ?? []);
  });
  define('pbo', async (...args) => {
    const named = takeNamed(args);
    return probabilityOfBacktestOverfitting(await toMatrix(args[0]), named.partitions ?? args[1]);
  });
  define('min_track_record_length', async (...args) => {
    const named = takeNamed(args);
    return minTrackRecordLength(await toReturnSeries(args[0]), named.target_sharpe ?? args[1], named.confidence ?? args[2]);
  });

  define('risk_parity', async (...args) => riskParity(await toCovariance(args[0])));
  define('hrp', async (...args) => hierarchicalRiskParity(await toCovariance(args[0])));
  define('mean_variance', async (...args) => {
    const named = takeNamed(args);
    return meanVariance(named.mu ?? args[0], await toCovariance(named.cov ?? args[1]));
  });

  define('quill', source => makeProductHandle(takeNamed, parseAndCheck(source)));
  define('load_quill', path => {
    if (typeof path !== 'string') throw new Error('load_quill() requires a file path string');
    const data = fs.readFile(path);
    const source = typeof data === 'string' ? data : new TextDecoder().decode(data);
    return makeProductHandle(takeNamed, parseAndCheck(source));
  });
}

function teraRecord(source) {
  const map = new Map(source instanceof Map ? source : Object.entries(source));
  for (const [key, value] of map) map[key] = value;
  return map;
}

export const QUANT_SIGNATURES = {
  backtest: [{ name: 'prices' }, { name: 'signal', defaultValue: '"momentum"', isOptional: true }, { name: 'portfolio', defaultValue: '"long_short"', isOptional: true }, { name: 'lookback', isOptional: true }, { name: 'fraction', isOptional: true }, { name: 'cost', isOptional: true }],
  walk_forward: [{ name: 'prices' }, { name: 'signal', defaultValue: '"momentum"', isOptional: true }, { name: 'portfolio', defaultValue: '"long_short"', isOptional: true }, { name: 'folds', defaultValue: '4', isOptional: true }, { name: 'min_train_fraction', defaultValue: '0.5', isOptional: true }],
  momentum: [{ name: 'lookback', defaultValue: '20', isOptional: true }],
  mean_reversion: [{ name: 'lookback', defaultValue: '20', isOptional: true }],
  zscore: [{ name: 'window', defaultValue: '20', isOptional: true }],
  equal_weight: [],
  cross_sectional: [],
  long_short: [{ name: 'fraction', defaultValue: '0.2', isOptional: true }],
  sharpe: [{ name: 'returns' }, { name: 'periods_per_year', defaultValue: '252', isOptional: true }],
  deflated_sharpe: [{ name: 'returns' }, { name: 'trial_sharpes' }],
  pbo: [{ name: 'trial_returns' }, { name: 'partitions', defaultValue: '10', isOptional: true }],
  min_track_record_length: [{ name: 'returns' }, { name: 'target_sharpe', defaultValue: '0', isOptional: true }, { name: 'confidence', defaultValue: '0.95', isOptional: true }],
  risk_parity: [{ name: 'covariance' }],
  hrp: [{ name: 'covariance' }],
  mean_variance: [{ name: 'mu' }, { name: 'cov' }],
  quill: [{ name: 'source' }],
  load_quill: [{ name: 'path' }],
};
