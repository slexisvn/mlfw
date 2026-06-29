var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);

// src/ir/topo.ts
function topoSort(graph) {
  const indegree = /* @__PURE__ */ new Map();
  for (const node of graph.nodes) {
    let dependencies = 0;
    for (const operand of node.operands) {
      if (operand.producer !== null) dependencies += 1;
    }
    indegree.set(node.id, dependencies);
  }
  const order = [];
  const queue = [];
  for (const node of graph.nodes) {
    if (indegree.get(node.id) === 0) queue.push(node);
  }
  let head = 0;
  while (head < queue.length) {
    const node = queue[head];
    head += 1;
    order.push(node);
    for (const use of node.result.uses) {
      const consumer = use.node;
      const remaining = (indegree.get(consumer.id) ?? 0) - 1;
      indegree.set(consumer.id, remaining);
      if (remaining === 0) queue.push(consumer);
    }
  }
  if (order.length !== graph.nodes.length) throw new Error("graph contains a cycle");
  return order;
}

// src/ir/op-registry.ts
var OpRegistry = class {
  constructor() {
    __publicField(this, "definitions", /* @__PURE__ */ new Map());
  }
  register(definition) {
    if (this.definitions.has(definition.name)) throw new Error(`duplicate op ${definition.name}`);
    this.definitions.set(definition.name, definition);
  }
  get(name) {
    const definition = this.definitions.get(name);
    if (definition === void 0) throw new Error(`unknown op ${name}`);
    return definition;
  }
  has(name) {
    return this.definitions.has(name);
  }
};
var registry = new OpRegistry();

// src/eval/interpreter.ts
function require2(values, id) {
  const value = values.get(id);
  if (value === void 0) throw new Error(`unbound value ${id}`);
  return value;
}
function evaluate(graph, bindings, batchSize) {
  const values = new Map(bindings);
  for (const node of topoSort(graph)) {
    const operands = node.operands.map((operand) => require2(values, operand.id));
    const outLen = node.result.kind === "batch" ? batchSize : 1;
    const out = registry.get(node.op).evalFn(operands, outLen, node.attrs);
    values.set(node.result.id, out);
  }
  return values;
}

// src/aad/grad-accumulator.ts
function addArrays(a, b) {
  const n = a.length >= b.length ? a.length : b.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    out[i] = (a.length === 1 ? a[0] : a[i]) + (b.length === 1 ? b[0] : b[i]);
  }
  return out;
}
function treeReduce(parts) {
  let level = parts;
  while (level.length > 1) {
    const next = [];
    for (let i = 0; i < level.length; i += 2) {
      if (i + 1 < level.length) next.push(addArrays(level[i], level[i + 1]));
      else next.push(level[i]);
    }
    level = next;
  }
  return level[0];
}
var GradAccumulator = class {
  constructor() {
    __publicField(this, "pending", /* @__PURE__ */ new Map());
  }
  add(id, grad) {
    const existing = this.pending.get(id);
    if (existing !== void 0) existing.push(grad);
    else this.pending.set(id, [grad]);
  }
  get(id) {
    const parts = this.pending.get(id);
    if (parts === void 0 || parts.length === 0) return null;
    return treeReduce(parts);
  }
};

// src/aad/reverse.ts
function require3(values, id) {
  const value = values.get(id);
  if (value === void 0) throw new Error(`missing forward value ${id}`);
  return value;
}
function reduceToLength(grad, target) {
  if (grad.length === target) return grad;
  if (target === 1) {
    let total = 0;
    for (let i = 0; i < grad.length; i += 1) total += grad[i];
    return new Float64Array([total]);
  }
  throw new Error(`cannot reduce gradient of length ${grad.length} to ${target}`);
}
function reverse(graph, forward, seeds) {
  const accumulator = new GradAccumulator();
  for (const [id, grad] of seeds) accumulator.add(id, grad);
  const order = topoSort(graph);
  for (let k = order.length - 1; k >= 0; k -= 1) {
    const node = order[k];
    if (node.operands.length === 0) continue;
    const adjOut = accumulator.get(node.result.id);
    if (adjOut === null) continue;
    const operands = node.operands.map((operand) => require3(forward, operand.id));
    const result = require3(forward, node.result.id);
    const raw = registry.get(node.op).adjointFn(operands, result, adjOut, node.attrs);
    for (let i = 0; i < node.operands.length; i += 1) {
      accumulator.add(node.operands[i].id, reduceToLength(raw[i], operands[i].length));
    }
  }
  const gradients = /* @__PURE__ */ new Map();
  for (const input of graph.inputs) {
    const grad = accumulator.get(input.id);
    if (grad !== null) gradients.set(input.id, grad);
  }
  return gradients;
}

// src/ir/value.ts
var Value = class {
  constructor(id, kind, label) {
    __publicField(this, "id");
    __publicField(this, "kind");
    __publicField(this, "label");
    __publicField(this, "producer");
    __publicField(this, "uses");
    this.id = id;
    this.kind = kind;
    this.label = label;
    this.producer = null;
    this.uses = [];
  }
};

// src/ir/node.ts
var Node = class {
  constructor(id, op, operands, result, attrs) {
    __publicField(this, "id");
    __publicField(this, "op");
    __publicField(this, "operands");
    __publicField(this, "result");
    __publicField(this, "attrs");
    this.id = id;
    this.op = op;
    this.operands = operands;
    this.result = result;
    this.attrs = attrs;
  }
};

// src/ir/graph.ts
var OPERATOR_OPS = {
  "+": "add",
  "-": "sub",
  "*": "mul",
  "/": "div"
};
var Graph = class {
  constructor() {
    __publicField(this, "inputs", []);
    __publicField(this, "nodes", []);
    __publicField(this, "output", null);
    __publicField(this, "valueSeq", 0);
    __publicField(this, "nodeSeq", 0);
  }
  input(kind, label) {
    const value = new Value(this.valueSeq, kind, label);
    this.valueSeq += 1;
    this.inputs.push(value);
    return value;
  }
  emit(op, operands, attrs, label) {
    const definition = registry.get(op);
    if (definition.arity >= 0 && operands.length !== definition.arity) {
      throw new Error(`op ${op} expects ${definition.arity} operands, received ${operands.length}`);
    }
    const kind = definition.inferKind(operands.map((operand) => operand.kind), attrs);
    const result = new Value(this.valueSeq, kind, label);
    this.valueSeq += 1;
    const node = new Node(this.nodeSeq, op, [...operands], result, attrs);
    this.nodeSeq += 1;
    result.producer = node;
    for (let i = 0; i < operands.length; i += 1) operands[i].uses.push({ node, index: i });
    this.nodes.push(node);
    return result;
  }
  constant(value) {
    return this.emit("const", [], { value }, `const_${value}`);
  }
  operator(symbol, left, right) {
    const op = OPERATOR_OPS[symbol];
    if (op === void 0) throw new Error(`unknown operator ${symbol}`);
    return this.emit(op, [left, right], {}, op);
  }
  add(left, right) {
    return this.emit("add", [left, right], {}, "add");
  }
  sub(left, right) {
    return this.emit("sub", [left, right], {}, "sub");
  }
  mul(left, right) {
    return this.emit("mul", [left, right], {}, "mul");
  }
  div(left, right) {
    return this.emit("div", [left, right], {}, "div");
  }
  neg(operand) {
    return this.emit("neg", [operand], {}, "neg");
  }
  exp(operand) {
    return this.emit("exp", [operand], {}, "exp");
  }
  log(operand) {
    return this.emit("log", [operand], {}, "log");
  }
  sqrt(operand) {
    return this.emit("sqrt", [operand], {}, "sqrt");
  }
  max(left, right) {
    return this.emit("max", [left, right], {}, "max");
  }
  min(left, right) {
    return this.emit("min", [left, right], {}, "min");
  }
  sigmoid(operand) {
    return this.emit("sigmoid", [operand], {}, "sigmoid");
  }
  softplus(operand) {
    return this.emit("softplus", [operand], {}, "softplus");
  }
  mean(operand) {
    return this.emit("mean", [operand], {}, "mean");
  }
  sum(operand) {
    return this.emit("sum", [operand], {}, "sum");
  }
};

// src/ir/ops/core.ts
var constOp = {
  name: "const",
  arity: 0,
  inferKind: () => "scalar",
  evalFn: (_operands, _outLen, attrs) => new Float64Array([attrs.value]),
  adjointFn: () => [],
  jvpFn: () => new Float64Array([0])
};

// src/ir/ops/helpers.ts
function elementwiseKind(kinds) {
  for (const kind of kinds) {
    if (kind === "batch") return "batch";
  }
  return "scalar";
}
function at(array, index) {
  return array.length === 1 ? array[0] : array[index];
}

// src/ir/ops/factories.ts
function binaryOp(name, forward, adjoint) {
  return {
    name,
    arity: 2,
    inferKind: (kinds) => elementwiseKind(kinds),
    evalFn: (operands, outLen) => {
      const a = operands[0];
      const b = operands[1];
      const out = new Float64Array(outLen);
      for (let i = 0; i < outLen; i += 1) out[i] = forward(at(a, i), at(b, i));
      return out;
    },
    adjointFn: (operands, result, adjOut) => {
      const a = operands[0];
      const b = operands[1];
      const n = adjOut.length;
      const gradA = new Float64Array(n);
      const gradB = new Float64Array(n);
      for (let i = 0; i < n; i += 1) {
        const pair = adjoint(at(a, i), at(b, i), at(result, i), adjOut[i]);
        gradA[i] = pair[0];
        gradB[i] = pair[1];
      }
      return [gradA, gradB];
    },
    jvpFn: (operands, tangents, result, outLen) => {
      const a = operands[0];
      const b = operands[1];
      const da = tangents[0];
      const db = tangents[1];
      const out = new Float64Array(outLen);
      for (let i = 0; i < outLen; i += 1) {
        const partial = adjoint(at(a, i), at(b, i), at(result, i), 1);
        out[i] = partial[0] * at(da, i) + partial[1] * at(db, i);
      }
      return out;
    }
  };
}
function unaryOp(name, forward, adjoint) {
  return {
    name,
    arity: 1,
    inferKind: (kinds) => elementwiseKind(kinds),
    evalFn: (operands, outLen) => {
      const a = operands[0];
      const out = new Float64Array(outLen);
      for (let i = 0; i < outLen; i += 1) out[i] = forward(at(a, i));
      return out;
    },
    adjointFn: (operands, result, adjOut) => {
      const a = operands[0];
      const n = adjOut.length;
      const gradA = new Float64Array(n);
      for (let i = 0; i < n; i += 1) gradA[i] = adjoint(at(a, i), at(result, i), adjOut[i]);
      return [gradA];
    },
    jvpFn: (operands, tangents, result, outLen) => {
      const a = operands[0];
      const da = tangents[0];
      const out = new Float64Array(outLen);
      for (let i = 0; i < outLen; i += 1) out[i] = adjoint(at(a, i), at(result, i), 1) * at(da, i);
      return out;
    }
  };
}

// src/ir/ops/arithmetic.ts
function registerArithmetic(target = registry) {
  target.register(binaryOp("add", (a, b) => a + b, (_a, _b, _y, g) => [g, g]));
  target.register(binaryOp("sub", (a, b) => a - b, (_a, _b, _y, g) => [g, -g]));
  target.register(binaryOp("mul", (a, b) => a * b, (a, b, _y, g) => [g * b, g * a]));
  target.register(binaryOp("div", (a, b) => a / b, (a, b, _y, g) => [g / b, -g * a / (b * b)]));
  target.register(unaryOp("neg", (a) => -a, (_a, _y, g) => -g));
}

// src/ir/ops/transcendental.ts
function registerTranscendental(target = registry) {
  target.register(unaryOp("exp", (a) => Math.exp(a), (_a, y, g) => g * y));
  target.register(unaryOp("log", (a) => Math.log(a), (a, _y, g) => g / a));
  target.register(unaryOp("sqrt", (a) => Math.sqrt(a), (_a, y, g) => g / (2 * y)));
  target.register(binaryOp("max", (a, b) => a >= b ? a : b, (a, b, _y, g) => a >= b ? [g, 0] : [0, g]));
  target.register(binaryOp("min", (a, b) => a <= b ? a : b, (a, b, _y, g) => a <= b ? [g, 0] : [0, g]));
  target.register(binaryOp("ge", (a, b) => a >= b ? 1 : 0, (_a, _b, _y, _g) => [0, 0]));
}

// src/ir/ops/reduction.ts
var meanOp = {
  name: "mean",
  arity: 1,
  inferKind: () => "scalar",
  evalFn: (operands) => {
    const a = operands[0];
    let total = 0;
    for (let i = 0; i < a.length; i += 1) total += a[i];
    return new Float64Array([total / a.length]);
  },
  adjointFn: (operands, _result, adjOut) => {
    const a = operands[0];
    const grad = new Float64Array(a.length);
    const share = adjOut[0] / a.length;
    for (let i = 0; i < a.length; i += 1) grad[i] = share;
    return [grad];
  },
  jvpFn: (_operands, tangents) => {
    const da = tangents[0];
    let total = 0;
    for (let i = 0; i < da.length; i += 1) total += da[i];
    return new Float64Array([total / da.length]);
  }
};
var sumOp = {
  name: "sum",
  arity: 1,
  inferKind: () => "scalar",
  evalFn: (operands) => {
    const a = operands[0];
    let total = 0;
    for (let i = 0; i < a.length; i += 1) total += a[i];
    return new Float64Array([total]);
  },
  adjointFn: (operands, _result, adjOut) => {
    const a = operands[0];
    const grad = new Float64Array(a.length);
    const share = adjOut[0];
    for (let i = 0; i < a.length; i += 1) grad[i] = share;
    return [grad];
  },
  jvpFn: (_operands, tangents) => {
    const da = tangents[0];
    let total = 0;
    for (let i = 0; i < da.length; i += 1) total += da[i];
    return new Float64Array([total]);
  }
};
function registerReduction(target = registry) {
  target.register(meanOp);
  target.register(sumOp);
}

// src/ir/ops/smoothing.ts
function sigmoid(x) {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}
function softplus(x) {
  return x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x));
}
function registerSmoothing(target = registry) {
  target.register(unaryOp("sigmoid", sigmoid, (_a, y, g) => g * y * (1 - y)));
  target.register(unaryOp("softplus", softplus, (a, _y, g) => g * sigmoid(a)));
}

// src/ir/ops/comparison.ts
function registerComparison(target = registry) {
  target.register(binaryOp("lt", (a, b) => a < b ? 1 : 0, (_a, _b, _y, _g) => [0, 0]));
  target.register(binaryOp("le", (a, b) => a <= b ? 1 : 0, (_a, _b, _y, _g) => [0, 0]));
  target.register(binaryOp("gt", (a, b) => a > b ? 1 : 0, (_a, _b, _y, _g) => [0, 0]));
  target.register(binaryOp("eq", (a, b) => a === b ? 1 : 0, (_a, _b, _y, _g) => [0, 0]));
  target.register(binaryOp("ne", (a, b) => a !== b ? 1 : 0, (_a, _b, _y, _g) => [0, 0]));
  target.register(unaryOp("abs", (a) => Math.abs(a), (a, _y, g) => g * Math.sign(a)));
}

// src/ir/ops/index.ts
var registered = false;
function registerBuiltinOps() {
  if (registered) return;
  registry.register(constOp);
  registerArithmetic(registry);
  registerTranscendental(registry);
  registerReduction(registry);
  registerSmoothing(registry);
  registerComparison(registry);
  registered = true;
}

// src/models/model-registry.ts
var ModelRegistry = class {
  constructor() {
    __publicField(this, "models", /* @__PURE__ */ new Map());
  }
  register(model) {
    if (this.models.has(model.name)) throw new Error(`duplicate model ${model.name}`);
    this.models.set(model.name, model);
  }
  get(name) {
    const model = this.models.get(name);
    if (model === void 0) throw new Error(`unknown model ${name}`);
    return model;
  }
};
var models = new ModelRegistry();

// src/models/equity/gbm.ts
var gbm = {
  name: "gbm",
  terminal(context) {
    const g = context.graph;
    const half = g.constant(0.5);
    const variance = g.mul(g.mul(half, context.vol), context.vol);
    const drift = g.mul(g.sub(context.rate, variance), context.maturity);
    const diffusion = g.mul(g.mul(context.vol, g.sqrt(context.maturity)), context.normals);
    const logTerminal = g.add(g.add(g.log(context.spot), drift), diffusion);
    return g.exp(logTerminal);
  }
};

// src/numerics/rng/mersenne-twister.ts
var N = 624;
var M = 397;
var MATRIX_A = 2567483615;
var UPPER_MASK = 2147483648;
var LOWER_MASK = 2147483647;
var TWO_POW_53 = 9007199254740992;
var TWO_POW_26 = 67108864;
var MersenneTwister = class {
  constructor(seed) {
    __publicField(this, "state", new Uint32Array(N));
    __publicField(this, "index", N + 1);
    this.seed(seed >>> 0);
  }
  seed(value) {
    this.state[0] = value >>> 0;
    for (let i = 1; i < N; i += 1) {
      const previous = this.state[i - 1] ^ this.state[i - 1] >>> 30;
      this.state[i] = Math.imul(1812433253, previous) + i >>> 0;
    }
    this.index = N;
  }
  nextUint32() {
    if (this.index >= N) {
      let y;
      let kk = 0;
      for (; kk < N - M; kk += 1) {
        y = this.state[kk] & UPPER_MASK | this.state[kk + 1] & LOWER_MASK;
        this.state[kk] = (this.state[kk + M] ^ y >>> 1 ^ ((y & 1) !== 0 ? MATRIX_A : 0)) >>> 0;
      }
      for (; kk < N - 1; kk += 1) {
        y = this.state[kk] & UPPER_MASK | this.state[kk + 1] & LOWER_MASK;
        this.state[kk] = (this.state[kk + (M - N)] ^ y >>> 1 ^ ((y & 1) !== 0 ? MATRIX_A : 0)) >>> 0;
      }
      y = this.state[N - 1] & UPPER_MASK | this.state[0] & LOWER_MASK;
      this.state[N - 1] = (this.state[M - 1] ^ y >>> 1 ^ ((y & 1) !== 0 ? MATRIX_A : 0)) >>> 0;
      this.index = 0;
    }
    let x = this.state[this.index];
    this.index += 1;
    x ^= x >>> 11;
    x ^= x << 7 & 2636928640;
    x ^= x << 15 & 4022730752;
    x ^= x >>> 18;
    return x >>> 0;
  }
  nextDouble() {
    const high = this.nextUint32() >>> 5;
    const low = this.nextUint32() >>> 6;
    return (high * TWO_POW_26 + low) / TWO_POW_53;
  }
};

// src/numerics/rng/inverse-normal-cdf.ts
var A = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
var B = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
var C = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
var D = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
var LOW = 0.02425;
var HIGH = 1 - LOW;
function inverseNormalCdf(probability) {
  let p = probability;
  if (p <= 0) p = Number.MIN_VALUE;
  if (p >= 1) p = 1 - Number.EPSILON;
  if (p < LOW) {
    const q2 = Math.sqrt(-2 * Math.log(p));
    return (((((C[0] * q2 + C[1]) * q2 + C[2]) * q2 + C[3]) * q2 + C[4]) * q2 + C[5]) / ((((D[0] * q2 + D[1]) * q2 + D[2]) * q2 + D[3]) * q2 + 1);
  }
  if (p <= HIGH) {
    const q2 = p - 0.5;
    const r = q2 * q2;
    return (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q2 / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1);
  }
  const q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
}

// src/numerics/stats/welford.ts
var Welford = class {
  constructor() {
    __publicField(this, "n", 0);
    __publicField(this, "runningMean", 0);
    __publicField(this, "m2", 0);
  }
  push(x) {
    this.n += 1;
    const delta = x - this.runningMean;
    this.runningMean += delta / this.n;
    this.m2 += delta * (x - this.runningMean);
  }
  get count() {
    return this.n;
  }
  get mean() {
    return this.runningMean;
  }
  get variance() {
    return this.n > 1 ? this.m2 / (this.n - 1) : 0;
  }
  get standardError() {
    return this.n > 0 ? Math.sqrt(this.variance / this.n) : 0;
  }
  get sumSquaredDeviations() {
    return this.m2;
  }
};
function combineMoments(states) {
  let count = 0;
  let mean = 0;
  let m2 = 0;
  for (const state of states) {
    if (state.count === 0) continue;
    const delta = state.mean - mean;
    const combined = count + state.count;
    mean += delta * state.count / combined;
    m2 += state.sumSquaredDeviations + delta * delta * count * state.count / combined;
    count = combined;
  }
  return { count, mean, sumSquaredDeviations: m2 };
}
function standardErrorOf(state) {
  if (state.count < 2) return 0;
  return Math.sqrt(state.sumSquaredDeviations / (state.count - 1) / state.count);
}

// src/models/equity/heston.ts
function priceHestonCall(spec) {
  const dt = spec.maturity / spec.steps;
  const sqrtDt = Math.sqrt(dt);
  const rhoComplement = Math.sqrt(1 - spec.correlation * spec.correlation);
  const generator = new MersenneTwister(spec.seed);
  const discount = Math.exp(-spec.rate * spec.maturity);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    let logSpot = Math.log(spec.spot);
    let variance = spec.initialVariance;
    for (let step = 0; step < spec.steps; step += 1) {
      const z1 = inverseNormalCdf(generator.nextDouble());
      const z2 = spec.correlation * z1 + rhoComplement * inverseNormalCdf(generator.nextDouble());
      const positiveVariance = variance > 0 ? variance : 0;
      logSpot += (spec.rate - 0.5 * positiveVariance) * dt + Math.sqrt(positiveVariance) * sqrtDt * z1;
      variance += spec.meanReversion * (spec.longVariance - positiveVariance) * dt + spec.volOfVol * Math.sqrt(positiveVariance) * sqrtDt * z2;
    }
    estimator.push(discount * Math.max(Math.exp(logSpot) - spec.strike, 0));
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}

// src/models/rates/hull-white.ts
function integratedMean(spec) {
  const { shortRate, meanReversion: meanReversion2, longRate, maturity } = spec;
  return longRate * maturity + (shortRate - longRate) * (1 - Math.exp(-meanReversion2 * maturity)) / meanReversion2;
}
function integratedVariance(spec) {
  const { meanReversion: meanReversion2, vol, maturity } = spec;
  const a = meanReversion2;
  return vol * vol / (a * a) * (maturity - 2 * (1 - Math.exp(-a * maturity)) / a + (1 - Math.exp(-2 * a * maturity)) / (2 * a));
}
function zeroCouponBond(spec) {
  return Math.exp(-integratedMean(spec) + 0.5 * integratedVariance(spec));
}

// src/models/fx/sabr.ts
function sabrImpliedVol(spec) {
  const { forward, strike, maturity, alpha: alpha2, beta, rho, volOfVol } = spec;
  const oneMinusBeta = 1 - beta;
  if (Math.abs(forward - strike) < 1e-12) {
    const term12 = oneMinusBeta * oneMinusBeta / 24 * (alpha2 * alpha2) / Math.pow(forward, 2 * oneMinusBeta);
    const term22 = rho * beta * volOfVol * alpha2 / (4 * Math.pow(forward, oneMinusBeta));
    const term32 = (2 - 3 * rho * rho) / 24 * volOfVol * volOfVol;
    return alpha2 / Math.pow(forward, oneMinusBeta) * (1 + (term12 + term22 + term32) * maturity);
  }
  const logFK = Math.log(forward / strike);
  const fkMid = Math.pow(forward * strike, oneMinusBeta / 2);
  const z = volOfVol / alpha2 * fkMid * logFK;
  const xz = Math.log((Math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho));
  const denominator = fkMid * (1 + oneMinusBeta * oneMinusBeta / 24 * logFK * logFK + Math.pow(oneMinusBeta, 4) / 1920 * Math.pow(logFK, 4));
  const term1 = oneMinusBeta * oneMinusBeta / 24 * (alpha2 * alpha2) / (fkMid * fkMid);
  const term2 = rho * beta * volOfVol * alpha2 / (4 * fkMid);
  const term3 = (2 - 3 * rho * rho) / 24 * volOfVol * volOfVol;
  return alpha2 / denominator * (z / xz) * (1 + (term1 + term2 + term3) * maturity);
}

// src/models/index.ts
var registered2 = false;
function registerBuiltinModels() {
  if (registered2) return;
  models.register(gbm);
  registered2 = true;
}

// src/script/lexer.ts
var OPERATORS = /* @__PURE__ */ new Set(["+", "-", "*", "/"]);
function isDigit(c) {
  return c >= "0" && c <= "9";
}
function isIdentStart(c) {
  return c >= "a" && c <= "z" || c >= "A" && c <= "Z" || c === "_";
}
function isIdentPart(c) {
  return isIdentStart(c) || isDigit(c);
}
function isExponentSign(source, index) {
  const c = source[index];
  if (c !== "+" && c !== "-") return false;
  const previous = source[index - 1];
  return previous === "e" || previous === "E";
}
function tokenize(source) {
  const tokens = [];
  let i = 0;
  const n = source.length;
  while (i < n) {
    const c = source[i];
    if (c === " " || c === "	" || c === "\n" || c === "\r") {
      i += 1;
      continue;
    }
    if (c === "(") {
      tokens.push({ type: "lparen", value: c });
      i += 1;
      continue;
    }
    if (c === ")") {
      tokens.push({ type: "rparen", value: c });
      i += 1;
      continue;
    }
    if (c === ",") {
      tokens.push({ type: "comma", value: c });
      i += 1;
      continue;
    }
    if (OPERATORS.has(c)) {
      tokens.push({ type: "op", value: c });
      i += 1;
      continue;
    }
    if (isDigit(c) || c === ".") {
      let j = i + 1;
      while (j < n && (isDigit(source[j]) || source[j] === "." || source[j] === "e" || source[j] === "E" || isExponentSign(source, j))) j += 1;
      tokens.push({ type: "number", value: source.slice(i, j) });
      i = j;
      continue;
    }
    if (isIdentStart(c)) {
      let j = i + 1;
      while (j < n && isIdentPart(source[j])) j += 1;
      tokens.push({ type: "ident", value: source.slice(i, j) });
      i = j;
      continue;
    }
    throw new Error(`unexpected character '${c}'`);
  }
  tokens.push({ type: "eof", value: "" });
  return tokens;
}

// src/script/parser.ts
var INFIX_PRECEDENCE = {
  "+": 10,
  "-": 10,
  "*": 20,
  "/": 20
};
var UNARY_PRECEDENCE = 30;
var Parser = class {
  constructor(tokens) {
    this.tokens = tokens;
    __publicField(this, "position", 0);
  }
  peek() {
    return this.tokens[this.position];
  }
  advance() {
    const token = this.tokens[this.position];
    this.position += 1;
    return token;
  }
  expect(type) {
    const token = this.advance();
    if (token.type !== type) throw new Error(`expected ${type}, found ${token.type}`);
    return token;
  }
  parseExpression(minPrecedence) {
    let left = this.parsePrefix();
    for (; ; ) {
      const token = this.peek();
      if (token.type !== "op") break;
      const precedence = INFIX_PRECEDENCE[token.value];
      if (precedence === void 0 || precedence < minPrecedence) break;
      this.advance();
      const right = this.parseExpression(precedence + 1);
      left = { kind: "binary", op: token.value, left, right };
    }
    return left;
  }
  parsePrefix() {
    const token = this.peek();
    if (token.type === "op" && token.value === "-") {
      this.advance();
      return { kind: "unary", op: "-", operand: this.parseExpression(UNARY_PRECEDENCE) };
    }
    if (token.type === "number") {
      this.advance();
      return { kind: "num", value: Number(token.value) };
    }
    if (token.type === "ident") {
      this.advance();
      if (this.peek().type === "lparen") {
        this.advance();
        const args = [];
        if (this.peek().type !== "rparen") {
          args.push(this.parseExpression(0));
          while (this.peek().type === "comma") {
            this.advance();
            args.push(this.parseExpression(0));
          }
        }
        this.expect("rparen");
        return { kind: "call", callee: token.value, args };
      }
      return { kind: "ident", name: token.value };
    }
    if (token.type === "lparen") {
      this.advance();
      const inner = this.parseExpression(0);
      this.expect("rparen");
      return inner;
    }
    throw new Error(`unexpected token ${token.type}`);
  }
};
function parsePayoff(source) {
  const parser = new Parser(tokenize(source));
  const expression = parser.parseExpression(0);
  parser.expect("eof");
  return expression;
}

// src/script/lower.ts
var CALL_ARITY = {
  max: 2,
  min: 2,
  exp: 1,
  log: 1,
  sqrt: 1,
  sigmoid: 1,
  softplus: 1
};
function lowerPayoff(expr, graph, environment) {
  switch (expr.kind) {
    case "num":
      return graph.constant(expr.value);
    case "ident": {
      const value = environment.get(expr.name);
      if (value === void 0) throw new Error(`unknown identifier '${expr.name}'`);
      return value;
    }
    case "unary":
      return graph.neg(lowerPayoff(expr.operand, graph, environment));
    case "binary":
      return graph.operator(expr.op, lowerPayoff(expr.left, graph, environment), lowerPayoff(expr.right, graph, environment));
    case "call": {
      const arity = CALL_ARITY[expr.callee];
      if (arity === void 0) throw new Error(`unknown function '${expr.callee}'`);
      if (expr.args.length !== arity) throw new Error(`function '${expr.callee}' expects ${arity} arguments`);
      const args = expr.args.map((argument) => lowerPayoff(argument, graph, environment));
      return graph.emit(expr.callee, args, {}, expr.callee);
    }
  }
}

// src/engines/european-graph.ts
function buildEuropeanGraph(payoff2, model) {
  registerBuiltinOps();
  registerBuiltinModels();
  const graph = new Graph();
  const spot2 = graph.input("scalar", "spot");
  const strike = graph.input("scalar", "strike");
  const rate = graph.input("scalar", "rate");
  const vol = graph.input("scalar", "vol");
  const maturity = graph.input("scalar", "maturity");
  const normals = graph.input("batch", "normals");
  const context = { graph, spot: spot2, rate, vol, maturity, normals };
  const terminal2 = models.get(model).terminal(context);
  const environment = /* @__PURE__ */ new Map([
    ["spot", terminal2],
    ["strike", strike],
    ["rate", rate],
    ["vol", vol],
    ["maturity", maturity]
  ]);
  const payoffValue = lowerPayoff(parsePayoff(payoff2), graph, environment);
  const discount = graph.exp(graph.neg(graph.mul(rate, maturity)));
  const discounted = graph.mul(discount, payoffValue);
  const price = graph.mean(discounted);
  graph.output = price;
  return { graph, inputs: { spot: spot2, strike, rate, vol, maturity, normals }, price, discounted };
}

// src/numerics/sampling.ts
function standardNormals(count, seed) {
  const generator = new MersenneTwister(seed);
  const draws = new Float64Array(count);
  for (let i = 0; i < count; i += 1) draws[i] = inverseNormalCdf(generator.nextDouble());
  return draws;
}

// src/engines/mc-engine.ts
function scalar(value) {
  return new Float64Array([value]);
}
function requireValue(values, id) {
  const value = values.get(id);
  if (value === void 0) throw new Error(`missing value ${id}`);
  return value;
}
function valueAndGradient(spec) {
  const built = buildEuropeanGraph(spec.payoff, spec.model);
  const draws = standardNormals(spec.paths, spec.seed);
  const bindings = /* @__PURE__ */ new Map([
    [built.inputs.spot.id, scalar(spec.spot)],
    [built.inputs.strike.id, scalar(spec.strike)],
    [built.inputs.rate.id, scalar(spec.rate)],
    [built.inputs.vol.id, scalar(spec.vol)],
    [built.inputs.maturity.id, scalar(spec.maturity)],
    [built.inputs.normals.id, draws]
  ]);
  const forward = evaluate(built.graph, bindings, spec.paths);
  const price = requireValue(forward, built.price.id)[0];
  const samples = requireValue(forward, built.discounted.id);
  const welford = new Welford();
  for (let i = 0; i < samples.length; i += 1) welford.push(samples[i]);
  const gradients = reverse(built.graph, forward, /* @__PURE__ */ new Map([[built.price.id, scalar(1)]]));
  const read = (value) => {
    const grad = gradients.get(value.id);
    return grad === void 0 ? 0 : grad[0];
  };
  return {
    price,
    standardError: welford.standardError,
    gradient: {
      delta: read(built.inputs.spot),
      vega: read(built.inputs.vol),
      rho: read(built.inputs.rate),
      theta: -read(built.inputs.maturity)
    }
  };
}
function priceEuropean(spec) {
  const base = valueAndGradient(spec);
  const spotBump = Math.max(spec.spot * 1e-4, 1e-6);
  const volBump = Math.max(spec.vol * 1e-3, 1e-6);
  const deltaUp = valueAndGradient({ ...spec, spot: spec.spot + spotBump }).gradient.delta;
  const deltaDown = valueAndGradient({ ...spec, spot: spec.spot - spotBump }).gradient.delta;
  const upVol = valueAndGradient({ ...spec, vol: spec.vol + volBump }).gradient;
  const downVol = valueAndGradient({ ...spec, vol: spec.vol - volBump }).gradient;
  const gamma = (deltaUp - deltaDown) / (2 * spotBump);
  const vanna = (upVol.delta - downVol.delta) / (2 * volBump);
  const volga = (upVol.vega - downVol.vega) / (2 * volBump);
  return {
    price: base.price,
    standardError: base.standardError,
    greeks: {
      delta: base.gradient.delta,
      vega: base.gradient.vega,
      rho: base.gradient.rho,
      theta: base.gradient.theta,
      gamma,
      vanna,
      volga
    }
  };
}

// src/backend/emit.ts
var REDUCTIONS = /* @__PURE__ */ new Set(["mean", "sum"]);
function isReduction(op) {
  return REDUCTIONS.has(op);
}
function scalarExpression(op, refs, attrs) {
  switch (op) {
    case "const":
      return `(${attrs.value})`;
    case "add":
      return `(${refs[0]} + ${refs[1]})`;
    case "sub":
      return `(${refs[0]} - ${refs[1]})`;
    case "mul":
      return `(${refs[0]} * ${refs[1]})`;
    case "div":
      return `(${refs[0]} / ${refs[1]})`;
    case "neg":
      return `(-${refs[0]})`;
    case "exp":
      return `Math.exp(${refs[0]})`;
    case "log":
      return `Math.log(${refs[0]})`;
    case "sqrt":
      return `Math.sqrt(${refs[0]})`;
    case "max":
      return `Math.max(${refs[0]}, ${refs[1]})`;
    case "min":
      return `Math.min(${refs[0]}, ${refs[1]})`;
    case "sigmoid":
      return `(1 / (1 + Math.exp(-(${refs[0]}))))`;
    case "softplus":
      return `((${refs[0]}) > 0 ? (${refs[0]}) + Math.log1p(Math.exp(-(${refs[0]}))) : Math.log1p(Math.exp(${refs[0]})))`;
    case "ge":
      return `((${refs[0]}) >= (${refs[1]}) ? 1 : 0)`;
    default:
      throw new Error(`no scalar emitter for op ${op}`);
  }
}
function cudaExpression(op, refs, attrs) {
  switch (op) {
    case "const":
      return `(${attrs.value.toExponential()})`;
    case "add":
      return `(${refs[0]} + ${refs[1]})`;
    case "sub":
      return `(${refs[0]} - ${refs[1]})`;
    case "mul":
      return `(${refs[0]} * ${refs[1]})`;
    case "div":
      return `(${refs[0]} / ${refs[1]})`;
    case "neg":
      return `(-${refs[0]})`;
    case "exp":
      return `exp(${refs[0]})`;
    case "log":
      return `log(${refs[0]})`;
    case "sqrt":
      return `sqrt(${refs[0]})`;
    case "max":
      return `fmax(${refs[0]}, ${refs[1]})`;
    case "min":
      return `fmin(${refs[0]}, ${refs[1]})`;
    case "sigmoid":
      return `(1.0 / (1.0 + exp(-(${refs[0]}))))`;
    case "softplus":
      return `log1p(exp(${refs[0]}))`;
    case "ge":
      return `((${refs[0]}) >= (${refs[1]}) ? 1.0 : 0.0)`;
    default:
      throw new Error(`no cuda emitter for op ${op}`);
  }
}

// src/backend/cpu/codegen.ts
function reference(value, postSet) {
  if (value.producer === null) {
    return value.kind === "batch" ? `x${value.id}[i]` : `s${value.id}`;
  }
  if (postSet.has(value.id) || value.kind === "scalar") return `v${value.id}`;
  return `b${value.id}`;
}
function compileCpuMultiKernel(graph, outputs) {
  for (const output of outputs) {
    if (output.kind !== "scalar") throw new Error("cpu kernel outputs must be scalar");
  }
  const order = topoSort(graph);
  const postSet = /* @__PURE__ */ new Set();
  const scalarPre = [];
  const batch = [];
  const reductions = [];
  const scalarPost = [];
  for (const node of order) {
    if (isReduction(node.op)) {
      reductions.push(node);
      postSet.add(node.result.id);
      continue;
    }
    if (node.result.kind === "batch") {
      for (const operand of node.operands) {
        if (postSet.has(operand.id)) throw new Error("batch op depends on a reduction; unsupported fusion pattern");
      }
      batch.push(node);
      continue;
    }
    const dependsOnReduction = node.operands.some((operand) => postSet.has(operand.id));
    if (dependsOnReduction) {
      scalarPost.push(node);
      postSet.add(node.result.id);
    } else {
      scalarPre.push(node);
    }
  }
  const lines = [];
  for (const input of graph.inputs) {
    if (input.kind === "scalar") lines.push(`const s${input.id} = inputs.get(${input.id})[0];`);
    else lines.push(`const x${input.id} = inputs.get(${input.id});`);
  }
  const emit = (node) => scalarExpression(node.op, node.operands.map((operand) => reference(operand, postSet)), node.attrs);
  for (const node of scalarPre) lines.push(`const v${node.result.id} = ${emit(node)};`);
  for (const node of reductions) lines.push(`let acc${node.result.id} = 0;`);
  lines.push("for (let i = 0; i < N; i += 1) {");
  for (const node of batch) lines.push(`  const b${node.result.id} = ${emit(node)};`);
  for (const node of reductions) lines.push(`  acc${node.result.id} += ${reference(node.operands[0], postSet)};`);
  lines.push("}");
  for (const node of reductions) lines.push(`const v${node.result.id} = acc${node.result.id}${node.op === "mean" ? " / N" : ""};`);
  for (const node of scalarPost) lines.push(`const v${node.result.id} = ${emit(node)};`);
  lines.push(`return [${outputs.map((output) => `v${output.id}`).join(", ")}];`);
  const source = lines.join("\n");
  const factory = new Function("inputs", "N", source);
  return { source, run: (inputs, batchSize) => Float64Array.from(factory(inputs, batchSize)) };
}
function compileCpuKernel(graph) {
  if (graph.output === null) throw new Error("graph has no output");
  const multi = compileCpuMultiKernel(graph, [graph.output]);
  return { source: multi.source, run: (inputs, batchSize) => multi.run(inputs, batchSize)[0] };
}

// src/engines/compiled-pricer.ts
var CompiledEuropeanPricer = class {
  constructor(payoff2, model, paths, seed) {
    __publicField(this, "built");
    __publicField(this, "kernel");
    __publicField(this, "normals");
    __publicField(this, "paths");
    this.built = buildEuropeanGraph(payoff2, model);
    this.kernel = compileCpuKernel(this.built.graph);
    this.normals = standardNormals(paths, seed);
    this.paths = paths;
  }
  reprice(market) {
    const inputs = /* @__PURE__ */ new Map([
      [this.built.inputs.spot.id, new Float64Array([market.spot])],
      [this.built.inputs.strike.id, new Float64Array([market.strike])],
      [this.built.inputs.rate.id, new Float64Array([market.rate])],
      [this.built.inputs.vol.id, new Float64Array([market.vol])],
      [this.built.inputs.maturity.id, new Float64Array([market.maturity])],
      [this.built.inputs.normals.id, this.normals]
    ]);
    return this.kernel.run(inputs, this.paths);
  }
  get source() {
    return this.kernel.source;
  }
};

// data/sobol/joe-kuo.json
var joe_kuo_default = {
  bits: 32,
  dimensions: [
    { d: 2, s: 1, a: 0, m: [1] },
    { d: 3, s: 2, a: 1, m: [1, 3] },
    { d: 4, s: 3, a: 1, m: [1, 3, 1] },
    { d: 5, s: 3, a: 2, m: [1, 1, 1] },
    { d: 6, s: 4, a: 1, m: [1, 1, 3, 3] },
    { d: 7, s: 4, a: 4, m: [1, 3, 5, 13] },
    { d: 8, s: 5, a: 2, m: [1, 1, 5, 5, 17] },
    { d: 9, s: 5, a: 4, m: [1, 1, 5, 5, 5] },
    { d: 10, s: 5, a: 7, m: [1, 1, 7, 11, 13] }
  ]
};

// src/numerics/rng/sobol.ts
var TABLE = joe_kuo_default;
var SCALE = 1 / 2 ** 32;
function trailingZeros(value) {
  let count = 0;
  let n = value;
  while ((n & 1) === 0) {
    count += 1;
    n >>>= 1;
  }
  return count;
}
function buildDirection(entry, bits) {
  const v = new Uint32Array(bits + 1);
  if (entry === null) {
    for (let i = 1; i <= bits; i += 1) v[i] = 1 << 32 - i >>> 0;
    return v;
  }
  const { s, a, m } = entry;
  for (let i = 1; i <= s; i += 1) v[i] = m[i - 1] << 32 - i >>> 0;
  for (let i = s + 1; i <= bits; i += 1) {
    let value = v[i - s] ^ v[i - s] >>> s;
    for (let k = 1; k <= s - 1; k += 1) value ^= (a >>> s - 1 - k & 1) * v[i - k];
    v[i] = value >>> 0;
  }
  return v;
}
var Sobol = class {
  constructor(dimension) {
    __publicField(this, "dimension");
    __publicField(this, "directions");
    __publicField(this, "state");
    __publicField(this, "count", 0);
    if (dimension > TABLE.dimensions.length + 1) throw new Error(`Sobol dimension ${dimension} exceeds available direction numbers`);
    this.dimension = dimension;
    this.directions = [];
    for (let d = 0; d < dimension; d += 1) {
      const entry = d === 0 ? null : TABLE.dimensions[d - 1];
      this.directions.push(buildDirection(entry, TABLE.bits));
    }
    this.state = new Uint32Array(dimension);
  }
  next() {
    this.count += 1;
    const column2 = trailingZeros(this.count) + 1;
    const point = new Float64Array(this.dimension);
    for (let d = 0; d < this.dimension; d += 1) {
      this.state[d] = (this.state[d] ^ this.directions[d][column2]) >>> 0;
      point[d] = this.state[d] * SCALE;
    }
    return point;
  }
};

// src/engines/european-samples.ts
function terminal(market, z) {
  const drift = (market.rate - 0.5 * market.vol * market.vol) * market.maturity;
  const diffusion = market.vol * Math.sqrt(market.maturity) * z;
  return market.spot * Math.exp(drift + diffusion);
}
function payoff(market, spotAtMaturity) {
  return Math.exp(-market.rate * market.maturity) * Math.max(spotAtMaturity - market.strike, 0);
}
function plainPayoffs(market) {
  const generator = new MersenneTwister(market.seed);
  const out = new Float64Array(market.paths);
  for (let i = 0; i < market.paths; i += 1) out[i] = payoff(market, terminal(market, inverseNormalCdf(generator.nextDouble())));
  return out;
}

// src/engines/parallel-mc.ts
var SHARD_SEED_STRIDE = 2654435761;
function pooledEuropeanCall(market, shards) {
  const states = [];
  for (let shard = 0; shard < shards; shard += 1) {
    const payoffs = plainPayoffs({ ...market, seed: market.seed + shard * SHARD_SEED_STRIDE >>> 0 });
    const welford = new Welford();
    for (let i = 0; i < payoffs.length; i += 1) welford.push(payoffs[i]);
    states.push({ count: welford.count, mean: welford.mean, sumSquaredDeviations: welford.sumSquaredDeviations });
  }
  const combined = combineMoments(states);
  return { price: combined.mean, standardError: standardErrorOf(combined), paths: combined.count };
}

// src/engines/path-engine.ts
function buildGbmPath(config) {
  registerBuiltinOps();
  const graph = new Graph();
  const spot2 = graph.input("scalar", "spot");
  const strike = graph.input("scalar", "strike");
  const rate = graph.input("scalar", "rate");
  const vol = graph.input("scalar", "vol");
  const maturity = graph.input("scalar", "maturity");
  const steps = config.steps;
  const dt = graph.div(maturity, graph.constant(steps));
  const half = graph.constant(0.5);
  const driftStep = graph.mul(graph.sub(rate, graph.mul(graph.mul(half, vol), vol)), dt);
  const volStep = graph.mul(vol, graph.sqrt(dt));
  const normals = [];
  const logFixings = [];
  const fixings = [];
  let logState = graph.log(spot2);
  for (let i = 0; i < steps; i += 1) {
    const z = graph.input("batch", `z_${i}`);
    normals.push(z);
    logState = graph.add(graph.add(logState, driftStep), graph.mul(volStep, z));
    logFixings.push(logState);
    fixings.push(graph.exp(logState));
  }
  return { graph, inputs: { spot: spot2, strike, rate, vol, maturity }, normals, logFixings, fixings, terminal: fixings[fixings.length - 1] };
}
function pathBindings(path, market) {
  const bindings = /* @__PURE__ */ new Map([
    [path.inputs.spot.id, new Float64Array([market.spot])],
    [path.inputs.strike.id, new Float64Array([market.strike])],
    [path.inputs.rate.id, new Float64Array([market.rate])],
    [path.inputs.vol.id, new Float64Array([market.vol])],
    [path.inputs.maturity.id, new Float64Array([market.maturity])]
  ]);
  for (let i = 0; i < path.normals.length; i += 1) {
    bindings.set(path.normals[i].id, standardNormals(market.paths, market.seed + i * 2654435761));
  }
  return bindings;
}

// src/engines/mc-core.ts
function requireValue2(values, id) {
  const value = values.get(id);
  if (value === void 0) throw new Error(`missing value ${id}`);
  return value;
}
function runMonteCarlo(graph, bindings, batchSize, price, sample, riskFactors) {
  const forward = evaluate(graph, bindings, batchSize);
  const priceValue = requireValue2(forward, price.id)[0];
  const samples = requireValue2(forward, sample.id);
  const welford = new Welford();
  for (let i = 0; i < samples.length; i += 1) welford.push(samples[i]);
  const adjoints = reverse(graph, forward, /* @__PURE__ */ new Map([[price.id, new Float64Array([1])]]));
  const gradients = /* @__PURE__ */ new Map();
  for (const factor of riskFactors) {
    const grad = adjoints.get(factor.id);
    gradients.set(factor.id, grad === void 0 ? 0 : grad[0]);
  }
  return { price: priceValue, standardError: welford.standardError, gradients };
}

// src/engines/exotics.ts
function present(output, spotId, volId) {
  return {
    price: output.price,
    standardError: output.standardError,
    delta: output.gradients.get(spotId) ?? 0,
    vega: output.gradients.get(volId) ?? 0
  };
}
function priceGeometricAsianCall(market, steps) {
  const path = buildGbmPath({ steps });
  const g = path.graph;
  let logSum = path.logFixings[0];
  for (let i = 1; i < path.logFixings.length; i += 1) logSum = g.add(logSum, path.logFixings[i]);
  const average = g.exp(g.mul(logSum, g.constant(1 / steps)));
  const payoff2 = g.max(g.sub(average, path.inputs.strike), g.constant(0));
  const discount = g.exp(g.neg(g.mul(path.inputs.rate, path.inputs.maturity)));
  const discounted = g.mul(discount, payoff2);
  const price = g.mean(discounted);
  g.output = price;
  const output = runMonteCarlo(g, pathBindings(path, market), market.paths, price, discounted, [path.inputs.spot, path.inputs.vol]);
  return present(output, path.inputs.spot.id, path.inputs.vol.id);
}
function priceArithmeticAsianCall(market, steps) {
  const path = buildGbmPath({ steps });
  const g = path.graph;
  let sum = path.fixings[0];
  for (let i = 1; i < path.fixings.length; i += 1) sum = g.add(sum, path.fixings[i]);
  const average = g.mul(sum, g.constant(1 / steps));
  const payoff2 = g.max(g.sub(average, path.inputs.strike), g.constant(0));
  const discount = g.exp(g.neg(g.mul(path.inputs.rate, path.inputs.maturity)));
  const discounted = g.mul(discount, payoff2);
  const price = g.mean(discounted);
  g.output = price;
  const output = runMonteCarlo(g, pathBindings(path, market), market.paths, price, discounted, [path.inputs.spot, path.inputs.vol]);
  return present(output, path.inputs.spot.id, path.inputs.vol.id);
}
function priceUpAndOutCall(market, steps, barrier, smoothing) {
  const path = buildGbmPath({ steps });
  const g = path.graph;
  const barrierValue = g.constant(barrier);
  const width = g.constant(smoothing);
  let survival2 = g.sigmoid(g.div(g.sub(barrierValue, path.fixings[0]), width));
  for (let i = 1; i < path.fixings.length; i += 1) {
    survival2 = g.mul(survival2, g.sigmoid(g.div(g.sub(barrierValue, path.fixings[i]), width)));
  }
  const intrinsic3 = g.max(g.sub(path.terminal, path.inputs.strike), g.constant(0));
  const payoff2 = g.mul(survival2, intrinsic3);
  const discount = g.exp(g.neg(g.mul(path.inputs.rate, path.inputs.maturity)));
  const discounted = g.mul(discount, payoff2);
  const price = g.mean(discounted);
  g.output = price;
  const output = runMonteCarlo(g, pathBindings(path, market), market.paths, price, discounted, [path.inputs.spot, path.inputs.vol]);
  return present(output, path.inputs.spot.id, path.inputs.vol.id);
}

// src/numerics/linalg/solve.ts
function solveLinearSystem(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let column2 = 0; column2 < n; column2 += 1) {
    let pivot = column2;
    for (let row = column2 + 1; row < n; row += 1) {
      if (Math.abs(a[row][column2]) > Math.abs(a[pivot][column2])) pivot = row;
    }
    const temp = a[column2];
    a[column2] = a[pivot];
    a[pivot] = temp;
    const diagonal = a[column2][column2];
    if (Math.abs(diagonal) < 1e-14) continue;
    for (let row = 0; row < n; row += 1) {
      if (row === column2) continue;
      const factor = a[row][column2] / diagonal;
      for (let k = column2; k <= n; k += 1) a[row][k] -= factor * a[column2][k];
    }
  }
  const solution = new Array(n);
  for (let i = 0; i < n; i += 1) {
    const diagonal = a[i][i];
    solution[i] = Math.abs(diagonal) < 1e-14 ? 0 : a[i][n] / diagonal;
  }
  return solution;
}

// src/engines/longstaff-schwartz.ts
function intrinsic(spot2, strike, isCall) {
  return isCall ? Math.max(spot2 - strike, 0) : Math.max(strike - spot2, 0);
}
function simulatePaths(spec) {
  const steps = spec.exerciseDates;
  const dt = spec.maturity / steps;
  const drift = (spec.rate - 0.5 * spec.vol * spec.vol) * dt;
  const diffusion = spec.vol * Math.sqrt(dt);
  const generator = new MersenneTwister(spec.seed);
  const logState = new Float64Array(spec.paths).fill(Math.log(spec.spot));
  const fixings = [];
  for (let step = 0; step < steps; step += 1) {
    const level = new Float64Array(spec.paths);
    for (let p = 0; p < spec.paths; p += 1) {
      logState[p] += drift + diffusion * inverseNormalCdf(generator.nextDouble());
      level[p] = Math.exp(logState[p]);
    }
    fixings.push(level);
  }
  return fixings;
}
function priceBermudanLsm(spec) {
  const steps = spec.exerciseDates;
  const dt = spec.maturity / steps;
  const stepDiscount = Math.exp(-spec.rate * dt);
  const fixings = simulatePaths(spec);
  const value = new Float64Array(spec.paths);
  const last = fixings[steps - 1];
  for (let p = 0; p < spec.paths; p += 1) value[p] = intrinsic(last[p], spec.strike, spec.isCall);
  for (let step = steps - 2; step >= 0; step -= 1) {
    for (let p = 0; p < spec.paths; p += 1) value[p] *= stepDiscount;
    const level = fixings[step];
    const ata = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
    ];
    const aty = [0, 0, 0];
    const inMoney = [];
    for (let p = 0; p < spec.paths; p += 1) {
      if (intrinsic(level[p], spec.strike, spec.isCall) <= 0) continue;
      inMoney.push(p);
      const x = level[p] / spec.strike;
      const basis = [1, x, x * x];
      for (let i = 0; i < 3; i += 1) {
        for (let j = 0; j < 3; j += 1) ata[i][j] += basis[i] * basis[j];
        aty[i] += basis[i] * value[p];
      }
    }
    if (inMoney.length < 3) continue;
    const coefficients = solveLinearSystem(ata, aty);
    for (const p of inMoney) {
      const x = level[p] / spec.strike;
      const continuation2 = coefficients[0] + coefficients[1] * x + coefficients[2] * x * x;
      const exercise = intrinsic(level[p], spec.strike, spec.isCall);
      if (exercise > continuation2) value[p] = exercise;
    }
  }
  let total = 0;
  for (let p = 0; p < spec.paths; p += 1) total += value[p] * stepDiscount;
  return total / spec.paths;
}

// src/engines/pde/thomas.ts
function solveTridiagonal(lower, diag, upper, rhs) {
  const n = diag.length;
  const cPrime = new Float64Array(n);
  const dPrime = new Float64Array(n);
  const solution = new Float64Array(n);
  cPrime[0] = upper[0] / diag[0];
  dPrime[0] = rhs[0] / diag[0];
  for (let i = 1; i < n; i += 1) {
    const denominator = diag[i] - lower[i] * cPrime[i - 1];
    cPrime[i] = upper[i] / denominator;
    dPrime[i] = (rhs[i] - lower[i] * dPrime[i - 1]) / denominator;
  }
  solution[n - 1] = dPrime[n - 1];
  for (let i = n - 2; i >= 0; i -= 1) solution[i] = dPrime[i] - cPrime[i] * solution[i + 1];
  return solution;
}

// src/engines/pde-engine.ts
function payoffAt(logSpot, strike, isCall) {
  const spot2 = Math.exp(logSpot);
  return isCall ? Math.max(spot2 - strike, 0) : Math.max(strike - spot2, 0);
}
function boundary(logSpot, spec, tau) {
  const spot2 = Math.exp(logSpot);
  const discountedStrike = spec.strike * Math.exp(-spec.rate * tau);
  if (spec.isCall) return Math.max(spot2 - discountedStrike, 0);
  return Math.max(discountedStrike - spot2, 0);
}
function pricePde(spec) {
  const m = spec.spaceSteps;
  const center = Math.log(spec.spot);
  const halfWidth = spec.widthStdDev * spec.vol * Math.sqrt(spec.maturity);
  const dx = 2 * halfWidth / (m - 1);
  const dt = spec.maturity / spec.timeSteps;
  const x = new Float64Array(m);
  for (let i = 0; i < m; i += 1) x[i] = center - halfWidth + i * dx;
  let values = new Float64Array(m);
  for (let i = 0; i < m; i += 1) values[i] = payoffAt(x[i], spec.strike, spec.isCall);
  const interior = m - 2;
  const subL = new Float64Array(m);
  const diagL = new Float64Array(m);
  const supL = new Float64Array(m);
  for (let i = 1; i <= interior; i += 1) {
    const nodeVol = spec.localVol === void 0 ? spec.vol : spec.localVol(Math.exp(x[i]));
    const variance = nodeVol * nodeVol;
    const drift = spec.rate - 0.5 * variance;
    subL[i] = 0.5 * variance / (dx * dx) - drift / (2 * dx);
    diagL[i] = -variance / (dx * dx) - spec.rate;
    supL[i] = 0.5 * variance / (dx * dx) + drift / (2 * dx);
  }
  const lower = new Float64Array(interior);
  const diag = new Float64Array(interior);
  const upper = new Float64Array(interior);
  const rhs = new Float64Array(interior);
  for (let step = 0; step < spec.timeSteps; step += 1) {
    const theta = step < spec.rannacherSteps ? 1 : 0.5;
    const tauNext = (step + 1) * dt;
    const leftNext = boundary(x[0], spec, tauNext);
    const rightNext = boundary(x[m - 1], spec, tauNext);
    for (let i = 1; i <= interior; i += 1) {
      lower[i - 1] = -theta * dt * subL[i];
      diag[i - 1] = 1 - theta * dt * diagL[i];
      upper[i - 1] = -theta * dt * supL[i];
      rhs[i - 1] = (1 - theta) * dt * subL[i] * values[i - 1] + (1 + (1 - theta) * dt * diagL[i]) * values[i] + (1 - theta) * dt * supL[i] * values[i + 1];
    }
    rhs[0] += theta * dt * subL[1] * leftNext;
    rhs[interior - 1] += theta * dt * supL[interior] * rightNext;
    const solved = solveTridiagonal(lower, diag, upper, rhs);
    const next = new Float64Array(m);
    next[0] = leftNext;
    next[m - 1] = rightNext;
    for (let i = 1; i <= interior; i += 1) next[i] = solved[i - 1];
    if (spec.american) {
      for (let i = 0; i < m; i += 1) {
        const intrinsic3 = payoffAt(x[i], spec.strike, spec.isCall);
        if (intrinsic3 > next[i]) next[i] = intrinsic3;
      }
    }
    values = next;
  }
  const j = Math.round((center - (center - halfWidth)) / dx);
  const delta = (values[j + 1] - values[j - 1]) / (2 * dx) / spec.spot;
  const valueXX = (values[j + 1] - 2 * values[j] + values[j - 1]) / (dx * dx);
  const valueX = (values[j + 1] - values[j - 1]) / (2 * dx);
  const gamma = (valueXX - valueX) / (spec.spot * spec.spot);
  return { price: values[j], delta, gamma };
}

// src/numerics/analytic/black-scholes.ts
var INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);
var HART_NUMERATOR = [0.0352624965998911, 0.700383064443688, 6.37396220353165, 33.912866078383, 112.079291497871, 221.213596169931, 220.206867912376];
var HART_DENOMINATOR = [0.0883883476483184, 1.75566716318264, 16.064177579207, 86.7807322029461, 296.564248779674, 637.333633378831, 793.826512519948, 440.413735824752];
function normalPdf(x) {
  return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}
function normalCdf(x) {
  const absX = Math.abs(x);
  if (absX > 37) return x > 0 ? 1 : 0;
  const exponential = Math.exp(-0.5 * absX * absX);
  let lowerTail;
  if (absX < 7.07106781186547) {
    let numerator = HART_NUMERATOR[0];
    for (let i = 1; i < HART_NUMERATOR.length; i += 1) numerator = numerator * absX + HART_NUMERATOR[i];
    let denominator = HART_DENOMINATOR[0];
    for (let i = 1; i < HART_DENOMINATOR.length; i += 1) denominator = denominator * absX + HART_DENOMINATOR[i];
    lowerTail = exponential * numerator / denominator;
  } else {
    let fraction = absX + 0.65;
    fraction = absX + 4 / fraction;
    fraction = absX + 3 / fraction;
    fraction = absX + 2 / fraction;
    fraction = absX + 1 / fraction;
    lowerTail = exponential / fraction / 2.506628274631;
  }
  return x > 0 ? 1 - lowerTail : lowerTail;
}
function blackScholes(input) {
  const { spot: spot2, strike, rate, vol, maturity, isCall } = input;
  const sqrtT = Math.sqrt(maturity);
  const d1 = (Math.log(spot2 / strike) + (rate + 0.5 * vol * vol) * maturity) / (vol * sqrtT);
  const d2 = d1 - vol * sqrtT;
  const discount = Math.exp(-rate * maturity);
  const pdfD1 = normalPdf(d1);
  const vega = spot2 * pdfD1 * sqrtT;
  const gamma = pdfD1 / (spot2 * vol * sqrtT);
  const vanna = -pdfD1 * d2 / vol;
  const volga = vega * d1 * d2 / vol;
  const sign = isCall ? 1 : -1;
  const delta = isCall ? normalCdf(d1) : normalCdf(d1) - 1;
  const theta = -spot2 * pdfD1 * vol / (2 * sqrtT) - sign * rate * strike * discount * normalCdf(sign * d2);
  const rho = sign * strike * maturity * discount * normalCdf(sign * d2);
  const price = isCall ? spot2 * normalCdf(d1) - strike * discount * normalCdf(d2) : strike * discount * normalCdf(-d2) - spot2 * normalCdf(-d1);
  return { price, delta, vega, gamma, vanna, volga, rho, theta };
}

// src/engines/calibration.ts
var DEFAULT_OPTIONS = { maxIterations: 200, tolerance: 1e-12, initialDamping: 1e-3 };
function sumSquares(values) {
  let total = 0;
  for (const value of values) total += value * value;
  return total;
}
function jacobian(residual, x, base) {
  const m = base.length;
  const n = x.length;
  const result = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let j = 0; j < n; j += 1) {
    const step = 1e-6 * Math.max(1, Math.abs(x[j]));
    const up = [...x];
    const down = [...x];
    up[j] += step;
    down[j] -= step;
    const fUp = residual(up);
    const fDown = residual(down);
    for (let i = 0; i < m; i += 1) result[i][j] = (fUp[i] - fDown[i]) / (2 * step);
  }
  return result;
}
function levenbergMarquardt(residual, initial, options = {}) {
  const config = { ...DEFAULT_OPTIONS, ...options };
  const n = initial.length;
  let x = [...initial];
  let residuals = residual(x);
  let cost = sumSquares(residuals);
  let damping = config.initialDamping;
  let iterations = 0;
  for (; iterations < config.maxIterations; iterations += 1) {
    const j = jacobian(residual, x, residuals);
    const m = residuals.length;
    const jtj = Array.from({ length: n }, () => new Array(n).fill(0));
    const jtr = new Array(n).fill(0);
    for (let a = 0; a < n; a += 1) {
      for (let b = 0; b < n; b += 1) {
        let sum = 0;
        for (let i = 0; i < m; i += 1) sum += j[i][a] * j[i][b];
        jtj[a][b] = sum;
      }
      let sumR = 0;
      for (let i = 0; i < m; i += 1) sumR += j[i][a] * residuals[i];
      jtr[a] = sumR;
    }
    let improved = false;
    for (let attempt = 0; attempt < 30; attempt += 1) {
      const system = jtj.map((row, i) => row.map((value, k) => i === k ? value + damping * (value + 1e-12) : value));
      const delta = solveLinearSystem(system, jtr.map((value) => -value));
      const candidate = x.map((value, i) => value + delta[i]);
      const candidateResiduals = residual(candidate);
      const candidateCost = sumSquares(candidateResiduals);
      if (candidateCost < cost) {
        x = candidate;
        const previousCost = cost;
        residuals = candidateResiduals;
        cost = candidateCost;
        damping *= 0.7;
        improved = true;
        if (previousCost - candidateCost < config.tolerance) iterations += 1;
        break;
      }
      damping *= 2.5;
    }
    if (!improved) break;
    if (cost < config.tolerance) break;
  }
  return { parameters: x, residualNorm: Math.sqrt(cost), iterations };
}
function impliedVolatility(input) {
  let vol = 0.2;
  for (let iteration = 0; iteration < 100; iteration += 1) {
    const valued = blackScholes({ spot: input.spot, strike: input.strike, rate: input.rate, vol, maturity: input.maturity, isCall: input.isCall });
    const diff = valued.price - input.price;
    if (Math.abs(diff) < 1e-12) break;
    if (valued.vega < 1e-12) break;
    vol -= diff / valued.vega;
    if (vol < 1e-6) vol = 1e-6;
  }
  return vol;
}

// src/engines/sabr-calibration.ts
function calibrateSabr(input) {
  const residual = (p) => input.quotes.map(
    (quote) => sabrImpliedVol({ forward: input.forward, strike: quote.strike, maturity: input.maturity, alpha: p[0], beta: input.beta, rho: p[1], volOfVol: p[2] }) - quote.vol
  );
  const result = levenbergMarquardt(residual, [input.initialAlpha, input.initialRho, input.initialVolOfVol]);
  return { alpha: result.parameters[0], rho: result.parameters[1], volOfVol: result.parameters[2], residualNorm: result.residualNorm };
}

// src/risk/calibrate-vol-aad.ts
function calibrateVolToPrice(spec, targetPrice) {
  let vol = 0.2;
  let iterations = 0;
  let priceError = Number.POSITIVE_INFINITY;
  for (; iterations < 50; iterations += 1) {
    const valued = priceEuropean({ ...spec, vol });
    priceError = valued.price - targetPrice;
    if (Math.abs(priceError) < 1e-7) break;
    if (Math.abs(valued.greeks.vega) < 1e-10) break;
    vol -= priceError / valued.greeks.vega;
    if (vol < 1e-4) vol = 1e-4;
  }
  return { vol, iterations, priceError };
}

// src/risk/xva.ts
function computeCva(spec) {
  const steps = spec.exposureDates;
  const dt = spec.maturity / steps;
  const drift = (spec.rate - 0.5 * spec.vol * spec.vol) * dt;
  const diffusion = spec.vol * Math.sqrt(dt);
  const generator = new MersenneTwister(spec.seed);
  const exposureSum = new Float64Array(steps);
  const logState = new Float64Array(spec.paths).fill(Math.log(spec.spot));
  for (let step = 0; step < steps; step += 1) {
    const time = (step + 1) * dt;
    const remaining = spec.maturity - time;
    const stepDiscount = Math.exp(-spec.rate * time);
    for (let p = 0; p < spec.paths; p += 1) {
      logState[p] += drift + diffusion * inverseNormalCdf(generator.nextDouble());
      const spotAtTime = Math.exp(logState[p]);
      const value = remaining > 1e-10 ? blackScholes({ spot: spotAtTime, strike: spec.strike, rate: spec.rate, vol: spec.vol, maturity: remaining, isCall: true }).price : Math.max(spotAtTime - spec.strike, 0);
      exposureSum[step] += stepDiscount * Math.max(value, 0);
    }
  }
  const expectedExposure = [];
  for (let step = 0; step < steps; step += 1) expectedExposure.push(exposureSum[step] / spec.paths);
  let cva = 0;
  for (let step = 0; step < steps; step += 1) {
    const start = step * dt;
    const end = (step + 1) * dt;
    const defaultProbability2 = Math.exp(-spec.hazardRate * start) - Math.exp(-spec.hazardRate * end);
    cva += expectedExposure[step] * defaultProbability2;
  }
  cva *= 1 - spec.recovery;
  return { cva, expectedExposure };
}

// src/numerics/analytic/geometric-asian.ts
function geometricAsianCall(input) {
  const { spot: spot2, strike, rate, vol, maturity, fixings } = input;
  const dt = maturity / fixings;
  let timeSum = 0;
  let covariance = 0;
  for (let k = 0; k < fixings; k += 1) {
    const tk = (k + 1) * dt;
    timeSum += tk;
    covariance += tk * (2 * (fixings - k) - 1);
  }
  const muG = Math.log(spot2) + (rate - 0.5 * vol * vol) / fixings * timeSum;
  const varG = vol * vol / (fixings * fixings) * covariance;
  const sigmaG = Math.sqrt(varG);
  const d1 = (muG - Math.log(strike) + varG) / sigmaG;
  const d2 = d1 - sigmaG;
  const expectation = Math.exp(muG + 0.5 * varG) * normalCdf(d1) - strike * normalCdf(d2);
  return Math.exp(-rate * maturity) * expectation;
}

// src/numerics/rng/brownian-bridge.ts
var BrownianBridge = class {
  constructor(steps) {
    __publicField(this, "steps");
    __publicField(this, "leftIndex");
    __publicField(this, "rightIndex");
    __publicField(this, "bridgeIndex");
    __publicField(this, "leftWeight");
    __publicField(this, "rightWeight");
    __publicField(this, "stdDev");
    this.steps = steps;
    this.leftIndex = new Int32Array(steps);
    this.rightIndex = new Int32Array(steps);
    this.bridgeIndex = new Int32Array(steps);
    this.leftWeight = new Float64Array(steps);
    this.rightWeight = new Float64Array(steps);
    this.stdDev = new Float64Array(steps);
    const map = new Int32Array(steps);
    map[steps - 1] = 1;
    this.bridgeIndex[0] = steps - 1;
    this.stdDev[0] = Math.sqrt(steps);
    this.leftWeight[0] = 0;
    this.rightWeight[0] = 0;
    let j = 0;
    for (let i = 1; i < steps; i += 1) {
      while (map[j] !== 0) j += 1;
      let k = j;
      while (map[k] === 0) k += 1;
      const l = j + (k - 1 - j >> 1);
      map[l] = i;
      this.bridgeIndex[i] = l;
      this.leftIndex[i] = j;
      this.rightIndex[i] = k;
      this.leftWeight[i] = (k - l) / (k + 1 - j);
      this.rightWeight[i] = (l + 1 - j) / (k + 1 - j);
      this.stdDev[i] = Math.sqrt((l + 1 - j) * (k - l) / (k + 1 - j));
      j = k + 1;
      if (j >= steps) j = 0;
    }
  }
  buildPath(normals) {
    const path = new Float64Array(this.steps);
    path[this.bridgeIndex[0]] = this.stdDev[0] * normals[0];
    for (let i = 1; i < this.steps; i += 1) {
      const l = this.leftIndex[i];
      const r = this.rightIndex[i];
      const b = this.bridgeIndex[i];
      const left = l === 0 ? 0 : path[l - 1];
      path[b] = this.leftWeight[i] * left + this.rightWeight[i] * path[r] + this.stdDev[i] * normals[i];
    }
    return path;
  }
  increments(normals) {
    const path = this.buildPath(normals);
    const result = new Float64Array(this.steps);
    result[0] = path[0];
    for (let i = 1; i < this.steps; i += 1) result[i] = path[i] - path[i - 1];
    return result;
  }
};

// src/numerics/variance-reduction/estimators.ts
function plainEstimate(samples) {
  const welford = new Welford();
  for (let i = 0; i < samples.length; i += 1) welford.push(samples[i]);
  return { mean: welford.mean, standardError: welford.standardError };
}
function antitheticEstimate(samples, antithetic) {
  const welford = new Welford();
  for (let i = 0; i < samples.length; i += 1) welford.push(0.5 * (samples[i] + antithetic[i]));
  return { mean: welford.mean, standardError: welford.standardError };
}
function controlVariateEstimate(samples, controls, controlMean) {
  const n = samples.length;
  let sampleMean = 0;
  let controlSampleMean = 0;
  for (let i = 0; i < n; i += 1) {
    sampleMean += samples[i];
    controlSampleMean += controls[i];
  }
  sampleMean /= n;
  controlSampleMean /= n;
  let covariance = 0;
  let controlVariance = 0;
  for (let i = 0; i < n; i += 1) {
    const dc = controls[i] - controlSampleMean;
    covariance += (samples[i] - sampleMean) * dc;
    controlVariance += dc * dc;
  }
  const beta = controlVariance > 0 ? covariance / controlVariance : 0;
  const welford = new Welford();
  for (let i = 0; i < n; i += 1) welford.push(samples[i] - beta * (controls[i] - controlMean));
  return { mean: welford.mean, standardError: welford.standardError };
}

// src/aad/jvp.ts
function requireValue3(values, id) {
  const value = values.get(id);
  if (value === void 0) throw new Error(`missing value ${id}`);
  return value;
}
function forwardMode(graph, bindings, seeds, batchSize) {
  const primals = evaluate(graph, bindings, batchSize);
  const tangents = /* @__PURE__ */ new Map();
  for (const input of graph.inputs) {
    const seed = seeds.get(input.id);
    const primal = requireValue3(primals, input.id);
    tangents.set(input.id, seed ?? new Float64Array(primal.length));
  }
  for (const node of topoSort(graph)) {
    const operandPrimals = node.operands.map((operand) => requireValue3(primals, operand.id));
    const operandTangents = node.operands.map((operand) => tangents.get(operand.id) ?? new Float64Array(requireValue3(primals, operand.id).length));
    const outLen = node.result.kind === "batch" ? batchSize : 1;
    const tangent = registry.get(node.op).jvpFn(operandPrimals, operandTangents, requireValue3(primals, node.result.id), outLen, node.attrs);
    tangents.set(node.result.id, tangent);
  }
  return { primals, tangents };
}

// src/backend/cuda/codegen.ts
function reference2(value) {
  if (value.producer === null) {
    return value.kind === "batch" ? `x${value.id}[i]` : `s${value.id}`;
  }
  return value.kind === "batch" ? `b${value.id}` : `v${value.id}`;
}
function compileCudaKernel(graph, name = "price_kernel") {
  if (graph.output === null || graph.output.kind !== "scalar") throw new Error("cuda kernel output must be a scalar reduction");
  const order = topoSort(graph);
  const scalarInputs = graph.inputs.filter((value) => value.kind === "scalar").map((value) => value.id);
  const batchInputs = graph.inputs.filter((value) => value.kind === "batch").map((value) => value.id);
  const signatureParts = [
    ...scalarInputs.map((id) => `const double s${id}`),
    ...batchInputs.map((id) => `const double* __restrict__ x${id}`),
    "double* __restrict__ result",
    "const int N"
  ];
  const body = [];
  const reductions = [];
  for (const node of order) {
    if (isReduction(node.op)) reductions.push(node);
    else if (node.result.kind === "scalar") body.push(`  const double v${node.result.id} = ${cudaExpression(node.op, node.operands.map(reference2), node.attrs)};`);
  }
  const loopBody = [];
  for (const node of order) {
    if (node.result.kind === "batch" && !isReduction(node.op)) {
      loopBody.push(`    const double b${node.result.id} = ${cudaExpression(node.op, node.operands.map(reference2), node.attrs)};`);
    }
  }
  const reduction = reductions[reductions.length - 1];
  const reductionOperand = reference2(reduction.operands[0]);
  const lines = [
    `extern "C" __global__ void ${name}(${signatureParts.join(", ")}) {`,
    "  const int i = blockIdx.x * blockDim.x + threadIdx.x;",
    "  if (i >= N) return;",
    ...body,
    ...loopBody,
    `  atomicAdd(result, ${reductionOperand}${reduction.op === "mean" ? " / (double) N" : ""});`,
    "}"
  ];
  return { source: lines.join("\n"), name, batchInputs, scalarInputs };
}

// src/runtime/cuda/driver.browser.ts
var UNAVAILABLE = "CUDA is unavailable in the browser build";
function launchReductionKernel(_launch, _scalarValues, _batchData, _batchSize, _targetArchitecture) {
  throw new Error(UNAVAILABLE);
}

// src/backend/codegen-registry.ts
var CodegenRegistry = class {
  constructor() {
    __publicField(this, "backends", /* @__PURE__ */ new Map());
  }
  register(kind, codegen) {
    if (this.backends.has(kind)) throw new Error(`duplicate codegen ${kind}`);
    this.backends.set(kind, codegen);
  }
  get(kind) {
    const codegen = this.backends.get(kind);
    if (codegen === void 0) throw new Error(`no codegen for ${kind}`);
    return codegen;
  }
};
var codegenRegistry = new CodegenRegistry();
var registered3 = false;
function registerBuiltinBackends() {
  if (registered3) return;
  codegenRegistry.register("cpu", (graph) => {
    const kernel = compileCpuKernel(graph);
    return { source: kernel.source, run: kernel.run };
  });
  codegenRegistry.register("cuda", (graph) => {
    const kernel = compileCudaKernel(graph);
    return {
      source: kernel.source,
      run: (inputs, batchSize) => {
        const scalarValues = /* @__PURE__ */ new Map();
        for (const id of kernel.scalarInputs) {
          const value = inputs.get(id);
          if (value === void 0) throw new Error(`missing scalar input ${id}`);
          scalarValues.set(id, value[0]);
        }
        const batchData = /* @__PURE__ */ new Map();
        for (const id of kernel.batchInputs) {
          const value = inputs.get(id);
          if (value === void 0) throw new Error(`missing batch input ${id}`);
          batchData.set(id, value);
        }
        return launchReductionKernel(kernel, scalarValues, batchData, batchSize);
      }
    };
  });
  registered3 = true;
}
function compileFor(target, graph) {
  registerBuiltinBackends();
  return codegenRegistry.get(target.kind)(graph);
}

// src/backend/target.ts
var cpuTarget = {
  name: "cpu",
  kind: "cpu",
  vectorWidth: 4,
  fp64: "full",
  threadModel: "simd"
};
var cudaTarget = {
  name: "cuda",
  kind: "cuda",
  vectorWidth: 32,
  fp64: "throttled",
  threadModel: "warp"
};

// src/backend/schedule/autotuner.ts
function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? 0.5 * (sorted[middle - 1] + sorted[middle]) : sorted[middle];
}
function autotune(candidates, repeats) {
  let best = null;
  const timings = [];
  for (const candidate of candidates) {
    const samples = [];
    let lastValue = 0;
    for (let r = 0; r < repeats; r += 1) {
      const start = performance.now();
      lastValue = candidate.run();
      samples.push(performance.now() - start);
    }
    const milliseconds = median(samples);
    timings.push({ name: candidate.name, milliseconds });
    if (best === null || milliseconds < best.milliseconds) {
      best = { name: candidate.name, value: lastValue, milliseconds, timings };
    }
  }
  if (best === null) throw new Error("no autotune candidates");
  return { ...best, timings };
}

// src/cli/index.ts
function parseArguments(argv) {
  const args = /* @__PURE__ */ new Map();
  for (let i = 0; i < argv.length - 1; i += 1) {
    const token = argv[i];
    if (token.startsWith("--")) args.set(token.slice(2), argv[i + 1]);
  }
  return args;
}
function number(args, key, fallback) {
  const value = args.get(key);
  return value === void 0 ? fallback : Number(value);
}
function runCli(argv) {
  const args = parseArguments(argv);
  const spec = {
    payoff: args.get("payoff") ?? "max(spot - strike, 0)",
    spot: number(args, "spot", 100),
    strike: number(args, "strike", 100),
    rate: number(args, "rate", 0.03),
    vol: number(args, "vol", 0.2),
    maturity: number(args, "maturity", 1),
    paths: number(args, "paths", 2e5),
    seed: number(args, "seed", 12345),
    model: args.get("model") ?? "gbm"
  };
  const result = priceEuropean(spec);
  const lines = [
    `payoff       ${spec.payoff}`,
    `price        ${result.price.toFixed(6)}`,
    `stderr       ${result.standardError.toFixed(6)}`,
    `delta        ${result.greeks.delta.toFixed(6)}`,
    `gamma        ${result.greeks.gamma.toFixed(6)}`,
    `vega         ${result.greeks.vega.toFixed(6)}`,
    `rho          ${result.greeks.rho.toFixed(6)}`,
    `theta        ${result.greeks.theta.toFixed(6)}`
  ];
  return lines.join("\n");
}

// src/passes/pass.ts
var PassManager = class {
  constructor() {
    __publicField(this, "passes", []);
  }
  add(pass) {
    this.passes.push(pass);
    return this;
  }
  run(graph) {
    let changed = false;
    for (const pass of this.passes) {
      if (pass.run(graph)) changed = true;
    }
    return changed;
  }
};
var FixedPointGroup = class {
  constructor(name, passes, maxIterations = 16) {
    __publicField(this, "name");
    __publicField(this, "passes");
    __publicField(this, "maxIterations");
    this.name = name;
    this.passes = passes;
    this.maxIterations = maxIterations;
  }
  run(graph) {
    let everChanged = false;
    for (let iteration = 0; iteration < this.maxIterations; iteration += 1) {
      let changed = false;
      for (const pass of this.passes) {
        if (pass.run(graph)) changed = true;
      }
      if (!changed) break;
      everChanged = true;
    }
    return everChanged;
  }
};

// src/ir/rewrite.ts
function isConstant(value) {
  return value.producer !== null && value.producer.op === "const";
}
function constantValue(value) {
  if (value.producer === null) throw new Error("value is not a constant");
  return value.producer.attrs.value;
}
function replaceAllUsesWith(oldValue, newValue) {
  if (oldValue === newValue) return;
  for (const use of oldValue.uses) {
    use.node.operands[use.index] = newValue;
    newValue.uses.push(use);
  }
  oldValue.uses.length = 0;
}
function rebuildUses(graph) {
  for (const input of graph.inputs) input.uses.length = 0;
  for (const node of graph.nodes) node.result.uses.length = 0;
  for (const node of graph.nodes) {
    for (let i = 0; i < node.operands.length; i += 1) node.operands[i].uses.push({ node, index: i });
  }
}
function pruneDeadNodes(graph) {
  const reachable = /* @__PURE__ */ new Set();
  const stack = [];
  if (graph.output !== null) stack.push(graph.output);
  while (stack.length > 0) {
    const value = stack.pop();
    if (value === void 0) continue;
    const producer = value.producer;
    if (producer === null || reachable.has(producer.id)) continue;
    reachable.add(producer.id);
    for (const operand of producer.operands) stack.push(operand);
  }
  const before = graph.nodes.length;
  const live = graph.nodes.filter((node) => reachable.has(node.id));
  graph.nodes.length = 0;
  for (const node of live) graph.nodes.push(node);
  rebuildUses(graph);
  return before - graph.nodes.length;
}

// src/passes/constant-fold.ts
var ConstantFoldPass = class {
  constructor() {
    __publicField(this, "name", "constant-fold");
  }
  run(graph) {
    let changed = false;
    for (const node of topoSort(graph)) {
      if (node.op === "const" || node.operands.length === 0) continue;
      if (!node.operands.every((operand) => isConstant(operand))) continue;
      const operandValues = node.operands.map((operand) => new Float64Array([constantValue(operand)]));
      const folded = registry.get(node.op).evalFn(operandValues, 1, node.attrs);
      const replacement = graph.constant(folded[0]);
      replaceAllUsesWith(node.result, replacement);
      changed = true;
    }
    if (changed) pruneDeadNodes(graph);
    return changed;
  }
};

// src/passes/algebraic.ts
function isConstEqual(value, target) {
  return isConstant(value) && constantValue(value) === target;
}
function simplify(node, graph) {
  const operands = node.operands;
  switch (node.op) {
    case "add":
      if (isConstEqual(operands[1], 0)) return operands[0];
      if (isConstEqual(operands[0], 0)) return operands[1];
      return null;
    case "sub":
      if (isConstEqual(operands[1], 0)) return operands[0];
      return null;
    case "mul":
      if (isConstEqual(operands[1], 1)) return operands[0];
      if (isConstEqual(operands[0], 1)) return operands[1];
      if (isConstEqual(operands[1], 0) || isConstEqual(operands[0], 0)) return graph.constant(0);
      return null;
    case "div":
      if (isConstEqual(operands[1], 1)) return operands[0];
      return null;
    case "neg": {
      const inner = operands[0].producer;
      if (inner !== null && inner.op === "neg") return inner.operands[0];
      return null;
    }
    default:
      return null;
  }
}
var AlgebraicSimplificationPass = class {
  constructor() {
    __publicField(this, "name", "algebraic");
  }
  run(graph) {
    let changed = false;
    for (const node of topoSort(graph)) {
      const replacement = simplify(node, graph);
      if (replacement === null || replacement === node.result) continue;
      if (replacement.kind !== node.result.kind) continue;
      replaceAllUsesWith(node.result, replacement);
      changed = true;
    }
    if (changed) pruneDeadNodes(graph);
    return changed;
  }
};

// src/passes/cse.ts
function attributeKey(attrs) {
  const keys = Object.keys(attrs).sort();
  return keys.map((key) => `${key}=${String(attrs[key])}`).join(",");
}
var CommonSubexpressionPass = class {
  constructor() {
    __publicField(this, "name", "cse");
  }
  run(graph) {
    let changed = false;
    const seen = /* @__PURE__ */ new Map();
    for (const node of topoSort(graph)) {
      const key = `${node.op}|${node.operands.map((operand) => operand.id).join(",")}|${attributeKey(node.attrs)}`;
      const existing = seen.get(key);
      if (existing === void 0) {
        seen.set(key, node.result);
        continue;
      }
      replaceAllUsesWith(node.result, existing);
      changed = true;
    }
    if (changed) pruneDeadNodes(graph);
    return changed;
  }
};

// src/passes/dce.ts
var DeadCodeEliminationPass = class {
  constructor() {
    __publicField(this, "name", "dce");
  }
  run(graph) {
    return pruneDeadNodes(graph) > 0;
  }
};

// src/passes/pipeline.ts
function optimize(graph) {
  const manager = new PassManager();
  manager.add(
    new FixedPointGroup("simplify", [
      new ConstantFoldPass(),
      new AlgebraicSimplificationPass(),
      new CommonSubexpressionPass(),
      new DeadCodeEliminationPass()
    ])
  );
  return manager.run(graph);
}

// src/aad/vjp-symbolic.ts
var rules = /* @__PURE__ */ new Map();
function registerSymbolicVjp(op, rule) {
  if (rules.has(op)) throw new Error(`duplicate symbolic VJP ${op}`);
  rules.set(op, rule);
}
function getSymbolicVjp(op) {
  const rule = rules.get(op);
  if (rule === void 0) throw new Error(`no symbolic VJP for op ${op}`);
  return rule;
}
var registered4 = false;
function registerSymbolicVjps() {
  if (registered4) return;
  const indicator = (g, left, right) => g.emit("ge", [left, right], {}, "ge");
  registerSymbolicVjp("add", ({ adjOut }) => [adjOut, adjOut]);
  registerSymbolicVjp("sub", ({ graph, adjOut }) => [adjOut, graph.neg(adjOut)]);
  registerSymbolicVjp("mul", ({ graph, operands, adjOut }) => [graph.mul(adjOut, operands[1]), graph.mul(adjOut, operands[0])]);
  registerSymbolicVjp("div", ({ graph, operands, adjOut }) => [
    graph.div(adjOut, operands[1]),
    graph.neg(graph.div(graph.mul(adjOut, operands[0]), graph.mul(operands[1], operands[1])))
  ]);
  registerSymbolicVjp("neg", ({ graph, adjOut }) => [graph.neg(adjOut)]);
  registerSymbolicVjp("exp", ({ graph, result, adjOut }) => [graph.mul(adjOut, result)]);
  registerSymbolicVjp("log", ({ graph, operands, adjOut }) => [graph.div(adjOut, operands[0])]);
  registerSymbolicVjp("sqrt", ({ graph, result, adjOut }) => [graph.div(adjOut, graph.mul(graph.constant(2), result))]);
  registerSymbolicVjp("sigmoid", ({ graph, result, adjOut }) => [graph.mul(adjOut, graph.mul(result, graph.sub(graph.constant(1), result)))]);
  registerSymbolicVjp("softplus", ({ graph, operands, adjOut }) => [graph.mul(adjOut, graph.sigmoid(operands[0]))]);
  registerSymbolicVjp("max", ({ graph, operands, adjOut }) => {
    const ind = indicator(graph, operands[0], operands[1]);
    return [graph.mul(adjOut, ind), graph.mul(adjOut, graph.sub(graph.constant(1), ind))];
  });
  registerSymbolicVjp("min", ({ graph, operands, adjOut }) => {
    const ind = indicator(graph, operands[1], operands[0]);
    return [graph.mul(adjOut, ind), graph.mul(adjOut, graph.sub(graph.constant(1), ind))];
  });
  registerSymbolicVjp("mean", ({ graph, adjOut, batchSize }) => [graph.div(adjOut, batchSize)]);
  registerSymbolicVjp("sum", ({ adjOut }) => [adjOut]);
  registered4 = true;
}

// src/aad/backward-graph.ts
var SymbolicAccumulator = class {
  constructor(graph) {
    this.graph = graph;
    __publicField(this, "pending", /* @__PURE__ */ new Map());
  }
  add(id, value) {
    const existing = this.pending.get(id);
    if (existing !== void 0) existing.push(value);
    else this.pending.set(id, [value]);
  }
  get(id) {
    const values = this.pending.get(id);
    if (values === void 0 || values.length === 0) return null;
    let level = values;
    while (level.length > 1) {
      const next = [];
      for (let i = 0; i < level.length; i += 2) {
        if (i + 1 < level.length) next.push(this.graph.add(level[i], level[i + 1]));
        else next.push(level[i]);
      }
      level = next;
    }
    return level[0];
  }
};
function buildBackwardGraph(graph, output, inputs) {
  registerSymbolicVjps();
  const batchSize = graph.input("scalar", "batchSize");
  const accumulator = new SymbolicAccumulator(graph);
  accumulator.add(output.id, graph.constant(1));
  const order = topoSort(graph);
  for (let k = order.length - 1; k >= 0; k -= 1) {
    const node = order[k];
    if (node.operands.length === 0) continue;
    const adjOut = accumulator.get(node.result.id);
    if (adjOut === null) continue;
    const adjoints = getSymbolicVjp(node.op)({ graph, operands: node.operands, result: node.result, adjOut, batchSize });
    for (let i = 0; i < node.operands.length; i += 1) {
      const operand = node.operands[i];
      let adjoint = adjoints[i];
      if (operand.kind === "scalar" && adjoint.kind === "batch") adjoint = graph.sum(adjoint);
      accumulator.add(operand.id, adjoint);
    }
  }
  const gradients = /* @__PURE__ */ new Map();
  for (const input of inputs) {
    const gradient = accumulator.get(input.id);
    if (gradient !== null) gradients.set(input.id, gradient);
  }
  return { graph, gradients, batchSize };
}

// src/aad/second-order.ts
function forwardOverReverse(backward, bindings, seedInputId, batchSize) {
  const seeds = /* @__PURE__ */ new Map([[seedInputId, new Float64Array([1])]]);
  const { tangents } = forwardMode(backward.graph, bindings, seeds, batchSize);
  const hessianColumn = /* @__PURE__ */ new Map();
  for (const [inputId, gradient] of backward.gradients) {
    const tangent = tangents.get(gradient.id);
    hessianColumn.set(inputId, tangent === void 0 ? 0 : tangent[0]);
  }
  return hessianColumn;
}

// src/aad/tape.ts
var Tape = class {
  constructor() {
    __publicField(this, "values", []);
    __publicField(this, "parents", []);
    __publicField(this, "partials", []);
  }
  push(value, parents, partials) {
    const index = this.values.length;
    this.values.push(value);
    this.parents.push(parents);
    this.partials.push(partials);
    return index;
  }
  gradients(outputIndex) {
    const adjoint = new Float64Array(this.values.length);
    adjoint[outputIndex] = 1;
    for (let i = this.values.length - 1; i >= 0; i -= 1) {
      const seed = adjoint[i];
      if (seed === 0) continue;
      const parents = this.parents[i];
      const partials = this.partials[i];
      for (let k = 0; k < parents.length; k += 1) adjoint[parents[k]] += seed * partials[k];
    }
    return adjoint;
  }
};
var ANumber = class _ANumber {
  constructor(tape, index) {
    this.tape = tape;
    this.index = index;
  }
  get value() {
    return this.tape.values[this.index];
  }
  unary(value, partial) {
    return new _ANumber(this.tape, this.tape.push(value, [this.index], [partial]));
  }
  binary(other, value, selfPartial, otherPartial) {
    return new _ANumber(this.tape, this.tape.push(value, [this.index, other.index], [selfPartial, otherPartial]));
  }
  add(other) {
    return this.binary(other, this.value + other.value, 1, 1);
  }
  sub(other) {
    return this.binary(other, this.value - other.value, 1, -1);
  }
  mul(other) {
    return this.binary(other, this.value * other.value, other.value, this.value);
  }
  div(other) {
    return this.binary(other, this.value / other.value, 1 / other.value, -this.value / (other.value * other.value));
  }
  neg() {
    return this.unary(-this.value, -1);
  }
  exp() {
    const value = Math.exp(this.value);
    return this.unary(value, value);
  }
  log() {
    return this.unary(Math.log(this.value), 1 / this.value);
  }
  sqrt() {
    const value = Math.sqrt(this.value);
    return this.unary(value, 1 / (2 * value));
  }
  max(other) {
    return this.value >= other.value ? this.binary(other, this.value, 1, 0) : this.binary(other, other.value, 0, 1);
  }
};
function constant(tape, value) {
  return new ANumber(tape, tape.push(value, [], []));
}
function variable(tape, value) {
  return new ANumber(tape, tape.push(value, [], []));
}
function gradientOf(tape, output, variables) {
  const adjoint = tape.gradients(output.index);
  return variables.map((variable2) => adjoint[variable2.index]);
}

// src/aad/checkpoint.ts
function checkpointedGradient(steps, carry0, param, forwardStep, segment) {
  const segmentLength = segment ?? Math.max(1, Math.round(Math.sqrt(steps)));
  const checkpoints = [carry0];
  let carry = carry0;
  for (let t = 0; t < steps; t += 1) {
    carry = forwardStep(carry, param, t).carry;
    if ((t + 1) % segmentLength === 0 && t + 1 < steps) checkpoints.push(carry);
  }
  const finalCarry = carry;
  let adjointCarry = 1;
  let dParam = 0;
  let maxStored = checkpoints.length;
  const numSegments = Math.ceil(steps / segmentLength);
  for (let s = numSegments - 1; s >= 0; s -= 1) {
    const start = s * segmentLength;
    const end = Math.min(start + segmentLength, steps);
    const dCarryPrev = [];
    const dParamStep = [];
    let segmentCarry = checkpoints[s];
    for (let t = start; t < end; t += 1) {
      const step = forwardStep(segmentCarry, param, t);
      dCarryPrev.push(step.dCarryPrev);
      dParamStep.push(step.dParam);
      segmentCarry = step.carry;
    }
    maxStored = Math.max(maxStored, checkpoints.length + dCarryPrev.length);
    for (let i = end - start - 1; i >= 0; i -= 1) {
      dParam += adjointCarry * dParamStep[i];
      adjointCarry *= dCarryPrev[i];
    }
  }
  return { finalCarry, dParam, dCarry0: adjointCarry, maxStored };
}

// src/risk/implicit-calibration.ts
function jacobian2(evaluate3, x, rows2) {
  const columns = x.length;
  const result = Array.from({ length: rows2 }, () => new Array(columns).fill(0));
  for (let j = 0; j < columns; j += 1) {
    const step = 1e-6 * Math.max(1, Math.abs(x[j]));
    const up = [...x];
    const down = [...x];
    up[j] += step;
    down[j] -= step;
    const fUp = evaluate3(up);
    const fDown = evaluate3(down);
    for (let i = 0; i < rows2; i += 1) result[i][j] = (fUp[i] - fDown[i]) / (2 * step);
  }
  return result;
}
function parameterQuoteSensitivity(residual, theta, quotes) {
  const n = theta.length;
  const m = quotes.length;
  const jTheta = jacobian2((t) => residual(t, quotes), theta, n);
  const jQuote = jacobian2((q) => residual(theta, q), quotes, n);
  const sensitivity = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let j = 0; j < m; j += 1) {
    const rhs = new Array(n);
    for (let i = 0; i < n; i += 1) rhs[i] = -jQuote[i][j];
    const solution = solveLinearSystem(jTheta.map((row) => [...row]), rhs);
    for (let i = 0; i < n; i += 1) sensitivity[i][j] = solution[i];
  }
  return sensitivity;
}
function priceQuoteSensitivity(priceParameterGradient, parameterQuoteJacobian) {
  const m = parameterQuoteJacobian[0].length;
  const result = new Array(m).fill(0);
  for (let j = 0; j < m; j += 1) {
    for (let i = 0; i < priceParameterGradient.length; i += 1) result[j] += priceParameterGradient[i] * parameterQuoteJacobian[i][j];
  }
  return result;
}

// src/risk/aad-xva.ts
function computeCvaSensitivities(spec) {
  const steps = spec.exposureDates;
  const dt = spec.maturity / steps;
  const drift = (spec.rate - 0.5 * spec.vol * spec.vol) * dt;
  const diffusion = spec.vol * Math.sqrt(dt);
  const generator = new MersenneTwister(spec.seed);
  const exposureSum = new Float64Array(steps);
  const exposureSpotSum = new Float64Array(steps);
  const logState = new Float64Array(spec.paths).fill(Math.log(spec.spot));
  for (let step = 0; step < steps; step += 1) {
    const time = (step + 1) * dt;
    const remaining = spec.maturity - time;
    const stepDiscount = Math.exp(-spec.rate * time);
    for (let p = 0; p < spec.paths; p += 1) {
      logState[p] += drift + diffusion * inverseNormalCdf(generator.nextDouble());
      const spotAtTime = Math.exp(logState[p]);
      if (remaining > 1e-10) {
        const valued = blackScholes({ spot: spotAtTime, strike: spec.strike, rate: spec.rate, vol: spec.vol, maturity: remaining, isCall: true });
        exposureSum[step] += stepDiscount * Math.max(valued.price, 0);
        exposureSpotSum[step] += stepDiscount * valued.delta * (spotAtTime / spec.spot);
      } else {
        const intrinsic3 = Math.max(spotAtTime - spec.strike, 0);
        exposureSum[step] += stepDiscount * intrinsic3;
        exposureSpotSum[step] += stepDiscount * (spotAtTime > spec.strike ? 1 : 0) * (spotAtTime / spec.spot);
      }
    }
  }
  let cva = 0;
  let dSpot = 0;
  let dHazard = 0;
  for (let step = 0; step < steps; step += 1) {
    const start = step * dt;
    const end = (step + 1) * dt;
    const defaultProbability2 = Math.exp(-spec.hazardRate * start) - Math.exp(-spec.hazardRate * end);
    const dDefaultProbability = -start * Math.exp(-spec.hazardRate * start) + end * Math.exp(-spec.hazardRate * end);
    const expectedExposure = exposureSum[step] / spec.paths;
    const expectedExposureSpot = exposureSpotSum[step] / spec.paths;
    cva += expectedExposure * defaultProbability2;
    dSpot += expectedExposureSpot * defaultProbability2;
    dHazard += expectedExposure * dDefaultProbability;
  }
  const factor = 1 - spec.recovery;
  return { cva: cva * factor, dSpot: dSpot * factor, dHazard: dHazard * factor };
}

// src/numerics/variance-reduction/likelihood-ratio.ts
function digitalCallLrDelta(market) {
  const generator = new MersenneTwister(market.seed);
  const sqrtT = Math.sqrt(market.maturity);
  const drift = (market.rate - 0.5 * market.vol * market.vol) * market.maturity;
  const diffusion = market.vol * sqrtT;
  const discount = Math.exp(-market.rate * market.maturity);
  const scoreScale = 1 / (market.spot * market.vol * sqrtT);
  const estimator = new Welford();
  for (let p = 0; p < market.paths; p += 1) {
    const z = inverseNormalCdf(generator.nextDouble());
    const terminal2 = market.spot * Math.exp(drift + diffusion * z);
    const indicator = terminal2 > market.strike ? 1 : 0;
    estimator.push(discount * indicator * z * scoreScale);
  }
  return { delta: estimator.mean, standardError: estimator.standardError };
}

// src/marketdata/date.ts
function daysFromCivil(year, month, day) {
  const y = month <= 2 ? year - 1 : year;
  const era = Math.floor((y >= 0 ? y : y - 399) / 400);
  const yearOfEra = y - era * 400;
  const dayOfYear = Math.floor((153 * (month > 2 ? month - 3 : month + 9) + 2) / 5) + day - 1;
  const dayOfEra = yearOfEra * 365 + Math.floor(yearOfEra / 4) - Math.floor(yearOfEra / 100) + dayOfYear;
  return era * 146097 + dayOfEra - 719468;
}
function civilFromDays(serial) {
  const z = serial + 719468;
  const era = Math.floor((z >= 0 ? z : z - 146096) / 146097);
  const dayOfEra = z - era * 146097;
  const yearOfEra = Math.floor((dayOfEra - Math.floor(dayOfEra / 1460) + Math.floor(dayOfEra / 36524) - Math.floor(dayOfEra / 146096)) / 365);
  const year = yearOfEra + era * 400;
  const dayOfYear = dayOfEra - (365 * yearOfEra + Math.floor(yearOfEra / 4) - Math.floor(yearOfEra / 100));
  const monthProgress = Math.floor((5 * dayOfYear + 2) / 153);
  const day = dayOfYear - Math.floor((153 * monthProgress + 2) / 5) + 1;
  const month = monthProgress < 10 ? monthProgress + 3 : monthProgress - 9;
  return { year: month <= 2 ? year + 1 : year, month, day };
}
function weekday(serial) {
  return ((serial % 7 + 4) % 7 + 7) % 7;
}
function isLeapYear(year) {
  return year % 4 === 0 && year % 100 !== 0 || year % 400 === 0;
}
var MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
function addMonths(serial, months) {
  const date = civilFromDays(serial);
  const total = date.year * 12 + (date.month - 1) + months;
  const year = Math.floor(total / 12);
  const month = total % 12 + 1;
  const maxDay = month === 2 && isLeapYear(year) ? 29 : MONTH_LENGTHS[month - 1];
  const day = Math.min(date.day, maxDay);
  return daysFromCivil(year, month, day);
}

// src/marketdata/daycount.ts
function thirty360(start, end) {
  const a = civilFromDays(start);
  const b = civilFromDays(end);
  const d1 = Math.min(a.day, 30);
  const d2 = d1 === 30 ? Math.min(b.day, 30) : b.day;
  return ((b.year - a.year) * 360 + (b.month - a.month) * 30 + (d2 - d1)) / 360;
}
function actActIsda(start, end) {
  if (end <= start) return 0;
  const a = civilFromDays(start);
  const b = civilFromDays(end);
  if (a.year === b.year) return (end - start) / (isLeapYear(a.year) ? 366 : 365);
  let fraction = 0;
  const endOfStartYear = civilFromDaysValue(a.year, 12, 31) + 1;
  fraction += (endOfStartYear - start) / (isLeapYear(a.year) ? 366 : 365);
  const startOfEndYear = civilFromDaysValue(b.year, 1, 1);
  fraction += (end - startOfEndYear) / (isLeapYear(b.year) ? 366 : 365);
  for (let year = a.year + 1; year < b.year; year += 1) fraction += 1;
  return fraction;
}
function civilFromDaysValue(year, month, day) {
  const y = month <= 2 ? year - 1 : year;
  const era = Math.floor((y >= 0 ? y : y - 399) / 400);
  const yearOfEra = y - era * 400;
  const dayOfYear = Math.floor((153 * (month > 2 ? month - 3 : month + 9) + 2) / 5) + day - 1;
  const dayOfEra = yearOfEra * 365 + Math.floor(yearOfEra / 4) - Math.floor(yearOfEra / 100) + dayOfYear;
  return era * 146097 + dayOfEra - 719468;
}
var conventions = /* @__PURE__ */ new Map([
  ["ACT/365", (start, end) => (end - start) / 365],
  ["ACT/360", (start, end) => (end - start) / 360],
  ["30/360", thirty360],
  ["ACT/ACT", actActIsda]
]);
function dayCountFraction(convention, start, end) {
  const fn = conventions.get(convention);
  if (fn === void 0) throw new Error(`unknown day-count convention ${convention}`);
  return fn(start, end);
}

// src/marketdata/calendar.ts
var Calendar = class {
  constructor(holidays = []) {
    __publicField(this, "holidays");
    this.holidays = new Set(holidays);
  }
  isBusinessDay(serial) {
    const day = weekday(serial);
    return day !== 0 && day !== 6 && !this.holidays.has(serial);
  }
  next(serial) {
    let s = serial;
    while (!this.isBusinessDay(s)) s += 1;
    return s;
  }
  previous(serial) {
    let s = serial;
    while (!this.isBusinessDay(s)) s -= 1;
    return s;
  }
  adjust(serial, convention) {
    if (convention === "none" || this.isBusinessDay(serial)) return serial;
    if (convention === "preceding") return this.previous(serial);
    const forward = this.next(serial);
    if (convention === "following") return forward;
    const sameMonth = civilFromDays(forward).month === civilFromDays(serial).month;
    return sameMonth ? forward : this.previous(serial);
  }
};

// src/marketdata/schedule.ts
function generateSchedule(spec) {
  const unadjusted = [];
  let count = 1;
  for (; ; ) {
    const date = addMonths(spec.start, count * spec.frequencyMonths);
    if (date >= spec.end) break;
    unadjusted.push(date);
    count += 1;
  }
  unadjusted.push(spec.end);
  return unadjusted.map((date) => spec.calendar.adjust(date, spec.convention));
}

// src/marketdata/curve.ts
function locate(times, t) {
  let low = 0;
  let high = times.length - 1;
  while (high - low > 1) {
    const mid = low + high >> 1;
    if (times[mid] <= t) low = mid;
    else high = mid;
  }
  return low;
}
var DiscountCurve = class {
  constructor(times, zeroRates) {
    __publicField(this, "times");
    __publicField(this, "zeroRates");
    if (times.length !== zeroRates.length || times.length === 0) throw new Error("curve requires matching non-empty pillars");
    this.times = times;
    this.zeroRates = zeroRates;
  }
  zeroRate(t) {
    if (t <= this.times[0]) return this.zeroRates[0];
    if (t >= this.times[this.times.length - 1]) return this.zeroRates[this.zeroRates.length - 1];
    const i = locate(this.times, t);
    const weight = (t - this.times[i]) / (this.times[i + 1] - this.times[i]);
    return this.zeroRates[i] + weight * (this.zeroRates[i + 1] - this.zeroRates[i]);
  }
  discountFactor(t) {
    return Math.exp(-this.zeroRate(t) * t);
  }
  forwardRate(start, end) {
    return (this.discountFactor(start) / this.discountFactor(end) - 1) / (end - start);
  }
};

// src/marketdata/bootstrap.ts
function bootstrapFromParSwaps(quotes, accrual) {
  const times = [];
  const discountFactors = [];
  for (const quote of quotes) {
    const periods = Math.round(quote.maturity / accrual);
    let annuity2 = 0;
    for (let i = 0; i < discountFactors.length; i += 1) annuity2 += accrual * discountFactors[i];
    const discountFactor = (1 - quote.rate * annuity2) / (1 + quote.rate * accrual);
    times.push(periods * accrual);
    discountFactors.push(discountFactor);
  }
  const zeroRates = discountFactors.map((df, i) => -Math.log(df) / times[i]);
  return new DiscountCurve(times, zeroRates);
}

// src/instruments/swap.ts
function fixedLegValue(swap, curve) {
  let value = 0;
  for (let i = 0; i < swap.times.length; i += 1) value += swap.fixedRate * swap.accruals[i] * curve.discountFactor(swap.times[i]);
  return value;
}
function floatLegValue(swap, curve) {
  return 1 - curve.discountFactor(swap.times[swap.times.length - 1]);
}
function swapValue(swap, curve) {
  return floatLegValue(swap, curve) - fixedLegValue(swap, curve);
}
function parSwapRate(times, accruals, curve) {
  let annuity2 = 0;
  for (let i = 0; i < times.length; i += 1) annuity2 += accruals[i] * curve.discountFactor(times[i]);
  return (1 - curve.discountFactor(times[times.length - 1])) / annuity2;
}

// src/risk/curve-risk.ts
function swapKeyRateSensitivities(times, zeroRates, accruals, fixedRate, maturityIndex) {
  const tape = new Tape();
  const rates = zeroRates.map((rate) => variable(tape, rate));
  const discountFactor = (i) => rates[i].mul(constant(tape, -times[i])).exp();
  let value = constant(tape, 1).sub(discountFactor(maturityIndex));
  for (let i = 0; i <= maturityIndex; i += 1) {
    value = value.sub(constant(tape, fixedRate * accruals[i]).mul(discountFactor(i)));
  }
  return { value: value.value, sensitivities: gradientOf(tape, value, rates) };
}

// src/marketdata/svi.ts
function totalVariance(params, k) {
  const d = k - params.m;
  return params.a + params.b * (params.rho * d + Math.sqrt(d * d + params.sigma * params.sigma));
}
function dTotalVariance(params, k) {
  const d = k - params.m;
  return params.b * (params.rho + d / Math.sqrt(d * d + params.sigma * params.sigma));
}
function d2TotalVariance(params, k) {
  const d = k - params.m;
  const root = Math.sqrt(d * d + params.sigma * params.sigma);
  return params.b * params.sigma * params.sigma / (root * root * root);
}
function sviImpliedVol(params, k, maturity) {
  return Math.sqrt(totalVariance(params, k) / maturity);
}
function butterflyG(params, k) {
  const w = totalVariance(params, k);
  const wp = dTotalVariance(params, k);
  const wpp = d2TotalVariance(params, k);
  const term = 1 - k * wp / (2 * w);
  return term * term - wp * wp / 4 * (1 / w + 0.25) + wpp / 2;
}
function fitSvi(logMoneyness, vols, maturity, initial) {
  const residual = (p) => logMoneyness.map((k, i) => sviImpliedVol({ a: p[0], b: p[1], rho: p[2], m: p[3], sigma: p[4] }, k, maturity) - vols[i]);
  const result = levenbergMarquardt(residual, [initial.a, initial.b, initial.rho, initial.m, initial.sigma], { maxIterations: 400 });
  const [a, b, rho, m, sigma] = result.parameters;
  return { params: { a, b, rho, m, sigma }, residualNorm: result.residualNorm };
}

// src/marketdata/vol-surface.ts
function bracket(slices, maturity) {
  if (maturity <= slices[0].maturity) return { lower: 0, upper: 0, weight: 0 };
  if (maturity >= slices[slices.length - 1].maturity) return { lower: slices.length - 1, upper: slices.length - 1, weight: 0 };
  let i = 0;
  while (i < slices.length - 1 && slices[i + 1].maturity <= maturity) i += 1;
  const weight = (maturity - slices[i].maturity) / (slices[i + 1].maturity - slices[i].maturity);
  return { lower: i, upper: i + 1, weight };
}
var VolSurface = class {
  constructor(slices) {
    __publicField(this, "slices");
    this.slices = [...slices].sort((a, b) => a.maturity - b.maturity);
  }
  interpolate(read, k, maturity) {
    const { lower, upper, weight } = bracket(this.slices, maturity);
    const low = read(this.slices[lower].params, k);
    if (lower === upper) return low;
    const high = read(this.slices[upper].params, k);
    return low + weight * (high - low);
  }
  totalVariance(k, maturity) {
    return this.interpolate(totalVariance, k, maturity);
  }
  dTotalVarianceDk(k, maturity) {
    return this.interpolate(dTotalVariance, k, maturity);
  }
  d2TotalVarianceDk(k, maturity) {
    return this.interpolate(d2TotalVariance, k, maturity);
  }
  dTotalVarianceDt(k, maturity) {
    const { lower, upper } = bracket(this.slices, maturity);
    if (lower === upper) {
      const index = lower === 0 ? 0 : lower - 1;
      const next = index + 1 < this.slices.length ? index + 1 : index;
      if (next === index) return totalVariance(this.slices[index].params, k) / this.slices[index].maturity;
      return (totalVariance(this.slices[next].params, k) - totalVariance(this.slices[index].params, k)) / (this.slices[next].maturity - this.slices[index].maturity);
    }
    return (totalVariance(this.slices[upper].params, k) - totalVariance(this.slices[lower].params, k)) / (this.slices[upper].maturity - this.slices[lower].maturity);
  }
  impliedVol(strike, forward, maturity) {
    const k = Math.log(strike / forward);
    return Math.sqrt(this.totalVariance(k, maturity) / maturity);
  }
};

// src/marketdata/dupire.ts
function localVariance(surface, k, maturity) {
  const w = surface.totalVariance(k, maturity);
  const wt = surface.dTotalVarianceDt(k, maturity);
  const wk = surface.dTotalVarianceDk(k, maturity);
  const wkk = surface.d2TotalVarianceDk(k, maturity);
  const denominator = 1 - k * wk / (2 * w) + 0.25 * (-0.25 - 1 / w + k * k / (w * w)) * wk * wk + 0.5 * wkk;
  return wt / denominator;
}
function localVol(surface, strike, forward, maturity) {
  const k = Math.log(strike / forward);
  return Math.sqrt(localVariance(surface, k, maturity));
}

// src/numerics/complex.ts
function complex(re, im) {
  return { re, im };
}
function add(a, b) {
  return { re: a.re + b.re, im: a.im + b.im };
}
function sub(a, b) {
  return { re: a.re - b.re, im: a.im - b.im };
}
function mul(a, b) {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}
function div(a, b) {
  const denominator = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / denominator, im: (a.im * b.re - a.re * b.im) / denominator };
}
function scale(a, s) {
  return { re: a.re * s, im: a.im * s };
}
function exp(a) {
  const magnitude = Math.exp(a.re);
  return { re: magnitude * Math.cos(a.im), im: magnitude * Math.sin(a.im) };
}
function log(a) {
  return { re: 0.5 * Math.log(a.re * a.re + a.im * a.im), im: Math.atan2(a.im, a.re) };
}
function sqrt(a) {
  const magnitude = Math.sqrt(Math.sqrt(a.re * a.re + a.im * a.im));
  const angle = Math.atan2(a.im, a.re) / 2;
  return { re: magnitude * Math.cos(angle), im: magnitude * Math.sin(angle) };
}

// src/analytics/characteristic.ts
function blackScholesCf(rate, vol, maturity) {
  const drift = rate - 0.5 * vol * vol;
  return {
    cf: (u) => exp(complex(-0.5 * vol * vol * u * u * maturity, u * drift * maturity)),
    c1: drift * maturity,
    c2: vol * vol * maturity
  };
}
function mertonCf(rate, maturity, p) {
  const kappa = Math.exp(p.jumpMean + 0.5 * p.jumpVol * p.jumpVol) - 1;
  const drift = rate - 0.5 * p.vol * p.vol - p.jumpIntensity * kappa;
  return {
    cf: (u) => {
      const diffusion = complex(-0.5 * p.vol * p.vol * u * u * maturity, u * drift * maturity);
      const jumpCf = exp(complex(-0.5 * p.jumpVol * p.jumpVol * u * u, u * p.jumpMean));
      const jump = scale(sub(jumpCf, complex(1, 0)), p.jumpIntensity * maturity);
      return exp(add(diffusion, jump));
    },
    c1: drift * maturity + p.jumpIntensity * maturity * p.jumpMean,
    c2: p.vol * p.vol * maturity + p.jumpIntensity * maturity * (p.jumpMean * p.jumpMean + p.jumpVol * p.jumpVol)
  };
}
function hestonCf(rate, maturity, p) {
  const { initialVariance: v0, meanReversion: kappa, longVariance: theta, volOfVol: xi, correlation: rho } = p;
  return {
    cf: (u) => {
      const iu = complex(0, u);
      const rhoXiIu = scale(iu, rho * xi);
      const kMinus = sub(complex(kappa, 0), rhoXiIu);
      const rhoXiIuMinusKappa = sub(rhoXiIu, complex(kappa, 0));
      const term1 = mul(rhoXiIuMinusKappa, rhoXiIuMinusKappa);
      const term2 = scale(complex(u * u, u), xi * xi);
      const d = sqrt(add(term1, term2));
      const kMinusMinusD = sub(kMinus, d);
      const kMinusPlusD = add(kMinus, d);
      const g = div(kMinusMinusD, kMinusPlusD);
      const eDt = exp(scale(d, -maturity));
      const oneMinusGeDt = sub(complex(1, 0), mul(g, eDt));
      const oneMinusG = sub(complex(1, 0), g);
      const logTerm = log(div(oneMinusGeDt, oneMinusG));
      const cBracket = sub(scale(kMinusMinusD, maturity), scale(logTerm, 2));
      const c = add(scale(iu, rate * maturity), scale(cBracket, kappa * theta / (xi * xi)));
      const dCoefficient = mul(scale(kMinusMinusD, 1 / (xi * xi)), div(sub(complex(1, 0), eDt), oneMinusGeDt));
      return exp(add(c, scale(dCoefficient, v0)));
    },
    c1: (rate - 0.5 * theta) * maturity,
    c2: theta * maturity + (v0 - theta) * (1 - Math.exp(-kappa * maturity)) / kappa
  };
}

// src/analytics/cos-method.ts
function chi(k, a, b, c, d) {
  const omega = k * Math.PI / (b - a);
  const cosD = Math.cos(omega * (d - a));
  const cosC = Math.cos(omega * (c - a));
  const sinD = Math.sin(omega * (d - a));
  const sinC = Math.sin(omega * (c - a));
  const expD = Math.exp(d);
  const expC = Math.exp(c);
  return 1 / (1 + omega * omega) * (cosD * expD - cosC * expC + omega * sinD * expD - omega * sinC * expC);
}
function psi(k, a, b, c, d) {
  if (k === 0) return d - c;
  const omega = k * Math.PI / (b - a);
  return (Math.sin(omega * (d - a)) - Math.sin(omega * (c - a))) / omega;
}
function payoffCoefficient(k, a, b, isCall) {
  if (isCall) return 2 / (b - a) * (chi(k, a, b, 0, b) - psi(k, a, b, 0, b));
  return 2 / (b - a) * (-chi(k, a, b, a, 0) + psi(k, a, b, a, 0));
}
function cosEuropeanPrice(model, spot2, strike, rate, maturity, isCall, terms = 1024, width = 12) {
  const a = model.c1 - width * Math.sqrt(Math.abs(model.c2));
  const b = model.c1 + width * Math.sqrt(Math.abs(model.c2));
  const x = Math.log(spot2 / strike);
  let sum = 0;
  for (let k = 0; k < terms; k += 1) {
    const omega = k * Math.PI / (b - a);
    const phi = model.cf(omega);
    const phase = omega * (x - a);
    const real = phi.re * Math.cos(phase) - phi.im * Math.sin(phase);
    const coefficient = payoffCoefficient(k, a, b, isCall);
    sum += (k === 0 ? 0.5 : 1) * real * coefficient;
  }
  return strike * Math.exp(-rate * maturity) * sum;
}

// src/models/equity/merton.ts
function poissonSample(mean, generator) {
  const threshold = Math.exp(-mean);
  let count = 0;
  let product = 1;
  for (; ; ) {
    product *= generator.nextDouble();
    if (product <= threshold) return count;
    count += 1;
  }
}
function priceMertonCall(spec) {
  const jumpCompensator = Math.exp(spec.jumpMean + 0.5 * spec.jumpVol * spec.jumpVol) - 1;
  const drift = (spec.rate - 0.5 * spec.vol * spec.vol - spec.jumpIntensity * jumpCompensator) * spec.maturity;
  const diffusion = spec.vol * Math.sqrt(spec.maturity);
  const discount = Math.exp(-spec.rate * spec.maturity);
  const generator = new MersenneTwister(spec.seed);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    const z = inverseNormalCdf(generator.nextDouble());
    const jumps = poissonSample(spec.jumpIntensity * spec.maturity, generator);
    let jumpSum = 0;
    for (let j = 0; j < jumps; j += 1) jumpSum += spec.jumpMean + spec.jumpVol * inverseNormalCdf(generator.nextDouble());
    const terminal2 = spec.spot * Math.exp(drift + diffusion * z + jumpSum);
    estimator.push(discount * Math.max(terminal2 - spec.strike, 0));
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}

// src/calibration/heston-calibration.ts
function calibrateHeston(spot2, rate, quotes, fixed, initial) {
  const residual = (p) => quotes.map((quote) => {
    const model = hestonCf(rate, quote.maturity, {
      initialVariance: p[0],
      meanReversion: fixed.meanReversion,
      longVariance: p[1],
      volOfVol: p[2],
      correlation: fixed.correlation
    });
    return cosEuropeanPrice(model, spot2, quote.strike, rate, quote.maturity, true) - quote.price;
  });
  const result = levenbergMarquardt(residual, [initial.initialVariance, initial.longVariance, initial.volOfVol], { maxIterations: 300 });
  return { initialVariance: result.parameters[0], longVariance: result.parameters[1], volOfVol: result.parameters[2], residualNorm: result.residualNorm };
}

// src/numerics/linalg/cholesky.ts
function cholesky(matrix) {
  const n = matrix.length;
  const lower = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j <= i; j += 1) {
      let sum = 0;
      for (let k = 0; k < j; k += 1) sum += lower[i][k] * lower[j][k];
      if (i === j) lower[i][j] = Math.sqrt(Math.max(matrix[i][i] - sum, 0));
      else lower[i][j] = lower[j][j] > 1e-14 ? (matrix[i][j] - sum) / lower[j][j] : 0;
    }
  }
  return lower;
}
function correlate(factor, independent) {
  const n = factor.length;
  const result = new Array(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j <= i; j += 1) result[i] += factor[i][j] * independent[j];
  }
  return result;
}

// src/models/rates/vasicek-swaption.ts
function bFactor(a, tenor) {
  return (1 - Math.exp(-a * tenor)) / a;
}
function discountBond(model, rate, t, maturity) {
  const a = model.meanReversion;
  const tenor = maturity - t;
  const b = bFactor(a, tenor);
  const logA = (b - tenor) * (a * a * model.longRate - 0.5 * model.vol * model.vol) / (a * a) - model.vol * model.vol * b * b / (4 * a);
  return Math.exp(logA - b * rate);
}
function bondOptionVol(model, expiry, bondMaturity) {
  const a = model.meanReversion;
  return model.vol * Math.sqrt((1 - Math.exp(-2 * a * expiry)) / (2 * a)) * bFactor(a, bondMaturity - expiry);
}
function zeroBondPut(model, expiry, bondMaturity, strike) {
  const pBond = discountBond(model, model.shortRate, 0, bondMaturity);
  const pExpiry = discountBond(model, model.shortRate, 0, expiry);
  const sigma = bondOptionVol(model, expiry, bondMaturity);
  const h = Math.log(pBond / (strike * pExpiry)) / sigma + 0.5 * sigma;
  return strike * pExpiry * normalCdf(-h + sigma) - pBond * normalCdf(-h);
}
function couponFlows(swaption) {
  return swaption.accruals.map((accrual, i) => i === swaption.accruals.length - 1 ? 1 + swaption.fixedRate * accrual : swaption.fixedRate * accrual);
}
function couponBond(model, rate, swaption, flows) {
  let value = 0;
  for (let i = 0; i < swaption.times.length; i += 1) value += flows[i] * discountBond(model, rate, swaption.expiry, swaption.times[i]);
  return value;
}
function jamshidianPayerSwaption(model, swaption) {
  const flows = couponFlows(swaption);
  let low = -1;
  let high = 1;
  for (let iteration = 0; iteration < 200; iteration += 1) {
    const mid = 0.5 * (low + high);
    if (couponBond(model, mid, swaption, flows) > 1) low = mid;
    else high = mid;
  }
  const criticalRate = 0.5 * (low + high);
  let price = 0;
  for (let i = 0; i < swaption.times.length; i += 1) {
    const strike = discountBond(model, criticalRate, swaption.expiry, swaption.times[i]);
    price += flows[i] * zeroBondPut(model, swaption.expiry, swaption.times[i], strike);
  }
  return price;
}

// src/models/rates/lmm.ts
function blackCaplet(initialForward, strike, vol, fixingTime, accrual, discount) {
  const sqrtT = Math.sqrt(fixingTime);
  const d1 = (Math.log(initialForward / strike) + 0.5 * vol * vol * fixingTime) / (vol * sqrtT);
  const d2 = d1 - vol * sqrtT;
  return discount * accrual * (initialForward * normalCdf(d1) - strike * normalCdf(d2));
}
function priceLmmCaplet(spec) {
  const n = spec.rateCount;
  const tau = spec.accrual;
  const sigma = spec.vol;
  const k = spec.capletIndex;
  const generator = new MersenneTwister(spec.seed);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    const forwards = new Array(n).fill(spec.initialForward);
    let bank = 1;
    for (let m = 0; m <= k; m += 1) {
      bank *= 1 + tau * forwards[m];
      if (m < k) {
        const z = inverseNormalCdf(generator.nextDouble());
        for (let i = m + 1; i < n; i += 1) {
          let drift = 0;
          for (let j = m + 1; j <= i; j += 1) drift += tau * forwards[j] / (1 + tau * forwards[j]);
          drift *= sigma * sigma;
          forwards[i] *= Math.exp((drift - 0.5 * sigma * sigma) * tau + sigma * Math.sqrt(tau) * z);
        }
      }
    }
    estimator.push(tau * Math.max(forwards[k] - spec.strike, 0) / bank);
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}

// src/models/credit/hazard-curve.ts
var HazardCurve = class {
  constructor(times, hazards) {
    __publicField(this, "times");
    __publicField(this, "hazards");
    if (times.length !== hazards.length) throw new Error("hazard curve requires matching pillars");
    this.times = times;
    this.hazards = hazards;
  }
  survival(t) {
    let integral = 0;
    let previous = 0;
    for (let i = 0; i < this.times.length; i += 1) {
      const segmentEnd = Math.min(this.times[i], t);
      if (segmentEnd > previous) integral += this.hazards[i] * (segmentEnd - previous);
      previous = this.times[i];
      if (this.times[i] >= t) break;
    }
    if (t > previous) integral += this.hazards[this.hazards.length - 1] * (t - previous);
    return Math.exp(-integral);
  }
};

// src/instruments/cds.ts
function schedule(spec) {
  const dates = [];
  const periods = Math.round(spec.maturity / spec.frequency);
  for (let i = 1; i <= periods; i += 1) dates.push(i * spec.frequency);
  return dates;
}
function premiumAnnuity(curve, spec) {
  let annuity2 = 0;
  let previous = 0;
  for (const date of schedule(spec)) {
    const discount = Math.exp(-spec.discountRate * date);
    annuity2 += (date - previous) * discount * 0.5 * (curve.survival(previous) + curve.survival(date));
    previous = date;
  }
  return annuity2;
}
function protectionLeg(curve, spec) {
  let protection = 0;
  let previous = 0;
  for (const date of schedule(spec)) {
    const discount = Math.exp(-spec.discountRate * date);
    protection += (1 - spec.recovery) * discount * (curve.survival(previous) - curve.survival(date));
    previous = date;
  }
  return protection;
}
function cdsParSpread(curve, spec) {
  return protectionLeg(curve, spec) / premiumAnnuity(curve, spec);
}
function bootstrapHazardCurve(quotes, frequency, recovery, discountRate) {
  const times = [];
  const hazards = [];
  for (const quote of quotes) {
    times.push(quote.maturity);
    hazards.push(0.01);
    let low = 1e-6;
    let high = 5;
    for (let iteration = 0; iteration < 200; iteration += 1) {
      const mid = 0.5 * (low + high);
      hazards[hazards.length - 1] = mid;
      const spec = { maturity: quote.maturity, frequency, recovery, discountRate };
      const spread = cdsParSpread(new HazardCurve(times, hazards), spec);
      if (spread > quote.spread) high = mid;
      else low = mid;
    }
    hazards[hazards.length - 1] = 0.5 * (low + high);
  }
  return new HazardCurve(times, hazards);
}

// src/models/credit/gaussian-copula.ts
function conditionalDefaultProbability(defaultProbability2, correlation, factor) {
  const threshold = inverseNormalCdf(defaultProbability2);
  return normalCdf((threshold - Math.sqrt(correlation) * factor) / Math.sqrt(1 - correlation));
}

// src/models/credit/cdo.ts
function conditionalLossDistribution(names, probability) {
  const distribution = new Float64Array(names + 1);
  distribution[0] = 1;
  for (let name = 0; name < names; name += 1) {
    for (let k = name + 1; k >= 1; k -= 1) distribution[k] = distribution[k] * (1 - probability) + distribution[k - 1] * probability;
    distribution[0] *= 1 - probability;
  }
  return distribution;
}
function trancheExpectedLoss(spec) {
  const lossGivenDefault = 1 - spec.recovery;
  const width = spec.detachment - spec.attachment;
  const lower = -6;
  const upper = 6;
  const step = (upper - lower) / spec.quadraturePoints;
  let expectedLoss = 0;
  let weightSum = 0;
  for (let q = 0; q < spec.quadraturePoints; q += 1) {
    const factor = lower + (q + 0.5) * step;
    const weight = normalPdf(factor) * step;
    weightSum += weight;
    const probability = conditionalDefaultProbability(spec.defaultProbability, spec.correlation, factor);
    const distribution = conditionalLossDistribution(spec.names, probability);
    let conditionalLoss = 0;
    for (let k = 0; k <= spec.names; k += 1) {
      const portfolioLoss = lossGivenDefault * k / spec.names;
      const trancheLoss = Math.min(Math.max(portfolioLoss - spec.attachment, 0), width);
      conditionalLoss += distribution[k] * trancheLoss;
    }
    expectedLoss += weight * conditionalLoss;
  }
  return expectedLoss / weightSum;
}

// src/numerics/analytic/margrabe.ts
function margrabeExchange(input) {
  const vol = Math.sqrt(input.vol1 * input.vol1 + input.vol2 * input.vol2 - 2 * input.correlation * input.vol1 * input.vol2);
  const sqrtT = Math.sqrt(input.maturity);
  const d1 = (Math.log(input.spot1 / input.spot2) + 0.5 * vol * vol * input.maturity) / (vol * sqrtT);
  const d2 = d1 - vol * sqrtT;
  return input.spot1 * normalCdf(d1) - input.spot2 * normalCdf(d2);
}

// src/engines/multi-asset-mc.ts
function simulateTerminals(spec, factor, generator) {
  const independent = spec.spots.map(() => inverseNormalCdf(generator.nextDouble()));
  const correlated = correlate(factor, independent);
  return spec.spots.map((spot2, i) => spot2 * Math.exp((spec.rate - 0.5 * spec.vols[i] * spec.vols[i]) * spec.maturity + spec.vols[i] * Math.sqrt(spec.maturity) * correlated[i]));
}
function exchangeOptionMc(spec) {
  const factor = cholesky(spec.correlation);
  const generator = new MersenneTwister(spec.seed);
  const discount = Math.exp(-spec.rate * spec.maturity);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    const terminals = simulateTerminals(spec, factor, generator);
    estimator.push(discount * Math.max(terminals[0] - terminals[1], 0));
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}
function basketCallMc(spec, strike) {
  const factor = cholesky(spec.correlation);
  const generator = new MersenneTwister(spec.seed);
  const discount = Math.exp(-spec.rate * spec.maturity);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    const terminals = simulateTerminals(spec, factor, generator);
    let average = 0;
    for (const terminal2 of terminals) average += terminal2;
    average /= terminals.length;
    estimator.push(discount * Math.max(average - strike, 0));
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}

// src/engines/pde-2d.ts
function exchangeOptionAdi(spec) {
  const m = spec.gridPoints;
  const dt = spec.maturity / spec.timeSteps;
  const theta = 0.5;
  const center1 = Math.log(spec.spot1);
  const center2 = Math.log(spec.spot2);
  const half1 = spec.widthStdDev * spec.vol1 * Math.sqrt(spec.maturity);
  const half2 = spec.widthStdDev * spec.vol2 * Math.sqrt(spec.maturity);
  const dx1 = 2 * half1 / (m - 1);
  const dx2 = 2 * half2 / (m - 1);
  const x1 = new Float64Array(m);
  const x2 = new Float64Array(m);
  for (let i = 0; i < m; i += 1) x1[i] = center1 - half1 + i * dx1;
  for (let j = 0; j < m; j += 1) x2[j] = center2 - half2 + j * dx2;
  const payoff2 = (i, j) => Math.max(Math.exp(x1[i]) - Math.exp(x2[j]), 0);
  let value = Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_2, j) => payoff2(i, j)));
  const v1 = spec.vol1 * spec.vol1;
  const v2 = spec.vol2 * spec.vol2;
  const drift1 = spec.rate - 0.5 * v1;
  const drift2 = spec.rate - 0.5 * v2;
  const crossCoefficient = spec.correlation * spec.vol1 * spec.vol2 / (4 * dx1 * dx2);
  const a1Sub = 0.5 * v1 / (dx1 * dx1) - drift1 / (2 * dx1);
  const a1Diag = -v1 / (dx1 * dx1) - 0.5 * spec.rate;
  const a1Sup = 0.5 * v1 / (dx1 * dx1) + drift1 / (2 * dx1);
  const a2Sub = 0.5 * v2 / (dx2 * dx2) - drift2 / (2 * dx2);
  const a2Diag = -v2 / (dx2 * dx2) - 0.5 * spec.rate;
  const a2Sup = 0.5 * v2 / (dx2 * dx2) + drift2 / (2 * dx2);
  const applyA1 = (grid, i, j) => a1Sub * grid[i - 1][j] + a1Diag * grid[i][j] + a1Sup * grid[i + 1][j];
  const applyA2 = (grid, i, j) => a2Sub * grid[i][j - 1] + a2Diag * grid[i][j] + a2Sup * grid[i][j + 1];
  const applyA0 = (grid, i, j) => crossCoefficient * (grid[i + 1][j + 1] - grid[i + 1][j - 1] - grid[i - 1][j + 1] + grid[i - 1][j - 1]);
  const interior = m - 2;
  const lower = new Float64Array(interior);
  const diag = new Float64Array(interior);
  const upper = new Float64Array(interior);
  const rhs = new Float64Array(interior);
  for (let step = 0; step < spec.timeSteps; step += 1) {
    const fixed = value;
    const a0v = Array.from({ length: m }, () => new Array(m).fill(0));
    const a1v = Array.from({ length: m }, () => new Array(m).fill(0));
    const a2v = Array.from({ length: m }, () => new Array(m).fill(0));
    for (let i = 1; i <= interior; i += 1) {
      for (let j = 1; j <= interior; j += 1) {
        a0v[i][j] = applyA0(fixed, i, j);
        a1v[i][j] = applyA1(fixed, i, j);
        a2v[i][j] = applyA2(fixed, i, j);
      }
    }
    const sweep1 = (source) => {
      const result = fixed.map((row) => row.slice());
      for (let j = 1; j <= interior; j += 1) {
        for (let i = 1; i <= interior; i += 1) {
          lower[i - 1] = -theta * dt * a1Sub;
          diag[i - 1] = 1 - theta * dt * a1Diag;
          upper[i - 1] = -theta * dt * a1Sup;
          rhs[i - 1] = source[i][j] - theta * dt * a1v[i][j];
        }
        rhs[0] -= -theta * dt * a1Sub * fixed[0][j];
        rhs[interior - 1] -= -theta * dt * a1Sup * fixed[m - 1][j];
        const solved = solveTridiagonal(lower, diag, upper, rhs);
        for (let i = 1; i <= interior; i += 1) result[i][j] = solved[i - 1];
      }
      return result;
    };
    const sweep2 = (source) => {
      const result = fixed.map((row) => row.slice());
      for (let i = 1; i <= interior; i += 1) {
        for (let j = 1; j <= interior; j += 1) {
          lower[j - 1] = -theta * dt * a2Sub;
          diag[j - 1] = 1 - theta * dt * a2Diag;
          upper[j - 1] = -theta * dt * a2Sup;
          rhs[j - 1] = source[i][j] - theta * dt * a2v[i][j];
        }
        rhs[0] -= -theta * dt * a2Sub * fixed[i][0];
        rhs[interior - 1] -= -theta * dt * a2Sup * fixed[i][m - 1];
        const solved = solveTridiagonal(lower, diag, upper, rhs);
        for (let j = 1; j <= interior; j += 1) result[i][j] = solved[j - 1];
      }
      return result;
    };
    const y0 = fixed.map((row) => row.slice());
    for (let i = 1; i <= interior; i += 1) {
      for (let j = 1; j <= interior; j += 1) y0[i][j] = fixed[i][j] + dt * (a0v[i][j] + a1v[i][j] + a2v[i][j]);
    }
    const y2 = sweep2(sweep1(y0));
    const corrected = y0.map((row) => row.slice());
    for (let i = 1; i <= interior; i += 1) {
      for (let j = 1; j <= interior; j += 1) corrected[i][j] = y0[i][j] + 0.5 * dt * (applyA0(y2, i, j) - a0v[i][j]);
    }
    value = sweep2(sweep1(corrected));
  }
  const i0 = Math.round((center1 - (center1 - half1)) / dx1);
  const j0 = Math.round((center2 - (center2 - half2)) / dx2);
  return value[i0][j0];
}

// src/engines/pde/psor.ts
var DEFAULT_OPTIONS2 = { relaxation: 1.4, maxIterations: 1e4, tolerance: 1e-12 };
function projectedSor(lower, diag, upper, rhs, floor, options = {}) {
  const config = { ...DEFAULT_OPTIONS2, ...options };
  const n = diag.length;
  const solution = new Float64Array(n);
  for (let i = 0; i < n; i += 1) solution[i] = Math.max(rhs[i] / diag[i], floor[i]);
  for (let iteration = 0; iteration < config.maxIterations; iteration += 1) {
    let error = 0;
    for (let i = 0; i < n; i += 1) {
      const left = i > 0 ? lower[i] * solution[i - 1] : 0;
      const right = i < n - 1 ? upper[i] * solution[i + 1] : 0;
      const gaussSeidel = (rhs[i] - left - right) / diag[i];
      const relaxed = solution[i] + config.relaxation * (gaussSeidel - solution[i]);
      const projected = Math.max(relaxed, floor[i]);
      error += (projected - solution[i]) * (projected - solution[i]);
      solution[i] = projected;
    }
    if (error < config.tolerance) break;
  }
  return solution;
}

// src/risk/var.ts
function historicalVar(pnl, confidence) {
  const sorted = Float64Array.from(pnl).sort();
  const index = Math.max(0, Math.floor((1 - confidence) * sorted.length));
  return -sorted[index];
}
function historicalExpectedShortfall(pnl, confidence) {
  const sorted = Float64Array.from(pnl).sort();
  const cutoff = Math.max(1, Math.floor((1 - confidence) * sorted.length));
  let total = 0;
  for (let i = 0; i < cutoff; i += 1) total += sorted[i];
  return -total / cutoff;
}
function parametricVar(sensitivities, covariance, confidence) {
  let variance = 0;
  for (let i = 0; i < sensitivities.length; i += 1) {
    for (let j = 0; j < sensitivities.length; j += 1) variance += sensitivities[i] * covariance[i][j] * sensitivities[j];
  }
  return inverseNormalCdf(confidence) * Math.sqrt(variance);
}

// src/risk/scenarios.ts
function generateScenarios(covariance, count, seed) {
  const factor = cholesky(covariance);
  const generator = new MersenneTwister(seed);
  const dimension = covariance.length;
  const scenarios = [];
  for (let s = 0; s < count; s += 1) {
    const independent = new Array(dimension);
    for (let i = 0; i < dimension; i += 1) independent[i] = inverseNormalCdf(generator.nextDouble());
    scenarios.push(correlate(factor, independent));
  }
  return scenarios;
}
function portfolioPnl(scenarios, sensitivities) {
  const pnl = new Float64Array(scenarios.length);
  for (let s = 0; s < scenarios.length; s += 1) {
    let value = 0;
    for (let i = 0; i < sensitivities.length; i += 1) value += sensitivities[i] * scenarios[s][i];
    pnl[s] = value;
  }
  return pnl;
}

// src/risk/pca.ts
function jacobiEigen(symmetric, maxSweeps = 100, tolerance = 1e-14) {
  const n = symmetric.length;
  const a = symmetric.map((row) => [...row]);
  const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_2, j) => i === j ? 1 : 0));
  for (let sweep = 0; sweep < maxSweeps; sweep += 1) {
    let offDiagonal = 0;
    for (let p = 0; p < n; p += 1) {
      for (let q = p + 1; q < n; q += 1) offDiagonal += a[p][q] * a[p][q];
    }
    if (offDiagonal < tolerance) break;
    for (let p = 0; p < n; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const phi = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
        const cos = Math.cos(phi);
        const sin = Math.sin(phi);
        for (let k = 0; k < n; k += 1) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = cos * akp - sin * akq;
          a[k][q] = sin * akp + cos * akq;
        }
        for (let k = 0; k < n; k += 1) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = cos * apk - sin * aqk;
          a[q][k] = sin * apk + cos * aqk;
        }
        for (let k = 0; k < n; k += 1) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = cos * vkp - sin * vkq;
          v[k][q] = sin * vkp + cos * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
  return {
    values: order.map((i) => a[i][i]),
    vectors: order.map((i) => v.map((row) => row[i]))
  };
}
function reconstructCovariance(system) {
  const n = system.values.length;
  const result = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      let sum = 0;
      for (let k = 0; k < n; k += 1) sum += system.values[k] * system.vectors[k][i] * system.vectors[k][j];
      result[i][j] = sum;
    }
  }
  return result;
}

// src/lang/token.ts
var KEYWORDS = /* @__PURE__ */ new Set([
  "product",
  "underlying",
  "model",
  "param",
  "var",
  "event",
  "in",
  "schedule",
  "if",
  "then",
  "else",
  "pay",
  "at",
  "stop",
  "let",
  "exercise",
  "and",
  "or",
  "not"
]);

// src/lang/lexer.ts
var TWO_CHAR_OPS = /* @__PURE__ */ new Set(["<=", ">=", "==", "!="]);
var ONE_CHAR_OPS = /* @__PURE__ */ new Set(["<", ">", "+", "-", "*", "/", "="]);
function isDigit2(c) {
  return c >= "0" && c <= "9";
}
function isIdentStart2(c) {
  return c >= "a" && c <= "z" || c >= "A" && c <= "Z" || c === "_";
}
function isIdentPart2(c) {
  return isIdentStart2(c) || isDigit2(c);
}
function isExponentSign2(source, index) {
  const c = source[index];
  if (c !== "+" && c !== "-") return false;
  const previous = source[index - 1];
  return previous === "e" || previous === "E";
}
function tokenize2(source) {
  const tokens = [];
  let i = 0;
  let line = 1;
  let col = 1;
  const n = source.length;
  const push = (type, value, startCol) => {
    tokens.push({ type, value, line, col: startCol });
  };
  while (i < n) {
    const c = source[i];
    if (c === "\n") {
      line += 1;
      col = 1;
      i += 1;
      continue;
    }
    if (c === " " || c === "	" || c === "\r") {
      i += 1;
      col += 1;
      continue;
    }
    if (c === "/" && source[i + 1] === "/") {
      while (i < n && source[i] !== "\n") i += 1;
      continue;
    }
    const startCol = col;
    if (c === "(") {
      push("lparen", c, startCol);
      i += 1;
      col += 1;
      continue;
    }
    if (c === ")") {
      push("rparen", c, startCol);
      i += 1;
      col += 1;
      continue;
    }
    if (c === "{") {
      push("lbrace", c, startCol);
      i += 1;
      col += 1;
      continue;
    }
    if (c === "}") {
      push("rbrace", c, startCol);
      i += 1;
      col += 1;
      continue;
    }
    if (c === ",") {
      push("comma", c, startCol);
      i += 1;
      col += 1;
      continue;
    }
    if (c === "?") {
      push("question", c, startCol);
      i += 1;
      col += 1;
      continue;
    }
    if (c === ":") {
      push("colon", c, startCol);
      i += 1;
      col += 1;
      continue;
    }
    const twoChar = source.slice(i, i + 2);
    if (TWO_CHAR_OPS.has(twoChar)) {
      push("op", twoChar, startCol);
      i += 2;
      col += 2;
      continue;
    }
    if (ONE_CHAR_OPS.has(c)) {
      push("op", c, startCol);
      i += 1;
      col += 1;
      continue;
    }
    if (isDigit2(c) || c === "." && isDigit2(source[i + 1])) {
      let j = i + 1;
      while (j < n && (isDigit2(source[j]) || source[j] === "." || source[j] === "e" || source[j] === "E" || isExponentSign2(source, j))) j += 1;
      push("number", source.slice(i, j), startCol);
      col += j - i;
      i = j;
      continue;
    }
    if (isIdentStart2(c)) {
      let j = i + 1;
      while (j < n && isIdentPart2(source[j])) j += 1;
      const text = source.slice(i, j);
      push(KEYWORDS.has(text) ? "keyword" : "ident", text, startCol);
      col += j - i;
      i = j;
      continue;
    }
    throw new Error(`unexpected character '${c}' at ${line}:${col}`);
  }
  tokens.push({ type: "eof", value: "", line, col });
  return tokens;
}

// src/lang/parser.ts
var INFIX_PRECEDENCE2 = {
  or: 10,
  and: 20,
  "==": 30,
  "!=": 30,
  "<": 40,
  "<=": 40,
  ">": 40,
  ">=": 40,
  "+": 50,
  "-": 50,
  "*": 60,
  "/": 60
};
var Parser2 = class {
  constructor(tokens) {
    this.tokens = tokens;
    __publicField(this, "index", 0);
  }
  peek() {
    return this.tokens[this.index];
  }
  advance() {
    const token = this.tokens[this.index];
    this.index += 1;
    return token;
  }
  position() {
    const token = this.peek();
    return { line: token.line, col: token.col };
  }
  fail(message) {
    const token = this.peek();
    throw new Error(`${message} at ${token.line}:${token.col} (found '${token.value || token.type}')`);
  }
  expect(type) {
    if (this.peek().type !== type) this.fail(`expected ${type}`);
    return this.advance();
  }
  expectKeyword(value) {
    const token = this.peek();
    if (token.type !== "keyword" || token.value !== value) this.fail(`expected '${value}'`);
    this.advance();
  }
  isKeyword(value) {
    const token = this.peek();
    return token.type === "keyword" && token.value === value;
  }
  parseProduct() {
    const pos = this.position();
    this.expectKeyword("product");
    const name = this.expect("ident").value;
    this.expect("lbrace");
    const underlyings = [];
    const params = [];
    const vars = [];
    const events = [];
    while (this.peek().type !== "rbrace") {
      if (this.isKeyword("underlying")) underlyings.push(this.parseUnderlying());
      else if (this.isKeyword("param")) params.push(this.parseParam());
      else if (this.isKeyword("var")) vars.push(this.parseVarDecl());
      else if (this.isKeyword("event")) events.push(this.parseEvent());
      else this.fail("expected declaration or event");
    }
    this.expect("rbrace");
    this.expect("eof");
    return { name, underlyings, params, vars, events, pos };
  }
  parseUnderlying() {
    const pos = this.position();
    this.expectKeyword("underlying");
    const name = this.expect("ident").value;
    this.expectKeyword("model");
    const model = this.expect("ident").value;
    const modelParams2 = [];
    if (this.peek().type === "lparen") {
      this.advance();
      if (this.peek().type !== "rparen") {
        modelParams2.push(this.parseModelParam());
        while (this.peek().type === "comma") {
          this.advance();
          modelParams2.push(this.parseModelParam());
        }
      }
      this.expect("rparen");
    }
    return { name, model, modelParams: modelParams2, pos };
  }
  parseModelParam() {
    const pos = this.position();
    const name = this.expect("ident").value;
    this.expectOperator("=");
    return { name, value: this.parseExpression(), pos };
  }
  parseParam() {
    const pos = this.position();
    this.expectKeyword("param");
    const name = this.expect("ident").value;
    this.expectOperator("=");
    return { name, value: this.parseExpression(), pos };
  }
  parseVarDecl() {
    const pos = this.position();
    this.expectKeyword("var");
    const name = this.expect("ident").value;
    this.expectOperator("=");
    return { name, init: this.parseExpression(), pos };
  }
  parseEvent() {
    const pos = this.position();
    this.expectKeyword("event");
    const variable2 = this.expect("ident").value;
    let schedule2;
    if (this.isKeyword("in")) {
      this.advance();
      this.expectKeyword("schedule");
      this.expect("lparen");
      const start = this.parseExpression();
      this.expect("comma");
      const end = this.parseExpression();
      this.expect("comma");
      const step = this.parseExpression();
      this.expect("rparen");
      schedule2 = { kind: "schedule", start, end, step };
    } else {
      this.expectOperator("=");
      schedule2 = { kind: "single", date: this.parseExpression() };
    }
    const body = this.parseBlock();
    return { variable: variable2, schedule: schedule2, body, pos };
  }
  parseBlock() {
    this.expect("lbrace");
    const statements = [];
    while (this.peek().type !== "rbrace") statements.push(this.parseStatement());
    this.expect("rbrace");
    return statements;
  }
  parseStatement() {
    const pos = this.position();
    if (this.isKeyword("var")) {
      this.advance();
      const name = this.expect("ident").value;
      this.expectOperator("=");
      return { kind: "assign", name, expr: this.parseExpression(), declare: true, pos };
    }
    if (this.isKeyword("if")) {
      this.advance();
      const cond = this.parseExpression();
      this.expectKeyword("then");
      const body = this.parseBlock();
      let otherwise = null;
      if (this.isKeyword("else")) {
        this.advance();
        otherwise = this.parseBlock();
      }
      return { kind: "if", cond, body, otherwise, pos };
    }
    if (this.isKeyword("pay")) {
      this.advance();
      const amount = this.parseExpression();
      let currency = null;
      if (this.isKeyword("in")) {
        this.advance();
        currency = this.expect("ident").value;
      }
      this.expectKeyword("at");
      return { kind: "pay", amount, date: this.parseExpression(), currency, pos };
    }
    if (this.isKeyword("stop")) {
      this.advance();
      return { kind: "stop", pos };
    }
    if (this.isKeyword("exercise")) {
      this.advance();
      const name = this.expect("ident").value;
      return { kind: "exercise", name, body: this.parseBlock(), pos };
    }
    if (this.peek().type === "ident") {
      const name = this.advance().value;
      this.expectOperator("=");
      return { kind: "assign", name, expr: this.parseExpression(), declare: false, pos };
    }
    this.fail("expected statement");
  }
  expectOperator(value) {
    const token = this.peek();
    if (token.type !== "op" || token.value !== value) this.fail(`expected '${value}'`);
    this.advance();
  }
  parseExpression() {
    if (this.isKeyword("let")) {
      const pos = this.position();
      this.advance();
      const name = this.expect("ident").value;
      this.expectOperator("=");
      const value = this.parseExpression();
      this.expectKeyword("in");
      const body = this.parseExpression();
      return { kind: "let", name, value, body, pos };
    }
    const cond = this.parseBinary(0);
    if (this.peek().type === "question") {
      const pos = this.position();
      this.advance();
      const whenTrue = this.parseExpression();
      this.expect("colon");
      const whenFalse = this.parseExpression();
      return { kind: "ternary", cond, whenTrue, whenFalse, pos };
    }
    return cond;
  }
  operatorOf(token) {
    if (token.type === "op") return token.value;
    if (token.type === "keyword" && (token.value === "and" || token.value === "or")) return token.value;
    return null;
  }
  parseBinary(minPrecedence) {
    let left = this.parseUnary();
    for (; ; ) {
      const op = this.operatorOf(this.peek());
      if (op === null) break;
      const precedence = INFIX_PRECEDENCE2[op];
      if (precedence === void 0 || precedence < minPrecedence) break;
      const pos = this.position();
      this.advance();
      const right = this.parseBinary(precedence + 1);
      left = { kind: "binary", op, left, right, pos };
    }
    return left;
  }
  parseUnary() {
    const token = this.peek();
    if (token.type === "op" && token.value === "-") {
      const pos = this.position();
      this.advance();
      return { kind: "unary", op: "-", operand: this.parseUnary(), pos };
    }
    if (token.type === "keyword" && token.value === "not") {
      const pos = this.position();
      this.advance();
      return { kind: "unary", op: "not", operand: this.parseUnary(), pos };
    }
    return this.parsePrimary();
  }
  parsePrimary() {
    const token = this.peek();
    const pos = this.position();
    if (token.type === "number") {
      this.advance();
      return { kind: "num", value: Number(token.value), pos };
    }
    if (token.type === "ident") {
      this.advance();
      if (this.peek().type === "lparen") {
        this.advance();
        const args = [];
        if (this.peek().type !== "rparen") {
          args.push(this.parseExpression());
          while (this.peek().type === "comma") {
            this.advance();
            args.push(this.parseExpression());
          }
        }
        this.expect("rparen");
        return { kind: "call", callee: token.value, args, pos };
      }
      return { kind: "ident", name: token.value, pos };
    }
    if (token.type === "lparen") {
      this.advance();
      const inner = this.parseExpression();
      this.expect("rparen");
      return inner;
    }
    this.fail("expected expression");
  }
};
function parseProduct(source) {
  return new Parser2(tokenize2(source)).parseProduct();
}

// src/lang/printer.ts
function printExpr(expr) {
  switch (expr.kind) {
    case "num":
      return String(expr.value);
    case "ident":
      return expr.name;
    case "unary":
      return expr.op === "not" ? `(not ${printExpr(expr.operand)})` : `(-${printExpr(expr.operand)})`;
    case "binary":
      return `(${printExpr(expr.left)} ${expr.op} ${printExpr(expr.right)})`;
    case "ternary":
      return `(${printExpr(expr.cond)} ? ${printExpr(expr.whenTrue)} : ${printExpr(expr.whenFalse)})`;
    case "call":
      return `${expr.callee}(${expr.args.map(printExpr).join(", ")})`;
    case "let":
      return `(let ${expr.name} = ${printExpr(expr.value)} in ${printExpr(expr.body)})`;
  }
}
function indent(depth) {
  return "  ".repeat(depth);
}
function printStmt(stmt, depth) {
  const pad = indent(depth);
  switch (stmt.kind) {
    case "assign":
      return `${pad}${stmt.declare ? "var " : ""}${stmt.name} = ${printExpr(stmt.expr)}`;
    case "pay":
      return `${pad}pay ${printExpr(stmt.amount)}${stmt.currency !== null ? ` in ${stmt.currency}` : ""} at ${printExpr(stmt.date)}`;
    case "stop":
      return `${pad}stop`;
    case "if": {
      const head = `${pad}if ${printExpr(stmt.cond)} then {
${printBlock(stmt.body, depth + 1)}
${pad}}`;
      if (stmt.otherwise === null) return head;
      return `${head} else {
${printBlock(stmt.otherwise, depth + 1)}
${pad}}`;
    }
    case "exercise":
      return `${pad}exercise ${stmt.name} {
${printBlock(stmt.body, depth + 1)}
${pad}}`;
  }
}
function printBlock(body, depth) {
  return body.map((stmt) => printStmt(stmt, depth)).join("\n");
}
function printEvent(event, depth) {
  const pad = indent(depth);
  const header = event.schedule.kind === "single" ? `event ${event.variable} = ${printExpr(event.schedule.date)}` : `event ${event.variable} in schedule(${printExpr(event.schedule.start)}, ${printExpr(event.schedule.end)}, ${printExpr(event.schedule.step)})`;
  return `${pad}${header} {
${printBlock(event.body, depth + 1)}
${pad}}`;
}
function printProduct(product) {
  const lines = [`product ${product.name} {`];
  for (const underlying of product.underlyings) {
    const params = underlying.modelParams.length > 0 ? `(${underlying.modelParams.map((p) => `${p.name} = ${printExpr(p.value)}`).join(", ")})` : "";
    lines.push(`${indent(1)}underlying ${underlying.name} model ${underlying.model}${params}`);
  }
  for (const param of product.params) lines.push(`${indent(1)}param ${param.name} = ${printExpr(param.value)}`);
  for (const declaration of product.vars) lines.push(`${indent(1)}var ${declaration.name} = ${printExpr(declaration.init)}`);
  for (const event of product.events) lines.push(printEvent(event, 1));
  lines.push("}");
  return lines.join("\n");
}

// src/lang/types.ts
var ARITHMETIC_OPS = /* @__PURE__ */ new Set(["+", "-", "*", "/"]);
var COMPARISON_OPS = /* @__PURE__ */ new Set(["<", "<=", ">", ">=", "==", "!="]);
var LOGICAL_OPS = /* @__PURE__ */ new Set(["and", "or"]);
var FUNCTIONS = {
  max: { args: ["Number", "Number"], result: "Number" },
  min: { args: ["Number", "Number"], result: "Number" },
  exp: { args: ["Number"], result: "Number" },
  log: { args: ["Number"], result: "Number" },
  sqrt: { args: ["Number"], result: "Number" },
  abs: { args: ["Number"], result: "Number" }
};
var PATH_FUNCTIONS = /* @__PURE__ */ new Set(["runningMax", "runningMin", "average"]);

// src/lang/models.ts
var MODELS = {
  gbm: { params: [] },
  bachelier: { params: [] },
  displaced: { params: ["shift"] },
  heston: { params: ["kappa", "theta", "xi", "rho", "v0"], extraNormals: 1, noVol: true, fineGrid: true },
  merton: { params: ["jumpIntensity", "jumpMean", "jumpVol"] },
  cev: { params: ["beta"], fineGrid: true },
  hw1f: { params: ["meanReversion", "vol"], rate: true },
  g2pp: { params: ["meanReversionA", "meanReversionB", "volA", "volB", "correlation"], rate: true, extraNormals: 1 },
  fx: { params: ["foreignRate"] }
};
var isModel = (model) => model in MODELS;
var modelParams = (model) => MODELS[model]?.params ?? [];
var isRateModel = (model) => MODELS[model]?.rate === true;
var extraNormalsPerStep = (model) => MODELS[model]?.extraNormals ?? 0;
var modelNeedsVol = (model) => isModel(model) && !isRateModel(model) && !MODELS[model].noVol;
var needsFineGrid = (model) => isRateModel(model) || MODELS[model]?.fineGrid === true;

// src/lang/typecheck.ts
var Checker = class {
  constructor(scope) {
    this.scope = scope;
    __publicField(this, "errors", []);
  }
  report(message, pos) {
    this.errors.push({ message, line: pos.line, col: pos.col });
  }
  checkExpr(expr) {
    switch (expr.kind) {
      case "num":
        return "Number";
      case "ident": {
        if (this.scope.underlyings.has(expr.name)) {
          this.report(`underlying '${expr.name}' must be observed as ${expr.name}(t)`, expr.pos);
          return "Number";
        }
        const type = this.scope.bindings.get(expr.name);
        if (type === void 0) {
          this.report(`undeclared identifier '${expr.name}'`, expr.pos);
          return "Number";
        }
        return type;
      }
      case "unary": {
        const operand = this.checkExpr(expr.operand);
        if (expr.op === "not") {
          if (operand !== "Bool") this.report("operator not expects Bool", expr.pos);
          return "Bool";
        }
        if (operand !== "Number") this.report("unary minus expects Number", expr.pos);
        return "Number";
      }
      case "binary": {
        const left = this.checkExpr(expr.left);
        const right = this.checkExpr(expr.right);
        if (ARITHMETIC_OPS.has(expr.op)) {
          if (left !== "Number" || right !== "Number") this.report(`operator ${expr.op} expects Number operands`, expr.pos);
          return "Number";
        }
        if (COMPARISON_OPS.has(expr.op)) {
          if (left !== "Number" || right !== "Number") this.report(`comparison ${expr.op} expects Number operands`, expr.pos);
          return "Bool";
        }
        if (LOGICAL_OPS.has(expr.op)) {
          if (left !== "Bool" || right !== "Bool") this.report(`operator ${expr.op} expects Bool operands`, expr.pos);
          return "Bool";
        }
        this.report(`unknown operator ${expr.op}`, expr.pos);
        return "Number";
      }
      case "ternary": {
        if (this.checkExpr(expr.cond) !== "Bool") this.report("ternary condition must be Bool", expr.pos);
        const whenTrue = this.checkExpr(expr.whenTrue);
        const whenFalse = this.checkExpr(expr.whenFalse);
        if (whenTrue !== whenFalse) this.report("ternary branches must have the same type", expr.pos);
        return whenTrue;
      }
      case "call":
        return this.checkCall(expr);
      case "let": {
        const valueType = this.checkExpr(expr.value);
        const previous = this.scope.bindings.get(expr.name);
        this.scope.bindings.set(expr.name, valueType);
        const bodyType = this.checkExpr(expr.body);
        if (previous === void 0) this.scope.bindings.delete(expr.name);
        else this.scope.bindings.set(expr.name, previous);
        return bodyType;
      }
    }
  }
  checkCall(expr) {
    if (this.scope.underlyings.has(expr.callee)) {
      if (expr.args.length !== 1) this.report(`observable ${expr.callee}(t) takes one time argument`, expr.pos);
      else if (this.checkExpr(expr.args[0]) !== "Number") this.report("observation time must be Number", expr.pos);
      return "Number";
    }
    if (expr.callee === "bond") {
      const underlying = expr.args[0];
      if (underlying === void 0 || underlying.kind !== "ident" || !this.scope.underlyings.has(underlying.name)) {
        this.report("bond expects a rate underlying as its first argument", expr.pos);
      }
      if (expr.args.length !== 2) this.report("bond(rate, maturity) takes two arguments", expr.pos);
      else if (this.checkExpr(expr.args[1]) !== "Number") this.report("bond maturity must be Number", expr.pos);
      return "Number";
    }
    if (PATH_FUNCTIONS.has(expr.callee)) {
      const underlying = expr.args[0];
      if (underlying === void 0 || underlying.kind !== "ident" || !this.scope.underlyings.has(underlying.name)) {
        this.report(`${expr.callee} expects an underlying as its first argument`, expr.pos);
      }
      for (let i = 1; i < expr.args.length; i += 1) {
        if (this.checkExpr(expr.args[i]) !== "Number") this.report(`${expr.callee} time argument must be Number`, expr.pos);
      }
      return "Number";
    }
    const signature = FUNCTIONS[expr.callee];
    if (signature === void 0) {
      this.report(`unknown function '${expr.callee}'`, expr.pos);
      return "Number";
    }
    if (expr.args.length !== signature.args.length) this.report(`${expr.callee} expects ${signature.args.length} arguments`, expr.pos);
    expr.args.forEach((arg, i) => {
      const type = this.checkExpr(arg);
      if (signature.args[i] !== void 0 && type !== signature.args[i]) this.report(`argument ${i + 1} of ${expr.callee} must be ${signature.args[i]}`, expr.pos);
    });
    return signature.result;
  }
  checkStmt(stmt) {
    switch (stmt.kind) {
      case "assign": {
        const type = this.checkExpr(stmt.expr);
        if (type !== "Number") this.report(`assignment to '${stmt.name}' must be Number`, stmt.pos);
        if (stmt.declare) this.scope.bindings.set(stmt.name, "Number");
        else if (!this.scope.bindings.has(stmt.name)) this.report(`assignment to undeclared variable '${stmt.name}'`, stmt.pos);
        return;
      }
      case "if": {
        if (this.checkExpr(stmt.cond) !== "Bool") this.report("if condition must be Bool", stmt.pos);
        for (const inner of stmt.body) this.checkStmt(inner);
        if (stmt.otherwise !== null) for (const inner of stmt.otherwise) this.checkStmt(inner);
        return;
      }
      case "pay":
        if (this.checkExpr(stmt.amount) !== "Number") this.report("pay amount must be Number", stmt.pos);
        if (this.checkExpr(stmt.date) !== "Number") this.report("pay date must be Number", stmt.pos);
        if (stmt.currency !== null && !this.scope.underlyings.has(stmt.currency)) this.report(`pay currency '${stmt.currency}' must be an FX underlying`, stmt.pos);
        return;
      case "stop":
        return;
      case "exercise":
        for (const inner of stmt.body) this.checkStmt(inner);
        return;
    }
  }
};
function checkProduct(product) {
  const underlyings = new Set(product.underlyings.map((u) => u.name));
  const bindings = /* @__PURE__ */ new Map();
  for (const param of product.params) bindings.set(param.name, "Number");
  for (const declaration of product.vars) bindings.set(declaration.name, "Number");
  const errors = [];
  for (const underlying of product.underlyings) {
    if (!isModel(underlying.model)) {
      errors.push({ message: `unknown model '${underlying.model}'`, line: underlying.pos.line, col: underlying.pos.col });
      continue;
    }
    const schema = modelParams(underlying.model);
    for (const param of underlying.modelParams) {
      if (!schema.includes(param.name)) errors.push({ message: `model '${underlying.model}' has no parameter '${param.name}'`, line: param.pos.line, col: param.pos.col });
    }
  }
  for (const event of product.events) {
    const scope = { underlyings, bindings: new Map(bindings) };
    scope.bindings.set(event.variable, "Number");
    const checker = new Checker(scope);
    for (const stmt of event.body) checker.checkStmt(stmt);
    errors.push(...checker.errors);
  }
  return errors;
}

// src/models/rates/hull-white-affine.ts
function hwBFactor(a, tenor) {
  return (1 - Math.exp(-a * tenor)) / a;
}
function hwInstantaneousForward(curve, t) {
  const h = 1e-4;
  const left = Math.max(t - h, 1e-8);
  const right = t + h;
  return (Math.log(curve.discountFactor(left)) - Math.log(curve.discountFactor(right))) / (right - left);
}
function hwAlpha(curve, a, vol, t) {
  const factor = 1 - Math.exp(-a * t);
  return hwInstantaneousForward(curve, t) + vol * vol / (2 * a * a) * factor * factor;
}
function hwDecay(a, dt) {
  return Math.exp(-a * dt);
}
function hwStepStd(a, vol, dt) {
  return vol * Math.sqrt((1 - Math.exp(-2 * a * dt)) / (2 * a));
}
function hwBondLogA(curve, a, vol, t, maturity) {
  const b = hwBFactor(a, maturity - t);
  const ratio = curve.discountFactor(maturity) / curve.discountFactor(t);
  return Math.log(ratio) + b * hwInstantaneousForward(curve, t) - vol * vol / (4 * a) * (1 - Math.exp(-2 * a * t)) * b * b;
}

// src/models/rates/g2pp.ts
function bFactor2(rate, tenor) {
  return (1 - Math.exp(-rate * tenor)) / rate;
}
function g2ppVariance(spec, tenor) {
  const { meanReversionA: a, meanReversionB: b, volA: sigma, volB: eta, correlation: rho } = spec;
  const termA = sigma * sigma / (a * a) * (tenor + 2 / a * Math.exp(-a * tenor) - 1 / (2 * a) * Math.exp(-2 * a * tenor) - 3 / (2 * a));
  const termB = eta * eta / (b * b) * (tenor + 2 / b * Math.exp(-b * tenor) - 1 / (2 * b) * Math.exp(-2 * b * tenor) - 3 / (2 * b));
  const cross = 2 * rho * sigma * eta / (a * b) * (tenor + (Math.exp(-a * tenor) - 1) / a + (Math.exp(-b * tenor) - 1) / b - (Math.exp(-(a + b) * tenor) - 1) / (a + b));
  return termA + termB + cross;
}
function g2ppBondToday(spec, maturity) {
  return Math.exp(-spec.flatRate * maturity + 0.5 * g2ppVariance(spec, maturity));
}
function g2ppBond(spec, t, maturity, x, y) {
  const tenor = maturity - t;
  const marketRatio = Math.exp(-spec.flatRate * maturity) / Math.exp(-spec.flatRate * t);
  const adjustment = 0.5 * (g2ppVariance(spec, tenor) - g2ppVariance(spec, maturity) + g2ppVariance(spec, t));
  return marketRatio * Math.exp(adjustment - bFactor2(spec.meanReversionA, tenor) * x - bFactor2(spec.meanReversionB, tenor) * y);
}
function couponFlows2(terms) {
  return terms.accruals.map((accrual, i) => i === terms.accruals.length - 1 ? 1 + terms.fixedRate * accrual : terms.fixedRate * accrual);
}
function forwardMean(primaryVol, primaryRev, secondaryVol, secondaryRev, rho, expiry) {
  const a = primaryRev;
  const b = secondaryRev;
  const sigma = primaryVol;
  const eta = secondaryVol;
  const m = (sigma * sigma / (a * a) + rho * sigma * eta / (a * b)) * (1 - Math.exp(-a * expiry)) - sigma * sigma / (2 * a * a) * (1 - Math.exp(-2 * a * expiry)) - rho * sigma * eta / (b * (a + b)) * (1 - Math.exp(-(a + b) * expiry));
  return -m;
}
function g2ppPayerSwaption(spec, terms, integrationPoints) {
  const { meanReversionA: a, meanReversionB: b, volA: sigma, volB: eta, correlation: rho } = spec;
  const expiry = terms.expiry;
  const flows = couponFlows2(terms);
  const factorA = terms.times.map((t) => bFactor2(a, t - expiry));
  const factorB = terms.times.map((t) => bFactor2(b, t - expiry));
  const amplitude = terms.times.map((t) => g2ppBond(spec, expiry, t, 0, 0));
  const sigmaX = sigma * Math.sqrt((1 - Math.exp(-2 * a * expiry)) / (2 * a));
  const sigmaY = eta * Math.sqrt((1 - Math.exp(-2 * b * expiry)) / (2 * b));
  const rhoXy = rho * sigma * eta / ((a + b) * sigmaX * sigmaY) * (1 - Math.exp(-(a + b) * expiry));
  const muX = forwardMean(sigma, a, eta, b, rho, expiry);
  const muY = forwardMean(eta, b, sigma, a, rho, expiry);
  const complement = Math.sqrt(1 - rhoXy * rhoXy);
  const solveBoundary = (x) => {
    const residual = (y) => {
      let sum = 0;
      for (let i = 0; i < flows.length; i += 1) sum += flows[i] * amplitude[i] * Math.exp(-factorA[i] * x - factorB[i] * y);
      return sum - 1;
    };
    let low = -2;
    let high = 2;
    while (residual(low) < 0) low -= 2;
    while (residual(high) > 0) high += 2;
    for (let iteration = 0; iteration < 80; iteration += 1) {
      const mid = 0.5 * (low + high);
      if (residual(mid) > 0) low = mid;
      else high = mid;
    }
    return 0.5 * (low + high);
  };
  const lower = muX - 8 * sigmaX;
  const upper = muX + 8 * sigmaX;
  const steps = integrationPoints;
  const step = (upper - lower) / steps;
  let integral = 0;
  for (let k = 0; k <= steps; k += 1) {
    const x = lower + k * step;
    const boundary2 = solveBoundary(x);
    const h1 = (boundary2 - muY) / (sigmaY * complement) - rhoXy * (x - muX) / (sigmaX * complement);
    let inner = normalCdf(-h1);
    for (let i = 0; i < flows.length; i += 1) {
      const lambda = flows[i] * amplitude[i] * Math.exp(-factorA[i] * x);
      const kappa = -factorB[i] * (muY - 0.5 * complement * complement * sigmaY * sigmaY * factorB[i] + rhoXy * sigmaY * (x - muX) / sigmaX);
      const h2 = h1 + factorB[i] * sigmaY * complement;
      inner -= lambda * Math.exp(kappa) * normalCdf(-h2);
    }
    const density = Math.exp(-0.5 * ((x - muX) / sigmaX) * ((x - muX) / sigmaX)) / (sigmaX * Math.sqrt(2 * Math.PI));
    const weight = k === 0 || k === steps ? 0.5 : 1;
    integral += weight * density * inner * step;
  }
  return Math.exp(-spec.flatRate * expiry) * integral;
}
function g2ppPayerSwaptionMc(spec, terms, steps, paths, seed) {
  const dt = terms.expiry / steps;
  const sqrtDt = Math.sqrt(dt);
  const rhoComplement = Math.sqrt(1 - spec.correlation * spec.correlation);
  const flows = couponFlows2(terms);
  const generator = new MersenneTwister(seed);
  let total = 0;
  for (let p = 0; p < paths; p += 1) {
    let x = 0;
    let y = 0;
    let integral = 0;
    for (let step = 0; step < steps; step += 1) {
      integral += (x + y + spec.flatRate) * dt;
      const z1 = inverseNormalCdf(generator.nextDouble());
      const z2 = spec.correlation * z1 + rhoComplement * inverseNormalCdf(generator.nextDouble());
      x += -spec.meanReversionA * x * dt + spec.volA * sqrtDt * z1;
      y += -spec.meanReversionB * y * dt + spec.volB * sqrtDt * z2;
    }
    let couponBond3 = 0;
    for (let i = 0; i < terms.times.length; i += 1) couponBond3 += flows[i] * g2ppBond(spec, terms.expiry, terms.times[i], x, y);
    total += Math.exp(-integral) * Math.max(1 - couponBond3, 0);
  }
  return total / paths;
}

// src/lang/compile.ts
var SEED_STRIDE = 2654435761;
var HESTON_STEPS_PER_YEAR = 16;
function refineGrid(eventTimes, stepsPerYear) {
  if (eventTimes.length === 0) return eventTimes;
  const maxDt = 1 / stepsPerYear;
  const points = new Set(eventTimes);
  let previous = 0;
  for (const t of eventTimes) {
    const span = t - previous;
    const sub2 = Math.max(1, Math.ceil(span / maxDt));
    for (let i = 1; i < sub2; i += 1) points.add(previous + span * i / sub2);
    previous = t;
  }
  return [...points].filter((x) => x > 0).sort((a, b) => a - b);
}
function evalConst(expr, params) {
  switch (expr.kind) {
    case "num":
      return expr.value;
    case "ident": {
      const value = params.get(expr.name);
      if (value === void 0) throw new Error(`'${expr.name}' is not a constant`);
      return value;
    }
    case "unary":
      return expr.op === "-" ? -evalConst(expr.operand, params) : evalConst(expr.operand, params);
    case "binary": {
      const left = evalConst(expr.left, params);
      const right = evalConst(expr.right, params);
      if (expr.op === "+") return left + right;
      if (expr.op === "-") return left - right;
      if (expr.op === "*") return left * right;
      if (expr.op === "/") return left / right;
      throw new Error(`operator ${expr.op} is not constant`);
    }
    default:
      throw new Error("expected a constant expression");
  }
}
function computeParamDefaults(product) {
  const defaults = /* @__PURE__ */ new Map();
  for (const param of product.params) defaults.set(param.name, evalConst(param.value, defaults));
  return defaults;
}
function expandEvents(product, defaults, timeShift) {
  const instances = [];
  for (const event of product.events) instances.push(...expandEvent(event, defaults));
  const shifted = timeShift === 0 ? instances : instances.map((instance) => ({ ...instance, time: instance.time - timeShift }));
  return shifted.sort((a, b) => a.time - b.time);
}
function expandEvent(event, defaults) {
  if (event.schedule.kind === "single") return [{ body: event.body, variable: event.variable, time: evalConst(event.schedule.date, defaults) }];
  const start = evalConst(event.schedule.start, defaults);
  const end = evalConst(event.schedule.end, defaults);
  const step = evalConst(event.schedule.step, defaults);
  const instances = [];
  const count = Math.round((end - start) / step);
  for (let i = 0; i <= count; i += 1) instances.push({ body: event.body, variable: event.variable, time: start + i * step });
  return instances;
}
function pushFixing(ctx, name, time, value) {
  ctx.fixingsByAsset.get(name).push({ time, value });
}
function lognormalKernel(displaced) {
  return {
    init(ctx, spec) {
      ctx.fixingsByAsset.set(spec.name, [{ time: 0, value: spec.spot }]);
      const base = displaced ? ctx.graph.add(spec.spot, spec.params.get("shift")) : spec.spot;
      ctx.logState.set(spec.name, ctx.graph.log(base));
    },
    step(ctx, spec, s) {
      const g = ctx.graph;
      const name = spec.name;
      const vol = spec.vol;
      const drift = g.mul(g.sub(s.effRate, g.mul(g.mul(ctx.half, vol), vol)), s.dtConst);
      const diffusion = g.mul(g.mul(vol, s.sqrtDt), s.increment);
      const newLog = g.add(g.add(ctx.logState.get(name), drift), diffusion);
      ctx.logState.set(name, newLog);
      const price = displaced ? g.sub(g.exp(newLog), spec.params.get("shift")) : g.exp(newLog);
      pushFixing(ctx, name, s.time, price);
    }
  };
}
var bachelierKernel = {
  init(ctx, spec) {
    ctx.fixingsByAsset.set(spec.name, [{ time: 0, value: spec.spot }]);
    ctx.level.set(spec.name, spec.spot);
  },
  step(ctx, spec, s) {
    const g = ctx.graph;
    const name = spec.name;
    const current = ctx.level.get(name);
    const drift = g.mul(g.mul(s.effRate, current), s.dtConst);
    const diffusion = g.mul(g.mul(spec.vol, s.sqrtDt), s.increment);
    const next = g.add(g.add(current, drift), diffusion);
    ctx.level.set(name, next);
    pushFixing(ctx, name, s.time, next);
  }
};
var hestonKernel = {
  init(ctx, spec) {
    ctx.fixingsByAsset.set(spec.name, [{ time: 0, value: spec.spot }]);
    ctx.logState.set(spec.name, ctx.graph.log(spec.spot));
    ctx.variance.set(spec.name, spec.params.get("v0"));
  },
  step(ctx, spec, s) {
    const g = ctx.graph;
    const name = spec.name;
    const kappa = spec.params.get("kappa");
    const theta = spec.params.get("theta");
    const xi = spec.params.get("xi");
    const rho = spec.params.get("rho");
    const v = ctx.variance.get(name);
    const vPlus = g.max(v, ctx.zero);
    const scale3 = g.mul(g.sqrt(vPlus), s.sqrtDt);
    const complement = g.sqrt(g.sub(ctx.one, g.mul(rho, rho)));
    const wV = g.add(g.mul(rho, s.increment), g.mul(complement, s.extraZ));
    const drift = g.mul(g.sub(s.effRate, g.mul(ctx.half, vPlus)), s.dtConst);
    const newLog = g.add(g.add(ctx.logState.get(name), drift), g.mul(scale3, s.increment));
    ctx.logState.set(name, newLog);
    const meanReversion2 = g.mul(g.mul(kappa, g.sub(theta, vPlus)), s.dtConst);
    const volOfVol = g.mul(g.mul(xi, scale3), wV);
    ctx.variance.set(name, g.add(g.add(v, meanReversion2), volOfVol));
    pushFixing(ctx, name, s.time, g.exp(newLog));
  }
};
var mertonKernel = {
  init(ctx, spec) {
    ctx.fixingsByAsset.set(spec.name, [{ time: 0, value: spec.spot }]);
    ctx.logState.set(spec.name, ctx.graph.log(spec.spot));
  },
  step(ctx, spec, s) {
    const g = ctx.graph;
    const name = spec.name;
    const sigma = spec.vol;
    const lambda = spec.params.get("jumpIntensity");
    const jm = spec.params.get("jumpMean");
    const jv = spec.params.get("jumpVol");
    const kappa = g.sub(g.exp(g.add(jm, g.mul(ctx.half, g.mul(jv, jv)))), ctx.one);
    const compensator = g.mul(lambda, kappa);
    const jumpInput = g.input("batch", `jump_${name}_${s.time}`);
    ctx.jumpInputs.push({ input: jumpInput, asset: name, step: s.step });
    const driftRate = g.sub(g.sub(s.effRate, g.mul(ctx.half, g.mul(sigma, sigma))), compensator);
    const drift = g.mul(driftRate, s.dtConst);
    const diffusion = g.mul(g.mul(sigma, s.sqrtDt), s.increment);
    const newLog = g.add(g.add(g.add(ctx.logState.get(name), drift), diffusion), jumpInput);
    ctx.logState.set(name, newLog);
    pushFixing(ctx, name, s.time, g.exp(newLog));
  }
};
var cevKernel = {
  init(ctx, spec) {
    ctx.fixingsByAsset.set(spec.name, [{ time: 0, value: spec.spot }]);
    ctx.level.set(spec.name, spec.spot);
  },
  step(ctx, spec, s) {
    const g = ctx.graph;
    const name = spec.name;
    const beta = spec.params.get("beta");
    const current = ctx.level.get(name);
    const safe = g.max(current, g.constant(1e-8));
    const localVol2 = g.mul(spec.vol, g.exp(g.mul(beta, g.log(safe))));
    const drift = g.mul(g.mul(s.effRate, current), s.dtConst);
    const diffusion = g.mul(g.mul(localVol2, s.sqrtDt), s.increment);
    const next = g.max(g.add(g.add(current, drift), diffusion), ctx.zero);
    ctx.level.set(name, next);
    pushFixing(ctx, name, s.time, next);
  }
};
var hw1fKernel = {
  init(ctx, spec) {
    const name = spec.name;
    ctx.xState.set(name, ctx.zero);
    ctx.integralState.set(name, ctx.zero);
    ctx.stochDiscount.set(name, /* @__PURE__ */ new Map());
    const factors = /* @__PURE__ */ new Map();
    ctx.rateFactors.set(name, factors);
    const r0 = ctx.alphaInput(name, 0);
    ctx.fixingsByAsset.set(name, [{ time: 0, value: r0 }]);
    factors.set(0, [r0]);
  },
  step(ctx, spec, s) {
    const g = ctx.graph;
    const name = spec.name;
    const decayInput = g.input("scalar", `hw_decay_${name}_${s.step}`);
    const stepStdInput = g.input("scalar", `hw_stepstd_${name}_${s.step}`);
    ctx.rateScalarInputs.push({ input: decayInput, kind: "decay", asset: name, dt: s.dt });
    ctx.rateScalarInputs.push({ input: stepStdInput, kind: "stepStd", asset: name, dt: s.dt });
    const x = ctx.xState.get(name);
    const integral = g.add(ctx.integralState.get(name), g.mul(g.add(x, ctx.alphaInput(name, s.previous)), s.dtConst));
    ctx.integralState.set(name, integral);
    const xNew = g.add(g.mul(x, decayInput), g.mul(stepStdInput, s.increment));
    ctx.xState.set(name, xNew);
    const rFix = g.add(xNew, ctx.alphaInput(name, s.time));
    pushFixing(ctx, name, s.time, rFix);
    ctx.stochDiscount.get(name).set(s.time, g.exp(g.neg(integral)));
    ctx.rateFactors.get(name).set(s.time, [rFix]);
  }
};
var g2ppKernel = {
  init(ctx, spec) {
    const name = spec.name;
    ctx.xState.set(name, ctx.zero);
    ctx.integralState.set(name, ctx.zero);
    ctx.stochDiscount.set(name, /* @__PURE__ */ new Map());
    const factors = /* @__PURE__ */ new Map();
    ctx.rateFactors.set(name, factors);
    ctx.yState.set(name, ctx.zero);
    ctx.fixingsByAsset.set(name, [{ time: 0, value: ctx.flatInput(name) }]);
    factors.set(0, [ctx.zero, ctx.zero]);
  },
  step(ctx, spec, s) {
    const g = ctx.graph;
    const name = spec.name;
    const aRev = spec.params.get("meanReversionA");
    const bRev = spec.params.get("meanReversionB");
    const sigma = spec.params.get("volA");
    const eta = spec.params.get("volB");
    const rho = spec.params.get("correlation");
    const flat = ctx.flatInput(name);
    const x = ctx.xState.get(name);
    const y = ctx.yState.get(name);
    const integral = g.add(ctx.integralState.get(name), g.mul(g.add(g.add(x, y), flat), s.dtConst));
    ctx.integralState.set(name, integral);
    const z2 = g.add(g.mul(rho, s.increment), g.mul(g.sqrt(g.sub(ctx.one, g.mul(rho, rho))), s.extraZ));
    const xNew = g.add(x, g.add(g.mul(g.neg(g.mul(aRev, x)), s.dtConst), g.mul(g.mul(sigma, s.sqrtDt), s.increment)));
    const yNew = g.add(y, g.add(g.mul(g.neg(g.mul(bRev, y)), s.dtConst), g.mul(g.mul(eta, s.sqrtDt), z2)));
    ctx.xState.set(name, xNew);
    ctx.yState.set(name, yNew);
    const rFix = g.add(g.add(xNew, yNew), flat);
    pushFixing(ctx, name, s.time, rFix);
    ctx.stochDiscount.get(name).set(s.time, g.exp(g.neg(integral)));
    ctx.rateFactors.get(name).set(s.time, [xNew, yNew]);
  }
};
var KERNELS = {
  gbm: lognormalKernel(false),
  fx: lognormalKernel(false),
  displaced: lognormalKernel(true),
  bachelier: bachelierKernel,
  heston: hestonKernel,
  merton: mertonKernel,
  cev: cevKernel,
  hw1f: hw1fKernel,
  g2pp: g2ppKernel
};
function buildPaths(graph, specs, rateInputs, times, cholInputs, rateScalarInputs) {
  const n = specs.length;
  const half = graph.constant(0.5);
  const one = graph.constant(1);
  const zero = graph.constant(0);
  const fixingsByAsset = /* @__PURE__ */ new Map();
  const logState = /* @__PURE__ */ new Map();
  const level = /* @__PURE__ */ new Map();
  const variance = /* @__PURE__ */ new Map();
  const xState = /* @__PURE__ */ new Map();
  const yState = /* @__PURE__ */ new Map();
  const integralState = /* @__PURE__ */ new Map();
  const stochDiscount = /* @__PURE__ */ new Map();
  const rateFactors = /* @__PURE__ */ new Map();
  const alphaByAsset = /* @__PURE__ */ new Map();
  const alphaInput = (asset, t) => {
    let perTime = alphaByAsset.get(asset);
    if (perTime === void 0) {
      perTime = /* @__PURE__ */ new Map();
      alphaByAsset.set(asset, perTime);
    }
    const existing = perTime.get(t);
    if (existing !== void 0) return existing;
    const input = graph.input("scalar", `hw_alpha_${asset}_${t}`);
    perTime.set(t, input);
    rateScalarInputs.push({ input, kind: "alpha", asset, time: t });
    return input;
  };
  const flatByAsset = /* @__PURE__ */ new Map();
  const flatInput = (asset) => {
    const existing = flatByAsset.get(asset);
    if (existing !== void 0) return existing;
    const input = graph.input("scalar", `g2_flat_${asset}`);
    flatByAsset.set(asset, input);
    rateScalarInputs.push({ input, kind: "flat", asset });
    return input;
  };
  const jumpInputs = [];
  const ctx = {
    graph,
    half,
    one,
    zero,
    logState,
    level,
    variance,
    xState,
    yState,
    integralState,
    stochDiscount,
    rateFactors,
    fixingsByAsset,
    jumpInputs,
    rateScalarInputs,
    alphaInput,
    flatInput
  };
  for (const spec of specs) KERNELS[spec.model].init(ctx, spec);
  const normals = [];
  let previous = 0;
  for (let step = 0; step < times.length; step += 1) {
    const time = times[step];
    const dt = time - previous;
    const dtConst = graph.constant(dt);
    const sqrtDt = graph.sqrt(dtConst);
    const stepRate = rateInputs[step];
    const spotZ = [];
    for (let i = 0; i < n; i += 1) {
      const z = graph.input("batch", `z_${specs[i].name}_${time}`);
      normals.push(z);
      spotZ.push(z);
    }
    const corr = [];
    if (cholInputs === null) {
      for (let i = 0; i < n; i += 1) corr.push(spotZ[i]);
    } else {
      for (let i = 0; i < n; i += 1) {
        let acc = graph.mul(cholInputs[i][0], spotZ[0]);
        for (let j = 1; j <= i; j += 1) acc = graph.add(acc, graph.mul(cholInputs[i][j], spotZ[j]));
        corr.push(acc);
      }
    }
    const extraZ = /* @__PURE__ */ new Map();
    for (let i = 0; i < n; i += 1) {
      if (extraNormalsPerStep(specs[i].model) > 0) {
        const z = graph.input("batch", `zx_${specs[i].name}_${time}`);
        normals.push(z);
        extraZ.set(specs[i].name, z);
      }
    }
    for (let i = 0; i < n; i += 1) {
      const spec = specs[i];
      const effRate = graph.add(stepRate, spec.driftAdjust);
      const s = { time, dt, dtConst, sqrtDt, step, previous, effRate, increment: corr[i], extraZ: extraZ.get(spec.name) };
      KERNELS[spec.model].step(ctx, spec, s);
    }
    previous = time;
  }
  const assets = /* @__PURE__ */ new Map();
  for (const spec of specs) assets.set(spec.name, { name: spec.name, fixings: fixingsByAsset.get(spec.name) });
  return { assets, normals, jumpInputs, stochDiscount, rateFactors };
}
function fixingAt(fixings, time) {
  for (const fixing of fixings) {
    if (Math.abs(fixing.time - time) < 1e-9) return fixing.value;
  }
  throw new Error(`no observation available at time ${time}`);
}
function timeOf(expr, context) {
  if (expr.kind === "ident" && expr.name === context.eventVar) return context.eventTime;
  const t = evalConst(expr, context.paramDefaults);
  return t <= 0 ? t : t - context.timeShift;
}
function blend(graph, condition, whenTrue, whenFalse) {
  return graph.add(graph.mul(condition, whenTrue), graph.mul(graph.sub(graph.constant(1), condition), whenFalse));
}
var BINARY_OPS = {
  "+": "add",
  "-": "sub",
  "*": "mul",
  "/": "div",
  "<": "lt",
  "<=": "le",
  ">": "gt",
  ">=": "ge",
  "==": "eq",
  "!=": "ne"
};
function lowerExpr(expr, context) {
  const g = context.graph;
  switch (expr.kind) {
    case "num":
      return g.constant(expr.value);
    case "ident": {
      if (expr.name === context.eventVar) return context.timeValue;
      const value = context.env.get(expr.name);
      if (value === void 0) throw new Error(`unknown identifier '${expr.name}'`);
      return value;
    }
    case "unary": {
      const operand = lowerExpr(expr.operand, context);
      return expr.op === "not" ? g.sub(g.constant(1), operand) : g.neg(operand);
    }
    case "binary": {
      if (expr.op === "and") return g.mul(lowerExpr(expr.left, context), lowerExpr(expr.right, context));
      if (expr.op === "or") {
        const a = lowerExpr(expr.left, context);
        const b = lowerExpr(expr.right, context);
        return g.sub(g.add(a, b), g.mul(a, b));
      }
      const op = BINARY_OPS[expr.op];
      if (op === void 0) throw new Error(`unknown operator ${expr.op}`);
      return g.emit(op, [lowerExpr(expr.left, context), lowerExpr(expr.right, context)], {}, op);
    }
    case "ternary": {
      const condition = lowerExpr(expr.cond, context);
      return blend(g, condition, lowerExpr(expr.whenTrue, context), lowerExpr(expr.whenFalse, context));
    }
    case "call":
      return lowerCall(expr, context);
    case "let": {
      const value = lowerExpr(expr.value, context);
      const env = new Map(context.env);
      env.set(expr.name, value);
      return lowerExpr(expr.body, { ...context, env });
    }
  }
}
function assetFixings(name, context) {
  const asset = context.assets.get(name);
  if (asset === void 0) throw new Error(`unknown underlying '${name}'`);
  return asset.fixings;
}
function lowerCall(expr, context) {
  const g = context.graph;
  if (context.assets.has(expr.callee)) return fixingAt(assetFixings(expr.callee, context), timeOf(expr.args[0], context));
  if (expr.callee === "bond") {
    const target = expr.args[0];
    const name = target.kind === "ident" ? target.name : context.assetOrder[0];
    const t = context.eventTime;
    const maturity = timeOf(expr.args[1], context);
    const key = `${name}|${t}|${maturity}`;
    const cached = context.bondCoeffs.get(key);
    if (cached !== void 0) return cached;
    const factors = context.rateFactors.get(name)?.get(t);
    if (factors === void 0) throw new Error(`bond requires a rate underlying observed at time ${t}`);
    const constInput = g.input("scalar", `bond_const_${name}_${t}_${maturity}`);
    context.rateScalarInputs.push({ input: constInput, kind: "bondConst", asset: name, time: t, maturity });
    let exponent = constInput;
    for (let k = 0; k < factors.length; k += 1) {
      const loading = g.input("scalar", `bond_load${k}_${name}_${t}_${maturity}`);
      context.rateScalarInputs.push({ input: loading, kind: "bondLoading", asset: name, time: t, maturity, factorIndex: k });
      exponent = g.add(exponent, g.mul(loading, factors[k]));
    }
    const result = g.exp(exponent);
    context.bondCoeffs.set(key, result);
    return result;
  }
  if (expr.callee === "runningMax" || expr.callee === "runningMin") {
    const target = expr.args[0];
    const name = target.kind === "ident" ? target.name : context.assetOrder[0];
    const limit = expr.args.length > 1 ? timeOf(expr.args[1], context) : context.eventTime;
    const relevant = assetFixings(name, context).filter((f) => f.time <= limit + 1e-9 && f.time > 0);
    let accumulator = relevant[0].value;
    for (let i = 1; i < relevant.length; i += 1) accumulator = expr.callee === "runningMax" ? g.max(accumulator, relevant[i].value) : g.min(accumulator, relevant[i].value);
    return accumulator;
  }
  if (expr.callee === "average") {
    const target = expr.args[0];
    const name = target.kind === "ident" ? target.name : context.assetOrder[0];
    const relevant = assetFixings(name, context).filter((f) => f.time <= context.eventTime + 1e-9 && f.time > 0);
    let sum = relevant[0].value;
    for (let i = 1; i < relevant.length; i += 1) sum = g.add(sum, relevant[i].value);
    return g.mul(sum, g.constant(1 / relevant.length));
  }
  const args = expr.args.map((arg) => lowerExpr(arg, context));
  if (expr.callee === "max") return g.max(args[0], args[1]);
  if (expr.callee === "min") return g.min(args[0], args[1]);
  if (expr.callee === "exp") return g.exp(args[0]);
  if (expr.callee === "log") return g.log(args[0]);
  if (expr.callee === "sqrt") return g.sqrt(args[0]);
  if (expr.callee === "abs") return g.emit("abs", [args[0]], {}, "abs");
  throw new Error(`unknown function '${expr.callee}'`);
}
function lowerStmt(stmt, context) {
  const g = context.graph;
  switch (stmt.kind) {
    case "assign": {
      const value = lowerExpr(stmt.expr, context);
      const old = context.env.get(stmt.name) ?? g.constant(0);
      context.env.set(stmt.name, blend(g, context.condition, value, old));
      return;
    }
    case "if": {
      const indicator = lowerExpr(stmt.cond, context);
      const thenContext = { ...context, condition: g.mul(context.condition, indicator) };
      for (const inner of stmt.body) lowerStmt(inner, thenContext);
      if (stmt.otherwise !== null) {
        const elseContext = { ...context, condition: g.mul(context.condition, g.sub(g.constant(1), indicator)) };
        for (const inner of stmt.otherwise) lowerStmt(inner, elseContext);
      }
      return;
    }
    case "pay": {
      let amount = lowerExpr(stmt.amount, context);
      if (stmt.currency !== null) amount = g.mul(amount, fixingAt(assetFixings(stmt.currency, context), timeOf(stmt.date, context)));
      const date = timeOf(stmt.date, context);
      const discount = context.stochasticDiscountFor?.(date) ?? context.discountFor(date);
      const contribution = g.mul(g.mul(g.mul(context.condition, context.alive.value), discount), amount);
      context.cashflow.value = g.add(context.cashflow.value, contribution);
      return;
    }
    case "stop":
      context.alive.value = g.mul(context.alive.value, g.sub(g.constant(1), context.condition));
      return;
    case "exercise": {
      const mask = g.input("batch", `exercise_${context.eventTime}`);
      const payStmt = stmt.body.find((inner) => inner.kind === "pay");
      const intrinsic3 = payStmt !== void 0 && payStmt.kind === "pay" ? lowerExpr(payStmt.amount, context) : g.constant(0);
      const level = fixingAt(assetFixings(context.assetOrder[0], context), context.eventTime);
      context.exercises.push({ date: context.eventTime, intrinsic: intrinsic3, underlying: level, mask });
      const exerciseContext = { ...context, condition: g.mul(context.condition, mask) };
      for (const inner of stmt.body) lowerStmt(inner, exerciseContext);
      context.alive.value = g.mul(context.alive.value, g.sub(g.constant(1), exerciseContext.condition));
      return;
    }
  }
}
function compileProduct(product, options = {}) {
  registerBuiltinOps();
  if (product.underlyings.length < 1) throw new Error("a product needs at least one underlying");
  for (const underlying of product.underlyings) {
    if (!isModel(underlying.model)) throw new Error(`model '${underlying.model}' is not supported`);
  }
  const assetOrder = product.underlyings.map((u) => u.name);
  const single = assetOrder.length === 1;
  const n = assetOrder.length;
  const paramDefaults = computeParamDefaults(product);
  const instances = expandEvents(product, paramDefaults, options.timeShift ?? 0);
  const eventTimes = [...new Set(instances.map((instance) => instance.time))].filter((t) => t > 0).sort((a, b) => a - b);
  const fineGrid = product.underlyings.some((u) => needsFineGrid(u.model));
  const times = fineGrid ? refineGrid(eventTimes, HESTON_STEPS_PER_YEAR) : eventTimes;
  const graph = new Graph();
  const spots = /* @__PURE__ */ new Map();
  const vols = /* @__PURE__ */ new Map();
  const driftAdjust = /* @__PURE__ */ new Map();
  const models2 = /* @__PURE__ */ new Map();
  const modelParamInputs = /* @__PURE__ */ new Map();
  const modelParamDefaults = /* @__PURE__ */ new Map();
  for (const underlying of product.underlyings) {
    const name = underlying.name;
    models2.set(name, underlying.model);
    spots.set(name, graph.input("scalar", single ? "spot" : `spot_${name}`));
    vols.set(name, graph.input("scalar", single ? "vol" : `vol_${name}`));
    driftAdjust.set(name, graph.input("scalar", `drift_adj_${name}`));
    const inputs = /* @__PURE__ */ new Map();
    const defaults = /* @__PURE__ */ new Map();
    const fileParams = new Map(underlying.modelParams.map((mp) => [mp.name, mp.value]));
    for (const param of modelParams(underlying.model)) {
      inputs.set(param, graph.input("scalar", `mparam_${name}_${param}`));
      const fileValue = fileParams.get(param);
      if (fileValue !== void 0) defaults.set(param, evalConst(fileValue, paramDefaults));
    }
    modelParamInputs.set(name, inputs);
    modelParamDefaults.set(name, defaults);
  }
  const rateInputs = times.map((_, k) => graph.input("scalar", `rate_step_${k}`));
  const discountInputs = /* @__PURE__ */ new Map();
  const discountFor = (time) => {
    if (Math.abs(time) < 1e-9) return graph.constant(1);
    const key = Math.round(time * 1e6);
    const hit = discountInputs.get(key);
    if (hit) return hit.value;
    const input = graph.input("scalar", `df_${time}`);
    discountInputs.set(key, { time, value: input });
    return input;
  };
  let cholInputs = null;
  if (!single) {
    cholInputs = [];
    for (let i = 0; i < n; i += 1) {
      const row = [];
      for (let j = 0; j <= i; j += 1) row.push(graph.input("scalar", `chol_${i}_${j}`));
      cholInputs.push(row);
    }
  }
  const params = /* @__PURE__ */ new Map();
  const env = /* @__PURE__ */ new Map();
  for (const param of product.params) {
    const value = graph.input("scalar", `param_${param.name}`);
    params.set(param.name, value);
    env.set(param.name, value);
  }
  const specs = product.underlyings.map((u) => ({
    name: u.name,
    model: u.model,
    spot: spots.get(u.name),
    vol: vols.get(u.name),
    driftAdjust: driftAdjust.get(u.name),
    params: modelParamInputs.get(u.name)
  }));
  const rateScalarInputs = [];
  const { assets, normals, jumpInputs, stochDiscount, rateFactors } = buildPaths(graph, specs, rateInputs, times, cholInputs, rateScalarInputs);
  const rateUnderlying = product.underlyings.find((u) => isRateModel(u.model));
  const stochasticDiscountFor = rateUnderlying ? (time) => stochDiscount.get(rateUnderlying.name).get(time) : null;
  const timeShift = options.timeShift ?? 0;
  const bondCoeffs = /* @__PURE__ */ new Map();
  const baseContext = { graph, env, discountFor, stochasticDiscountFor, assets, assetOrder, eventVar: "", eventTime: 0, timeValue: graph.constant(0), paramDefaults, timeShift, rateScalarInputs, bondCoeffs, rateFactors, models: models2 };
  for (const declaration of product.vars) env.set(declaration.name, lowerExpr(declaration.init, baseContext));
  const alive = { value: graph.constant(1) };
  const cashflow = { value: graph.constant(0) };
  const exercises = [];
  for (const instance of instances) {
    const context = {
      ...baseContext,
      eventVar: instance.variable,
      eventTime: instance.time,
      timeValue: graph.constant(instance.time),
      condition: graph.constant(1),
      alive,
      cashflow,
      exercises
    };
    for (const stmt of instance.body) lowerStmt(stmt, context);
  }
  const price = graph.mean(cashflow.value);
  graph.output = price;
  const equityNames = assetOrder.filter((name) => !isRateModel(models2.get(name)));
  const riskFactors = [...equityNames.map((name) => spots.get(name)), ...equityNames.map((name) => vols.get(name))];
  return { graph, spots, vols, driftAdjust, models: models2, modelParamInputs, modelParamDefaults, assetOrder, cholInputs, rateInputs, discountInputs, jumpInputs, rateScalarInputs, params, paramDefaults, normals, times, price, cashflow: cashflow.value, riskFactors, exercises };
}
function regress(intrinsic3, level, value) {
  const ata = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  const aty = [0, 0, 0];
  let count = 0;
  for (let p = 0; p < value.length; p += 1) {
    if (intrinsic3[p] <= 0) continue;
    count += 1;
    const basis = [1, level[p], level[p] * level[p]];
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) ata[i][j] += basis[i] * basis[j];
      aty[i] += basis[i] * value[p];
    }
  }
  return count >= 3 ? solveLinearSystem(ata, aty) : null;
}
function fillExerciseMasks(compiled, bindings, paths, discountTo) {
  const exercises = [...compiled.exercises].sort((a, b) => a.date - b.date);
  const masks = exercises.map(() => new Float64Array(paths));
  for (const exercise of exercises) bindings.set(exercise.mask.id, new Float64Array(paths));
  const forward = evaluate(compiled.graph, bindings, paths);
  const intrinsics = exercises.map((exercise) => forward.get(exercise.intrinsic.id));
  const levels = exercises.map((exercise) => forward.get(exercise.underlying.id));
  const value = new Float64Array(paths);
  const chosen = new Int32Array(paths).fill(-1);
  for (let k = exercises.length - 1; k >= 0; k -= 1) {
    if (k < exercises.length - 1) {
      const discount = discountTo(exercises[k + 1].date) / discountTo(exercises[k].date);
      for (let p = 0; p < paths; p += 1) value[p] *= discount;
    }
    const intrinsic3 = intrinsics[k];
    const level = levels[k];
    const coefficients = regress(intrinsic3, level, value);
    for (let p = 0; p < paths; p += 1) {
      if (intrinsic3[p] <= 0) continue;
      const continuation2 = coefficients === null ? 0 : coefficients[0] + coefficients[1] * level[p] + coefficients[2] * level[p] * level[p];
      if (intrinsic3[p] > continuation2) {
        value[p] = intrinsic3[p];
        chosen[p] = k;
      }
    }
  }
  for (let p = 0; p < paths; p += 1) {
    if (chosen[p] >= 0) masks[chosen[p]][p] = 1;
  }
  for (let k = 0; k < exercises.length; k += 1) bindings.set(exercises[k].mask.id, masks[k]);
}
function scalar2(value) {
  return new Float64Array([value]);
}
var JUMP_SEED_BASE = 2654435761;
function poissonSample2(mean, generator) {
  const threshold = Math.exp(-mean);
  let count = 0;
  let product = 1;
  for (; ; ) {
    product *= generator.nextDouble();
    if (product <= threshold) return count;
    count += 1;
  }
}
function compoundPoissonJumps(paths, meanCount, jumpMean, jumpVol, seed) {
  const out = new Float64Array(paths);
  if (meanCount <= 0) return out;
  const generator = new MersenneTwister(seed);
  for (let p = 0; p < paths; p += 1) {
    const count = poissonSample2(meanCount, generator);
    const z = inverseNormalCdf(generator.nextDouble());
    out[p] = count * jumpMean + jumpVol * Math.sqrt(count) * z;
  }
  return out;
}
function identityMatrix(n) {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_2, j) => i === j ? 1 : 0));
}
function discountFactorOf(market, t) {
  if (t <= 0) return 1;
  return market.curve ? market.curve.discountFactor(t) : Math.exp(-market.rate * t);
}
function resolveModelParam(market, compiled, asset, param) {
  const override = market.modelParams?.[asset]?.[param];
  if (override !== void 0) return override;
  const fallback = compiled.modelParamDefaults.get(asset)?.get(param);
  if (fallback !== void 0) return fallback;
  throw new Error(`missing model parameter '${param}' for underlying '${asset}'`);
}
function forwardRateOf(market, a, b) {
  if (b <= a) return market.rate;
  return (Math.log(discountFactorOf(market, a)) - Math.log(discountFactorOf(market, b))) / (b - a);
}
function valueCompiled(compiled, market, paths, seed) {
  const { assetOrder, times } = compiled;
  const single = assetOrder.length === 1;
  const bindings = /* @__PURE__ */ new Map();
  for (const name of assetOrder) {
    if (isRateModel(compiled.models.get(name))) {
      bindings.set(compiled.spots.get(name).id, scalar2(0));
      bindings.set(compiled.vols.get(name).id, scalar2(0));
      bindings.set(compiled.driftAdjust.get(name).id, scalar2(0));
      continue;
    }
    const spotValue = market.spots?.[name] ?? market.spot;
    if (spotValue === void 0) throw new Error(`missing spot for underlying '${name}'`);
    bindings.set(compiled.spots.get(name).id, scalar2(spotValue));
    const volValue = market.vols?.[name] ?? market.vol;
    if (volValue === void 0) {
      if (modelNeedsVol(compiled.models.get(name))) throw new Error(`missing vol for underlying '${name}'`);
      bindings.set(compiled.vols.get(name).id, scalar2(0));
    } else {
      bindings.set(compiled.vols.get(name).id, scalar2(volValue));
    }
    const model = compiled.models.get(name);
    const effectiveVol = volValue ?? 0;
    let adjust = 0;
    if (model === "fx") {
      adjust = -resolveModelParam(market, compiled, name, "foreignRate");
    } else {
      const quanto = market.quanto?.[name];
      adjust = quanto ? -quanto.correlation * effectiveVol * quanto.fxVol : 0;
    }
    bindings.set(compiled.driftAdjust.get(name).id, scalar2(adjust));
  }
  for (const [name, inputs] of compiled.modelParamInputs) {
    for (const [param, value] of inputs) bindings.set(value.id, scalar2(resolveModelParam(market, compiled, name, param)));
  }
  for (const [name, value] of compiled.params) {
    const override = market.params?.[name];
    bindings.set(value.id, scalar2(override ?? compiled.paramDefaults.get(name) ?? 0));
  }
  let previous = 0;
  for (let k = 0; k < times.length; k += 1) {
    bindings.set(compiled.rateInputs[k].id, scalar2(forwardRateOf(market, previous, times[k])));
    previous = times[k];
  }
  for (const { time, value } of compiled.discountInputs.values()) bindings.set(value.id, scalar2(discountFactorOf(market, time)));
  if (!single && compiled.cholInputs !== null) {
    const factor = cholesky(market.correlation ?? identityMatrix(assetOrder.length));
    for (let i = 0; i < compiled.cholInputs.length; i += 1) {
      for (let j = 0; j < compiled.cholInputs[i].length; j += 1) bindings.set(compiled.cholInputs[i][j].id, scalar2(factor[i][j]));
    }
  }
  for (let k = 0; k < compiled.normals.length; k += 1) {
    bindings.set(compiled.normals[k].id, standardNormals(paths, seed + k * SEED_STRIDE >>> 0));
  }
  for (let idx = 0; idx < compiled.jumpInputs.length; idx += 1) {
    const record = compiled.jumpInputs[idx];
    const get = (p) => resolveModelParam(market, compiled, record.asset, p);
    const lambda = get("jumpIntensity");
    const jumpMean = get("jumpMean");
    const jumpVol = get("jumpVol");
    const dt = times[record.step] - (record.step === 0 ? 0 : times[record.step - 1]);
    bindings.set(record.input.id, compoundPoissonJumps(paths, lambda * dt, jumpMean, jumpVol, seed + JUMP_SEED_BASE + idx * SEED_STRIDE >>> 0));
  }
  if (compiled.rateScalarInputs.length > 0) {
    const curve = market.curve ?? new DiscountCurve([1], [market.rate]);
    const rateParam = (asset, p) => resolveModelParam(market, compiled, asset, p);
    const g2spec = (asset) => ({
      flatRate: market.rate,
      meanReversionA: rateParam(asset, "meanReversionA"),
      meanReversionB: rateParam(asset, "meanReversionB"),
      volA: rateParam(asset, "volA"),
      volB: rateParam(asset, "volB"),
      correlation: rateParam(asset, "correlation")
    });
    for (const record of compiled.rateScalarInputs) {
      const model = compiled.models.get(record.asset);
      const a = model === "hw1f" ? rateParam(record.asset, "meanReversion") : 0;
      const vol = model === "hw1f" ? rateParam(record.asset, "vol") : 0;
      let value;
      switch (record.kind) {
        case "alpha":
          value = hwAlpha(curve, a, vol, record.time);
          break;
        case "decay":
          value = hwDecay(a, record.dt);
          break;
        case "stepStd":
          value = hwStepStd(a, vol, record.dt);
          break;
        case "flat":
          value = market.rate;
          break;
        case "bondConst":
          if (model === "g2pp") {
            const spec = g2spec(record.asset);
            const t = record.time;
            const tenor = record.maturity - t;
            value = -market.rate * tenor + 0.5 * (g2ppVariance(spec, tenor) - g2ppVariance(spec, record.maturity) + g2ppVariance(spec, t));
          } else {
            value = hwBondLogA(curve, a, vol, record.time, record.maturity);
          }
          break;
        case "bondLoading":
          if (model === "g2pp") {
            const rev = record.factorIndex === 0 ? rateParam(record.asset, "meanReversionA") : rateParam(record.asset, "meanReversionB");
            value = -hwBFactor(rev, record.maturity - record.time);
          } else {
            value = -hwBFactor(a, record.maturity - record.time);
          }
          break;
      }
      bindings.set(record.input.id, scalar2(value));
    }
  }
  if (compiled.exercises.length > 0) fillExerciseMasks(compiled, bindings, paths, (t) => discountFactorOf(market, t));
  return runMonteCarlo(compiled.graph, bindings, paths, compiled.price, compiled.cashflow, compiled.riskFactors);
}
var THETA_STEP = 1 / 365;
var RATE_BUMP = 1e-4;
function shiftRate(market, dr) {
  const curve = market.curve ? new DiscountCurve(market.curve.times, market.curve.zeroRates.map((z) => z + dr)) : void 0;
  return { ...market, rate: market.rate + dr, curve };
}
function shiftPillar(curve, pillar, dr) {
  const zeroRates = curve.zeroRates.slice();
  zeroRates[pillar] += dr;
  return new DiscountCurve(curve.times, zeroRates);
}
function priceProduct(product, market, paths, seed, options = {}) {
  const depth = options.greeks ?? "full";
  const compiled = compileProduct(product);
  const { assetOrder } = compiled;
  const single = assetOrder.length === 1;
  const base = valueCompiled(compiled, market, paths, seed);
  const grad = (g, id) => g.get(id) ?? 0;
  const greeks = {};
  if (single) {
    greeks.delta = grad(base.gradients, compiled.spots.get(assetOrder[0]).id);
    greeks.vega = grad(base.gradients, compiled.vols.get(assetOrder[0]).id);
  } else {
    for (const name of assetOrder) {
      greeks[`delta_${name}`] = grad(base.gradients, compiled.spots.get(name).id);
      greeks[`vega_${name}`] = grad(base.gradients, compiled.vols.get(name).id);
    }
  }
  if (depth === "price-only") return { price: base.price, standardError: base.standardError, greeks };
  const rhoUp = valueCompiled(compiled, shiftRate(market, RATE_BUMP), paths, seed).price;
  const rhoDown = valueCompiled(compiled, shiftRate(market, -RATE_BUMP), paths, seed).price;
  greeks.rho = (rhoUp - rhoDown) / (2 * RATE_BUMP);
  if (depth === "first-order") return { price: base.price, standardError: base.standardError, greeks };
  if (single) {
    const name = assetOrder[0];
    const spotId = compiled.spots.get(name).id;
    const volId = compiled.vols.get(name).id;
    const spot2 = market.spots?.[name] ?? market.spot;
    const vol = market.vols?.[name] ?? market.vol;
    const bumped = (s, v) => ({
      ...market,
      spot: s,
      vol: v,
      spots: market.spots ? { ...market.spots, [name]: s } : void 0,
      vols: market.vols ? { ...market.vols, [name]: v } : void 0
    });
    if (spot2 !== void 0) {
      const volRef = vol ?? 0;
      const spotBump = Math.max(Math.abs(spot2) * 1e-4, 1e-6);
      const deltaUp = grad(valueCompiled(compiled, bumped(spot2 + spotBump, volRef), paths, seed).gradients, spotId);
      const deltaDown = grad(valueCompiled(compiled, bumped(spot2 - spotBump, volRef), paths, seed).gradients, spotId);
      greeks.gamma = (deltaUp - deltaDown) / (2 * spotBump);
      if (vol !== void 0 && modelNeedsVol(compiled.models.get(name))) {
        const volBump = Math.max(Math.abs(vol) * 0.05, 1e-3);
        const up = valueCompiled(compiled, bumped(spot2, vol + volBump), paths, seed).gradients;
        const down = valueCompiled(compiled, bumped(spot2, vol - volBump), paths, seed).gradients;
        greeks.vanna = (grad(up, spotId) - grad(down, spotId)) / (2 * volBump);
        greeks.volga = (grad(up, volId) - grad(down, volId)) / (2 * volBump);
      }
    }
  }
  if (market.curve) {
    const curve = market.curve;
    for (let i = 0; i < curve.times.length; i += 1) {
      const up = valueCompiled(compiled, { ...market, curve: shiftPillar(curve, i, RATE_BUMP) }, paths, seed).price;
      const down = valueCompiled(compiled, { ...market, curve: shiftPillar(curve, i, -RATE_BUMP) }, paths, seed).price;
      greeks[`rho_pillar_${i}`] = (up - down) / (2 * RATE_BUMP);
    }
  }
  const shifted = compileProduct(product, { timeShift: THETA_STEP });
  const shiftedValue = valueCompiled(shifted, market, paths, seed);
  greeks.theta = (shiftedValue.price - base.price) / THETA_STEP;
  return { price: base.price, standardError: base.standardError, greeks };
}

// stub-node:node:fs
var fail = () => {
  throw new Error("Node built-ins are unavailable in the browser build");
};
var readFileSync = fail;
var node_fs_default = new Proxy({}, { get: () => fail });

// src/lang/cli.ts
function checkScript(source) {
  return checkProduct(parseProduct(source)).map((error) => `${error.line}:${error.col}: ${error.message}`);
}
function printScript(source) {
  return printProduct(parseProduct(source));
}
function priceScript(source, market, paths, seed) {
  const product = parseProduct(source);
  const errors = checkProduct(product);
  if (errors.length > 0) throw new Error(`type errors:
${errors.map((e) => `${e.line}:${e.col}: ${e.message}`).join("\n")}`);
  return priceProduct(product, market, paths, seed, { greeks: "first-order" });
}
function parseFlags(args) {
  const flags = /* @__PURE__ */ new Map();
  for (let i = 0; i < args.length - 1; i += 1) {
    if (args[i].startsWith("--")) flags.set(args[i].slice(2), args[i + 1]);
  }
  return flags;
}
function flagNumber(flags, key, fallback) {
  const value = flags.get(key);
  return value === void 0 ? fallback : Number(value);
}
function runLangCli(argv) {
  const command = argv[0];
  const file = argv[1];
  if (command === void 0 || file === void 0) return "usage: quill <run|check|print> file.quill";
  const source = readFileSync(file, "utf8");
  if (command === "check") {
    const errors = checkScript(source);
    return errors.length === 0 ? "ok" : errors.join("\n");
  }
  if (command === "print") return printScript(source);
  if (command === "run") {
    const flags = parseFlags(argv.slice(2));
    const market = { spot: flagNumber(flags, "spot", 100), rate: flagNumber(flags, "rate", 0.03), vol: flagNumber(flags, "vol", 0.2) };
    const result = priceScript(source, market, flagNumber(flags, "paths", 1e5), flagNumber(flags, "seed", 1));
    return [
      `price   ${result.price.toFixed(6)}`,
      `stderr  ${result.standardError.toFixed(6)}`,
      `delta   ${result.greeks.delta.toFixed(6)}`,
      `vega    ${result.greeks.vega.toFixed(6)}`,
      `rho     ${result.greeks.rho.toFixed(6)}`
    ].join("\n");
  }
  return `unknown command '${command}'`;
}

// src/lang/contracts.ts
var ORIGIN = { line: 0, col: 0 };
function num(value) {
  return { kind: "num", value, pos: ORIGIN };
}
function call(callee, args) {
  return { kind: "call", callee, args, pos: ORIGIN };
}
function binary(op, left, right) {
  return { kind: "binary", op, left, right, pos: ORIGIN };
}
var konst = (value) => () => num(value);
var spot = (date, underlying) => call(underlying, [num(date)]);
var minus = (a, b) => (date, underlying) => binary("-", a(date, underlying), b(date, underlying));
var maxObs = (a, b) => (date, underlying) => call("max", [a(date, underlying), b(date, underlying)]);
function pay(date, amount) {
  return { kind: "pay", date, amount };
}
function and(left, right) {
  return { kind: "and", left, right };
}
function give(inner) {
  return { kind: "give", inner };
}
function scale2(factor, inner) {
  return { kind: "scale", factor, inner };
}
function zeroCouponBond2(date, notional) {
  return pay(date, konst(notional));
}
function europeanCall(date, strike) {
  return pay(date, maxObs(minus(spot, konst(strike)), konst(0)));
}
function europeanPut(date, strike) {
  return pay(date, maxObs(minus(konst(strike), spot), konst(0)));
}
function collect(contract, multiplier) {
  switch (contract.kind) {
    case "zero":
      return [];
    case "pay": {
      const amount = contract.amount;
      return [{ date: contract.date, amount: (date, underlying) => binary("*", num(multiplier), amount(date, underlying)) }];
    }
    case "and":
      return [...collect(contract.left, multiplier), ...collect(contract.right, multiplier)];
    case "give":
      return collect(contract.inner, -multiplier);
    case "scale":
      return collect(contract.inner, multiplier * contract.factor);
  }
}
function desugar(contract, underlying = "S", model = "gbm") {
  const cashflows = collect(contract, 1);
  const events = cashflows.map((cashflow, i) => ({
    variable: `e${i}`,
    schedule: { kind: "single", date: num(cashflow.date) },
    body: [{ kind: "pay", amount: cashflow.amount(cashflow.date, underlying), date: num(cashflow.date), currency: null, pos: ORIGIN }],
    pos: ORIGIN
  }));
  return { name: "Contract", underlyings: [{ name: underlying, model, modelParams: [], pos: ORIGIN }], params: [], vars: [], events, pos: ORIGIN };
}

// src/lang/calibrate.ts
function calibrateHestonAndPrice(product, market, config, paths, seed, options = {}) {
  const hestonUnderlying = product.underlyings.find((u) => u.model === "heston");
  if (hestonUnderlying === void 0) throw new Error("calibrateHestonAndPrice requires an underlying with model heston");
  const spot2 = market.spots?.[hestonUnderlying.name] ?? market.spot;
  if (spot2 === void 0) throw new Error(`missing spot for underlying '${hestonUnderlying.name}'`);
  const calibration = calibrateHeston(spot2, market.rate, config.quotes, { meanReversion: config.meanReversion, correlation: config.correlation }, config.initial);
  const modelParams2 = {
    kappa: config.meanReversion,
    theta: calibration.longVariance,
    xi: calibration.volOfVol,
    rho: config.correlation,
    v0: calibration.initialVariance
  };
  const calibratedMarket = {
    ...market,
    modelParams: { ...market.modelParams, [hestonUnderlying.name]: { ...market.modelParams?.[hestonUnderlying.name], ...modelParams2 } }
  };
  return { result: priceProduct(product, calibratedMarket, paths, seed, options), calibration, modelParams: modelParams2 };
}

// src/marketdata/multi-curve.ts
function projectionForward(projection, start, end) {
  return (projection.discountFactor(start) / projection.discountFactor(end) - 1) / (end - start);
}
function floatingLegValue(swap, discount, projection) {
  let value = 0;
  let previous = 0;
  for (let i = 0; i < swap.times.length; i += 1) {
    const forward = projectionForward(projection, previous, swap.times[i]);
    value += swap.accruals[i] * forward * discount.discountFactor(swap.times[i]);
    previous = swap.times[i];
  }
  return value;
}
function annuity(swap, discount) {
  let total = 0;
  for (let i = 0; i < swap.times.length; i += 1) total += swap.accruals[i] * discount.discountFactor(swap.times[i]);
  return total;
}
function multiCurveSwapValue(swap, discount, projection) {
  return floatingLegValue(swap, discount, projection) - swap.fixedRate * annuity(swap, discount);
}
function multiCurveParRate(swap, discount, projection) {
  return floatingLegValue(swap, discount, projection) / annuity(swap, discount);
}
function bootstrapProjectionCurve(discount, quotes, accrual) {
  const times = [];
  const zeroRates = [];
  for (const quote of quotes) {
    const periods = Math.round(quote.maturity / accrual);
    const swapTimes = Array.from({ length: periods }, (_, i) => (i + 1) * accrual);
    const swapAccruals = swapTimes.map(() => accrual);
    const swap = { times: swapTimes, accruals: swapAccruals, fixedRate: quote.rate };
    let low = -0.02;
    let high = 0.5;
    for (let iteration = 0; iteration < 200; iteration += 1) {
      const mid = 0.5 * (low + high);
      times.push(quote.maturity);
      zeroRates.push(mid);
      const candidate = new DiscountCurve([...times], [...zeroRates]);
      const par = multiCurveParRate(swap, discount, candidate);
      times.pop();
      zeroRates.pop();
      if (par > quote.rate) high = mid;
      else low = mid;
    }
    times.push(quote.maturity);
    zeroRates.push(0.5 * (low + high));
  }
  return new DiscountCurve(times, zeroRates);
}

// src/marketdata/xccy.ts
function fxForward(spot2, domesticDiscount, foreignDiscount) {
  return spot2 * foreignDiscount / domesticDiscount;
}
function csaPresentValue(amount, time, collateral) {
  return amount * collateral.discountFactor(time);
}
function basisAdjustedCurve(base, basisSpread) {
  return new DiscountCurve([...base.times], base.zeroRates.map((rate) => rate + basisSpread));
}

// src/instruments/rates-instruments.ts
function depositRate(curve, tenor) {
  return (1 / curve.discountFactor(tenor) - 1) / tenor;
}
function fraRate(curve, start, end) {
  return projectionForward(curve, start, end);
}
function futureConvexityAdjustment(vol, start, end) {
  return 0.5 * vol * vol * start * end;
}
function futureRate(curve, vol, start, end) {
  return fraRate(curve, start, end) + futureConvexityAdjustment(vol, start, end);
}

// src/marketdata/interpolation.ts
function monotoneCubic(xs, ys) {
  const n = xs.length;
  const delta = new Array(n - 1);
  for (let i = 0; i < n - 1; i += 1) delta[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
  const slope = new Array(n);
  slope[0] = delta[0];
  slope[n - 1] = delta[n - 2];
  for (let i = 1; i < n - 1; i += 1) slope[i] = delta[i - 1] * delta[i] <= 0 ? 0 : (delta[i - 1] + delta[i]) / 2;
  for (let i = 0; i < n - 1; i += 1) {
    if (delta[i] === 0) {
      slope[i] = 0;
      slope[i + 1] = 0;
      continue;
    }
    const a = slope[i] / delta[i];
    const b = slope[i + 1] / delta[i];
    const magnitude = a * a + b * b;
    if (magnitude > 9) {
      const scale3 = 3 / Math.sqrt(magnitude);
      slope[i] = scale3 * a * delta[i];
      slope[i + 1] = scale3 * b * delta[i];
    }
  }
  return (x) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (i < n - 1 && xs[i + 1] < x) i += 1;
    const h = xs[i + 1] - xs[i];
    const t = (x - xs[i]) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    return h00 * ys[i] + h10 * h * slope[i] + h01 * ys[i + 1] + h11 * h * slope[i + 1];
  };
}

// src/marketdata/equity-forward.ts
function equityForward(spot2, rate, maturity, dividends) {
  let presentValue = 0;
  for (const dividend of dividends) {
    if (dividend.time <= maturity) presentValue += dividend.amount * Math.exp(-rate * dividend.time);
  }
  return (spot2 - presentValue) * Math.exp(rate * maturity);
}
function equityForwardWithYield(spot2, rate, dividendYield, maturity) {
  return spot2 * Math.exp((rate - dividendYield) * maturity);
}

// src/marketdata/ssvi.ts
function ssviTotalVariance(k, atmVariance, params) {
  const phi = params.eta / Math.pow(atmVariance, params.gamma);
  const term = phi * k + params.rho;
  return 0.5 * atmVariance * (1 + params.rho * phi * k + Math.sqrt(term * term + (1 - params.rho * params.rho)));
}
function ssviImpliedVol(k, maturity, atmVariance, params) {
  return Math.sqrt(ssviTotalVariance(k, atmVariance, params) / maturity);
}

// src/risk/exposure.ts
function europeanTrade(strike, maturity, rate, vol, isCall, quantity) {
  return {
    mtm(spot2, time) {
      if (time >= maturity) return quantity * Math.max(isCall ? spot2 - strike : strike - spot2, 0);
      return quantity * blackScholes({ spot: spot2, strike, rate, vol, maturity: maturity - time, isCall }).price;
    }
  };
}
function forwardTrade(strike, maturity, rate, quantity) {
  return {
    mtm(spot2, time) {
      if (time >= maturity) return quantity * (spot2 - strike);
      return quantity * (spot2 - strike * Math.exp(-rate * (maturity - time)));
    }
  };
}
function quantileOf(values, q) {
  const sorted = Float64Array.from(values).sort();
  const index = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
  return sorted[index];
}
function simulateExposure(portfolio, market, config) {
  const steps = config.grid.length;
  const generator = new MersenneTwister(config.seed);
  const mtmPaths = config.grid.map(() => new Float64Array(config.paths));
  const spotPaths = config.grid.map(() => new Float64Array(config.paths));
  for (let p = 0; p < config.paths; p += 1) {
    let logSpot = Math.log(market.spot);
    let previousTime2 = 0;
    for (let k = 0; k < steps; k += 1) {
      const dt = config.grid[k] - previousTime2;
      logSpot += (market.rate - 0.5 * market.vol * market.vol) * dt + market.vol * Math.sqrt(dt) * inverseNormalCdf(generator.nextDouble());
      const spot2 = Math.exp(logSpot);
      let value = 0;
      for (const trade of portfolio) value += trade.mtm(spot2, config.grid[k]);
      spotPaths[k][p] = spot2;
      mtmPaths[k][p] = value;
      previousTime2 = config.grid[k];
    }
  }
  const epe = [];
  const ene = [];
  const pfe = [];
  const collateralEpe = [];
  for (let k = 0; k < steps; k += 1) {
    const positive = new Float64Array(config.paths);
    let sumPositive = 0;
    let sumNegative = 0;
    let sumCollateral = 0;
    for (let p = 0; p < config.paths; p += 1) {
      const value = mtmPaths[k][p];
      const exposure = Math.max(value, 0);
      positive[p] = exposure;
      sumPositive += exposure;
      sumNegative += Math.min(value, 0);
      const collateral = k > 0 ? Math.max(mtmPaths[k - 1][p] - config.collateralThreshold, 0) : 0;
      sumCollateral += Math.max(value - collateral, 0);
    }
    epe.push(sumPositive / config.paths);
    ene.push(sumNegative / config.paths);
    pfe.push(quantileOf(positive, config.quantile));
    collateralEpe.push(sumCollateral / config.paths);
  }
  let runningMax = 0;
  let integral = 0;
  let previousTime = 0;
  for (let k = 0; k < steps; k += 1) {
    runningMax = Math.max(runningMax, epe[k]);
    integral += runningMax * (config.grid[k] - previousTime);
    previousTime = config.grid[k];
  }
  const eepe = integral / config.grid[steps - 1];
  return { times: config.grid, epe, ene, pfe, collateralEpe, eepe, spotPaths, mtmPaths };
}

// src/risk/xva-suite.ts
function survival(hazard, time) {
  return Math.exp(-hazard * time);
}
function computeXva(profile, spec) {
  let cva = 0;
  let dva = 0;
  let fva = 0;
  let mva = 0;
  let previousTime = 0;
  for (let k = 0; k < profile.times.length; k += 1) {
    const time = profile.times[k];
    const discount = Math.exp(-spec.rate * time);
    const counterpartyDefault = survival(spec.hazardRate, previousTime) - survival(spec.hazardRate, time);
    const ownDefault = survival(spec.ownHazardRate, previousTime) - survival(spec.ownHazardRate, time);
    const dt = time - previousTime;
    cva += (1 - spec.recovery) * profile.epe[k] * discount * counterpartyDefault;
    dva += (1 - spec.ownRecovery) * -profile.ene[k] * discount * ownDefault;
    const unfunded = Math.max(profile.collateralEpe[k], 0);
    fva += spec.fundingSpread * unfunded * discount * survival(spec.hazardRate, time) * dt;
    const initialMargin = Math.max(profile.pfe[k] - profile.epe[k], 0);
    mva += spec.fundingSpread * initialMargin * discount * survival(spec.hazardRate, time) * dt;
    previousTime = time;
  }
  return { cva, dva, fva, mva };
}
function wrongWayCva(profile, spec) {
  const steps = profile.times.length;
  const paths = profile.spotPaths[0].length;
  let independent = 0;
  let wrongWay = 0;
  let previousTime = 0;
  for (let k = 0; k < steps; k += 1) {
    const time = profile.times[k];
    const discount = Math.exp(-spec.rate * time);
    const baseDefault = Math.exp(-spec.hazardRate * previousTime) - Math.exp(-spec.hazardRate * time);
    let sumExposure = 0;
    let sumWeighted = 0;
    let sumWeights = 0;
    for (let p = 0; p < paths; p += 1) {
      const exposure = Math.max(profile.mtmPaths[k][p], 0);
      const stress = Math.exp(spec.correlation * (spec.spot - profile.spotPaths[k][p]) / spec.spot);
      sumExposure += exposure;
      sumWeighted += exposure * stress;
      sumWeights += stress;
    }
    independent += (1 - spec.recovery) * (sumExposure / paths) * discount * baseDefault;
    wrongWay += (1 - spec.recovery) * (sumWeighted / sumWeights) * discount * baseDefault;
    previousTime = time;
  }
  return { independent, wrongWay };
}

// src/risk/frtb.ts
function deltaRiskCharge(sensitivities, correlations) {
  const buckets = /* @__PURE__ */ new Map();
  for (const sensitivity of sensitivities) {
    const weighted = sensitivity.sensitivity * sensitivity.riskWeight;
    const existing = buckets.get(sensitivity.bucket);
    if (existing !== void 0) existing.push(weighted);
    else buckets.set(sensitivity.bucket, [weighted]);
  }
  const bucketKeys = [...buckets.keys()];
  const kb = [];
  const sb = [];
  for (const key of bucketKeys) {
    const weighted = buckets.get(key);
    let variance = 0;
    let sum = 0;
    for (let i = 0; i < weighted.length; i += 1) {
      sum += weighted[i];
      for (let j = 0; j < weighted.length; j += 1) variance += (i === j ? 1 : correlations.withinBucket) * weighted[i] * weighted[j];
    }
    kb.push(Math.sqrt(Math.max(variance, 0)));
    sb.push(sum);
  }
  let total = 0;
  for (let b = 0; b < bucketKeys.length; b += 1) {
    for (let c = 0; c < bucketKeys.length; c += 1) {
      total += (b === c ? 1 : correlations.acrossBucket) * (b === c ? kb[b] * kb[b] : sb[b] * sb[c]);
    }
  }
  return Math.sqrt(Math.max(total, 0));
}
function vegaRiskCharge(sensitivities, correlations) {
  return deltaRiskCharge(sensitivities, correlations);
}
function curvatureRiskCharge(positions, correlations) {
  const buckets = /* @__PURE__ */ new Map();
  for (const position of positions) {
    const cvr = -Math.min(position.cvrUp, position.cvrDown);
    const existing = buckets.get(position.bucket);
    if (existing !== void 0) existing.push(cvr);
    else buckets.set(position.bucket, [cvr]);
  }
  const keys = [...buckets.keys()];
  const kb = [];
  const sb = [];
  for (const key of keys) {
    const cvr = buckets.get(key);
    let variance = 0;
    let sum = 0;
    for (let i = 0; i < cvr.length; i += 1) {
      sum += cvr[i];
      for (let j = 0; j < cvr.length; j += 1) {
        const psi2 = cvr[i] < 0 && cvr[j] < 0 ? 0 : 1;
        variance += i === j ? Math.max(cvr[i], 0) * Math.max(cvr[i], 0) : correlations.withinBucket * cvr[i] * cvr[j] * psi2;
      }
    }
    kb.push(Math.sqrt(Math.max(variance, 0)));
    sb.push(sum);
  }
  let total = 0;
  for (let b = 0; b < keys.length; b += 1) {
    for (let c = 0; c < keys.length; c += 1) {
      if (b === c) total += kb[b] * kb[b];
      else {
        const psi2 = sb[b] < 0 && sb[c] < 0 ? 0 : 1;
        total += correlations.acrossBucket * sb[b] * sb[c] * psi2;
      }
    }
  }
  return Math.sqrt(Math.max(total, 0));
}
function standardisedCapital(delta, vega, curvature, defaultCharge) {
  return delta + vega + curvature + defaultCharge;
}
function defaultRiskCharge(exposures) {
  let longCharge = 0;
  let shortCharge = 0;
  for (const exposure of exposures) {
    const weighted = exposure.jtd * exposure.riskWeight;
    if (weighted >= 0) longCharge += weighted;
    else shortCharge += -weighted;
  }
  return Math.max(longCharge - 0.5 * shortCharge, 0);
}

// src/risk/xva-full.ts
function europeanFullTrade(strike, maturity, rate, vol, isCall, quantity) {
  return {
    mtm(spot2, time) {
      if (time >= maturity) return quantity * Math.max(isCall ? spot2 - strike : strike - spot2, 0);
      return quantity * blackScholes({ spot: spot2, strike, rate, vol, maturity: maturity - time, isCall }).price;
    },
    delta(spot2, time) {
      if (time >= maturity) return quantity * (isCall ? spot2 > strike ? 1 : 0 : spot2 < strike ? -1 : 0);
      return quantity * blackScholes({ spot: spot2, strike, rate, vol, maturity: maturity - time, isCall }).delta;
    }
  };
}
function simulateFullExposure(portfolio, market, config) {
  const steps = config.grid.length;
  const generator = new MersenneTwister(config.seed);
  const mtm = config.grid.map(() => new Float64Array(config.paths));
  const im = config.grid.map(() => new Float64Array(config.paths));
  for (let p = 0; p < config.paths; p += 1) {
    let logSpot = Math.log(market.spot);
    let previous = 0;
    for (let k = 0; k < steps; k += 1) {
      const dt = config.grid[k] - previous;
      logSpot += (market.rate - 0.5 * market.vol * market.vol) * dt + market.vol * Math.sqrt(dt) * inverseNormalCdf(generator.nextDouble());
      const spot2 = Math.exp(logSpot);
      let value = 0;
      let delta = 0;
      for (const trade of portfolio) {
        value += trade.mtm(spot2, config.grid[k]);
        delta += trade.delta(spot2, config.grid[k]);
      }
      mtm[k][p] = value;
      im[k][p] = config.imRiskWeight * Math.abs(delta) * spot2;
      previous = config.grid[k];
    }
  }
  const epe = [];
  const collateralEpe = [];
  const imProfile = [];
  for (let k = 0; k < steps; k += 1) {
    let sumPositive = 0;
    let sumCollateral = 0;
    let sumIm = 0;
    for (let p = 0; p < config.paths; p += 1) {
      sumPositive += Math.max(mtm[k][p], 0);
      const lagIndex = k - config.mporSteps;
      const lagged = lagIndex >= 0 ? mtm[lagIndex][p] : 0;
      const required = Math.max(lagged - config.threshold, 0);
      const collateral = required >= config.minimumTransfer ? required : 0;
      sumCollateral += Math.max(mtm[k][p] - collateral, 0);
      sumIm += im[k][p];
    }
    epe.push(sumPositive / config.paths);
    collateralEpe.push(sumCollateral / config.paths);
    imProfile.push(sumIm / config.paths);
  }
  return { times: config.grid, epe, collateralEpe, imProfile };
}
function computeFullXva(profile, spec) {
  let cva = 0;
  let fva = 0;
  let mva = 0;
  let previous = 0;
  for (let k = 0; k < profile.times.length; k += 1) {
    const time = profile.times[k];
    const discount = Math.exp(-spec.rate * time);
    const survival2 = Math.exp(-spec.hazardRate * time);
    const defaultProbability2 = Math.exp(-spec.hazardRate * previous) - survival2;
    const dt = time - previous;
    cva += (1 - spec.recovery) * profile.collateralEpe[k] * discount * defaultProbability2;
    fva += spec.fundingSpread * profile.collateralEpe[k] * discount * survival2 * dt;
    mva += spec.fundingSpread * profile.imProfile[k] * discount * survival2 * dt;
    previous = time;
  }
  return { cva, fva, mva };
}

// src/risk/irc.ts
function incrementalRiskCharge(spec) {
  const generator = new MersenneTwister(spec.seed);
  const factorLoading = Math.sqrt(spec.correlation);
  const idiosyncratic = Math.sqrt(1 - spec.correlation);
  const thresholds = spec.obligors.map((obligor) => inverseNormalCdf(obligor.defaultProbability));
  const losses = new Float64Array(spec.scenarios);
  let expected = 0;
  for (let s = 0; s < spec.scenarios; s += 1) {
    const systemic = inverseNormalCdf(generator.nextDouble());
    let loss = 0;
    for (let i = 0; i < spec.obligors.length; i += 1) {
      const latent = factorLoading * systemic + idiosyncratic * inverseNormalCdf(generator.nextDouble());
      if (latent < thresholds[i]) loss += spec.obligors[i].exposure * spec.obligors[i].lossGivenDefault;
    }
    losses[s] = loss;
    expected += loss;
  }
  expected /= spec.scenarios;
  const sorted = Float64Array.from(losses).sort();
  const index = Math.min(sorted.length - 1, Math.floor(spec.confidence * sorted.length));
  return sorted[index] - expected;
}

// src/risk/stressed-var.ts
function stressedVar(sensitivities, covariance, confidence, stressMultiplier) {
  const stressed = covariance.map((row) => row.map((value) => value * stressMultiplier * stressMultiplier));
  return parametricVar(sensitivities, stressed, confidence);
}

// src/risk/pnl-attribution.ts
function attributePnl(scenario) {
  const greeks = blackScholes(scenario);
  const after = blackScholes({
    spot: scenario.spot + scenario.spotMove,
    strike: scenario.strike,
    rate: scenario.rate,
    vol: scenario.vol + scenario.volMove,
    maturity: scenario.maturity - scenario.timeMove,
    isCall: scenario.isCall
  });
  const deltaPnl = greeks.delta * scenario.spotMove;
  const gammaPnl = 0.5 * greeks.gamma * scenario.spotMove * scenario.spotMove;
  const vegaPnl = greeks.vega * scenario.volMove;
  const thetaPnl = greeks.theta * scenario.timeMove;
  const explained = deltaPnl + gammaPnl + vegaPnl + thetaPnl;
  const actual = after.price - greeks.price;
  return { actual, delta: deltaPnl, gamma: gammaPnl, vega: vegaPnl, theta: thetaPnl, explained, unexplained: actual - explained };
}

// src/models/equity/hybrid.ts
function priceHybridCall(spec) {
  const dt = spec.maturity / spec.steps;
  const sqrtDt = Math.sqrt(dt);
  const rhoComplement = Math.sqrt(1 - spec.correlation * spec.correlation);
  const generator = new MersenneTwister(spec.seed);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    let logSpot = Math.log(spec.spot);
    let rate = spec.shortRate;
    let integral = 0;
    for (let step = 0; step < spec.steps; step += 1) {
      integral += rate * dt;
      const z1 = inverseNormalCdf(generator.nextDouble());
      const z2 = spec.correlation * z1 + rhoComplement * inverseNormalCdf(generator.nextDouble());
      logSpot += (rate - 0.5 * spec.equityVol * spec.equityVol) * dt + spec.equityVol * sqrtDt * z1;
      rate += spec.rateMeanReversion * (spec.rateLong - rate) * dt + spec.rateVol * sqrtDt * z2;
    }
    estimator.push(Math.exp(-integral) * Math.max(Math.exp(logSpot) - spec.strike, 0));
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}

// src/models/equity/rough-bergomi.ts
function priceRoughBergomiCall(spec) {
  const n = spec.steps;
  const dt = spec.maturity / n;
  const alpha2 = spec.hurst - 0.5;
  const sqrtDt = Math.sqrt(dt);
  const rhoComplement = Math.sqrt(1 - spec.correlation * spec.correlation);
  const kernel = new Float64Array(n + 1);
  for (let k = 1; k <= n; k += 1) kernel[k] = Math.pow(dt, alpha2 + 0.5) * (Math.pow(k, alpha2 + 1) - Math.pow(k - 1, alpha2 + 1)) / (alpha2 + 1);
  const orthogonalScale = Math.pow(dt, alpha2 + 0.5) * Math.sqrt(1 / (2 * alpha2 + 1) - 1 / ((alpha2 + 1) * (alpha2 + 1)));
  const scale3 = spec.volOfVol * Math.sqrt(2 * spec.hurst);
  const generator = new MersenneTwister(spec.seed);
  const discount = Math.exp(-spec.rate * spec.maturity);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    const driving = new Float64Array(n);
    const orthogonal = new Float64Array(n);
    const independent = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
      driving[i] = inverseNormalCdf(generator.nextDouble());
      orthogonal[i] = inverseNormalCdf(generator.nextDouble());
      independent[i] = inverseNormalCdf(generator.nextDouble());
    }
    let logSpot = Math.log(spec.spot);
    let variance = spec.forwardVariance;
    for (let i = 1; i <= n; i += 1) {
      const spotShock = spec.correlation * (driving[i - 1] * sqrtDt) + rhoComplement * (independent[i - 1] * sqrtDt);
      logSpot += (spec.rate - 0.5 * variance) * dt + Math.sqrt(variance) * spotShock;
      let volterra = orthogonalScale * orthogonal[i - 1];
      for (let k = 1; k <= i; k += 1) volterra += kernel[k] * driving[i - k];
      variance = spec.forwardVariance * Math.exp(scale3 * volterra - 0.5 * spec.volOfVol * spec.volOfVol * Math.pow(i * dt, 2 * spec.hurst));
    }
    estimator.push(discount * Math.max(Math.exp(logSpot) - spec.strike, 0));
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}

// src/models/equity/slv.ts
function priceStochasticLocalVolCall(spec, leverage) {
  const dt = spec.maturity / spec.steps;
  const sqrtDt = Math.sqrt(dt);
  const rhoComplement = Math.sqrt(1 - spec.correlation * spec.correlation);
  const generator = new MersenneTwister(spec.seed);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    let logSpot = Math.log(spec.spot);
    let variance = spec.initialVariance;
    for (let step = 0; step < spec.steps; step += 1) {
      const time = step * dt;
      const z1 = inverseNormalCdf(generator.nextDouble());
      const z2 = spec.correlation * z1 + rhoComplement * inverseNormalCdf(generator.nextDouble());
      const positiveVariance = variance > 0 ? variance : 0;
      const local = leverage(Math.exp(logSpot), time);
      const localVol2 = local * Math.sqrt(positiveVariance);
      logSpot += (spec.rate - 0.5 * localVol2 * localVol2) * dt + localVol2 * sqrtDt * z1;
      variance += spec.meanReversion * (spec.longVariance - positiveVariance) * dt + spec.volOfVol * Math.sqrt(positiveVariance) * sqrtDt * z2;
    }
    estimator.push(Math.exp(-spec.rate * spec.maturity) * Math.max(Math.exp(logSpot) - spec.strike, 0));
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}

// src/models/equity/slv-calibration.ts
function conditionalVariance(spots, variances, bins) {
  const paths = spots.length;
  const order = Array.from({ length: paths }, (_, i) => i).sort((a, b) => spots[a] - spots[b]);
  const conditional = new Float64Array(paths);
  const binSize = Math.ceil(paths / bins);
  for (let start = 0; start < paths; start += binSize) {
    const end = Math.min(start + binSize, paths);
    let sum = 0;
    for (let k = start; k < end; k += 1) sum += variances[order[k]];
    const mean = sum / (end - start);
    for (let k = start; k < end; k += 1) conditional[order[k]] = mean;
  }
  return conditional;
}
function calibrateAndPriceSlvCall(spec, strike, targetLocalVol) {
  const dt = spec.maturity / spec.steps;
  const sqrtDt = Math.sqrt(dt);
  const rhoComplement = Math.sqrt(1 - spec.correlation * spec.correlation);
  const generator = new MersenneTwister(spec.seed);
  const logSpot = new Float64Array(spec.paths).fill(Math.log(spec.spot));
  const variance = new Float64Array(spec.paths).fill(spec.initialVariance);
  const spots = new Float64Array(spec.paths);
  for (let step = 0; step < spec.steps; step += 1) {
    const time = step * dt;
    for (let p = 0; p < spec.paths; p += 1) spots[p] = Math.exp(logSpot[p]);
    const conditional = conditionalVariance(spots, variance, spec.bins);
    for (let p = 0; p < spec.paths; p += 1) {
      const leverage = targetLocalVol(spots[p], time) / Math.sqrt(Math.max(conditional[p], 1e-8));
      const localVol2 = leverage * Math.sqrt(Math.max(variance[p], 0));
      const z1 = inverseNormalCdf(generator.nextDouble());
      const z2 = spec.correlation * z1 + rhoComplement * inverseNormalCdf(generator.nextDouble());
      logSpot[p] += (spec.rate - 0.5 * localVol2 * localVol2) * dt + localVol2 * sqrtDt * z1;
      const positiveVariance = Math.max(variance[p], 0);
      variance[p] += spec.meanReversion * (spec.longVariance - positiveVariance) * dt + spec.volOfVol * Math.sqrt(positiveVariance) * sqrtDt * z2;
    }
  }
  const discount = Math.exp(-spec.rate * spec.maturity);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) estimator.push(discount * Math.max(Math.exp(logSpot[p]) - strike, 0));
  return { price: estimator.mean, standardError: estimator.standardError };
}

// src/models/equity/heston-qe.ts
var PSI_CRITICAL = 1.5;
function priceHestonCallQe(spec) {
  const dt = spec.maturity / spec.steps;
  const kappa = spec.meanReversion;
  const theta = spec.longVariance;
  const xi = spec.volOfVol;
  const rho = spec.correlation;
  const expKappa = Math.exp(-kappa * dt);
  const gamma = 0.5;
  const k0 = -rho * kappa * theta * dt / xi;
  const k1 = gamma * dt * (kappa * rho / xi - 0.5) - rho / xi;
  const k2 = gamma * dt * (kappa * rho / xi - 0.5) + rho / xi;
  const k3 = gamma * dt * (1 - rho * rho);
  const k4 = gamma * dt * (1 - rho * rho);
  const generator = new MersenneTwister(spec.seed);
  const discount = Math.exp(-spec.rate * spec.maturity);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    let logSpot = Math.log(spec.spot);
    let variance = spec.initialVariance;
    for (let step = 0; step < spec.steps; step += 1) {
      const mean = theta + (variance - theta) * expKappa;
      const s2 = variance * xi * xi * expKappa * (1 - expKappa) / kappa + theta * xi * xi * (1 - expKappa) * (1 - expKappa) / (2 * kappa);
      const psi2 = s2 / (mean * mean);
      let nextVariance;
      const zv = inverseNormalCdf(generator.nextDouble());
      if (psi2 <= PSI_CRITICAL) {
        const invPsi = 2 / psi2;
        const b2 = invPsi - 1 + Math.sqrt(invPsi) * Math.sqrt(invPsi - 1);
        const a = mean / (1 + b2);
        const b = Math.sqrt(b2);
        nextVariance = a * (b + zv) * (b + zv);
      } else {
        const pProb = (psi2 - 1) / (psi2 + 1);
        const beta = (1 - pProb) / mean;
        const uniform = generator.nextDouble();
        nextVariance = uniform <= pProb ? 0 : Math.log((1 - pProb) / (1 - uniform)) / beta;
      }
      const zs = inverseNormalCdf(generator.nextDouble());
      logSpot += spec.rate * dt + k0 + k1 * variance + k2 * nextVariance + Math.sqrt(k3 * variance + k4 * nextVariance) * zs;
      variance = nextVariance;
    }
    estimator.push(discount * Math.max(Math.exp(logSpot) - spec.strike, 0));
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}

// src/models/rates/hull-white-1f.ts
function bFactor3(a, tenor) {
  return (1 - Math.exp(-a * tenor)) / a;
}
function instantaneousForward(curve, t) {
  const h = 1e-4;
  const left = Math.max(t - h, 1e-8);
  const right = t + h;
  return (Math.log(curve.discountFactor(left)) - Math.log(curve.discountFactor(right))) / (right - left);
}
function alpha(spec, t) {
  const a = spec.meanReversion;
  return instantaneousForward(spec.curve, t) + spec.vol * spec.vol / (2 * a * a) * (1 - Math.exp(-a * t)) * (1 - Math.exp(-a * t));
}
function hullWhiteBond(spec, t, maturity, shortRate) {
  const a = spec.meanReversion;
  const b = bFactor3(a, maturity - t);
  const ratio = spec.curve.discountFactor(maturity) / spec.curve.discountFactor(t);
  const logA = Math.log(ratio) + b * instantaneousForward(spec.curve, t) - spec.vol * spec.vol / (4 * a) * (1 - Math.exp(-2 * a * t)) * b * b;
  return Math.exp(logA - b * shortRate);
}
function zeroBondPut2(spec, expiry, bondMaturity, strike) {
  const a = spec.meanReversion;
  const sigmaP = spec.vol * bFactor3(a, bondMaturity - expiry) * Math.sqrt((1 - Math.exp(-2 * a * expiry)) / (2 * a));
  const pBond = spec.curve.discountFactor(bondMaturity);
  const pExpiry = spec.curve.discountFactor(expiry);
  const h = Math.log(pBond / (strike * pExpiry)) / sigmaP + 0.5 * sigmaP;
  return strike * pExpiry * normalCdf(-h + sigmaP) - pBond * normalCdf(-h);
}
function couponFlows3(swaption) {
  return swaption.accruals.map((accrual, i) => i === swaption.accruals.length - 1 ? 1 + swaption.fixedRate * accrual : swaption.fixedRate * accrual);
}
function couponBond2(spec, swaption, flows, shortRate) {
  let value = 0;
  for (let i = 0; i < swaption.times.length; i += 1) value += flows[i] * hullWhiteBond(spec, swaption.expiry, swaption.times[i], shortRate);
  return value;
}
function hullWhitePayerSwaption(spec, swaption) {
  const flows = couponFlows3(swaption);
  let low = -1;
  let high = 1;
  for (let iteration = 0; iteration < 200; iteration += 1) {
    const mid = 0.5 * (low + high);
    if (couponBond2(spec, swaption, flows, mid) > 1) low = mid;
    else high = mid;
  }
  const criticalRate = 0.5 * (low + high);
  let price = 0;
  for (let i = 0; i < swaption.times.length; i += 1) {
    const strike = hullWhiteBond(spec, swaption.expiry, swaption.times[i], criticalRate);
    price += flows[i] * zeroBondPut2(spec, swaption.expiry, swaption.times[i], strike);
  }
  return price;
}
function hullWhitePayerSwaptionMc(spec, swaption, steps, paths, seed) {
  const a = spec.meanReversion;
  const dt = swaption.expiry / steps;
  const decay = Math.exp(-a * dt);
  const stepStd = spec.vol * Math.sqrt((1 - Math.exp(-2 * a * dt)) / (2 * a));
  const flows = couponFlows3(swaption);
  const generator = new MersenneTwister(seed);
  let total = 0;
  for (let p = 0; p < paths; p += 1) {
    let x = 0;
    let integral = 0;
    for (let step = 0; step < steps; step += 1) {
      const time = step * dt;
      integral += (x + alpha(spec, time)) * dt;
      x = x * decay + stepStd * inverseNormalCdf(generator.nextDouble());
    }
    const shortRate = x + alpha(spec, swaption.expiry);
    const swapValue2 = 1 - couponBond2(spec, swaption, flows, shortRate);
    total += Math.exp(-integral) * Math.max(swapValue2, 0);
  }
  return total / paths;
}

// src/models/rates/lmm-multifactor.ts
function factorMatrix(spec) {
  const n = spec.rateCount;
  const correlation = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_2, j) => Math.exp(-spec.correlationDecay * Math.abs(i - j))));
  return cholesky(correlation);
}
function simulate(spec, expiryIndex, observe) {
  const n = spec.rateCount;
  const tau = spec.accrual;
  const factor = factorMatrix(spec);
  const generator = new MersenneTwister(spec.seed);
  const estimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    const forwards = new Float64Array(n).fill(spec.initialForward);
    let bank = 1;
    for (let m = 0; m < expiryIndex; m += 1) {
      bank *= 1 + tau * forwards[m];
      const shocks = new Array(n);
      for (let q = 0; q < n; q += 1) shocks[q] = inverseNormalCdf(generator.nextDouble());
      for (let i = m + 1; i < n; i += 1) {
        let drift = 0;
        for (let j = m + 1; j <= i; j += 1) {
          let rho = 0;
          for (let q = 0; q <= Math.min(i, j); q += 1) rho += factor[i][q] * factor[j][q];
          drift += tau * rho * spec.vols[j] * forwards[j] / (1 + tau * forwards[j]);
        }
        drift *= spec.vols[i];
        let diffusion = 0;
        for (let q = 0; q <= i; q += 1) diffusion += factor[i][q] * shocks[q];
        forwards[i] *= Math.exp((drift - 0.5 * spec.vols[i] * spec.vols[i]) * tau + spec.vols[i] * Math.sqrt(tau) * diffusion);
      }
    }
    bank *= 1 + tau * forwards[expiryIndex];
    estimator.push(observe(forwards, bank));
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}
function lmmCaplet(spec, capletIndex, strike) {
  const tau = spec.accrual;
  return simulate(spec, capletIndex, (forwards, bank) => tau * Math.max(forwards[capletIndex] - strike, 0) / bank);
}
function lmmSwaption(spec, expiryIndex, strike) {
  const tau = spec.accrual;
  const n = spec.rateCount;
  return simulate(spec, expiryIndex, (forwards, bank) => {
    let bond = 1;
    let annuity2 = 0;
    for (let i = expiryIndex; i < n; i += 1) {
      bond /= 1 + tau * forwards[i];
      annuity2 += tau * bond;
    }
    const swapRate = (1 - bond) / annuity2;
    return annuity2 * Math.max(swapRate - strike, 0) / bank;
  });
}
function blackCapletPrice(initialForward, strike, vol, fixingTime, accrual, discount) {
  const sqrtT = Math.sqrt(fixingTime);
  const d1 = (Math.log(initialForward / strike) + 0.5 * vol * vol * fixingTime) / (vol * sqrtT);
  const d2 = d1 - vol * sqrtT;
  return discount * accrual * (initialForward * normalCdf(d1) - strike * normalCdf(d2));
}

// src/models/credit/cir.ts
function cirSurvival(spec, maturity) {
  const { initialIntensity: lambda0, meanReversion: kappa, longIntensity: theta, vol: sigma } = spec;
  const gamma = Math.sqrt(kappa * kappa + 2 * sigma * sigma);
  const expGammaT = Math.exp(gamma * maturity);
  const denominator = (gamma + kappa) * (expGammaT - 1) + 2 * gamma;
  const b = 2 * (expGammaT - 1) / denominator;
  const a = Math.pow(2 * gamma * Math.exp((kappa + gamma) * maturity / 2) / denominator, 2 * kappa * theta / (sigma * sigma));
  return a * Math.exp(-b * lambda0);
}

// src/models/credit/merton-structural.ts
function distanceTerms(spec) {
  const sqrtT = Math.sqrt(spec.maturity);
  const d1 = (Math.log(spec.firmValue / spec.debt) + (spec.rate + 0.5 * spec.assetVol * spec.assetVol) * spec.maturity) / (spec.assetVol * sqrtT);
  return { d1, d2: d1 - spec.assetVol * sqrtT };
}
function defaultProbability(spec) {
  return normalCdf(-distanceTerms(spec).d2);
}
function riskyBondValue(spec) {
  const { d1, d2 } = distanceTerms(spec);
  return spec.firmValue * normalCdf(-d1) + spec.debt * Math.exp(-spec.rate * spec.maturity) * normalCdf(d2);
}
function creditSpread(spec) {
  const riskFree = spec.debt * Math.exp(-spec.rate * spec.maturity);
  return -(1 / spec.maturity) * Math.log(riskyBondValue(spec) / riskFree);
}

// src/models/rates/jarrow-yildirim.ts
function nominalBond(spec) {
  return Math.exp(-spec.nominalRate * spec.maturity);
}
function realBond(spec) {
  return Math.exp(-spec.realRate * spec.maturity);
}
function forwardIndex(spec) {
  return spec.indexLevel * realBond(spec) / nominalBond(spec);
}
function breakevenInflation(spec) {
  return Math.pow(forwardIndex(spec) / spec.indexLevel, 1 / spec.maturity) - 1;
}
function inflationLinkedZeroCouponBond(spec) {
  return nominalBond(spec) * forwardIndex(spec) / spec.indexLevel;
}

// src/models/rates/jarrow-yildirim-stochastic.ts
function nominalAlpha(spec, t) {
  const a = spec.nominalMeanReversion;
  return spec.nominalForward + spec.nominalVol * spec.nominalVol / (2 * a * a) * (1 - Math.exp(-a * t)) * (1 - Math.exp(-a * t));
}
function realAlpha(spec, t) {
  const a = spec.realMeanReversion;
  return spec.realForward + spec.realVol * spec.realVol / (2 * a * a) * (1 - Math.exp(-a * t)) * (1 - Math.exp(-a * t));
}
function jyInflationCallMc(spec) {
  const dt = spec.maturity / spec.steps;
  const sqrtDt = Math.sqrt(dt);
  const factor = cholesky([
    [1, spec.corrNominalReal, spec.corrNominalCpi],
    [spec.corrNominalReal, 1, spec.corrRealCpi],
    [spec.corrNominalCpi, spec.corrRealCpi, 1]
  ]);
  const quantoDrift = spec.corrRealCpi * spec.realVol * spec.cpiVol;
  const generator = new MersenneTwister(spec.seed);
  const priceEstimator = new Welford();
  const ratioEstimator = new Welford();
  for (let p = 0; p < spec.paths; p += 1) {
    let xNominal = 0;
    let xReal = 0;
    let logIndex = Math.log(spec.indexLevel);
    let integral = 0;
    for (let step = 0; step < spec.steps; step += 1) {
      const time = step * dt;
      const nominal = xNominal + nominalAlpha(spec, time);
      const real = xReal + realAlpha(spec, time);
      integral += nominal * dt;
      const z0 = inverseNormalCdf(generator.nextDouble());
      const z1 = inverseNormalCdf(generator.nextDouble());
      const z2 = inverseNormalCdf(generator.nextDouble());
      const dWn = factor[0][0] * z0;
      const dWr = factor[1][0] * z0 + factor[1][1] * z1;
      const dWi = factor[2][0] * z0 + factor[2][1] * z1 + factor[2][2] * z2;
      xNominal += -spec.nominalMeanReversion * xNominal * dt + spec.nominalVol * sqrtDt * dWn;
      xReal += (-spec.realMeanReversion * xReal - quantoDrift) * dt + spec.realVol * sqrtDt * dWr;
      logIndex += (nominal - real - 0.5 * spec.cpiVol * spec.cpiVol) * dt + spec.cpiVol * sqrtDt * dWi;
    }
    const discount = Math.exp(-integral);
    const ratio = Math.exp(logIndex) / spec.indexLevel;
    priceEstimator.push(discount * Math.max(ratio - spec.strike, 0));
    ratioEstimator.push(discount * ratio);
  }
  return { price: priceEstimator.mean, standardError: priceEstimator.standardError, forwardIndexRatio: ratioEstimator.mean, realBond: Math.exp(-spec.realForward * spec.maturity) };
}

// src/aad/checkpoint-path.ts
function advance(logSpot, shock, market) {
  const variance = market.baseVol * market.baseVol * Math.exp(2 * market.elasticity * (logSpot - Math.log(market.reference)));
  const volatility = Math.sqrt(variance);
  const sqrtDt = Math.sqrt(market.dt);
  const next = logSpot + (market.rate - 0.5 * variance) * market.dt + volatility * sqrtDt * shock;
  const jacobian3 = 1 - market.elasticity * variance * market.dt + market.elasticity * volatility * sqrtDt * shock;
  return { logSpot: next, jacobian: jacobian3 };
}
function fullPathGradient(shocks, market, logSpot0) {
  const states = new Float64Array(market.steps + 1);
  const jacobians = new Float64Array(market.steps + 1);
  states[0] = logSpot0;
  for (let t = 1; t <= market.steps; t += 1) {
    const result = advance(states[t - 1], shocks[t - 1], market);
    states[t] = result.logSpot;
    jacobians[t] = result.jacobian;
  }
  let adjoint = 0;
  for (let t = market.steps; t >= 1; t -= 1) {
    const local = Math.exp(states[t]) / market.steps;
    adjoint = (local + adjoint) * jacobians[t];
  }
  return { gradient: adjoint, stored: market.steps + 1 };
}
function checkpointedPathGradient(shocks, market, logSpot0, segment) {
  const checkpointCount = Math.floor(market.steps / segment) + 1;
  const checkpoints = new Float64Array(checkpointCount);
  checkpoints[0] = logSpot0;
  let logSpot = logSpot0;
  let index = 1;
  for (let t = 1; t <= market.steps; t += 1) {
    logSpot = advance(logSpot, shocks[t - 1], market).logSpot;
    if (t % segment === 0 && index < checkpointCount) {
      checkpoints[index] = logSpot;
      index += 1;
    }
  }
  let adjoint = 0;
  let maxStored = checkpointCount;
  const lastStart = Math.floor((market.steps - 1) / segment) * segment;
  for (let start = lastStart; start >= 0; start -= segment) {
    const end = Math.min(start + segment, market.steps);
    const states = new Float64Array(end - start + 1);
    const jacobians = new Float64Array(end - start + 1);
    states[0] = checkpoints[start / segment];
    for (let t = start + 1; t <= end; t += 1) {
      const result = advance(states[t - start - 1], shocks[t - 1], market);
      states[t - start] = result.logSpot;
      jacobians[t - start] = result.jacobian;
    }
    maxStored = Math.max(maxStored, checkpointCount + (end - start + 1));
    for (let t = end; t > start; t -= 1) {
      const local = Math.exp(states[t - start]) / market.steps;
      adjoint = (local + adjoint) * jacobians[t - start];
    }
  }
  return { gradient: adjoint, maxStored };
}

// src/engines/aad-calibration.ts
var DEFAULT_OPTIONS3 = { maxIterations: 200, tolerance: 1e-12, initialDamping: 1e-3 };
function sumSquares2(values) {
  let total = 0;
  for (const value of values) total += value * value;
  return total;
}
function evaluateResidualJacobian(model, x) {
  const tape = new Tape();
  const parameters = x.map((value) => variable(tape, value));
  const residuals = model(parameters, tape);
  return {
    residuals: residuals.map((residual) => residual.value),
    jacobian: residuals.map((residual) => gradientOf(tape, residual, parameters))
  };
}
function aadLevenbergMarquardt(model, initial, options = {}) {
  const config = { ...DEFAULT_OPTIONS3, ...options };
  const n = initial.length;
  let x = [...initial];
  let evaluation = evaluateResidualJacobian(model, x);
  let residuals = evaluation.residuals;
  let cost = sumSquares2(residuals);
  let damping = config.initialDamping;
  let iterations = 0;
  for (; iterations < config.maxIterations; iterations += 1) {
    const j = evaluation.jacobian;
    const m = residuals.length;
    const jtj = Array.from({ length: n }, () => new Array(n).fill(0));
    const jtr = new Array(n).fill(0);
    for (let a = 0; a < n; a += 1) {
      for (let b = 0; b < n; b += 1) {
        let sum = 0;
        for (let i = 0; i < m; i += 1) sum += j[i][a] * j[i][b];
        jtj[a][b] = sum;
      }
      let sumR = 0;
      for (let i = 0; i < m; i += 1) sumR += j[i][a] * residuals[i];
      jtr[a] = sumR;
    }
    let improved = false;
    for (let attempt = 0; attempt < 30; attempt += 1) {
      const system = jtj.map((row, i) => row.map((value, k) => i === k ? value + damping * (value + 1e-12) : value));
      const delta = solveLinearSystem(system, jtr.map((value) => -value));
      const candidate = x.map((value, i) => value + delta[i]);
      const candidateEvaluation = evaluateResidualJacobian(model, candidate);
      const candidateCost = sumSquares2(candidateEvaluation.residuals);
      if (candidateCost < cost) {
        x = candidate;
        const previousCost = cost;
        evaluation = candidateEvaluation;
        residuals = candidateEvaluation.residuals;
        cost = candidateCost;
        damping *= 0.7;
        improved = true;
        if (previousCost - candidateCost < config.tolerance) iterations += 1;
        break;
      }
      damping *= 2.5;
    }
    if (!improved) break;
    if (cost < config.tolerance) break;
  }
  return { parameters: x, residualNorm: Math.sqrt(cost), iterations };
}

// src/alpha/types.ts
function compose(signal, portfolio) {
  return { signal, portfolio };
}

// src/alpha/linalg.ts
function matVec(m, v) {
  return m.map((row) => row.reduce((sum, x, j) => sum + x * v[j], 0));
}
function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
  return sum;
}
function transpose(m) {
  const r = m.length;
  const c = r === 0 ? 0 : m[0].length;
  const out = Array.from({ length: c }, () => new Array(r).fill(0));
  for (let i = 0; i < r; i += 1) for (let j = 0; j < c; j += 1) out[j][i] = m[i][j];
  return out;
}
function matMul(a, b) {
  const r = a.length;
  const inner = b.length;
  const c = inner === 0 ? 0 : b[0].length;
  const out = Array.from({ length: r }, () => new Array(c).fill(0));
  for (let i = 0; i < r; i += 1) {
    for (let k = 0; k < inner; k += 1) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < c; j += 1) out[i][j] += aik * b[k][j];
    }
  }
  return out;
}
function scaleMatrix(m, factor) {
  return m.map((row) => row.map((x) => x * factor));
}
function addMatrices(a, b) {
  return a.map((row, i) => row.map((x, j) => x + b[i][j]));
}
function identity(n) {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_2, j) => i === j ? 1 : 0));
}
function invert(m) {
  const n = m.length;
  const inv = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let j = 0; j < n; j += 1) {
    const basis = new Array(n).fill(0);
    basis[j] = 1;
    const col = solveLinearSystem(m, basis);
    for (let i = 0; i < n; i += 1) inv[i][j] = col[i];
  }
  return inv;
}

// src/alpha/stats.ts
function moments(xs) {
  const n = xs.length;
  if (n === 0) return { mean: 0, variance: 0, skew: 0, kurt: 3 };
  let mean = 0;
  for (const x of xs) mean += x;
  mean /= n;
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (const x of xs) {
    const d = x - mean;
    const d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  const std = Math.sqrt(m2);
  const skew = std < 1e-12 ? 0 : m3 / (std * std * std);
  const kurt = m2 < 1e-24 ? 3 : m4 / (m2 * m2);
  return { mean, variance: m2, skew, kurt };
}
function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 0;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i += 1) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i += 1) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  return va < 1e-24 || vb < 1e-24 ? 0 : cov / Math.sqrt(va * vb);
}
function ranks(xs) {
  const order = xs.map((x, i2) => [x, i2]).sort((p, q) => p[0] - q[0]);
  const result = new Array(xs.length).fill(0);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j += 1;
    const average = (i + j) / 2;
    for (let k = i; k <= j; k += 1) result[order[k][1]] = average;
    i = j + 1;
  }
  return result;
}
function spearman(a, b) {
  return pearson(ranks(a), ranks(b));
}
function percentile(xs, quantile) {
  if (xs.length === 0) return 0;
  const sorted = [...xs].sort((p, q) => p - q);
  const position = quantile * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(sorted.length - 1, lower + 1);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

// src/alpha/regression.ts
function withIntercept(design) {
  return design.map((row) => [1, ...row]);
}
function ols(design, response) {
  const n = design.length;
  const k = n === 0 ? 0 : design[0].length;
  const xtx = Array.from({ length: k }, () => new Array(k).fill(0));
  const xty = new Array(k).fill(0);
  for (let i = 0; i < n; i += 1) {
    for (let a = 0; a < k; a += 1) {
      xty[a] += design[i][a] * response[i];
      for (let b = a; b < k; b += 1) xtx[a][b] += design[i][a] * design[i][b];
    }
  }
  for (let a = 0; a < k; a += 1) for (let b = a + 1; b < k; b += 1) xtx[b][a] = xtx[a][b];
  const inverse = invert(xtx);
  const coefficients = new Array(k).fill(0);
  for (let a = 0; a < k; a += 1) for (let b = 0; b < k; b += 1) coefficients[a] += inverse[a][b] * xty[b];
  let meanY = 0;
  for (const y of response) meanY += y;
  meanY /= n || 1;
  const residuals = new Array(n).fill(0);
  let sse = 0;
  let sst = 0;
  for (let i = 0; i < n; i += 1) {
    let fitted = 0;
    for (let a = 0; a < k; a += 1) fitted += design[i][a] * coefficients[a];
    const error = response[i] - fitted;
    residuals[i] = error;
    sse += error * error;
    sst += (response[i] - meanY) * (response[i] - meanY);
  }
  const sigmaSquared = sse / Math.max(1, n - k);
  const standardErrors = new Array(k).fill(0);
  const tStats = new Array(k).fill(0);
  for (let a = 0; a < k; a += 1) {
    const se = Math.sqrt(Math.max(sigmaSquared * inverse[a][a], 0));
    standardErrors[a] = se;
    tStats[a] = se < 1e-12 ? 0 : coefficients[a] / se;
  }
  const rSquared = sst < 1e-24 ? 0 : 1 - sse / sst;
  return { coefficients, residuals, rSquared, tStats, standardErrors };
}

// src/alpha/util.ts
var rows = (m) => m.length;
var cols = (m) => m.length === 0 ? 0 : m[0].length;
function zeros(t, n) {
  return Array.from({ length: t }, () => new Array(n).fill(0));
}
function mapMatrix(m, fn) {
  return m.map((row, t) => row.map((x, j) => fn(x, t, j)));
}
function column(m, j) {
  return m.map((row) => row[j]);
}
function perColumn(m, transform) {
  const t = rows(m);
  const n = cols(m);
  const out = zeros(t, n);
  for (let j = 0; j < n; j += 1) {
    const series = transform(column(m, j), j);
    for (let i = 0; i < t; i += 1) out[i][j] = series[i];
  }
  return out;
}
function meanStd(xs) {
  const accumulator = new Welford();
  for (const x of xs) accumulator.push(x);
  return { mean: accumulator.mean, std: Math.sqrt(accumulator.variance) };
}
function toReturns(prices) {
  const t = rows(prices);
  const n = cols(prices);
  const out = zeros(t, n);
  for (let i = 1; i < t; i += 1) {
    for (let j = 0; j < n; j += 1) out[i][j] = prices[i - 1][j] === 0 ? 0 : prices[i][j] / prices[i - 1][j] - 1;
  }
  return out;
}
function capLeverage(weights, maxLeverage) {
  if (!Number.isFinite(maxLeverage)) return weights;
  return weights.map((row) => {
    const gross = row.reduce((sum, w) => sum + Math.abs(w), 0);
    const scale3 = gross > maxLeverage ? maxLeverage / gross : 1;
    return row.map((w) => w * scale3);
  });
}
function rollingMeanStd(series, window) {
  const t = series.length;
  const prefix = new Float64Array(t + 1);
  const prefixSquared = new Float64Array(t + 1);
  for (let i = 0; i < t; i += 1) {
    prefix[i + 1] = prefix[i] + series[i];
    prefixSquared[i + 1] = prefixSquared[i] + series[i] * series[i];
  }
  const mean = new Array(t).fill(0);
  const std = new Array(t).fill(0);
  for (let i = 0; i < t; i += 1) {
    const start = Math.max(0, i - window + 1);
    const count = i - start + 1;
    const sum = prefix[i + 1] - prefix[start];
    const sumSquared = prefixSquared[i + 1] - prefixSquared[start];
    const m = sum / count;
    mean[i] = m;
    std[i] = Math.sqrt(Math.max(sumSquared / count - m * m, 0));
  }
  return { mean, std };
}

// src/alpha/covariance.ts
function columnMeans(returns) {
  const t = rows(returns);
  const n = cols(returns);
  const mean = new Array(n).fill(0);
  for (let i = 0; i < t; i += 1) for (let j = 0; j < n; j += 1) mean[j] += returns[i][j];
  for (let j = 0; j < n; j += 1) mean[j] /= t || 1;
  return mean;
}
function sampleCovariance(returns) {
  const t = rows(returns);
  const n = cols(returns);
  const mean = columnMeans(returns);
  const cov = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < t; i += 1) {
    for (let a = 0; a < n; a += 1) {
      const da = returns[i][a] - mean[a];
      for (let b = a; b < n; b += 1) cov[a][b] += da * (returns[i][b] - mean[b]);
    }
  }
  const denom = Math.max(1, t - 1);
  for (let a = 0; a < n; a += 1) {
    for (let b = a; b < n; b += 1) {
      cov[a][b] /= denom;
      cov[b][a] = cov[a][b];
    }
  }
  return cov;
}
function correlationFromCovariance(cov) {
  const n = cov.length;
  const corr = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const denom = Math.sqrt(cov[i][i] * cov[j][j]);
      corr[i][j] = denom < 1e-24 ? 0 : cov[i][j] / denom;
    }
  }
  return corr;
}
function ledoitWolf(returns) {
  const t = rows(returns);
  const n = cols(returns);
  const mean = columnMeans(returns);
  const demeaned = returns.map((row) => row.map((x, j) => x - mean[j]));
  const sample = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < t; i += 1) {
    for (let a = 0; a < n; a += 1) {
      const da = demeaned[i][a];
      for (let b = a; b < n; b += 1) sample[a][b] += da * demeaned[i][b];
    }
  }
  for (let a = 0; a < n; a += 1) {
    for (let b = a; b < n; b += 1) {
      sample[a][b] /= t || 1;
      sample[b][a] = sample[a][b];
    }
  }
  let mu = 0;
  for (let i = 0; i < n; i += 1) mu += sample[i][i];
  mu /= n || 1;
  let d2 = 0;
  for (let a = 0; a < n; a += 1) {
    for (let b = 0; b < n; b += 1) {
      const target = a === b ? mu : 0;
      d2 += (sample[a][b] - target) * (sample[a][b] - target);
    }
  }
  d2 /= n || 1;
  let b2 = 0;
  for (let i = 0; i < t; i += 1) {
    for (let a = 0; a < n; a += 1) {
      const da = demeaned[i][a];
      for (let b = 0; b < n; b += 1) {
        const term = da * demeaned[i][b] - sample[a][b];
        b2 += term * term;
      }
    }
  }
  b2 /= (n || 1) * (t || 1) * (t || 1);
  b2 = Math.min(b2, d2);
  const shrinkage = d2 < 1e-24 ? 0 : b2 / d2;
  const covariance = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_2, b) => {
    const target = a === b ? mu : 0;
    return shrinkage * target + (1 - shrinkage) * sample[a][b];
  }));
  return { covariance, shrinkage };
}

// src/alpha/signals.ts
function momentum(lookback = 20) {
  return (prices) => perColumn(prices, (series) => {
    const t = series.length;
    const out = new Array(t).fill(0);
    for (let i = lookback; i < t; i += 1) out[i] = series[i - lookback] === 0 ? 0 : series[i] / series[i - lookback] - 1;
    return out;
  });
}
function meanReversion(lookback = 20) {
  const trend = momentum(lookback);
  return (prices) => mapMatrix(trend(prices), (x) => -x);
}
function zscore(window = 20) {
  return (prices) => perColumn(prices, (series) => {
    const { mean, std } = rollingMeanStd(series, window);
    return series.map((x, i) => std[i] > 1e-12 ? (x - mean[i]) / std[i] : 0);
  });
}
var SIGNALS = {
  momentum: (params = {}) => momentum(params.lookback),
  meanReversion: (params = {}) => meanReversion(params.lookback),
  zscore: (params = {}) => zscore(params.window)
};

// src/alpha/portfolio.ts
function equalWeight() {
  return (score) => score.map((row) => {
    const active = row.reduce((sum, x) => sum + (x === 0 ? 0 : 1), 0);
    return active === 0 ? row.map(() => 0) : row.map((x) => Math.sign(x) / active);
  });
}
function crossSectional() {
  return (score) => score.map((row) => {
    const n = row.length;
    const mean = n === 0 ? 0 : row.reduce((sum, x) => sum + x, 0) / n;
    const centered = row.map((x) => x - mean);
    const gross = centered.reduce((sum, x) => sum + Math.abs(x), 0);
    return gross < 1e-12 ? row.map(() => 0) : centered.map((x) => x / gross);
  });
}
function longShortRank(fraction = 0.2) {
  return (score) => score.map((row) => {
    const n = row.length;
    if (n < 2) return row.map(() => 0);
    const k = Math.max(1, Math.min(Math.floor(n * fraction), Math.floor(n / 2)));
    const order = row.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
    const weights = new Array(n).fill(0);
    for (let r = 0; r < k; r += 1) {
      weights[order[r][1]] = -1 / k;
      weights[order[n - 1 - r][1]] = 1 / k;
    }
    return weights;
  });
}
function meanVariance(expectedReturns, covariance) {
  const raw = solveLinearSystem(covariance, expectedReturns);
  const gross = raw.reduce((sum, x) => sum + Math.abs(x), 0);
  return gross < 1e-12 ? raw.map(() => 0) : raw.map((x) => x / gross);
}
function volTarget(weights, returns, target, periodsPerYear = 252) {
  const t = rows(weights);
  const n = cols(weights);
  const port = [];
  for (let i = 1; i < t; i += 1) {
    let r = 0;
    for (let j = 0; j < n; j += 1) r += weights[i - 1][j] * returns[i][j];
    port.push(r);
  }
  const realized = meanStd(port).std * Math.sqrt(periodsPerYear);
  const scale3 = realized < 1e-12 ? 1 : target / realized;
  return weights.map((row) => row.map((w) => w * scale3));
}
var PORTFOLIOS = {
  equalWeight: () => equalWeight(),
  crossSectional: () => crossSectional(),
  longShortRank: (params = {}) => longShortRank(params.fraction)
};

// src/numerics/optimization/differential-evolution.ts
var DEFAULTS = { populationSize: 40, generations: 300, differentialWeight: 0.7, crossoverProbability: 0.9, seed: 1 };
function differentialEvolution(objective, lower, upper, options = {}) {
  const config = { ...DEFAULTS, ...options };
  const dimension = lower.length;
  const generator = new MersenneTwister(config.seed);
  const random = () => generator.nextDouble();
  const population = [];
  const fitness = [];
  for (let i = 0; i < config.populationSize; i += 1) {
    const candidate = lower.map((low, j) => low + random() * (upper[j] - low));
    population.push(candidate);
    fitness.push(objective(candidate));
  }
  let bestIndex = 0;
  for (let i = 1; i < config.populationSize; i += 1) if (fitness[i] < fitness[bestIndex]) bestIndex = i;
  for (let generation = 0; generation < config.generations; generation += 1) {
    for (let i = 0; i < config.populationSize; i += 1) {
      let a = i;
      let b = i;
      let c = i;
      while (a === i) a = Math.floor(random() * config.populationSize);
      while (b === i || b === a) b = Math.floor(random() * config.populationSize);
      while (c === i || c === a || c === b) c = Math.floor(random() * config.populationSize);
      const trial = population[i].slice();
      const forced = Math.floor(random() * dimension);
      for (let j = 0; j < dimension; j += 1) {
        if (j === forced || random() < config.crossoverProbability) {
          const mutated = population[a][j] + config.differentialWeight * (population[b][j] - population[c][j]);
          trial[j] = Math.min(Math.max(mutated, lower[j]), upper[j]);
        }
      }
      const trialFitness = objective(trial);
      if (trialFitness < fitness[i]) {
        population[i] = trial;
        fitness[i] = trialFitness;
        if (trialFitness < fitness[bestIndex]) bestIndex = i;
      }
    }
  }
  return { point: population[bestIndex], value: fitness[bestIndex] };
}

// src/alpha/allocation.ts
function riskContributions(covariance, weights) {
  const sigmaW = matVec(covariance, weights);
  const portfolioVariance = dot(weights, sigmaW);
  return weights.map((w, i) => portfolioVariance < 1e-24 ? 0 : w * sigmaW[i] / portfolioVariance);
}
function riskParity(covariance, budget, iterations = 2e3, tolerance = 1e-12) {
  const n = covariance.length;
  const target = budget ?? new Array(n).fill(1 / n);
  let weights = new Array(n).fill(1 / n);
  for (let iter = 0; iter < iterations; iter += 1) {
    const sigmaW = matVec(covariance, weights);
    const portfolioVariance = dot(weights, sigmaW);
    const next = weights.map((w, i) => {
      const contribution = portfolioVariance < 1e-24 ? target[i] : w * sigmaW[i] / portfolioVariance;
      return w * Math.sqrt(target[i] / Math.max(contribution, 1e-18));
    });
    const sum = next.reduce((s, x) => s + x, 0);
    let maxDiff = 0;
    for (let i = 0; i < n; i += 1) {
      next[i] /= sum;
      maxDiff = Math.max(maxDiff, Math.abs(next[i] - weights[i]));
    }
    weights = next;
    if (maxDiff < tolerance) break;
  }
  return weights;
}
function singleLinkageOrder(distance) {
  const n = distance.length;
  const clusters = [];
  for (let i = 0; i < n; i += 1) clusters.push({ members: [i], node: i });
  const linkage = (a, b) => {
    let minimum = Infinity;
    for (const x of a) for (const y of b) if (distance[x][y] < minimum) minimum = distance[x][y];
    return minimum;
  };
  while (clusters.length > 1) {
    let bestI = 0;
    let bestJ = 1;
    let best = Infinity;
    for (let i = 0; i < clusters.length; i += 1) {
      for (let j = i + 1; j < clusters.length; j += 1) {
        const d = linkage(clusters[i].members, clusters[j].members);
        if (d < best) {
          best = d;
          bestI = i;
          bestJ = j;
        }
      }
    }
    const merged = {
      members: [...clusters[bestI].members, ...clusters[bestJ].members],
      node: { left: clusters[bestI].node, right: clusters[bestJ].node }
    };
    clusters.splice(bestJ, 1);
    clusters.splice(bestI, 1);
    clusters.push(merged);
  }
  const order = [];
  const flatten = (node) => {
    if (typeof node === "number") order.push(node);
    else {
      flatten(node.left);
      flatten(node.right);
    }
  };
  if (clusters.length > 0) flatten(clusters[0].node);
  return order;
}
function clusterVariance(covariance, items) {
  const inverse = items.map((i) => 1 / Math.max(covariance[i][i], 1e-18));
  const sum = inverse.reduce((s, x) => s + x, 0);
  const weights = inverse.map((x) => x / sum);
  let variance = 0;
  for (let a = 0; a < items.length; a += 1) {
    for (let b = 0; b < items.length; b += 1) variance += weights[a] * covariance[items[a]][items[b]] * weights[b];
  }
  return variance;
}
function hierarchicalRiskParity(covariance) {
  const n = covariance.length;
  const correlation = correlationFromCovariance(covariance);
  const distance = correlation.map((row) => row.map((c) => Math.sqrt(Math.max(0.5 * (1 - c), 0))));
  const order = singleLinkageOrder(distance);
  const weights = new Array(n).fill(1);
  let clusters = order.length > 0 ? [order] : [];
  while (clusters.length > 0) {
    const next = [];
    for (const cluster of clusters) {
      if (cluster.length <= 1) continue;
      const half = Math.floor(cluster.length / 2);
      const left = cluster.slice(0, half);
      const right = cluster.slice(half);
      const leftVariance = clusterVariance(covariance, left);
      const rightVariance = clusterVariance(covariance, right);
      const alpha2 = leftVariance + rightVariance < 1e-24 ? 0.5 : 1 - leftVariance / (leftVariance + rightVariance);
      for (const i of left) weights[i] *= alpha2;
      for (const i of right) weights[i] *= 1 - alpha2;
      next.push(left, right);
    }
    clusters = next;
  }
  return weights;
}
function blackLitterman(priorCovariance, marketWeights, riskAversion = 2.5, tau = 0.05, views) {
  const equilibrium = matVec(priorCovariance, marketWeights).map((x) => riskAversion * x);
  if (!views || views.views.length === 0) return { expectedReturns: equilibrium, weights: marketWeights };
  const tauSigma = scaleMatrix(priorCovariance, tau);
  const pick = views.pick;
  const pickTransposed = transpose(pick);
  const scaled = matMul(matMul(pick, tauSigma), pickTransposed);
  const omega = views.omega ?? scaled.map((row, i) => row.map((x, j) => i === j ? x : 0));
  const middle = invert(addMatrices(scaled, omega));
  const pickEquilibrium = matVec(pick, equilibrium);
  const surprise = views.views.map((q, i) => q - pickEquilibrium[i]);
  const adjustment = matVec(matMul(tauSigma, pickTransposed), matVec(middle, surprise));
  const expectedReturns = equilibrium.map((x, i) => x + adjustment[i]);
  const raw = matVec(invert(scaleMatrix(priorCovariance, riskAversion)), expectedReturns);
  const gross = raw.reduce((s, x) => s + Math.abs(x), 0);
  const weights = gross < 1e-12 ? raw : raw.map((x) => x / gross);
  return { expectedReturns, weights };
}
function constrainedMeanVariance(expectedReturns, covariance, constraints = {}) {
  const { longOnly = false, maxWeight = 1, maxLeverage = 1, riskAversion = 1, penaltyWeight = 1e3, seed = 1 } = constraints;
  const n = expectedReturns.length;
  const lower = new Array(n).fill(longOnly ? 0 : -maxWeight);
  const upper = new Array(n).fill(maxWeight);
  const objective = (x) => {
    let expected = 0;
    for (let i = 0; i < n; i += 1) expected += expectedReturns[i] * x[i];
    let risk = 0;
    for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) risk += x[i] * covariance[i][j] * x[j];
    const gross = x.reduce((s, v) => s + Math.abs(v), 0);
    return -(expected - 0.5 * riskAversion * risk) + penaltyWeight * (gross - maxLeverage) ** 2;
  };
  return differentialEvolution(objective, lower, upper, { seed }).point;
}

// src/alpha/costs.ts
function linearCost(rate = 5e-4) {
  return (trade) => rate * trade.reduce((sum, x) => sum + Math.abs(x), 0);
}
function squareRootImpact(coefficient = 0.1) {
  return (trade) => trade.reduce((sum, x) => sum + coefficient * Math.abs(x) ** 1.5, 0);
}
function borrowCost(annualRate = 0.01, periodsPerYear = 252) {
  const perPeriod = annualRate / periodsPerYear;
  return (_trade, weights) => perPeriod * weights.reduce((sum, w) => sum + Math.max(-w, 0), 0);
}
function combineCosts(...models2) {
  return (trade, weights, context) => models2.reduce((sum, model) => sum + model(trade, weights, context), 0);
}
var COSTS = {
  linear: (params = {}) => linearCost(params.rate),
  squareRootImpact: (params = {}) => squareRootImpact(params.coefficient),
  borrow: (params = {}) => borrowCost(params.annualRate, params.periodsPerYear)
};

// src/alpha/metrics.ts
function sharpe(returns, periodsPerYear = 252) {
  const { mean, std } = meanStd(returns);
  return std < 1e-12 ? 0 : mean / std * Math.sqrt(periodsPerYear);
}
function sortino(returns, periodsPerYear = 252) {
  if (returns.length === 0) return 0;
  const { mean } = meanStd(returns);
  let downsideSquared = 0;
  for (const r of returns) if (r < 0) downsideSquared += r * r;
  const downside = Math.sqrt(downsideSquared / returns.length);
  return downside < 1e-12 ? 0 : mean / downside * Math.sqrt(periodsPerYear);
}
function maxDrawdown(equity) {
  let peak = -Infinity;
  let worst = 0;
  for (const value of equity) {
    if (value > peak) peak = value;
    const drawdown = peak > 0 ? 1 - value / peak : 0;
    if (drawdown > worst) worst = drawdown;
  }
  return worst;
}
function calmar(returns, equity, periodsPerYear = 252) {
  const drawdown = maxDrawdown(equity);
  const annualized = meanStd(returns).mean * periodsPerYear;
  return drawdown < 1e-12 ? 0 : annualized / drawdown;
}
function hitRate(returns) {
  if (returns.length === 0) return 0;
  let wins = 0;
  for (const r of returns) if (r > 0) wins += 1;
  return wins / returns.length;
}
function summarizePerformance(portReturns, equity, turnover, periodsPerYear = 252) {
  return {
    sharpe: sharpe(portReturns, periodsPerYear),
    sortino: sortino(portReturns, periodsPerYear),
    maxDrawdown: maxDrawdown(equity),
    calmar: calmar(portReturns, equity, periodsPerYear),
    hitRate: hitRate(portReturns),
    turnover
  };
}
function valueAtRisk(returns, confidence = 0.95) {
  return historicalVar(Float64Array.from(returns), confidence);
}
function expectedShortfall(returns, confidence = 0.95) {
  return historicalExpectedShortfall(Float64Array.from(returns), confidence);
}

// src/alpha/overfit.ts
var EULER_MASCHERONI = 0.5772156649015329;
function sharpePerPeriod(returns) {
  const { mean, variance } = moments(returns);
  const std = Math.sqrt(variance);
  return std < 1e-12 ? 0 : mean / std;
}
function sharpeTStat(returns) {
  return sharpePerPeriod(returns) * Math.sqrt(returns.length);
}
function probabilisticSharpe(returns, benchmark = 0) {
  const n = returns.length;
  if (n < 2) return 0;
  const { mean, variance, skew, kurt } = moments(returns);
  const std = Math.sqrt(variance);
  const sharpe2 = std < 1e-12 ? 0 : mean / std;
  const denominator = Math.sqrt(Math.max(1 - skew * sharpe2 + (kurt - 1) / 4 * sharpe2 * sharpe2, 1e-12));
  return normalCdf((sharpe2 - benchmark) * Math.sqrt(n - 1) / denominator);
}
function expectedMaxSharpe(trials, variance) {
  if (trials < 2) return 0;
  const sd = Math.sqrt(Math.max(variance, 0));
  const high = inverseNormalCdf(1 - 1 / trials);
  const low = inverseNormalCdf(1 - 1 / (trials * Math.E));
  return sd * ((1 - EULER_MASCHERONI) * high + EULER_MASCHERONI * low);
}
function deflatedSharpe(returns, trialSharpes) {
  const expectedMax = expectedMaxSharpe(trialSharpes.length, moments(trialSharpes).variance);
  return probabilisticSharpe(returns, expectedMax);
}
function minTrackRecordLength(returns, targetSharpe = 0, confidence = 0.95) {
  const { mean, variance, skew, kurt } = moments(returns);
  const std = Math.sqrt(variance);
  const sharpe2 = std < 1e-12 ? 0 : mean / std;
  if (Math.abs(sharpe2 - targetSharpe) < 1e-12) return Infinity;
  const z = inverseNormalCdf(confidence);
  return 1 + (1 - skew * sharpe2 + (kurt - 1) / 4 * sharpe2 * sharpe2) * (z / (sharpe2 - targetSharpe)) ** 2;
}
function combinations(n, k) {
  const result = [];
  const chosen = [];
  const recurse = (start) => {
    if (chosen.length === k) {
      result.push(chosen.slice());
      return;
    }
    for (let i = start; i < n; i += 1) {
      chosen.push(i);
      recurse(i + 1);
      chosen.pop();
    }
  };
  recurse(0);
  return result;
}
function sharpeOverRows(returns, rowIndices, column2) {
  const series = rowIndices.map((i) => returns[i][column2]);
  return sharpePerPeriod(series);
}
function probabilityOfBacktestOverfitting(trialReturns, partitions = 10) {
  const t = rows(trialReturns);
  const m = cols(trialReturns);
  const blocks = Math.max(2, partitions - partitions % 2);
  const blockOf = Array.from({ length: blocks }, () => []);
  for (let i = 0; i < t; i += 1) blockOf[Math.min(blocks - 1, Math.floor(i * blocks / t))].push(i);
  const combos = combinations(blocks, blocks / 2);
  let belowMedian = 0;
  let counted = 0;
  for (const inSample of combos) {
    const isSet = new Set(inSample);
    const isRows = [];
    const oosRows = [];
    for (let b = 0; b < blocks; b += 1) (isSet.has(b) ? isRows : oosRows).push(...blockOf[b]);
    if (isRows.length < 2 || oosRows.length < 2) continue;
    let best = 0;
    let bestPerf = -Infinity;
    for (let strategy = 0; strategy < m; strategy += 1) {
      const perf = sharpeOverRows(trialReturns, isRows, strategy);
      if (perf > bestPerf) {
        bestPerf = perf;
        best = strategy;
      }
    }
    const oosBest = sharpeOverRows(trialReturns, oosRows, best);
    let rank = 1;
    for (let strategy = 0; strategy < m; strategy += 1) if (sharpeOverRows(trialReturns, oosRows, strategy) < oosBest) rank += 1;
    const relative = rank / (m + 1);
    const logit = Math.log(relative / (1 - relative));
    if (logit <= 0) belowMedian += 1;
    counted += 1;
  }
  return counted === 0 ? 0 : belowMedian / counted;
}
function bonferroni(pValues, alpha2 = 0.05) {
  const threshold = alpha2 / Math.max(1, pValues.length);
  return pValues.map((p) => p <= threshold);
}
function benjaminiHochberg(pValues, alpha2 = 0.05) {
  const m = pValues.length;
  const order = pValues.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]);
  let cutoff = -1;
  for (let r = 0; r < m; r += 1) if (order[r][0] <= (r + 1) / m * alpha2) cutoff = r;
  const reject = new Array(m).fill(false);
  for (let r = 0; r <= cutoff; r += 1) reject[order[r][1]] = true;
  return reject;
}

// src/alpha/factor.ts
function factorRegression(assetReturns, factorReturns) {
  const design = withIntercept(factorReturns);
  const fit = ols(design, assetReturns);
  const dof = Math.max(1, fit.residuals.length - design[0].length);
  let sse = 0;
  for (const e of fit.residuals) sse += e * e;
  return {
    alpha: fit.coefficients[0],
    betas: fit.coefficients.slice(1),
    residualVol: Math.sqrt(sse / dof),
    rSquared: fit.rSquared,
    alphaTStat: fit.tStats[0],
    betaTStats: fit.tStats.slice(1)
  };
}
function factorRiskAttribution(betas, factorCovariance, specificVariance) {
  let factorVariance = 0;
  for (let i = 0; i < betas.length; i += 1) {
    for (let j = 0; j < betas.length; j += 1) factorVariance += betas[i] * factorCovariance[i][j] * betas[j];
  }
  return {
    factor: Math.sqrt(Math.max(factorVariance, 0)),
    specific: Math.sqrt(Math.max(specificVariance, 0)),
    total: Math.sqrt(Math.max(factorVariance + specificVariance, 0))
  };
}
function pcaFactors(returns, count) {
  const covariance = sampleCovariance(returns);
  const eigen = jacobiEigen(covariance);
  const n = covariance.length;
  const top = Math.min(count, n);
  const totalVariance2 = eigen.values.reduce((sum, v) => sum + Math.max(v, 0), 0) || 1;
  const loadings = Array.from({ length: n }, () => new Array(top).fill(0));
  for (let f = 0; f < top; f += 1) for (let i = 0; i < n; i += 1) loadings[i][f] = eigen.vectors[f][i];
  const t = rows(returns);
  const assets = cols(returns);
  const factors = Array.from({ length: t }, () => new Array(top).fill(0));
  for (let s = 0; s < t; s += 1) {
    for (let f = 0; f < top; f += 1) {
      let value = 0;
      for (let i = 0; i < assets; i += 1) value += loadings[i][f] * returns[s][i];
      factors[s][f] = value;
    }
  }
  const explained = eigen.values.slice(0, top).map((v) => Math.max(v, 0) / totalVariance2);
  return { factors, loadings, explained };
}

// src/alpha/ic.ts
function informationCoefficient(signal, forwardReturns) {
  return pearson(signal, forwardReturns);
}
function rankInformationCoefficient(signal, forwardReturns) {
  return spearman(signal, forwardReturns);
}
function icSeries(signal, returns, horizon = 1, rank = true) {
  const t = Math.min(signal.length, returns.length);
  const series = [];
  for (let i = 0; i + horizon < t; i += 1) series.push(rank ? spearman(signal[i], returns[i + horizon]) : pearson(signal[i], returns[i + horizon]));
  return series;
}
function informationRatioOfIC(ic) {
  const { mean, std } = meanStd(ic);
  return std < 1e-12 ? 0 : mean / std;
}
function alphaDecay(signal, returns, horizons, rank = true) {
  return horizons.map((horizon) => ({ horizon, ic: meanStd(icSeries(signal, returns, horizon, rank)).mean }));
}
function combineSignals(signals, weights) {
  const count = signals.length;
  if (count === 0) return [];
  const blend2 = weights ?? new Array(count).fill(1 / count);
  const t = signals[0].length;
  const n = t === 0 ? 0 : signals[0][0].length;
  const out = Array.from({ length: t }, () => new Array(n).fill(0));
  for (let s = 0; s < count; s += 1) {
    for (let i = 0; i < t; i += 1) {
      const row = signals[s][i];
      const { mean, std } = meanStd(row);
      for (let j = 0; j < n; j += 1) out[i][j] += blend2[s] * (std < 1e-12 ? 0 : (row[j] - mean) / std);
    }
  }
  return out;
}

// src/alpha/backtest.ts
function computeWeights(prices, strategy, maxLeverage) {
  return capLeverage(strategy.portfolio(strategy.signal(prices)), maxLeverage);
}
function evaluate2(returns, weights, costModel, start, end) {
  const n = cols(returns);
  const portReturns = [];
  const equity = [];
  let level = 1;
  let turnover = 0;
  for (let i = Math.max(1, start + 1); i < end; i += 1) {
    const current = weights[i - 1];
    const trade = new Array(n).fill(0);
    let gross = 0;
    for (let j = 0; j < n; j += 1) {
      gross += current[j] * returns[i][j];
      trade[j] = current[j] - (i - 2 >= 0 ? weights[i - 2][j] : 0);
    }
    let traded = 0;
    for (let j = 0; j < n; j += 1) traded += Math.abs(trade[j]);
    const net = gross - costModel(trade, current);
    turnover += traded;
    level *= 1 + net;
    portReturns.push(net);
    equity.push(level);
  }
  return { portReturns, equity, turnover };
}
function backtest(prices, strategy, config = {}) {
  const { cost = 0, maxLeverage = Infinity, start = 0, periodsPerYear = 252, costModel } = config;
  const model = costModel ?? linearCost(cost);
  const returns = toReturns(prices);
  const weights = computeWeights(prices, strategy, maxLeverage);
  const track = evaluate2(returns, weights, model, start, rows(prices));
  return { equity: track.equity, weights, portReturns: track.portReturns, turnover: track.turnover, metrics: summarizePerformance(track.portReturns, track.equity, track.turnover, periodsPerYear) };
}
function walkForward(prices, makeStrategy, config = {}) {
  const { cost = 0, maxLeverage = Infinity, folds = 4, minTrainFraction = 0.5, periodsPerYear = 252, costModel } = config;
  const model = costModel ?? linearCost(cost);
  const t = rows(prices);
  const returns = toReturns(prices);
  const trainEnd = Math.max(1, Math.floor(t * minTrainFraction));
  const segment = Math.max(1, Math.floor((t - trainEnd) / folds));
  const portReturns = [];
  let turnover = 0;
  for (let f = 0; f < folds; f += 1) {
    const segStart = trainEnd + f * segment;
    if (segStart >= t) break;
    const segEnd = f === folds - 1 ? t : Math.min(t, segStart + segment);
    const strategy = makeStrategy(prices.slice(0, segStart));
    const weights = computeWeights(prices, strategy, maxLeverage);
    const track = evaluate2(returns, weights, model, segStart, segEnd);
    for (const r of track.portReturns) portReturns.push(r);
    turnover += track.turnover;
  }
  const equity = [];
  let level = 1;
  for (const r of portReturns) {
    level *= 1 + r;
    equity.push(level);
  }
  return { equity, weights: [], portReturns, turnover, metrics: summarizePerformance(portReturns, equity, turnover, periodsPerYear) };
}

// src/alpha/execution.ts
function eventBacktest(prices, strategy, config = {}) {
  const { costModel, slippageBps = 0, financingRate = 0, periodsPerYear = 252, rebalanceEvery = 1, maxLeverage = Infinity } = config;
  const model = costModel ?? linearCost(0);
  const t = rows(prices);
  const n = cols(prices);
  const targets = capLeverage(strategy.portfolio(strategy.signal(prices)), maxLeverage);
  const slippage = slippageBps / 1e4;
  const financePerPeriod = financingRate / periodsPerYear;
  let weights = new Array(n).fill(0);
  let level = 1;
  let turnover = 0;
  const equity = [];
  const portReturns = [];
  for (let i = 1; i < t; i += 1) {
    let drift = 0;
    const stepReturns = new Array(n).fill(0);
    for (let j = 0; j < n; j += 1) {
      stepReturns[j] = prices[i - 1][j] === 0 ? 0 : prices[i][j] / prices[i - 1][j] - 1;
      drift += weights[j] * stepReturns[j];
    }
    const drifted = weights.map((w, j) => 1 + drift === 0 ? 0 : w * (1 + stepReturns[j]) / (1 + drift));
    const financeCost = financePerPeriod * drifted.reduce((sum, w) => sum + Math.max(-w, 0), 0);
    let tradeCost = 0;
    let traded = 0;
    let positioned = drifted;
    if (i % rebalanceEvery === 0) {
      const target = targets[i];
      const trade = target.map((w, j) => w - drifted[j]);
      for (const x of trade) traded += Math.abs(x);
      tradeCost = model(trade, drifted) + slippage * traded;
      positioned = target;
    }
    const net = drift - financeCost - tradeCost;
    level *= 1 + net;
    turnover += traded;
    weights = positioned;
    equity.push(level);
    portReturns.push(net);
  }
  return { equity, weights: targets, portReturns, turnover, metrics: summarizePerformance(portReturns, equity, turnover, periodsPerYear) };
}

// src/alpha/benchmark.ts
function alphaBeta(strategy, benchmark, periodsPerYear = 252) {
  const design = withIntercept(benchmark.map((b) => [b]));
  const fit = ols(design, strategy);
  return { alpha: fit.coefficients[0] * periodsPerYear, beta: fit.coefficients[1], rSquared: fit.rSquared };
}
function trackingError(strategy, benchmark, periodsPerYear = 252) {
  const active = strategy.map((s, i) => s - benchmark[i]);
  return meanStd(active).std * Math.sqrt(periodsPerYear);
}
function informationRatio(strategy, benchmark, periodsPerYear = 252) {
  const active = strategy.map((s, i) => s - benchmark[i]);
  const { mean, std } = meanStd(active);
  return std < 1e-12 ? 0 : mean / std * Math.sqrt(periodsPerYear);
}

// src/alpha/analytics.ts
function rollingSharpe(returns, window = 60, periodsPerYear = 252) {
  const { mean, std } = rollingMeanStd(returns, window);
  return mean.map((m, i) => std[i] < 1e-12 ? 0 : m / std[i] * Math.sqrt(periodsPerYear));
}
function rollingVolatility(returns, window = 60, periodsPerYear = 252) {
  return rollingMeanStd(returns, window).std.map((s) => s * Math.sqrt(periodsPerYear));
}
function underwater(equity) {
  let peak = -Infinity;
  return equity.map((value) => {
    if (value > peak) peak = value;
    return peak > 0 ? value / peak - 1 : 0;
  });
}
function drawdownTable(equity) {
  const episodes = [];
  if (equity.length === 0) return episodes;
  let peak = equity[0];
  let peakIndex = 0;
  let troughIndex = 0;
  let troughValue = Infinity;
  let inDrawdown = false;
  for (let i = 0; i < equity.length; i += 1) {
    if (equity[i] >= peak) {
      if (inDrawdown) {
        episodes.push({ start: peakIndex, trough: troughIndex, recovery: i, depth: 1 - troughValue / peak, length: i - peakIndex });
        inDrawdown = false;
      }
      peak = equity[i];
      peakIndex = i;
    } else if (!inDrawdown) {
      inDrawdown = true;
      troughValue = equity[i];
      troughIndex = i;
    } else if (equity[i] < troughValue) {
      troughValue = equity[i];
      troughIndex = i;
    }
  }
  if (inDrawdown) episodes.push({ start: peakIndex, trough: troughIndex, recovery: -1, depth: 1 - troughValue / peak, length: equity.length - 1 - peakIndex });
  return episodes;
}

// src/alpha/report.ts
function computeExposure(weights, turnover) {
  const t = weights.length;
  if (t === 0) return { grossMean: 0, netMean: 0, turnover };
  let gross = 0;
  let net = 0;
  for (const row of weights) {
    for (const w of row) {
      gross += Math.abs(w);
      net += w;
    }
  }
  return { grossMean: gross / t, netMean: net / t, turnover };
}
function tearSheet(result, options = {}) {
  const { benchmarkReturns, rollingWindow = 60, periodsPerYear = 252 } = options;
  let benchmark;
  if (benchmarkReturns) {
    const ab = alphaBeta(result.portReturns, benchmarkReturns, periodsPerYear);
    benchmark = {
      alpha: ab.alpha,
      beta: ab.beta,
      informationRatio: informationRatio(result.portReturns, benchmarkReturns, periodsPerYear),
      trackingError: trackingError(result.portReturns, benchmarkReturns, periodsPerYear)
    };
  }
  return {
    metrics: result.metrics,
    drawdowns: drawdownTable(result.equity),
    rollingSharpe: rollingSharpe(result.portReturns, rollingWindow, periodsPerYear),
    exposure: computeExposure(result.weights, result.turnover),
    benchmark
  };
}

// src/numerics/mlmc.ts
function eulerAsian(market, increments) {
  const steps = increments.length;
  const dt = market.maturity / steps;
  let spot2 = market.spot;
  let sum = 0;
  for (let i = 0; i < steps; i += 1) {
    spot2 += market.rate * spot2 * dt + market.vol * spot2 * increments[i];
    sum += spot2;
  }
  return Math.exp(-market.rate * market.maturity) * Math.max(sum / steps - market.strike, 0);
}
function asianLevel(market, level, paths, seed) {
  const fineSteps = 2 ** level;
  const dtFine = market.maturity / fineSteps;
  const sqrtDtFine = Math.sqrt(dtFine);
  const generator = new MersenneTwister(seed);
  let sum = 0;
  let sumSquares3 = 0;
  for (let p = 0; p < paths; p += 1) {
    const fine = new Float64Array(fineSteps);
    for (let i = 0; i < fineSteps; i += 1) fine[i] = inverseNormalCdf(generator.nextDouble()) * sqrtDtFine;
    let difference;
    if (level === 0) {
      difference = eulerAsian(market, fine);
    } else {
      const coarse = new Float64Array(fineSteps / 2);
      for (let i = 0; i < coarse.length; i += 1) coarse[i] = fine[2 * i] + fine[2 * i + 1];
      difference = eulerAsian(market, fine) - eulerAsian(market, coarse);
    }
    sum += difference;
    sumSquares3 += difference * difference;
  }
  const mean = sum / paths;
  return { mean, variance: sumSquares3 / paths - mean * mean };
}
function mlmcAsian(market, levels, pathsPerLevel, seed) {
  let estimate = 0;
  for (let level = 0; level <= levels; level += 1) estimate += asianLevel(market, level, pathsPerLevel[level], seed + level * 7919).mean;
  return estimate;
}

// src/numerics/variance-reduction/importance-sampling.ts
function importanceSampledCall(market) {
  const sqrtT = Math.sqrt(market.maturity);
  const diffusion = market.vol * sqrtT;
  const drift = (market.rate - 0.5 * market.vol * market.vol) * market.maturity;
  const discount = Math.exp(-market.rate * market.maturity);
  const generator = new MersenneTwister(market.seed);
  const estimator = new Welford();
  for (let p = 0; p < market.paths; p += 1) {
    const shifted = inverseNormalCdf(generator.nextDouble()) + market.drift;
    const terminal2 = market.spot * Math.exp(drift + diffusion * shifted);
    const payoff2 = Math.max(terminal2 - market.strike, 0);
    const likelihood = Math.exp(-market.drift * shifted + 0.5 * market.drift * market.drift);
    estimator.push(discount * payoff2 * likelihood);
  }
  return { price: estimator.mean, standardError: estimator.standardError };
}

// src/engines/qmc-engine.ts
function qmcEuropeanCall(market, points) {
  const generator = new Sobol(1);
  const drift = (market.rate - 0.5 * market.vol * market.vol) * market.maturity;
  const diffusion = market.vol * Math.sqrt(market.maturity);
  const discount = Math.exp(-market.rate * market.maturity);
  let total = 0;
  for (let i = 0; i < points; i += 1) {
    const z = inverseNormalCdf(generator.next()[0]);
    total += discount * Math.max(market.spot * Math.exp(drift + diffusion * z) - market.strike, 0);
  }
  return total / points;
}
function qmcAsianCall(market, steps, points) {
  const generator = new Sobol(steps);
  const bridge = new BrownianBridge(steps);
  const dt = market.maturity / steps;
  const sqrtDt = Math.sqrt(dt);
  const discount = Math.exp(-market.rate * market.maturity);
  let total = 0;
  for (let i = 0; i < points; i += 1) {
    const uniforms = generator.next();
    const normals = new Float64Array(steps);
    for (let s = 0; s < steps; s += 1) normals[s] = inverseNormalCdf(uniforms[s]);
    const path = bridge.buildPath(normals);
    let sum = 0;
    for (let s = 0; s < steps; s += 1) {
      const time = (s + 1) * dt;
      sum += market.spot * Math.exp((market.rate - 0.5 * market.vol * market.vol) * time + market.vol * sqrtDt * path[s]);
    }
    total += discount * Math.max(sum / steps - market.strike, 0);
  }
  return total / points;
}

// src/analytics/convexity.ts
function swapAnnuity(rate, tenor) {
  return (1 - 1 / Math.pow(1 + rate, tenor)) / rate;
}
function cmsConvexityAdjustment(swapRate, vol, expiry, tenor) {
  const h = 1e-5;
  const base = swapAnnuity(swapRate, tenor);
  const up = swapAnnuity(swapRate + h, tenor);
  const down = swapAnnuity(swapRate - h, tenor);
  const firstDerivative = (up - down) / (2 * h);
  const secondDerivative = (up - 2 * base + down) / (h * h);
  return -0.5 * vol * vol * swapRate * swapRate * expiry * (secondDerivative / firstDerivative);
}
function quantoForward(forward, assetVol, fxVol, correlation, maturity) {
  return forward * Math.exp(-correlation * assetVol * fxVol * maturity);
}

// src/engines/pide-jump.ts
function mertonPideCall(spec) {
  const m = spec.spaceSteps;
  const dt = spec.maturity / spec.timeSteps;
  const center = Math.log(spec.spot);
  const effectiveVol = Math.sqrt(spec.vol * spec.vol + spec.jumpIntensity * (spec.jumpMean * spec.jumpMean + spec.jumpVol * spec.jumpVol));
  const halfWidth = spec.widthStdDev * effectiveVol * Math.sqrt(spec.maturity);
  const dx = 2 * halfWidth / (m - 1);
  const x = new Float64Array(m);
  for (let i = 0; i < m; i += 1) x[i] = center - halfWidth + i * dx;
  const kappa = Math.exp(spec.jumpMean + 0.5 * spec.jumpVol * spec.jumpVol) - 1;
  const drift = spec.rate - 0.5 * spec.vol * spec.vol - spec.jumpIntensity * kappa;
  const jumpHalf = spec.widthStdDev * spec.jumpVol + Math.abs(spec.jumpMean);
  const dy = 2 * jumpHalf / (spec.quadraturePoints - 1);
  const y = new Float64Array(spec.quadraturePoints);
  const weight = new Float64Array(spec.quadraturePoints);
  let weightSum = 0;
  for (let q = 0; q < spec.quadraturePoints; q += 1) {
    y[q] = -jumpHalf + q * dy;
    const standardized = (y[q] - spec.jumpMean) / spec.jumpVol;
    weight[q] = Math.exp(-0.5 * standardized * standardized) / (spec.jumpVol * Math.sqrt(2 * Math.PI)) * dy;
    weightSum += weight[q];
  }
  for (let q = 0; q < spec.quadraturePoints; q += 1) weight[q] /= weightSum;
  let values = new Float64Array(m);
  for (let i = 0; i < m; i += 1) values[i] = Math.max(Math.exp(x[i]) - spec.strike, 0);
  const interpolate = (grid, point) => {
    if (point <= x[0]) return grid[0];
    if (point >= x[m - 1]) return Math.max(Math.exp(point) - spec.strike, 0);
    const position = (point - x[0]) / dx;
    const lower2 = Math.floor(position);
    const frac = position - lower2;
    return grid[lower2] * (1 - frac) + grid[lower2 + 1] * frac;
  };
  const variance = spec.vol * spec.vol;
  const subL = 0.5 * variance / (dx * dx) - drift / (2 * dx);
  const diagL = -variance / (dx * dx) - (spec.rate + spec.jumpIntensity);
  const supL = 0.5 * variance / (dx * dx) + drift / (2 * dx);
  const interior = m - 2;
  const lower = new Float64Array(interior);
  const diag = new Float64Array(interior);
  const upper = new Float64Array(interior);
  const rhs = new Float64Array(interior);
  for (let step = 0; step < spec.timeSteps; step += 1) {
    const tau = (step + 1) * dt;
    const leftBoundary = 0;
    const rightBoundary = Math.exp(x[m - 1]) - spec.strike * Math.exp(-spec.rate * tau);
    const jump = new Float64Array(m);
    for (let i = 1; i <= interior; i += 1) {
      let integral = 0;
      for (let q = 0; q < spec.quadraturePoints; q += 1) integral += interpolate(values, x[i] + y[q]) * weight[q];
      jump[i] = integral;
    }
    for (let i = 1; i <= interior; i += 1) {
      lower[i - 1] = -dt * subL;
      diag[i - 1] = 1 - dt * diagL;
      upper[i - 1] = -dt * supL;
      rhs[i - 1] = values[i] + dt * spec.jumpIntensity * jump[i];
    }
    rhs[0] += dt * subL * leftBoundary;
    rhs[interior - 1] += dt * supL * rightBoundary;
    const solved = solveTridiagonal(lower, diag, upper, rhs);
    const next = new Float64Array(m);
    next[0] = leftBoundary;
    next[m - 1] = rightBoundary;
    for (let i = 1; i <= interior; i += 1) next[i] = solved[i - 1];
    values = next;
  }
  const j = Math.round((center - (center - halfWidth)) / dx);
  return values[j];
}

// src/engines/bermudan-dual.ts
function intrinsic2(spot2, strike, isCall) {
  return isCall ? Math.max(spot2 - strike, 0) : Math.max(strike - spot2, 0);
}
function regress2(level, value, mask) {
  const ata = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  const aty = [0, 0, 0];
  let count = 0;
  for (let p = 0; p < value.length; p += 1) {
    if (!mask[p]) continue;
    count += 1;
    const basis = [1, level[p], level[p] * level[p]];
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) ata[i][j] += basis[i] * basis[j];
      aty[i] += basis[i] * value[p];
    }
  }
  return count >= 3 ? solveLinearSystem(ata, aty) : [0, 0, 0];
}
function continuation(coefficients, spot2) {
  return coefficients[0] + coefficients[1] * spot2 + coefficients[2] * spot2 * spot2;
}
function bermudanDualBound(spec) {
  const steps = spec.exerciseDates;
  const dt = spec.maturity / steps;
  const drift = (spec.rate - 0.5 * spec.vol * spec.vol) * dt;
  const diffusion = spec.vol * Math.sqrt(dt);
  const generator = new MersenneTwister(spec.seed);
  const spots = [];
  const logState = new Float64Array(spec.paths).fill(Math.log(spec.spot));
  for (let k = 0; k < steps; k += 1) {
    const level = new Float64Array(spec.paths);
    for (let p = 0; p < spec.paths; p += 1) {
      logState[p] += drift + diffusion * inverseNormalCdf(generator.nextDouble());
      level[p] = Math.exp(logState[p]);
    }
    spots.push(level);
  }
  const discounted = (value2, step) => value2 * Math.exp(-spec.rate * (step + 1) * dt);
  const allPaths = Array.from({ length: spec.paths }, () => true);
  const dualCoefficients = new Array(steps);
  const value = new Float64Array(spec.paths);
  for (let p = 0; p < spec.paths; p += 1) value[p] = discounted(intrinsic2(spots[steps - 1][p], spec.strike, spec.isCall), steps - 1);
  dualCoefficients[steps - 1] = [0, 0, 0];
  for (let k = steps - 2; k >= 0; k -= 1) {
    const level = spots[k];
    const inMoney = Array.from({ length: spec.paths }, (_, p) => intrinsic2(level[p], spec.strike, spec.isCall) > 0);
    dualCoefficients[k] = regress2(level, value, allPaths);
    const policy = regress2(level, value, inMoney);
    for (let p = 0; p < spec.paths; p += 1) {
      const immediate = discounted(intrinsic2(level[p], spec.strike, spec.isCall), k);
      if (inMoney[p] && immediate > continuation(policy, level[p])) value[p] = immediate;
    }
  }
  let lower = 0;
  for (let p = 0; p < spec.paths; p += 1) lower += value[p];
  lower /= spec.paths;
  let upper = 0;
  for (let p = 0; p < spec.paths; p += 1) {
    let martingale = 0;
    let best = Number.NEGATIVE_INFINITY;
    for (let k = 0; k < steps; k += 1) {
      const payoff2 = discounted(intrinsic2(spots[k][p], spec.strike, spec.isCall), k);
      const valueFunction = k === steps - 1 ? payoff2 : Math.max(payoff2, continuation(dualCoefficients[k], spots[k][p]));
      if (k > 0) martingale += valueFunction - continuation(dualCoefficients[k - 1], spots[k - 1][p]);
      best = Math.max(best, payoff2 - martingale);
    }
    upper += best;
  }
  upper /= spec.paths;
  return { lower, upper };
}

// src/runtime/calc-graph.ts
var CalcNode = class {
  constructor(deps, compute, isInput) {
    this.deps = deps;
    this.compute = compute;
    this.isInput = isInput;
    __publicField(this, "value");
    __publicField(this, "valid", false);
    __publicField(this, "dependents", []);
  }
};
var CalcGraph = class {
  input(value) {
    const node = new CalcNode([], () => value, true);
    node.value = value;
    node.valid = true;
    return node;
  }
  node(deps, compute) {
    const node = new CalcNode(deps, compute, false);
    for (const dep of deps) dep.dependents.push(node);
    return node;
  }
  get(node) {
    if (node.valid) return node.value;
    const depValues = node.deps.map((dep) => this.get(dep));
    node.value = node.compute(depValues);
    node.valid = true;
    return node.value;
  }
  set(input, value) {
    if (!input.isInput) throw new Error("only input nodes can be set");
    input.value = value;
    input.valid = true;
    this.invalidate(input);
  }
  invalidate(node) {
    for (const dependent of node.dependents) {
      if (dependent.valid) {
        dependent.valid = false;
        this.invalidate(dependent);
      }
    }
  }
};
export {
  BrownianBridge,
  COSTS,
  CalcGraph,
  Calendar,
  CompiledEuropeanPricer,
  DiscountCurve,
  FixedPointGroup,
  Graph,
  MersenneTwister,
  PORTFOLIOS,
  PassManager,
  SIGNALS,
  Sobol,
  Tape,
  VolSurface,
  aadLevenbergMarquardt,
  addMatrices,
  addMonths,
  alphaBeta,
  alphaDecay,
  and,
  antitheticEstimate,
  attributePnl,
  autotune,
  backtest,
  basisAdjustedCurve,
  basketCallMc,
  benjaminiHochberg,
  bermudanDualBound,
  blackCaplet,
  blackCapletPrice,
  blackLitterman,
  blackScholes,
  blackScholesCf,
  bonferroni,
  bootstrapFromParSwaps,
  bootstrapHazardCurve,
  bootstrapProjectionCurve,
  borrowCost,
  breakevenInflation,
  buildBackwardGraph,
  buildEuropeanGraph,
  butterflyG,
  calibrateAndPriceSlvCall,
  calibrateHeston,
  calibrateHestonAndPrice,
  calibrateSabr,
  calibrateVolToPrice,
  calmar,
  cdsParSpread,
  checkProduct,
  checkScript,
  checkpointedGradient,
  checkpointedPathGradient,
  cholesky,
  cirSurvival,
  civilFromDays,
  cmsConvexityAdjustment,
  combineCosts,
  combineSignals,
  compileCpuMultiKernel,
  compileFor,
  compileProduct,
  compose,
  computeCva,
  computeCvaSensitivities,
  computeFullXva,
  computeXva,
  constant,
  constrainedMeanVariance,
  controlVariateEstimate,
  correlationFromCovariance,
  cosEuropeanPrice,
  cpuTarget,
  creditSpread,
  crossSectional,
  csaPresentValue,
  cudaTarget,
  curvatureRiskCharge,
  dayCountFraction,
  daysFromCivil,
  defaultRiskCharge,
  deflatedSharpe,
  deltaRiskCharge,
  depositRate,
  desugar,
  differentialEvolution,
  digitalCallLrDelta,
  dot,
  drawdownTable,
  equalWeight,
  equityForward,
  equityForwardWithYield,
  europeanCall as europeanCallContract,
  europeanFullTrade,
  europeanPut as europeanPutContract,
  europeanTrade,
  evaluate,
  evaluateResidualJacobian,
  eventBacktest,
  exchangeOptionAdi,
  exchangeOptionMc,
  expectedShortfall,
  factorRegression,
  factorRiskAttribution,
  fitSvi,
  forwardMode,
  forwardOverReverse,
  forwardTrade,
  fraRate,
  fullPathGradient,
  futureConvexityAdjustment,
  futureRate,
  fxForward,
  g2ppBondToday,
  g2ppPayerSwaption,
  g2ppPayerSwaptionMc,
  generateScenarios,
  generateSchedule,
  geometricAsianCall,
  give,
  gradientOf,
  hestonCf,
  hierarchicalRiskParity,
  historicalExpectedShortfall,
  historicalVar,
  hitRate,
  hullWhiteBond,
  hullWhitePayerSwaption,
  hullWhitePayerSwaptionMc,
  icSeries,
  identity,
  impliedVolatility,
  importanceSampledCall,
  incrementalRiskCharge,
  inflationLinkedZeroCouponBond,
  informationCoefficient,
  informationRatio,
  informationRatioOfIC,
  invert,
  jacobiEigen,
  jamshidianPayerSwaption,
  jyInflationCallMc,
  ledoitWolf,
  levenbergMarquardt,
  linearCost,
  lmmCaplet,
  lmmSwaption,
  localVol,
  longShortRank,
  margrabeExchange,
  matMul,
  matVec,
  maxDrawdown,
  meanReversion,
  meanVariance,
  mertonCf,
  defaultProbability as mertonDefaultProbability,
  mertonPideCall,
  minTrackRecordLength,
  mlmcAsian,
  moments,
  momentum,
  monotoneCubic,
  multiCurveParRate,
  multiCurveSwapValue,
  normalCdf,
  normalPdf,
  ols,
  optimize,
  parSwapRate,
  parameterQuoteSensitivity,
  parametricVar,
  parseProduct,
  pcaFactors,
  pearson,
  percentile,
  plainEstimate,
  pooledEuropeanCall,
  portfolioPnl,
  priceArithmeticAsianCall,
  priceBermudanLsm,
  priceEuropean,
  priceGeometricAsianCall,
  priceHestonCall,
  priceHestonCallQe,
  priceHybridCall,
  priceLmmCaplet,
  priceMertonCall,
  pricePde,
  priceProduct,
  priceQuoteSensitivity,
  priceRoughBergomiCall,
  priceScript,
  priceStochasticLocalVolCall,
  priceUpAndOutCall,
  printProduct,
  printScript,
  probabilisticSharpe,
  probabilityOfBacktestOverfitting,
  projectedSor,
  projectionForward,
  qmcAsianCall,
  qmcEuropeanCall,
  quantoForward,
  rankInformationCoefficient,
  ranks,
  reconstructCovariance,
  registerBuiltinBackends,
  registerBuiltinOps,
  reverse,
  riskContributions,
  riskParity,
  rollingSharpe,
  rollingVolatility,
  runCli,
  runLangCli,
  sabrImpliedVol,
  sampleCovariance,
  scale2 as scale,
  scaleMatrix,
  sharpe,
  sharpePerPeriod,
  sharpeTStat,
  simulateExposure,
  simulateFullExposure,
  sortino,
  spearman,
  squareRootImpact,
  ssviImpliedVol,
  ssviTotalVariance,
  standardisedCapital,
  stressedVar,
  summarizePerformance,
  sviImpliedVol,
  swapKeyRateSensitivities,
  swapValue,
  tearSheet,
  trackingError,
  trancheExpectedLoss,
  transpose,
  underwater,
  valueAtRisk,
  variable,
  vegaRiskCharge,
  volTarget,
  walkForward,
  weekday,
  withIntercept,
  wrongWayCva,
  zeroCouponBond2 as zcbContract,
  zeroCouponBond,
  zscore
};
