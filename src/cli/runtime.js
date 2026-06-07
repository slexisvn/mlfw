import * as fw from '../index.js';
import { Module } from '../nn/module.js';
import { Tensor } from '../tensor/core/tensor.js';
import { SymbolicTensor } from '../tracing/symbolic_tensor.js';
import { executeCompiled, _traceCore } from '../tracing/compile.js';
import { Compiler } from '../compiler/pipeline/compiler.js';
import { TraceLevel } from '../compiler/pipeline/trace.js';
import { parse } from './parser.js';
import { CompiledProgramView, formatTrace } from './format.js';
import { installBuiltins, takeNamed } from './builtins.js';

class Environment {
  constructor(parent = null) {
    this.parent = parent;
    this.values = new Map();
  }
  define(name, value) { this.values.set(name, value); return value; }
  set(name, value) {
    if (this.values.has(name)) { this.values.set(name, value); return value; }
    if (this.parent) return this.parent.set(name, value);
    throw new Error(`Cannot assign to undefined variable '${name}'`);
  }
  get(name) {
    if (this.values.has(name)) return this.values.get(name);
    if (this.parent) return this.parent.get(name);
    throw new Error(`Unknown name '${name}'`);
  }
}

export class TensorLangRuntime {
  constructor({ output = console.log } = {}) {
    this.output = output;
    this.global = new Environment();
    this._installBuiltins();
  }

  execute(source) {
    try {
      return this.evaluateProgram(parse(source), this.global);
    } catch (error) {
      if (error.line !== undefined) throw error;
      throw new LangRuntimeError(error.message, this.currentNode?.line ?? 1, this.currentNode?.column ?? 1, error);
    }
  }

  getCompletionNames() {
    return [...this.global.values.keys()];
  }

  getVariable(name) {
    try { return this.global.get(name); } catch { return undefined; }
  }

  evaluateProgram(program, env) {
    let value;
    for (const statement of program.body) {
      const result = this.evaluateStatement(statement, env);
      if (result && (result.__return || result.__break || result.__continue)) return result;
      value = result;
    }
    return value;
  }

  evaluateStatement(node, env) {
    return this.withNode(node, () => {
      if (node.type === 'Assign') return env.define(node.name, this.evaluateExpression(node.value, env));
      if (node.type === 'CompoundAssign') {
        const current = env.get(node.name);
        const right = this.evaluateExpression(node.value, env);
        return env.set(node.name, this.applyBinary(node.op, current, right));
      }
      if (node.type === 'If') return this.evaluateIf(node, env);
      if (node.type === 'For') return this.evaluateFor(node, env);
      if (node.type === 'While') return this.evaluateWhile(node, env);
      if (node.type === 'Break') return { __break: true };
      if (node.type === 'Continue') return { __continue: true };
      if (node.type === 'ExpressionStatement') return this.evaluateExpression(node.expression, env);
      if (node.type === 'Return') return { __return: true, value: this.evaluateExpression(node.value, env) };
      if (node.type === 'FunctionDeclaration') return this.defineFunction(node, env);
      if (node.type === 'ModelDeclaration') return this.defineModel(node, env);
      if (node.type === 'ForwardDeclaration') throw new Error('forward can only appear inside model');
      throw new Error(`Unsupported statement ${node.type}`);
    });
  }

  evaluateExpression(node, env) {
    return this.withNode(node, () => {
      if (node.type === 'Literal') return node.value;
      if (node.type === 'Identifier') return env.get(node.name);
      if (node.type === 'Array') return node.elements.map(x => this.evaluateExpression(x, env));
      if (node.type === 'Unary') {
        const value = this.evaluateExpression(node.value, env);
        if (node.op === '-') return this.applyUnaryMinus(value);
        if (node.op === 'not') return this.applyUnaryNot(value);
        return value;
      }
      if (node.type === 'Binary') {
        if (node.op === 'and' || node.op === 'or') {
          const left = this.evaluateExpression(node.left, env);
          if (!isTensorValue(left)) {
            if (node.op === 'and') return left ? this.evaluateExpression(node.right, env) : left;
            return left ? left : this.evaluateExpression(node.right, env);
          }
          const right = this.evaluateExpression(node.right, env);
          return this.applyBinary(node.op, left, right);
        }
        return this.applyBinary(node.op, this.evaluateExpression(node.left, env), this.evaluateExpression(node.right, env));
      }
      if (node.type === 'Member') {
        const object = this.evaluateExpression(node.object, env);
        const value = object[node.property];
        return typeof value === 'function' ? value.bind(object) : value;
      }
      if (node.type === 'Index') return this.evaluateIndex(node, env);
      if (node.type === 'Call') return this.evaluateCall(node, env);
      throw new Error(`Unsupported expression ${node.type}`);
    });
  }

  evaluateCall(node, env) {
    const callable = this.evaluateExpression(node.callee, env);
    const positional = [];
    const named = {};
    for (const arg of node.args) {
      const value = this.evaluateExpression(arg.value, env);
      if (arg.name) named[arg.name] = value;
      else positional.push(value);
    }
    if (Object.keys(named).length > 0) positional.push({ __named: true, ...named });
    if (callable instanceof Module) return callable.forward(...positional);
    if (typeof callable !== 'function') throw new Error('Value is not callable');
    return callable(...positional);
  }

  applyUnaryMinus(value) {
    if (isTensorValue(value)) return fw.neg(value);
    return -value;
  }

  applyUnaryNot(value) {
    if (isTensorValue(value)) {
      const one = fw.ones(value.shape, { dtype: value.dtype, device: value.device });
      return fw.sub(one, value);
    }
    return !value;
  }

  applyBinary(op, left, right) {
    const tensor = isTensorValue(left) || isTensorValue(right);
    if (!tensor) {
      if (op === '+') return left + right;
      if (op === '-') return left - right;
      if (op === '*') return left * right;
      if (op === '/') return left / right;
      if (op === '**') return left ** right;
      if (op === '==') return left === right;
      if (op === '!=') return left !== right;
      if (op === '<') return left < right;
      if (op === '<=') return left <= right;
      if (op === '>') return left > right;
      if (op === '>=') return left >= right;
      if (op === 'and') return left && right;
      if (op === 'or') return left || right;
    }
    [left, right] = promoteScalars(left, right);
    const fn = {
      '+': fw.add, '-': fw.sub, '*': fw.mul, '/': fw.div, '**': fw.pow, '@': fw.matmul,
      '==': fw.eq, '!=': fw.ne, '<': fw.lt, '<=': fw.le, '>': fw.gt, '>=': fw.ge,
      'and': fw.mul,
      'or': (a, b) => fw.sub(fw.add(a, b), fw.mul(a, b)),
    }[op];
    if (!fn) throw new Error(`Unsupported operator '${op}'`);
    return fn(left, right);
  }

  isTruthy(value) {
    if (isTensorValue(value)) {
      if (value.numel !== 1) throw new Error('Condition tensor must be a scalar (single element)');
      return Boolean(value.item());
    }
    return Boolean(value);
  }

  evaluateIf(node, env) {
    if (this.isTruthy(this.evaluateExpression(node.condition, env))) {
      return this.evaluateProgram({ type: 'Program', body: node.body }, env);
    }
    for (const elif of node.elifs) {
      if (this.isTruthy(this.evaluateExpression(elif.condition, env))) {
        return this.evaluateProgram({ type: 'Program', body: elif.body }, env);
      }
    }
    if (node.elseBody) {
      return this.evaluateProgram({ type: 'Program', body: node.elseBody }, env);
    }
    return undefined;
  }

  evaluateFor(node, env) {
    const iterable = this.evaluateExpression(node.iterable, env);
    if (!Array.isArray(iterable)) throw new Error('for...in expects an array');
    let value;
    for (const item of iterable) {
      env.define(node.variable, item);
      const result = this.evaluateProgram({ type: 'Program', body: node.body }, env);
      if (result && result.__return) return result;
      if (result && result.__break) break;
      if (result && result.__continue) continue;
      value = result;
    }
    return value;
  }

  evaluateWhile(node, env) {
    let value;
    while (this.isTruthy(this.evaluateExpression(node.condition, env))) {
      const result = this.evaluateProgram({ type: 'Program', body: node.body }, env);
      if (result && result.__return) return result;
      if (result && result.__break) break;
      if (result && result.__continue) continue;
      value = result;
    }
    return value;
  }

  defineFunction(node, declarationEnv) {
    const runtime = this;
    const func = (...args) => {
      const callEnv = new Environment(declarationEnv);
      node.params.forEach((name, i) => callEnv.define(name, args[i]));
      const result = runtime.evaluateProgram({ type: 'Program', body: node.body }, callEnv);
      return result && result.__return ? result.value : result;
    };
    func._langName = node.name;
    declarationEnv.define(node.name, func);
    return func;
  }

  defineModel(node, declarationEnv) {
    const runtime = this;
    const forward = node.body.find(x => x.type === 'ForwardDeclaration');
    if (!forward) throw new Error(`Model ${node.name} needs a forward block`);
    const fields = node.body.filter(x => x.type !== 'ForwardDeclaration');

    const factory = (...args) => {
      const named = takeNamed(args);
      const modelEnv = new Environment(declarationEnv);
      node.params.forEach((name, i) => modelEnv.define(name, named[name] ?? args[i]));
      class LangModel extends Module {
        constructor() {
          super();
          this._langName = node.name;
          for (const field of fields) {
            const value = runtime.evaluateStatement(field, modelEnv);
            if (field.type === 'Assign') this[field.name] = value;
          }
        }
        forward(...inputs) {
          const callEnv = new Environment(modelEnv);
          for (const field of fields) {
            if (field.type === 'Assign') callEnv.define(field.name, this[field.name]);
          }
          forward.params.forEach((name, i) => callEnv.define(name, inputs[i]));
          const result = runtime.evaluateProgram({ type: 'Program', body: forward.body }, callEnv);
          return result && result.__return ? result.value : result;
        }
        toString() { return `${this._langName}${super.toString().slice(this.constructor.name.length)}`; }
      }
      return new LangModel();
    };
    factory._langName = node.name;
    declarationEnv.define(node.name, factory);
    return factory;
  }

  _installBuiltins() {
    const define = (name, value) => this.global.define(name, value);
    installBuiltins(this, define);
  }

  evaluateIndex(node, env) {
    let value = this.evaluateExpression(node.object, env);
    if (!(value instanceof Tensor)) throw new Error('Indexing currently expects a Tensor');
    let dim = 0;
    for (const item of node.items) {
      if (dim >= value.ndim) throw new Error(`Too many indices for tensor with ${value.ndim} dimensions`);
      if (item.type === 'Slice') {
        const start = item.start ? this.evaluateExpression(item.start, env) : 0;
        const end = item.end ? this.evaluateExpression(item.end, env) : value.shape[dim];
        const step = item.step ? this.evaluateExpression(item.step, env) : 1;
        if (![start, end, step].every(Number.isInteger)) throw new Error('Slice bounds must be integers');
        if (step <= 0) throw new Error('Slice step must be a positive integer');
        value = value.slice(dim, start, end, step);
        dim++;
      } else {
        let index = this.evaluateExpression(item, env);
        if (!Number.isInteger(index)) throw new Error('Tensor index must be an integer');
        if (index < 0) index += value.shape[dim];
        if (index < 0 || index >= value.shape[dim]) {
          throw new Error(`Index ${index} is out of bounds for dimension ${dim} with size ${value.shape[dim]}`);
        }
        value = value.select(dim, index);
      }
    }
    return value;
  }

  withNode(node, evaluate) {
    const previous = this.currentNode;
    this.currentNode = node;
    try {
      return evaluate();
    } catch (error) {
      if (error.line !== undefined) throw error;
      throw new LangRuntimeError(error.message, node.line ?? 1, node.column ?? 1, error);
    } finally {
      this.currentNode = previous;
    }
  }

  compile(model, ...args) {
    if (!(model instanceof Module)) {
      throw new Error('compile() currently expects a model. Example: compile(model, input=x)');
    }
    const named = takeNamed(args);
    const rawInput = named.input ?? args[0];
    let inputs = rawInput != null ? (Array.isArray(rawInput) ? rawInput : [rawInput]) : null;
    if (inputs && inputs.some(x => !(x instanceof Tensor))) {
      throw new Error('compile() input must be a tensor, for example compile(model, input=x)');
    }

    const targetName = named.target ?? 'cpu';
    const target = targetName === 'gpu' ? fw.GPUTarget() : targetName === 'wasm' ? fw.WasmTarget() : fw.CPUTarget();
    const runtime = this;

    const compilerOpts = {
      target,
      verify: named.verify ?? true,
      fusion: {
        enabled: named.fusion ?? false,
        epilogue: named.epilogue ?? false,
        strategy: named.fusionStrategy ?? 'xla',
      },
      scheduling: {
        enabled: named.scheduling ?? false,
        autotune: named.autotune ?? false,
        numTrials: named.numTrials ?? 64,
        timeBudgetMs: named.timeBudgetMs ?? 30000,
      },
      quantization: {
        enabled: named.quantization ?? false,
      },
      optimization: {
        layout: named.layout ?? false,
        rematerialization: named.rematerialization ?? false,
      },
      memory: {
        inplaceReuse: named.inplaceReuse ?? false,
      },
      partition: {
        enabled: named.partition ?? false,
        targets: [],
      },
    };

    let _compiled = null;
    let _shapeKey = null;
    let _view = null;

    const modelName = model._langName || model.constructor.name;

    function _getShapeKey(tensorInputs) {
      let key = '';
      for (const inp of tensorInputs) key += inp.shape.join(',') + ':' + inp.dtype + '|';
      return key;
    }

    function _doCompile(tensorInputs) {
      const events = [];
      const traced = _traceCore(
        (...values) => model.forward(...values),
        tensorInputs,
        { name: modelName },
      );
      const compiler = new Compiler({
        ...compilerOpts,
        trace: {
          level: TraceLevel.DEBUG,
          sink: event => events.push(event),
          irSnapshot: { afterGraphPasses: true, afterLowering: true, afterScheduling: true },
        },
      });
      const result = compiler.compile(traced.graph);
      _view = new CompiledProgramView({ model, inputs: tensorInputs, graph: traced.graph, result, events, target: targetName });
      runtime.output(formatTrace(events));
      return {
        result,
        graph: traced.graph,
        capturedParams: traced.capturedParams,
        numUserInputs: traced.numUserInputs,
        outputTypes: traced.outputTypes,
      };
    }

    // Eager compile if inputs provided
    if (inputs) {
      _compiled = _doCompile(inputs);
      _shapeKey = _getShapeKey(inputs);
    }

    // Callable: compiled(x) → Tensor
    const execFn = (...callInputs) => {
      const key = _getShapeKey(callInputs);
      if (!_compiled || _shapeKey !== key) {
        _compiled = _doCompile(callInputs);
        _shapeKey = key;
      }
      return executeCompiled(_compiled, callInputs);
    };

    execFn._isCompiled = true;
    execFn._langName = 'compiled';
    Object.defineProperty(execFn, '_compiledView', { get: () => _view });
    return execFn;
  }
}

function isTensorValue(value) {
  return value instanceof Tensor || value instanceof SymbolicTensor;
}

function promoteScalars(left, right) {
  const reference = left instanceof Tensor ? left : right instanceof Tensor ? right : null;
  if (!reference) return [left, right];
  const options = { dtype: reference.dtype, device: reference.device };
  if (!isTensorValue(left)) left = fw.tensor(left, options);
  if (!isTensorValue(right)) right = fw.tensor(right, options);
  return [left, right];
}

export class LangRuntimeError extends Error {
  constructor(message, line, column, cause) {
    super(`${message} at ${line}:${column}`, { cause });
    this.name = 'LangRuntimeError';
    this.line = line;
    this.column = column;
  }
}
