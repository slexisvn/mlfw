import {
  ANY, NUMBER, STRING, BOOL, NULL, NONE, TENSOR,
  listType, dictType, functionType, moduleType, unionType,
  isAssignable, isAny, typeToString, join,
} from './types.js';

export class LangTypeError extends Error {
  constructor(message, line, column) {
    super(`${message} at ${line}:${column}`);
    this.name = 'TypeError';
    this.line = line;
    this.column = column;
  }
}

const NAME_TYPES = {
  int: NUMBER, float: NUMBER, num: NUMBER, number: NUMBER,
  str: STRING, string: STRING, bool: BOOL,
  Tensor: TENSOR, tensor: TENSOR, none: NONE, null: NULL,
};

export const TYPE_NAMES = Object.freeze([...Object.keys(NAME_TYPES), 'list', 'dict']);

export const HOST_GLOBALS = Object.freeze(['chart']);

const TENSOR_MEMBERS = {
  shape: listType(NUMBER), ndim: NUMBER, numel: NUMBER,
  dtype: STRING, device: STRING, requiresGrad: BOOL, grad: TENSOR,
};

const COMPARISONS = new Set(['==', '!=', '<', '<=', '>', '>=']);
const ARITHMETIC = new Set(['-', '*', '/', '**']);

class TypeEnv {
  constructor(parent = null) {
    this.parent = parent;
    this.values = new Map();
  }
  define(name, type) { this.values.set(name, type); return type; }
  lookup(name) {
    if (this.values.has(name)) return this.values.get(name);
    if (this.parent) return this.parent.lookup(name);
    return undefined;
  }
}

class TypeChecker {
  constructor({ builtinNames, builtinTypes }) {
    this.diagnostics = [];
    this.root = new TypeEnv();
    for (const name of builtinNames) this.root.define(name, builtinTypes.get(name) ?? ANY);

    this.exprHandlers = {
      Literal: node => this.inferLiteral(node),
      Identifier: (node, env) => this.inferIdentifier(node, env),
      Array: (node, env) => this.inferArray(node, env),
      Dict: (node, env) => this.inferDict(node, env),
      ListComprehension: (node, env) => this.inferComprehension(node, env),
      Unary: (node, env) => this.inferUnary(node, env),
      Binary: (node, env) => this.inferBinary(node, env),
      Member: (node, env) => this.inferMember(node, env),
      Index: (node, env) => this.inferIndex(node, env),
      Call: (node, env) => this.inferCall(node, env),
    };
    this.stmtHandlers = {
      Assign: (node, env) => this.checkAssign(node, env),
      CompoundAssign: (node, env) => this.checkCompoundAssign(node, env),
      DestructureAssign: (node, env) => this.checkDestructure(node, env),
      IndexAssign: (node, env) => this.checkIndexAssign(node, env),
      If: (node, env, ctx) => this.checkIf(node, env, ctx),
      For: (node, env, ctx) => this.checkFor(node, env, ctx),
      While: (node, env, ctx) => this.checkWhile(node, env, ctx),
      Return: (node, env, ctx) => this.checkReturn(node, env, ctx),
      ExpressionStatement: (node, env) => this.infer(node.expression, env),
      FunctionDeclaration: (node, env) => this.checkFunction(node, env),
      ModelDeclaration: (node, env) => this.checkModel(node, env),
    };
  }

  run(program) {
    this.checkBlock(program.body, this.root, { returns: [], declaredReturn: ANY });
    return this.diagnostics;
  }

  report(message, node) {
    this.diagnostics.push(new LangTypeError(message, node?.line ?? 1, node?.column ?? 1));
  }

  checkBlock(body, env, ctx) {
    for (const statement of body) this.checkStatement(statement, env, ctx);
  }

  checkStatement(node, env, ctx) {
    const handler = this.stmtHandlers[node.type];
    if (handler) handler(node, env, ctx);
  }

  infer(node, env) {
    const handler = this.exprHandlers[node.type];
    return handler ? handler(node, env) : ANY;
  }

  isTensor(type) { return type && type.kind === 'tensor'; }

  resolveTypeNode(node) {
    if (!node) return ANY;
    if (node.kind === 'UnionType') return unionType(node.members.map(member => this.resolveTypeNode(member)));
    if (node.kind === 'FunctionType') {
      return functionType(node.params.map(param => this.resolveTypeNode(param)), this.resolveTypeNode(node.ret), false, node.params.length);
    }
    if (node.kind === 'GenericType') {
      if (node.name === 'list') return listType(this.resolveTypeNode(node.args[0]));
      if (node.name === 'dict') return dictType(this.resolveTypeNode(node.args[0]), this.resolveTypeNode(node.args[1]));
      if (node.name === 'Tensor' || node.name === 'tensor') return TENSOR;
      return moduleType(node.name);
    }
    if (node.name === 'list') return listType(ANY);
    if (node.name === 'dict') return dictType(ANY, ANY);
    return NAME_TYPES[node.name] ?? moduleType(node.name);
  }

  elementOf(type) {
    if (isAny(type)) return ANY;
    if (type.kind === 'list') return type.element;
    if (type.kind === 'dict') return type.key;
    if (type.kind === 'string') return STRING;
    return ANY;
  }

  inferLiteral(node) {
    const value = node.value;
    if (typeof value === 'number') return NUMBER;
    if (typeof value === 'string') return STRING;
    if (typeof value === 'boolean') return BOOL;
    if (value === null) return NULL;
    return ANY;
  }

  inferIdentifier(node, env) {
    const type = env.lookup(node.name);
    if (type === undefined) {
      this.report(`undefined name '${node.name}'`, node);
      return ANY;
    }
    return type;
  }

  inferArray(node, env) {
    let element = null;
    for (const item of node.elements) {
      const type = this.infer(item, env);
      element = element === null ? type : join(element, type);
    }
    return listType(element ?? ANY);
  }

  inferDict(node, env) {
    let key = null;
    let value = null;
    for (const entry of node.entries) {
      const keyType = this.infer(entry.key, env);
      const valueType = this.infer(entry.value, env);
      key = key === null ? keyType : join(key, keyType);
      value = value === null ? valueType : join(value, valueType);
    }
    return dictType(key ?? ANY, value ?? ANY);
  }

  inferComprehension(node, env) {
    const iterable = this.infer(node.iterable, env);
    const scope = new TypeEnv(env);
    scope.define(node.variable, this.elementOf(iterable));
    if (node.condition) this.infer(node.condition, scope);
    return listType(this.infer(node.expr, scope));
  }

  inferUnary(node, env) {
    const type = this.infer(node.value, env);
    if (node.op === 'not') return this.isTensor(type) ? TENSOR : BOOL;
    if (this.isTensor(type)) return TENSOR;
    if (isAny(type)) return ANY;
    if (type.kind === 'number') return NUMBER;
    return ANY;
  }

  inferBinary(node, env) {
    const left = this.infer(node.left, env);
    const right = this.infer(node.right, env);
    return this.binaryResult(node.op, left, right, node);
  }

  binaryResult(op, left, right, node) {
    const tensor = this.isTensor(left) || this.isTensor(right);
    if (op === 'and' || op === 'or') return tensor ? TENSOR : join(left, right);
    if (COMPARISONS.has(op)) return tensor ? TENSOR : BOOL;
    if (isAny(left) || isAny(right)) return ANY;
    if (op === '@') {
      if (!tensor) this.report(`operator '@' requires tensors, got ${typeToString(left)} and ${typeToString(right)}`, node);
      return TENSOR;
    }
    if (tensor) return TENSOR;
    if (left.kind === 'number' && right.kind === 'number') return NUMBER;
    if (op === '+') {
      if (left.kind === 'string' && right.kind === 'string') return STRING;
      if (left.kind === 'list' && right.kind === 'list') return listType(join(left.element, right.element));
    }
    if (op === '+' || ARITHMETIC.has(op)) {
      this.report(`operator '${op}' cannot be applied to ${typeToString(left)} and ${typeToString(right)}`, node);
    }
    return ANY;
  }

  inferMember(node, env) {
    const object = this.infer(node.object, env);
    if (isAny(object)) return ANY;
    if (object.kind === 'tensor') return TENSOR_MEMBERS[node.property] ?? ANY;
    if ((object.kind === 'list' || object.kind === 'string') && node.property === 'length') return NUMBER;
    return ANY;
  }

  inferIndexItems(items, env) {
    for (const item of items) {
      if (item.type === 'Slice') {
        if (item.start) this.infer(item.start, env);
        if (item.end) this.infer(item.end, env);
        if (item.step) this.infer(item.step, env);
      } else {
        this.infer(item, env);
      }
    }
  }

  inferIndex(node, env) {
    const object = this.infer(node.object, env);
    this.inferIndexItems(node.items, env);
    if (isAny(object)) return ANY;
    if (object.kind === 'list') return object.element;
    if (object.kind === 'dict') return object.value;
    if (object.kind === 'string') return STRING;
    return ANY;
  }

  inferCall(node, env) {
    const calleeType = this.infer(node.callee, env);
    const positional = [];
    let hasNamed = false;
    for (const arg of node.args) {
      const type = this.infer(arg.value, env);
      if (arg.name) hasNamed = true;
      else positional.push({ type, node: arg.value });
    }
    if (calleeType.kind !== 'function') return ANY;
    if (!hasNamed) {
      const min = calleeType.required;
      const max = calleeType.variadic ? Infinity : calleeType.params.length;
      if (positional.length < min || positional.length > max) {
        const expected = min === max ? `${min}` : max === Infinity ? `at least ${min}` : `${min}-${max}`;
        this.report(`expected ${expected} argument(s), got ${positional.length}`, node);
        return calleeType.ret;
      }
      for (let i = 0; i < positional.length && i < calleeType.params.length; i++) {
        if (!isAssignable(positional[i].type, calleeType.params[i])) {
          this.report(`argument ${i + 1} expects ${typeToString(calleeType.params[i])}, got ${typeToString(positional[i].type)}`, positional[i].node);
        }
      }
    }
    return calleeType.ret;
  }

  checkAssign(node, env) {
    const valueType = this.infer(node.value, env);
    if (node.annotation) {
      const declared = this.resolveTypeNode(node.annotation);
      if (!isAssignable(valueType, declared)) {
        this.report(`cannot assign ${typeToString(valueType)} to '${node.name}: ${typeToString(declared)}'`, node);
      }
      env.define(node.name, declared);
    } else {
      env.define(node.name, valueType);
    }
  }

  checkCompoundAssign(node, env) {
    const current = env.lookup(node.name);
    if (current === undefined) this.report(`undefined name '${node.name}'`, node);
    const value = this.infer(node.value, env);
    env.define(node.name, this.binaryResult(node.op, current ?? ANY, value, node));
  }

  checkDestructure(node, env) {
    const valueType = this.infer(node.value, env);
    const element = valueType.kind === 'list' ? valueType.element : ANY;
    for (const name of node.names) env.define(name, element);
  }

  checkIndexAssign(node, env) {
    this.infer(node.object, env);
    this.inferIndexItems(node.items, env);
    this.infer(node.value, env);
  }

  checkIf(node, env, ctx) {
    this.infer(node.condition, env);
    this.checkBlock(node.body, env, ctx);
    for (const elif of node.elifs) {
      this.infer(elif.condition, env);
      this.checkBlock(elif.body, env, ctx);
    }
    if (node.elseBody) this.checkBlock(node.elseBody, env, ctx);
  }

  checkFor(node, env, ctx) {
    const iterable = this.infer(node.iterable, env);
    env.define(node.variable, this.elementOf(iterable));
    this.checkBlock(node.body, env, ctx);
  }

  checkWhile(node, env, ctx) {
    this.infer(node.condition, env);
    this.checkBlock(node.body, env, ctx);
  }

  checkReturn(node, env, ctx) {
    ctx.returns.push({ type: this.infer(node.value, env), node });
  }

  resolveParams(names, annotations, owner, node, env) {
    return names.map((name, index) => {
      const annotation = annotations?.[index];
      if (!annotation) {
        this.report(`parameter '${name}' of ${owner} needs a type annotation`, node);
        return ANY;
      }
      return this.resolveTypeNode(annotation);
    });
  }

  checkReturnTypes(returns, declared) {
    for (const result of returns) {
      if (!isAssignable(result.type, declared)) {
        this.report(`return type ${typeToString(result.type)} is not assignable to declared ${typeToString(declared)}`, result.node);
      }
    }
  }

  checkFunction(node, env) {
    const paramTypes = this.resolveParams(node.params, node.paramTypes, `'${node.name}'`, node, env);
    if (!node.returnType) this.report(`function '${node.name}' needs a return type annotation`, node);
    const declaredReturn = node.returnType ? this.resolveTypeNode(node.returnType) : ANY;
    env.define(node.name, functionType(paramTypes, declaredReturn, false, node.params.length));
    const fnEnv = new TypeEnv(env);
    node.params.forEach((name, index) => fnEnv.define(name, paramTypes[index]));
    const ctx = { returns: [], declaredReturn };
    this.checkBlock(node.body, fnEnv, ctx);
    if (node.returnType) this.checkReturnTypes(ctx.returns, declaredReturn);
  }

  checkModel(node, env) {
    const paramTypes = this.resolveParams(node.params, node.paramTypes, `model '${node.name}'`, node, env);
    env.define(node.name, functionType(paramTypes, moduleType(node.name), false, node.params.length));
    const modelEnv = new TypeEnv(env);
    node.params.forEach((name, index) => modelEnv.define(name, paramTypes[index]));

    const blocks = new Set(['ForwardDeclaration', 'TrainDeclaration', 'ValidateDeclaration', 'OptimizerDeclaration']);
    for (const field of node.body) {
      if (!blocks.has(field.type)) this.checkStatement(field, modelEnv, { returns: [], declaredReturn: ANY });
    }
    for (const block of node.body) {
      if (block.type === 'ForwardDeclaration') this.checkModelBlock(block, modelEnv, node.name, false);
      else if (block.type === 'TrainDeclaration' || block.type === 'ValidateDeclaration') this.checkModelBlock(block, modelEnv, node.name, true);
      else if (block.type === 'OptimizerDeclaration') this.checkBlock(block.body, this.stepEnv(modelEnv, node.name), { returns: [], declaredReturn: ANY });
    }
  }

  stepEnv(modelEnv, modelName) {
    const env = new TypeEnv(modelEnv);
    env.define(modelName, moduleType(modelName));
    return env;
  }

  checkModelBlock(block, modelEnv, modelName, isStep) {
    const paramTypes = this.resolveParams(block.params, block.paramTypes, `'${modelName}'`, block, modelEnv);
    const env = this.stepEnv(modelEnv, modelName);
    block.params.forEach((name, index) => env.define(name, paramTypes[index]));
    if (isStep) env.define('log', functionType([STRING, ANY], NONE, true, 1));
    const declaredReturn = block.returnType ? this.resolveTypeNode(block.returnType) : ANY;
    const ctx = { returns: [], declaredReturn };
    this.checkBlock(block.body, env, ctx);
    if (block.returnType) this.checkReturnTypes(ctx.returns, declaredReturn);
  }
}

export function typecheck(program, builtinEnv) {
  return new TypeChecker(builtinEnv).run(program);
}
