export const ArgKind = Object.freeze({
  TENSOR: 'Tensor',
  SCALAR: 'Scalar',
  INT: 'int',
  FLOAT: 'float',
  BOOL: 'bool',
  INT_LIST: 'int[]',
  TENSOR_LIST: 'Tensor[]',
  DTYPE: 'Dtype',
  DEVICE: 'Device',
  STRING: 'str',
  MEMORY_FORMAT: 'MemoryFormat',
});

const _TENSOR_KINDS = new Set([ArgKind.TENSOR, ArgKind.TENSOR_LIST]);

export class SchemaArg {
  constructor(name, kind, defaultValue, isOut) {
    this.name = name;
    this.kind = kind;
    this.defaultValue = defaultValue ?? null;
    this.isOut = isOut ?? false;
  }

  get isTensor() {
    return _TENSOR_KINDS.has(this.kind);
  }
}

export class OperatorSchema {
  constructor(namespace, name, overload, args, returns) {
    this.namespace = namespace;
    this.name = name;
    this.overload = overload || '';
    this.args = Object.freeze(args);
    this.returns = Object.freeze(returns);

    this._key = null;
    this._tensorArgIndices = null;
  }

  qualifiedName() {
    return `${this.namespace}::${this.name}`;
  }

  key() {
    if (!this._key) {
      this._key = this.overload
        ? `${this.namespace}::${this.name}.${this.overload}`
        : `${this.namespace}::${this.name}`;
    }
    return this._key;
  }

  get tensorArgIndices() {
    if (!this._tensorArgIndices) {
      const indices = [];
      for (let i = 0; i < this.args.length; i++) {
        if (this.args[i].isTensor) indices.push(i);
      }
      this._tensorArgIndices = Object.freeze(indices);
    }
    return this._tensorArgIndices;
  }

  get numTensorArgs() {
    return this.tensorArgIndices.length;
  }
}

const _KIND_MAP = new Map([
  ['Tensor', ArgKind.TENSOR],
  ['Scalar', ArgKind.SCALAR],
  ['int', ArgKind.INT],
  ['float', ArgKind.FLOAT],
  ['bool', ArgKind.BOOL],
  ['int[]', ArgKind.INT_LIST],
  ['Tensor[]', ArgKind.TENSOR_LIST],
  ['Dtype', ArgKind.DTYPE],
  ['Device', ArgKind.DEVICE],
  ['str', ArgKind.STRING],
  ['MemoryFormat', ArgKind.MEMORY_FORMAT],
]);

export function parseSchema(str, namespace) {
  const ns = namespace || 'mlc';
  const arrowIdx = str.indexOf('->');
  const sigPart = arrowIdx >= 0 ? str.substring(0, arrowIdx).trim() : str.trim();
  const retPart = arrowIdx >= 0 ? str.substring(arrowIdx + 2).trim() : '';

  const parenOpen = sigPart.indexOf('(');
  const parenClose = sigPart.lastIndexOf(')');

  let name, overload = '';
  const namePart = sigPart.substring(0, parenOpen).trim();
  const dotIdx = namePart.indexOf('.');
  if (dotIdx >= 0) {
    name = namePart.substring(0, dotIdx);
    overload = namePart.substring(dotIdx + 1);
  } else {
    name = namePart;
  }

  const argStr = sigPart.substring(parenOpen + 1, parenClose).trim();
  const args = argStr.length > 0 ? _parseArgs(argStr) : [];

  const returns = retPart.length > 0 ? _parseReturns(retPart) : [{ kind: ArgKind.TENSOR }];

  return new OperatorSchema(ns, name, overload, args, returns);
}

function _parseArgs(str) {
  const parts = _splitTopLevel(str, ',');
  return parts.map(part => {
    const trimmed = part.trim();
    const eqIdx = trimmed.indexOf('=');
    let mainPart = trimmed;
    let defaultValue = null;
    if (eqIdx >= 0) {
      mainPart = trimmed.substring(0, eqIdx).trim();
      defaultValue = trimmed.substring(eqIdx + 1).trim();
    }

    const isOut = mainPart.endsWith('!');
    if (isOut) mainPart = mainPart.substring(0, mainPart.length - 1).trim();

    const spaceIdx = mainPart.lastIndexOf(' ');
    let typePart, namePart;
    if (spaceIdx >= 0) {
      typePart = mainPart.substring(0, spaceIdx).trim();
      namePart = mainPart.substring(spaceIdx + 1).trim();
    } else {
      typePart = mainPart;
      namePart = '';
    }

    const kind = _KIND_MAP.get(typePart) || ArgKind.SCALAR;

    return new SchemaArg(namePart, kind, defaultValue, isOut);
  });
}

function _parseReturns(str) {
  const trimmed = str.trim();
  if (trimmed.startsWith('(') && trimmed.endsWith(')')) {
    const inner = trimmed.substring(1, trimmed.length - 1);
    const parts = _splitTopLevel(inner, ',');
    return parts.map(p => {
      const t = p.trim();
      const kind = _KIND_MAP.get(t) || ArgKind.TENSOR;
      return { kind };
    });
  }
  const kind = _KIND_MAP.get(trimmed) || ArgKind.TENSOR;
  return [{ kind }];
}

function _splitTopLevel(str, sep) {
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (ch === '(' || ch === '[') depth++;
    else if (ch === ')' || ch === ']') depth--;
    else if (ch === sep && depth === 0) {
      parts.push(str.substring(start, i));
      start = i + 1;
    }
  }
  parts.push(str.substring(start));
  return parts;
}
