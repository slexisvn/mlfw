type TensorOpReturnSpec =
  | { kind: 'tensor'; arity: 1 }
  | { kind: 'tensor_list' }
  | { kind: 'tuple'; arity: number }
  | { kind: 'value'; arity: 1 };

type TensorOpSpec = Readonly<{
  schema: string;
  ir?: string;
  scalarArgs?: readonly string[];
  returns?: TensorOpReturnSpec;
}>;

type TensorOpMetadata = TensorOpSpec & Readonly<{
  name: string;
  returns: TensorOpReturnSpec;
}>;

export const TENSOR_OPS: readonly TensorOpSpec[] = Object.freeze([
  { schema: 'add(Tensor self, Tensor other) -> Tensor' },
  { schema: 'sub(Tensor self, Tensor other) -> Tensor' },
  { schema: 'mul(Tensor self, Tensor other) -> Tensor' },
  { schema: 'div(Tensor self, Tensor other) -> Tensor' },
  { schema: 'neg(Tensor self) -> Tensor' },
  { schema: 'pow(Tensor self, Tensor exponent) -> Tensor' },
  { schema: 'rem(Tensor self, Tensor other) -> Tensor' },
  { schema: 'maximum(Tensor self, Tensor other) -> Tensor' },
  { schema: 'minimum(Tensor self, Tensor other) -> Tensor' },

  { schema: 'exp(Tensor self) -> Tensor' },
  { schema: 'log(Tensor self) -> Tensor' },
  { schema: 'sqrt(Tensor self) -> Tensor' },
  { schema: 'rsqrt(Tensor self) -> Tensor' },
  { schema: 'abs(Tensor self) -> Tensor' },
  { schema: 'sin(Tensor self) -> Tensor' },
  { schema: 'cos(Tensor self) -> Tensor' },
  { schema: 'tanh(Tensor self) -> Tensor' },
  { schema: 'erf(Tensor self) -> Tensor' },
  { schema: 'erfc(Tensor self) -> Tensor' },
  { schema: 'lgamma(Tensor self) -> Tensor' },
  { schema: 'gamma(Tensor self) -> Tensor' },
  { schema: 'sigmoid(Tensor self) -> Tensor' },
  { schema: 'relu(Tensor self) -> Tensor' },
  { schema: 'gelu(Tensor self) -> Tensor' },
  { schema: 'silu(Tensor self) -> Tensor' },
  { schema: 'sign(Tensor self) -> Tensor' },
  { schema: 'floor(Tensor self) -> Tensor' },
  { schema: 'ceil(Tensor self) -> Tensor' },

  { schema: 'eq(Tensor self, Tensor other) -> Tensor' },
  { schema: 'ne(Tensor self, Tensor other) -> Tensor' },
  { schema: 'lt(Tensor self, Tensor other) -> Tensor' },
  { schema: 'le(Tensor self, Tensor other) -> Tensor' },
  { schema: 'gt(Tensor self, Tensor other) -> Tensor' },
  { schema: 'ge(Tensor self, Tensor other) -> Tensor' },
  { schema: 'where(Tensor condition, Tensor self, Tensor other) -> Tensor' },
  { schema: 'clamp(Tensor self, Tensor min, Tensor max) -> Tensor' },
  { schema: 'pad(Tensor self, Tensor value, int[] low, int[] high) -> Tensor' },
  { schema: 'one_hot(Tensor indices, int depth) -> Tensor' },
  { schema: 'index_select(Tensor self, Tensor index, int dim) -> Tensor' },
  { schema: 'gather(Tensor self, Tensor index, int dim) -> Tensor' },
  { schema: 'scatter_add(Tensor self, Tensor index, Tensor src, int dim) -> Tensor' },
  { schema: 'scatter(Tensor self, int dim, Tensor index, Tensor src) -> Tensor' },

  { schema: 'sum(Tensor self, int[] dim, bool keepdim) -> Tensor' },
  { schema: 'mean(Tensor self, int[] dim, bool keepdim) -> Tensor' },
  { schema: 'max(Tensor self, int[] dim, bool keepdim) -> Tensor' },
  { schema: 'min(Tensor self, int[] dim, bool keepdim) -> Tensor' },
  { schema: 'prod(Tensor self, int[] dim, bool keepdim) -> Tensor' },
  { schema: 'argmax(Tensor self, int dim, bool keepdim) -> Tensor' },
  { schema: 'argmin(Tensor self, int dim, bool keepdim) -> Tensor' },

  { schema: 'matmul(Tensor self, Tensor other) -> Tensor' },
  { schema: 'dot(Tensor self, Tensor other) -> Tensor' },

  { schema: 'cat(Tensor[] tensors, int dim) -> Tensor' },
  { schema: 'stack(Tensor[] tensors, int dim) -> Tensor' },

  { schema: 'clone(Tensor self) -> Tensor' },
  { schema: 'fill(Tensor self, Scalar value) -> Tensor' },

  { schema: 'reshape(Tensor self, int[] shape) -> Tensor', ir: 'reshape', scalarArgs: ['shape'] },
  { schema: 'transpose(Tensor self, int dim0, int dim1) -> Tensor', ir: 'transpose' },
  { schema: 'permute(Tensor self, int[] dims) -> Tensor', ir: 'permute' },
  { schema: 'broadcast_in_dim(Tensor self, int[] result_shape, int[] broadcast_dimensions) -> Tensor', ir: 'broadcast_in_dim' },
  { schema: 'expand(Tensor self, int[] shape) -> Tensor', ir: 'expand', scalarArgs: ['shape'] },
  { schema: 'slice(Tensor self, int dim, int start, int end, int step) -> Tensor' },
  { schema: 'unsqueeze(Tensor self, int dim) -> Tensor' },
  { schema: 'squeeze(Tensor self, int dim) -> Tensor' },
  { schema: 'narrow(Tensor self, int dim, int start, int length) -> Tensor' },
  { schema: 'select(Tensor self, int dim, int index) -> Tensor' },
  { schema: 'contiguous(Tensor self) -> Tensor' },

  { schema: 'repeat(Tensor self, int[] reps) -> Tensor' },
  { schema: 'tile(Tensor self, int[] reps) -> Tensor' },
  { schema: 'split(Tensor self, int[] sizes, int dim) -> Tensor[]', returns: { kind: 'tensor_list' } },
  { schema: 'chunk(Tensor self, int chunks, int dim) -> Tensor[]', returns: { kind: 'tensor_list' } },
  { schema: 'roll(Tensor self, int shift, int dim) -> Tensor' },
  { schema: 'flip(Tensor self, int[] dims) -> Tensor' },
  { schema: 'cumsum(Tensor self, int dim) -> Tensor' },
  { schema: 'sort(Tensor self, int dim, bool descending) -> Tensor' },
  { schema: 'argsort(Tensor self, int dim, bool descending) -> Tensor' },
  { schema: 'topk(Tensor self, int k, int dim, bool largest) -> (Tensor, Tensor)', returns: { kind: 'tuple', arity: 2 } },

  { schema: 'softmax(Tensor self, int dim) -> Tensor' },
  { schema: 'log_softmax(Tensor self, int dim) -> Tensor' },
  { schema: 'layer_norm(Tensor input, Tensor weight, Tensor bias, int axis, float eps) -> Tensor' },
  { schema: 'batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor mean, Tensor var, int axis, float eps) -> Tensor' },
  { schema: 'conv2d(Tensor input, Tensor weight, int[] strides, int[] padding, int[] dilation, int groups) -> Tensor' },
  { schema: 'pool2d(Tensor input, str pool_type, int[] kernel_size, int[] strides, int[] padding) -> Tensor' },
  { schema: 'embedding(Tensor weight, Tensor indices) -> Tensor' },
]);

const _ARG_OVERRIDES = new Map<string, readonly string[]>(TENSOR_OPS.filter(op => op.scalarArgs).map(op => [opNameFromSchema(op.schema), op.scalarArgs!]));
const _SCALAR_ARG_NAMES = new Map<string, readonly string[]>(TENSOR_OPS.map(op => [opNameFromSchema(op.schema), _ARG_OVERRIDES.get(opNameFromSchema(op.schema)) || scalarArgNamesFromSchema(op.schema)]));
const _METADATA = new Map<string, TensorOpMetadata>(TENSOR_OPS.map(op => {
  const name = opNameFromSchema(op.schema);
  return [name, Object.freeze({ ...op, name, returns: op.returns || returnsFromSchema(op.schema) })];
}));

export function opNameFromSchema(schema: string): string {
  return schema.slice(0, schema.indexOf('(')).split('.')[0].trim();
}

export function tensorOpSchemas(): string[] {
  return TENSOR_OPS.map(op => op.schema);
}

export function scalarArgNames(opName: string): readonly string[] | null {
  return _SCALAR_ARG_NAMES.get(opName) || null;
}

export function tensorOpMetadata(opName: string): TensorOpMetadata | null {
  return _METADATA.get(opName) || null;
}

export function returnSpec(opName: string): TensorOpReturnSpec | null {
  const meta = tensorOpMetadata(opName);
  return meta ? meta.returns : null;
}

function scalarArgNamesFromSchema(schema: string): string[] {
  const argsStart = schema.indexOf('(') + 1;
  const argsEnd = schema.lastIndexOf(')');
  const args = schema.slice(argsStart, argsEnd).trim();
  if (!args) return [];
  const names: string[] = [];
  for (const part of splitTopLevel(args)) {
    const trimmed = part.trim();
    const space = trimmed.lastIndexOf(' ');
    if (space < 0) continue;
    const kind = trimmed.slice(0, space).trim();
    if (kind === 'Tensor' || kind === 'Tensor[]') continue;
    const name = trimmed.slice(space + 1).trim().replace(/=.*/, '').trim();
    names.push(name);
  }
  return names;
}

function splitTopLevel(str: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (ch === '(' || ch === '[') depth++;
    else if (ch === ')' || ch === ']') depth--;
    else if (ch === ',' && depth === 0) {
      out.push(str.slice(start, i));
      start = i + 1;
    }
  }
  out.push(str.slice(start));
  return out;
}

function returnsFromSchema(schema: string): TensorOpReturnSpec {
  const marker = '->';
  const idx = schema.lastIndexOf(marker);
  const text = idx >= 0 ? schema.slice(idx + marker.length).trim() : 'Tensor';
  if (text === 'Tensor') return { kind: 'tensor', arity: 1 };
  if (text === 'Tensor[]') return { kind: 'tensor_list' };
  if (text.startsWith('(') && text.endsWith(')')) {
    return { kind: 'tuple', arity: splitTopLevel(text.slice(1, -1)).length };
  }
  return { kind: 'value', arity: 1 };
}
