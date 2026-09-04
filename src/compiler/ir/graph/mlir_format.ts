import { COMPARE_DIRECTIONS } from '../../../util/dtype_map.js';
import { TensorType, DYNAMIC, shapeProduct, typeToString } from './types.js';
import { reduceIdentity } from './ops/reduction.js';
import type { AttrValue, IRType, ScalarDType } from './types.js';
import type { Operation } from './operation.js';
import type { Value } from './value.js';

const MLIR_DIALECT = 'tera';

export type MlirAttrKind = 'i64' | 'i64array' | 'i64pairs' | 'any';

export type MlirDerivedSource = 'resultType' | 'resultShape' | 'resultDtype';

export type MlirTypeForm =
  | 'resultList'
  | 'result'
  | 'functional'
  | 'operandToResult'
  | 'operandsToResult'
  | 'firstAndResult'
  | 'operandsOptional'
  | 'elements';

export type MlirAttrBinding = Readonly<{ ir: string; mlir: string; kind: MlirAttrKind }>;

export type MlirDerivedAttr = Readonly<{ ir: string; from: MlirDerivedSource }>;

export type MlirFixedAttr = Readonly<{ ir: string; value: AttrValue }>;

export type MlirGroupSize =
  | Readonly<{ kind: 'fixed'; count: number }>
  | Readonly<{ kind: 'attr'; attr: string }>
  | Readonly<{ kind: 'rest' }>;

export type MlirOperandGroup = Readonly<{
  keyword: string | null;
  size: MlirGroupSize;
  optional: boolean;
  types: boolean;
}>;

export type MlirRegionForm = Readonly<{
  open: string;
  separators: readonly string[];
  close: string;
  repeat: boolean;
  labelDepth: number;
}>;

export type MlirKeyword = Readonly<{
  ir: string;
  toMlir: ReadonlyMap<string, string>;
  toIr: ReadonlyMap<string, string>;
}>;

export type MlirOpForm = Readonly<{
  opName: string;
  mnemonic: string;
  operands: number;
  results: number;
  types: MlirTypeForm;
  keyword: MlirKeyword | null;
  attrs: readonly MlirAttrBinding[];
  derived: readonly MlirDerivedAttr[];
  fixed: readonly MlirFixedAttr[];
  seedOperand: number;
  combiner: boolean;
  groups: readonly MlirOperandGroup[] | null;
  regions: MlirRegionForm | null;
  attrByIr: ReadonlyMap<string, MlirAttrBinding>;
  attrByMlir: ReadonlyMap<string, MlirAttrBinding>;
  consumedAttrs: ReadonlySet<string>;
  sizesGroup: number;
}>;

type FormSpec = Readonly<{
  ops: readonly string[];
  mnemonic?: string;
  operands: number;
  results?: number;
  types: MlirTypeForm;
  keyword?: MlirKeyword;
  attrs?: readonly MlirAttrBinding[];
  derived?: readonly MlirDerivedAttr[];
  fixed?: readonly MlirFixedAttr[];
  seedOperand?: number;
  combiner?: boolean;
  groups?: readonly MlirOperandGroup[];
  regions?: MlirRegionForm;
}>;

const VARIADIC = -1;

export const SIZES_GROUP = 'sizes';

export const GENERIC_REGIONS: MlirRegionForm = {
  open: ' ({', separators: ['}, {'], close: '})', repeat: true, labelDepth: 1,
};

const BODY_REGIONS: MlirRegionForm = {
  open: ' {', separators: [], close: '}', repeat: false, labelDepth: 0,
};
const BRANCH_REGIONS: MlirRegionForm = {
  open: ' {', separators: ['} else {'], close: '}', repeat: false, labelDepth: 0,
};

function group(keyword: string | null, size: MlirGroupSize, optional = false,
               types = true): MlirOperandGroup {
  return { keyword, size, optional, types };
}

function attr(ir: string, kind: MlirAttrKind, mlir: string = ir): MlirAttrBinding {
  return { ir, mlir, kind };
}

function derived(ir: string, from: MlirDerivedSource): MlirDerivedAttr {
  return { ir, from };
}

function fixed(ir: string, value: AttrValue): MlirFixedAttr {
  return { ir, value };
}

function keyword(ir: string, values: Readonly<Record<string, string>>): MlirKeyword {
  const entries = Object.entries(values);
  return { ir, toMlir: new Map(entries), toIr: new Map(entries.map(([k, v]) => [v, k])) };
}

function sameNames(names: Iterable<string>): Readonly<Record<string, string>> {
  const out: Record<string, string> = {};
  for (const name of names) out[name] = name;
  return out;
}

const SPECS: readonly FormSpec[] = [
  { ops: ['add', 'sub', 'mul', 'div', 'maximum'], operands: 2, types: 'result' },
  { ops: ['neg', 'exp', 'sqrt', 'rsqrt', 'tanh', 'stop_gradient'], operands: 1, types: 'result' },
  {
    ops: ['constant'], operands: 0, types: 'elements',
    derived: [derived('tensor_type', 'resultType')]
  },
  {
    ops: ['iota'], operands: VARIADIC, types: 'result',
    groups: [group(SIZES_GROUP, { kind: 'rest' }, true, false)],
    attrs: [attr('iota_dimension', 'i64')],
    derived: [derived('tensor_type', 'resultType')]
  },
  {
    ops: ['dot'], operands: 2, types: 'functional',
    attrs: [
      attr('lhs_batch', 'i64array'), attr('lhs_contracting', 'i64array'),
      attr('rhs_batch', 'i64array'), attr('rhs_contracting', 'i64array')
    ]
  },
  {
    ops: ['reduce'], operands: 1, types: 'operandToResult', seedOperand: 1, combiner: true,
    keyword: keyword('reduce_type', { sum: 'sum', prod: 'product', max: 'maximum', min: 'minimum', mean: 'mean' }),
    attrs: [attr('dimensions', 'i64array')]
  },
  {
    ops: ['compare'], operands: 2, types: 'operandToResult',
    keyword: keyword('direction', sameNames(COMPARE_DIRECTIONS))
  },
  { ops: ['select'], operands: 3, types: 'firstAndResult' },
  {
    ops: ['convert'], operands: 1, types: 'operandToResult',
    derived: [derived('target_dtype', 'resultDtype')]
  },
  {
    ops: ['dim'], operands: 1, types: 'operandToResult',
    attrs: [attr('dimension', 'i64')]
  },
  {
    ops: ['broadcast_in_dim'], operands: VARIADIC, types: 'operandToResult',
    groups: [
      group(null, { kind: 'fixed', count: 1 }, false, false),
      group(SIZES_GROUP, { kind: 'rest' }, true, false)
    ],
    attrs: [attr('broadcast_dimensions', 'i64array')],
    derived: [derived('result_shape', 'resultShape')]
  },
  {
    ops: ['transpose'], operands: 1, types: 'operandToResult',
    attrs: [attr('permutation', 'i64array')]
  },
  {
    ops: ['reshape'], operands: VARIADIC, types: 'operandToResult',
    groups: [
      group(null, { kind: 'fixed', count: 1 }, false, false),
      group(SIZES_GROUP, { kind: 'rest' }, true, false)
    ],
    derived: [derived('new_shape', 'resultShape')]
  },
  {
    ops: ['slice'], operands: 1, types: 'operandToResult',
    attrs: [
      attr('starts', 'i64array', 'start_indices'),
      attr('limits', 'i64array', 'limit_indices'),
      attr('strides', 'i64array')
    ]
  },
  {
    ops: ['concat'], operands: VARIADIC, types: 'operandsToResult',
    attrs: [attr('dimension', 'i64')]
  },
  {
    ops: ['reverse'], operands: 1, types: 'operandToResult',
    attrs: [attr('dimensions', 'i64array')]
  },
  {
    ops: ['conv'], operands: 2, types: 'functional',
    attrs: [
      attr('strides', 'i64array'), attr('padding', 'i64pairs'),
      attr('dilation', 'i64array'), attr('groups', 'i64')
    ],
    fixed: [fixed('input_layout', 'NCHW'), fixed('kernel_layout', 'OIHW')]
  },
  {
    ops: ['pool2d'], operands: 1, types: 'operandToResult',
    keyword: keyword('pool_type', { max: 'max', avg: 'average' }),
    attrs: [
      attr('kernel_size', 'i64array'), attr('strides', 'i64array'),
      attr('padding', 'i64pairs'), attr('ceil_mode', 'any'),
      attr('count_include_pad', 'any')
    ],
    fixed: [fixed('layout', 'NCHW')]
  },
  {
    ops: ['pad'], operands: 2, types: 'functional',
    attrs: [
      attr('low', 'i64array'), attr('high', 'i64array'),
      attr('interior', 'i64array')
    ]
  },
  {
    ops: ['gather'], operands: 2, types: 'functional',
    attrs: [
      attr('offset_dims', 'i64array'), attr('collapsed_slice_dims', 'i64array'),
      attr('start_index_map', 'i64array'), attr('slice_sizes', 'i64array'),
      attr('index_vector_dim', 'i64')
    ]
  },
  {
    ops: ['scatter'], operands: 3, types: 'functional', combiner: true,
    attrs: [
      attr('update_window_dims', 'i64array'), attr('inserted_window_dims', 'i64array'),
      attr('scatter_dims_to_operand_dims', 'i64array'), attr('index_vector_dim', 'i64')
    ]
  },
  {
    ops: ['scan'], operands: VARIADIC, results: VARIADIC, types: 'resultList',
    groups: [
      group('init', { kind: 'attr', attr: 'num_carry' }),
      group('xs', { kind: 'attr', attr: 'num_xs' }),
      group('consts', { kind: 'attr', attr: 'num_consts' }, true),
      group(SIZES_GROUP, { kind: 'rest' }, true, false)
    ],
    regions: BODY_REGIONS
  },
  { ops: ['if'], operands: VARIADIC, results: VARIADIC, types: 'functional', regions: BRANCH_REGIONS },
  { ops: ['yield'], operands: VARIADIC, results: 0, types: 'operandsOptional' },
  { ops: ['return'], mnemonic: 'return', operands: VARIADIC, results: 0, types: 'operandsOptional' },
];

const FORM_BY_OP = new Map<string, MlirOpForm>();
const FORM_BY_MNEMONIC = new Map<string, MlirOpForm>();

const DENSE_PAYLOAD_ATTR = 'value';

function consumedAttrsOf(spec: FormSpec): Set<string> {
  const consumed = new Set<string>();
  if (spec.keyword) consumed.add(spec.keyword.ir);
  for (const entry of spec.derived ?? []) consumed.add(entry.ir);
  for (const entry of spec.fixed ?? []) consumed.add(entry.ir);
  for (const group of spec.groups ?? []) {
    if (group.size.kind === 'attr') consumed.add(group.size.attr);
  }
  if (spec.types === 'elements') consumed.add(DENSE_PAYLOAD_ATTR);
  return consumed;
}

for (const spec of SPECS) {
  const attrs = spec.attrs ?? [];
  const groups = spec.groups ?? null;
  for (const opName of spec.ops) {
    const form: MlirOpForm = {
      opName,
      mnemonic: spec.mnemonic ?? `${MLIR_DIALECT}.${opName}`,
      operands: spec.operands,
      results: spec.results ?? 1,
      types: spec.types,
      keyword: spec.keyword ?? null,
      attrs,
      derived: spec.derived ?? [],
      fixed: spec.fixed ?? [],
      seedOperand: spec.seedOperand ?? VARIADIC,
      combiner: spec.combiner ?? false,
      groups,
      regions: spec.regions ?? null,
      attrByIr: new Map(attrs.map((binding) => [binding.ir, binding])),
      attrByMlir: new Map(attrs.map((binding) => [binding.mlir, binding])),
      consumedAttrs: consumedAttrsOf(spec),
      sizesGroup: groups ? groups.findIndex((group) => group.keyword === SIZES_GROUP) : -1,
    };
    if (form.sizesGroup >= 0
        && (form.sizesGroup !== (groups as readonly MlirOperandGroup[]).length - 1
            || (groups as readonly MlirOperandGroup[])[form.sizesGroup].size.kind !== 'rest')) {
      throw new Error(`'${form.mnemonic}' declares a '${SIZES_GROUP}' clause that is not its trailing variadic one`);
    }
    FORM_BY_OP.set(form.opName, form);
    FORM_BY_MNEMONIC.set(form.mnemonic, form);
  }
}

export function mlirFormOfMnemonic(mnemonic: string): MlirOpForm | undefined {
  return FORM_BY_MNEMONIC.get(mnemonic);
}

export function mlirFormOfOp(opName: string): MlirOpForm | undefined {
  return FORM_BY_OP.get(opName);
}

const QUALIFIER = `${MLIR_DIALECT}.`;

export function qualify(opName: string): string {
  return QUALIFIER + opName;
}

export function unqualify(mnemonic: string): string {
  return mnemonic.startsWith(QUALIFIER) ? mnemonic.slice(QUALIFIER.length) : mnemonic;
}

export function isNumberArray(value: AttrValue | undefined): value is readonly number[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === 'number');
}

export function isNumberPairs(value: AttrValue | undefined): value is readonly (readonly number[])[] {
  return Array.isArray(value)
    && value.every((entry) => isNumberArray(entry as AttrValue) && (entry as readonly number[]).length === 2);
}

export function flattenPairs(pairs: readonly (readonly number[])[]): number[] {
  const flat: number[] = [];
  for (const pair of pairs) flat.push(pair[0], pair[1]);
  return flat;
}

export function nestPairs(flat: readonly number[]): number[][] {
  const pairs: number[][] = [];
  for (let i = 0; i + 1 < flat.length; i += 2) pairs.push([flat[i], flat[i + 1]]);
  return pairs;
}

function staticNumel(type: TensorType): number {
  return shapeProduct(type.shape, -1);
}

function densePayloadFits(op: Operation): boolean {
  const type = op.getResult(0).type;
  if (!(type instanceof TensorType)) return false;
  const numel = staticNumel(type);
  if (numel < 0) return false;
  const value = op.getAttr('value');
  if (typeof value === 'number') return true;
  if (!ArrayBuffer.isView(value) || value instanceof DataView) return false;
  return type.shape.length > 0 && (value as ArrayBufferView & { length: number }).length === numel;
}

type ImpliedTypes = Readonly<{ from: 'result' | 'operand'; start: number }>;

const IMPLIED_OPERAND_TYPES: Readonly<Partial<Record<MlirTypeForm, ImpliedTypes>>> = {
  result: { from: 'result', start: 0 },
  firstAndResult: { from: 'result', start: 1 },
  operandToResult: { from: 'operand', start: 1 },
};

function carriesEveryOperandType(op: Operation, form: MlirOpForm): boolean {
  if (form.groups) return true;
  const implied = IMPLIED_OPERAND_TYPES[form.types];
  if (!implied) return true;
  const reference = typeToString(implied.from === 'result' ? op.getResult(0).type : op.getOperand(0).type);
  for (let i = implied.start; i < op.numOperands; i++) {
    if (i !== form.seedOperand && typeToString(op.getOperand(i).type) !== reference) return false;
  }
  return true;
}

function seedElementType(op: Operation): ScalarDType | null {
  const operand = op.getOperand(0).type;
  return operand instanceof TensorType ? operand.dtype : null;
}

function isScalarOf(type: IRType, dtype: ScalarDType | null): boolean {
  return type instanceof TensorType && type.shape.length === 0 && type.dtype === dtype;
}

function isElidableSeed(seed: Value, identity: number, dtype: ScalarDType | null): boolean {
  const producer = seed.definingOp;
  if (!producer || producer.opName !== 'constant' || !isScalarOf(seed.type, dtype)) return false;
  const value = producer.getAttr('value');
  return typeof value === 'number' && Object.is(value, identity);
}

function isElidableCombiner(op: Operation, dtype: ScalarDType | null): boolean {
  if (op.regions.length !== 1) return false;
  const blocks = op.getRegion(0).blocks;
  if (blocks.length !== 1 || blocks[0].opsArray().length !== 0) return false;
  const args = blocks[0].arguments;
  return args.length === 2 && args.every((arg) => isScalarOf(arg.type, dtype));
}

export function combinerScalarType(operandType: IRType): TensorType | null {
  return operandType instanceof TensorType ? new TensorType([], operandType.dtype) : null;
}

export function customFormOf(op: Operation): MlirOpForm | null {
  const form = FORM_BY_OP.get(op.opName);
  if (!form) return null;
  const seeded = form.seedOperand >= 0;
  const expectedOperands = form.operands < 0 ? op.numOperands : form.operands + (seeded ? 1 : 0);
  if (op.numOperands !== expectedOperands) return null;
  if (form.results >= 0 && op.numResults !== form.results) return null;
  if (!carriesEveryOperandType(op, form)) return null;

  let irKind: AttrValue | undefined;
  if (form.keyword) {
    irKind = op.getAttr(form.keyword.ir);
    if (typeof irKind !== 'string' || !form.keyword.toMlir.has(irKind)) return null;
  }
  for (const binding of form.attrs) {
    if (!op.hasAttr(binding.ir)) continue;
    const value = op.getAttr(binding.ir);
    if (binding.kind === 'i64' && typeof value !== 'number') return null;
    if (binding.kind === 'i64array' && !isNumberArray(value)) return null;
    if (binding.kind === 'i64pairs' && !isNumberPairs(value)) return null;
  }
  for (const entry of form.fixed) {
    if (op.getAttr(entry.ir) !== entry.value) return null;
  }
  if (form.combiner) {
    const dtype = seedElementType(op);
    if (!isElidableCombiner(op, dtype)) return null;
    if (seeded) {
      const identity = reduceIdentity(irKind as string);
      if (identity === undefined) return null;
      if (!isElidableSeed(op.getOperand(form.seedOperand), identity, dtype)) return null;
    }
  } else if (form.regions) {
    if (op.regions.length !== form.regions.separators.length + 1) return null;
    if (op.regions.some((region) => region.blocks.length !== 1)) return null;
  } else if (op.regions.length !== 0) {
    return null;
  }
  if (form.groups && groupSpans(op, form.groups) === null) return null;
  if (form.types === 'elements' && !densePayloadFits(op)) return null;
  return form;
}

export function groupSpans(op: Operation, groups: readonly MlirOperandGroup[]): number[] | null {
  const bounds: number[] = [0];
  let at = 0;
  for (let i = 0; i < groups.length; i++) {
    const size = groups[i].size;
    if (size.kind === 'rest') {
      if (i !== groups.length - 1) return null;
      at = op.numOperands;
    } else if (size.kind === 'fixed') {
      at += size.count;
    } else {
      const count = op.getAttr(size.attr);
      if (typeof count !== 'number' || !Number.isInteger(count) || count < 0) return null;
      at += count;
    }
    if (at > op.numOperands) return null;
    bounds.push(at);
  }
  return bounds[bounds.length - 1] === op.numOperands ? bounds : null;
}

export function sizesOperandSpan(op: Operation): { start: number; count: number } | null {
  const form = FORM_BY_OP.get(op.opName);
  if (!form || !form.groups || form.sizesGroup < 0) return null;
  const spans = groupSpans(op, form.groups);
  if (!spans) return null;
  const index = form.sizesGroup;
  return { start: spans[index], count: spans[index + 1] - spans[index] };
}

export function sizesClauseErrors(op: Operation): string[] {
  const form = FORM_BY_OP.get(op.opName);
  if (!form || !form.groups || form.sizesGroup < 0) return [];
  const spans = groupSpans(op, form.groups);
  if (!spans) {
    return [`${op.opName} declares operand clauses that do not fit its ${op.numOperands} operands`];
  }
  const given = spans[form.sizesGroup + 1] - spans[form.sizesGroup];
  const wanted = dynamicResultExtents(op).length;
  if (given === 0 || given === wanted) return [];
  return [`${op.opName} expects one size per dynamic result extent: ${wanted} expected, ${given} given`];
}

export function dynamicResultExtents(op: Operation): { result: number; axis: number }[] {
  const extents: { result: number; axis: number }[] = [];
  for (let result = 0; result < op.numResults; result++) {
    const type = op.getResult(result).type;
    if (!(type instanceof TensorType)) continue;
    for (let axis = 0; axis < type.shape.length; axis++) {
      if (type.shape[axis] === DYNAMIC) extents.push({ result, axis });
    }
  }
  return extents;
}

export function derivedAttrValue(source: MlirDerivedSource, resultType: IRType): AttrValue | undefined {
  if (source === 'resultType') return resultType as unknown as AttrValue;
  if (!(resultType instanceof TensorType)) return undefined;
  if (source === 'resultDtype') return resultType.dtype;
  return resultType.shape as unknown as AttrValue;
}

const FLOAT_LAYOUT: Readonly<Record<string, readonly [number, number]>> = {
  f16: [16, 10], bf16: [16, 7], f32: [32, 23], f64: [64, 52]
};

const DEFAULT_FLOAT: ScalarDType = 'f64';

function floatLayout(dtype: ScalarDType): readonly [number, number] {
  return FLOAT_LAYOUT[dtype] ?? FLOAT_LAYOUT[DEFAULT_FLOAT];
}

function nonFiniteBits(value: number, dtype: ScalarDType): bigint {
  const [total, mantissa] = floatLayout(dtype);
  const exponent = BigInt((1 << (total - 1 - mantissa)) - 1) << BigInt(mantissa);
  const magnitude = Number.isNaN(value) ? exponent | (1n << BigInt(mantissa - 1)) : exponent;
  return value < 0 ? magnitude | (1n << BigInt(total - 1)) : magnitude;
}

function decodeFloatBits(bits: bigint, dtype: ScalarDType): number {
  const [total, mantissa] = floatLayout(dtype);
  const exponentBits = total - 1 - mantissa;
  const sign = (bits >> BigInt(total - 1)) & 1n ? -1 : 1;
  const exponent = Number((bits >> BigInt(mantissa)) & ((1n << BigInt(exponentBits)) - 1n));
  const fraction = bits & ((1n << BigInt(mantissa)) - 1n);
  const bias = (1 << (exponentBits - 1)) - 1;
  if (exponent === (1 << exponentBits) - 1) return fraction === 0n ? sign * Infinity : NaN;
  if (exponent === 0) return sign * Number(fraction) * Math.pow(2, 1 - bias - mantissa);
  return sign * (1 + Number(fraction) / Math.pow(2, mantissa)) * Math.pow(2, exponent - bias);
}

export function formatFloatLiteral(value: number, dtype: ScalarDType): string {
  if (!Number.isFinite(value)) {
    const [total] = floatLayout(dtype);
    return `0x${nonFiniteBits(value, dtype).toString(16).toUpperCase().padStart(total / 4, '0')}`;
  }
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

export function formatNonFiniteAttr(value: number): string {
  return `${formatFloatLiteral(value, DEFAULT_FLOAT)} : ${DEFAULT_FLOAT}`;
}

export function parseFloatLiteral(text: string, dtype: ScalarDType = DEFAULT_FLOAT): number {
  const trimmed = text.trim();
  if (trimmed.startsWith('0x') || trimmed.startsWith('0X')) {
    return decodeFloatBits(BigInt(trimmed), dtype);
  }
  return Number(trimmed);
}

export function seedConstantAttrs(irKind: string, operandType: IRType): {
  attrs: Map<string, AttrValue>;
  type: TensorType;
} | null {
  const type = combinerScalarType(operandType);
  if (!type) return null;
  const identity = reduceIdentity(irKind);
  if (identity === undefined) return null;
  return { attrs: new Map<string, AttrValue>([['value', identity], ['tensor_type', type]]), type };
}
