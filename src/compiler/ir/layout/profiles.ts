import { MemoryScope } from '../tensor/tensor_types.js';
import { AxeAxis } from './axe.js';
import type { SymExpr } from '../sym_int.js';
import type { AxeAxisName, AxeLayout, Iter } from './axe.js';

const THREAD_SPACE_AXES: readonly AxeAxisName[] = [
  AxeAxis.LANE,
  AxeAxis.WARP,
  AxeAxis.REG,
  AxeAxis.THREAD_X,
  AxeAxis.THREAD_Y,
  AxeAxis.THREAD_Z
];

const SCOPE_AXES: Readonly<Record<string, readonly AxeAxisName[]>> = Object.freeze({
  [MemoryScope.GLOBAL]: [AxeAxis.MEM],
  [MemoryScope.SHARED]: [AxeAxis.MEM, AxeAxis.LANE, AxeAxis.WARP, AxeAxis.THREAD_X, AxeAxis.THREAD_Y, AxeAxis.THREAD_Z],
  [MemoryScope.LOCAL]: [AxeAxis.MEM, ...THREAD_SPACE_AXES],
  [MemoryScope.REGISTER]: [AxeAxis.MEM, ...THREAD_SPACE_AXES]
});

function isPositiveConst(x: SymExpr): boolean {
  return typeof x === 'number' && Number.isInteger(x) && x > 0;
}

function allIters(layout: AxeLayout): readonly Iter[] {
  return layout.shard.concat(layout.replica);
}

export function validateGraphProfile(layout: AxeLayout, shape: readonly SymExpr[] | null = null): string[] {
  const errors: string[] = [];
  if (layout.replica.length > 0) {
    errors.push('a graph-level layout may not replicate: nothing below the graph IR consumes a set-valued layout yet');
  }
  for (const it of allIters(layout)) {
    if (it.axis !== AxeAxis.MEM) {
      errors.push(`a graph-level layout may only use the '${AxeAxis.MEM}' axis, found '${it.axis}'`);
    }
    if (typeof it.extent === 'number' && !isPositiveConst(it.extent)) {
      errors.push(`iter extents must be positive, found ${it.extent}`);
    }
  }
  if (shape && shape.every(dim => typeof dim === 'number')) {
    if (!layout.group(shape as readonly number[])) {
      errors.push(`the layout does not group under shape [${shape.join(', ')}]`);
    }
  }
  return errors;
}

export function validateThreadProfile(layout: AxeLayout, scope: string = MemoryScope.LOCAL): string[] {
  const errors: string[] = [];
  const allowed = SCOPE_AXES[scope];
  if (!allowed) {
    errors.push(`unknown memory scope '${scope}'`);
    return errors;
  }
  for (const it of allIters(layout)) {
    if (!allowed.includes(it.axis)) {
      errors.push(`a '${scope}' buffer may not be laid out over the '${it.axis}' axis`);
    }
    if (!isPositiveConst(it.extent)) {
      errors.push(`a thread-level layout needs constant positive extents, found '${it.extent}'`);
    }
    if (typeof it.stride !== 'number' || !Number.isInteger(it.stride) || it.stride === 0) {
      errors.push(`a thread-level layout needs constant non-zero strides, found '${it.stride}'`);
    }
  }
  for (const [axis, value] of layout.offset) {
    if (!allowed.includes(axis)) {
      errors.push(`a '${scope}' buffer may not carry an offset on the '${axis}' axis`);
    }
    if (typeof value !== 'number' || !Number.isInteger(value)) {
      errors.push(`a thread-level offset must be a constant, found '${value}' on '${axis}'`);
    }
  }
  return errors;
}
