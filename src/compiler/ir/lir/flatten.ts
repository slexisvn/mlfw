import { MathOpNode, IntImmNode, VariableNode } from '../tensor/nodes.js';
import { SymInt, symVarName } from '../sym_int.js';
import { symIntToNode } from '../tensor/sym_lower.js';
import type { TirNode } from '../tensor/nodes.js';
import type { Buffer } from '../tensor/buffer.js';

export type ShapeParamMap = ReadonlyMap<string, { name: string }> | null | undefined;

export function flattenIndex(buffer: Buffer, indices: readonly TirNode[], shapeParamMap?: ShapeParamMap): TirNode {
  const baseOffset = typeof buffer.offset === 'number' ? buffer.offset : 0;

  if (indices.length === 0) return new IntImmNode(baseOffset);
  if (indices.length === 1) {
    if (baseOffset === 0) return indices[0];
    return new MathOpNode('+', indices[0], new IntImmNode(baseOffset));
  }

  const terms: TirNode[] = [];
  if (baseOffset !== 0) terms.push(new IntImmNode(baseOffset));
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    if (idx.type === 'IntImmNode' && (idx as IntImmNode).value === 0) continue;

    const stride = buffer.strides[i];
    if (stride === 1) {
      terms.push(idx);
    } else if (typeof stride === 'number' && stride >= 0) {
      terms.push(new MathOpNode('*', idx, new IntImmNode(stride as number)));
    } else {
      const dynStride = computeDynamicStride(buffer, i, shapeParamMap);
      terms.push(new MathOpNode('*', idx, dynStride));
    }
  }

  if (terms.length === 0) return new IntImmNode(0);
  return terms.reduce((acc, term) => new MathOpNode('+', acc, term));
}

export function computeDynamicStride(buffer: Buffer, dimIdx: number, shapeParamMap?: ShapeParamMap): TirNode {
  const factors: TirNode[] = [];
  for (let j = dimIdx + 1; j < buffer.shape.length; j++) {
    const d = buffer.shape[j];
    if (typeof d === 'number' && d >= 0) {
      factors.push(new IntImmNode(d as number));
    } else {
      factors.push(resolveShapeParam(buffer, j, shapeParamMap));
    }
  }
  if (factors.length === 0) return new IntImmNode(1);
  return factors.reduce((acc, f) => new MathOpNode('*', acc, f));
}

export function resolveShapeParam(buffer: Buffer, dimIdx: number, shapeParamMap?: ShapeParamMap): TirNode {
  const d = buffer.shape[dimIdx];
  if (d instanceof SymInt) {
    return symIntToNode(d, (name) => new VariableNode(symVarName(name), 'index'));
  }
  if (shapeParamMap) {
    const key = `${buffer.name}:${dimIdx}`;
    const v = shapeParamMap.get(key);
    if (v) return new VariableNode(v.name, 'index');
  }
  return new IntImmNode(1);
}

export function computeNumelExpr(buffer: Buffer, shapeParamMap?: ShapeParamMap): TirNode {
  if (buffer.shape.length === 0) return new IntImmNode(1);

  const factors: TirNode[] = [];
  for (let d = 0; d < buffer.shape.length; d++) {
    const dim = buffer.shape[d];
    if (typeof dim === 'number' && dim >= 0) {
      factors.push(new IntImmNode(dim as number));
    } else {
      factors.push(resolveShapeParam(buffer, d, shapeParamMap));
    }
  }
  return factors.reduce((acc, f) => new MathOpNode('*', acc, f));
}
