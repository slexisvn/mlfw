import { describe, it, expect } from 'vitest';
import { detectWmmaMatmul, WMMA_INTRIN } from '../../../../src/compiler/passes/schedule/tensorize_pass.js';
import { getCudaIntrinSpec } from '../../../../src/backend/cuda/tensor_intrin.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import { BlockNode, PrimFunc, SeqNode } from '../../../../src/compiler/ir/tensor/nodes.js';
import { Layout } from '../../../../src/compiler/ir/graph/types.js';
import {
  axeAxisOfThreadTag,
  threadTagOfAxeAxis
} from '../../../../src/compiler/analysis/thread_binding.js';
import { AxeAxis } from '../../../../src/compiler/ir/layout/axe.js';

function buf(name, shape, dtype, strides = null) {
  return new Buffer(name, shape, dtype, 'global', strides);
}

function matmulFunc(a, b, c) {
  const region = x => ({ buffer: x });
  const block = new BlockNode('matmul', [], [region(a), region(b)], [region(c)], new SeqNode([]));
  const pf = new PrimFunc('main', [], block);
  for (const x of [a, b, c]) pf.bufferMap.set(x.name, x);
  return pf;
}

const M = 32, N = 48, K = 16;

describe('WMMA tensorize legality follows the registered intrinsic', () => {
  it('the intrinsic declares its tile shape and the layout it needs its operands in', () => {
    const spec = getCudaIntrinSpec(WMMA_INTRIN);
    expect(spec.shape).toEqual({ m: 16, n: 16, k: 16 });
    expect(spec.operandLayout.equals(Layout.rowMajor(2))).toBe(true);
  });

  it('accepts a half GEMM whose operands are laid out the way the intrinsic reads them', () => {
    const pf = matmulFunc(buf('A', [M, K], 'f16'), buf('B', [K, N], 'f16'), buf('C', [M, N], 'f32'));
    expect(detectWmmaMatmul(pf)).toEqual({ M, N, K, a: 'A', b: 'B', c: 'C' });
  });

  it('refuses an operand stored column-major, which the leading-dimension arithmetic would misread', () => {
    const colMajorA = buf('A', [M, K], 'f16', Layout.columnMajor(2).computeStrides([M, K]));
    const pf = matmulFunc(colMajorA, buf('B', [K, N], 'f16'), buf('C', [M, N], 'f32'));
    expect(detectWmmaMatmul(pf)).toBeNull();
  });

  it('refuses a padded operand whose rows are not tightly packed', () => {
    const paddedC = buf('C', [M, N], 'f32', [N + 4, 1]);
    const pf = matmulFunc(buf('A', [M, K], 'f16'), buf('B', [K, N], 'f16'), paddedC);
    expect(detectWmmaMatmul(pf)).toBeNull();
  });

  it('refuses a shape that does not fill the declared tile', () => {
    const pf = matmulFunc(buf('A', [M, 8], 'f16'), buf('B', [8, N], 'f16'), buf('C', [M, N], 'f32'));
    expect(detectWmmaMatmul(pf)).toBeNull();
  });

  it('refuses operands that are not half precision', () => {
    const pf = matmulFunc(buf('A', [M, K], 'f32'), buf('B', [K, N], 'f32'), buf('C', [M, N], 'f32'));
    expect(detectWmmaMatmul(pf)).toBeNull();
  });
});

describe('thread tags and layout axes are the same vocabulary', () => {
  it('maps every CUDA thread tag onto a layout axis', () => {
    expect(axeAxisOfThreadTag('threadIdx.x')).toBe(AxeAxis.THREAD_X);
    expect(axeAxisOfThreadTag('threadIdx.z')).toBe(AxeAxis.THREAD_Z);
    expect(axeAxisOfThreadTag('blockIdx.y')).toBe(AxeAxis.BLOCK_Y);
  });

  it('round-trips every tag the schedule is allowed to bind', () => {
    for (const tag of ['threadIdx.x', 'threadIdx.y', 'threadIdx.z', 'blockIdx.x', 'blockIdx.y', 'blockIdx.z']) {
      expect(threadTagOfAxeAxis(axeAxisOfThreadTag(tag))).toBe(tag);
    }
  });

  it('has no axis for a tag the schedule would reject', () => {
    expect(axeAxisOfThreadTag('warpIdx.x')).toBeNull();
    expect(axeAxisOfThreadTag('threadIdx.w')).toBeNull();
  });

  it('has no tag for an axis that exists only inside a kernel', () => {
    expect(threadTagOfAxeAxis(AxeAxis.LANE)).toBeNull();
    expect(threadTagOfAxeAxis(AxeAxis.REG)).toBeNull();
  });
});
