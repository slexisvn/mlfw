import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CUDATarget, CPUTarget } from '../../../../src/backend/target.js';
import { Compiler } from '../../../../src/compiler/pipeline/compiler.js';
import { GraphModule } from '../../../../src/compiler/ir/graph/module.js';

const F32 = ScalarType.F32;
const F16 = ScalarType.F16;
const t = (shape, dtype = F32) => new TensorType(shape, dtype);

function halfMatmul(name, M, K, N, outDtype = F32) {
  const module = new GraphModule(name);
  module.addFunction(buildFunction(name, [t([M, K], F16), t([K, N], F16)], [t([M, N], outDtype)], (b, a) => {
    b.returnOp([b._inferAndBuild('dot', [a[0], a[1]], {
      lhs_contracting: [1], rhs_contracting: [0], out_dtype: outDtype,
    }).getResult(0)]);
  }));
  return module;
}

function compileCuda(module, opts = {}) {
  return new Compiler({ target: CUDATarget(), optimization: { tensorize: true }, fusion: { enabled: false }, ...opts })
    .compile(module);
}

describe('auto-tensorize reaches WMMA through the real graph path', () => {
  it('emits a WMMA kernel for an f16 x f16 matmul that accumulates in f32', () => {
    const src = compileCuda(halfMatmul('half_mm', 64, 64, 64)).getSource('half_mm');
    expect(src).toMatch(/fragment<accumulator, 16, 16, 16, float>/);
    expect(src).toMatch(/load_matrix_sync/);
    expect(src).toMatch(/mma_sync/);
  });

  it('leaves an f16-accumulating matmul alone, since WMMA needs an f32 accumulator', () => {
    expect(compileCuda(halfMatmul('half_acc', 64, 64, 64, F16)).getSource('half_acc'))
      .not.toMatch(/load_matrix_sync/);
  });

  it('does not tensorize when a dimension is not a multiple of the WMMA tile', () => {
    expect(compileCuda(halfMatmul('odd_mm', 64, 64, 40)).getSource('odd_mm'))
      .not.toMatch(/load_matrix_sync/);
  });

  it('promotes the f16 operands to the f32 accumulator on a backend without tensor cores', () => {
    const src = new Compiler({ target: CPUTarget(), fusion: { enabled: false } })
      .compile(halfMatmul('cpu_mm', 4, 4, 4)).getSource('cpu_mm');
    expect(src).toMatch(/__mlfw_f16_to_f32\(buf_1[^)]*\)[\s\S]*__mlfw_f16_to_f32\(buf_3/);
    expect(src).not.toMatch(/load_matrix_sync/);
  });
});
