import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import { applyAutocast } from '../../../../src/compiler/passes/precision/mixed_precision.js';
import { T } from '../../../_utils/ir_fixture.js';

const F = ScalarType.F32, F16 = ScalarType.F16;
const M = 4, K = 8, N = 4, P = 4;

function chain(name) {
  return buildFunction(name, [T([M, K]), T([K, N]), T([N, P])], [T([M, P])], (b, a) => {
    const c1 = b.matmul(a[0], a[1]).getResult(0);
    const r = b.relu(c1).getResult(0);
    b.returnOp([b.matmul(r, a[2]).getResult(0)]);
  });
}

function innerFunc(mod) {
  return mod.functions ? mod.functions().next().value : mod;
}

const A = new Float32Array(M * K).map((_, i) => Math.sin(i * 0.7) * 1.2);
const B = new Float32Array(K * N).map((_, i) => Math.cos(i * 0.4) * 1.1);
const E = new Float32Array(N * P).map((_, i) => Math.sin(i * 0.3) * 0.9);

function countConverts(func) {
  return [...func.ops()].filter(o => o.opName === 'convert').length;
}
function reluDtype(func) {
  const ops = [...func.ops()];
  const r = ops.find(o => o.opName === 'maximum' || o.opName === 'relu');
  return r ? r.getResult(0).type.dtype : null;
}

describe('mixed-precision FOLLOW propagation', () => {
  it('leaves FOLLOW ops in f32 by default (byte-identical to prior behavior)', () => {
    const func = innerFunc(chain('amp_default'));
    applyAutocast(func, { allow: new Set(['dot']) });
    expect(reluDtype(func)).toBe(F);
  });

  it('sinks the FOLLOW op into f16 and collapses the inter-matmul round-trip converts', () => {
    const base = innerFunc(chain('amp_base'));
    applyAutocast(base, { allow: new Set(['dot']) });
    const baseConverts = countConverts(base);

    const prop = innerFunc(chain('amp_prop'));
    applyAutocast(prop, { allow: new Set(['dot']), propagateFollow: true });

    expect(reluDtype(prop)).toBe(F16);
    expect(countConverts(prop)).toBeLessThan(baseConverts);
  });

  it('stays numerically close to the full-f32 reference after propagation', () => {
    const ref = new Float32Array(M * P);
    compileGraph(chain('ref'), CPUTarget(), {}).run('ref', A, B, E, ref);

    const mod = chain('prop');
    applyAutocast(innerFunc(mod), { allow: new Set(['dot']), propagateFollow: true });
    const out = new Float32Array(M * P);
    compileGraph(mod, CPUTarget(), {}).run('prop', A, B, E, out);

    let maxDiff = 0, scale = 0;
    for (let i = 0; i < M * P; i++) { maxDiff = Math.max(maxDiff, Math.abs(ref[i] - out[i])); scale = Math.max(scale, Math.abs(ref[i])); }
    expect(maxDiff).toBeGreaterThan(0);
    expect(maxDiff / (scale + 1e-9)).toBeLessThan(0.1);
  });
});
