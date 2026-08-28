import { describe, it, expect } from 'vitest';
import { dispatcher } from '../../src/dispatcher/dispatcher.js';
import { Library } from '../../src/dispatcher/library.js';
import { DispatchKey, DispatchKeySet, deviceForBackendKey } from '../../src/dispatcher/dispatch_key.js';
import { BACKEND_COVERAGE_KEYS } from '../../src/dispatcher/backend_coverage.js';
import { zeros, tensor, matmul } from '../../src/index.js';
import { qr } from '../../src/numeric/index.js';
import { LINALG_SCHEMAS } from '../../src/kernels/defs/linalg_defs.js';
import { ML_SCHEMAS } from '../../src/kernels/defs/ml_defs.js';
import { NUMERIC_SCHEMAS } from '../../src/kernels/defs/numeric_defs.js';

const SHIPPED_OPS = dispatcher.listOps();

const HOST_OP_NAMES = [...LINALG_SCHEMAS, ...ML_SCHEMAS, ...NUMERIC_SCHEMAS]
  .map(schema => schema.slice(0, schema.indexOf('(')))
  .sort();

const devicesOf = name => [...dispatcher.findOp(name).entry.devices].sort().join(',');

function keysWithoutKernel(handle) {
  return BACKEND_COVERAGE_KEYS
    .filter(key => !handle.entry.hasKernel(key))
    .map(key => deviceForBackendKey(key));
}

function dispatchError(name, key, ...args) {
  try {
    dispatcher.dispatch(dispatcher.findOp(name), DispatchKeySet.fromKeys(key), ...args);
    return null;
  } catch (e) {
    return e.message;
  }
}

function define(schema, devices) {
  new Library('mlc', 'DEF').def(schema, devices);
  const handle = dispatcher.findOp(schema.slice(0, schema.indexOf('(')));
  expect(handle).not.toBeNull();
  return handle;
}

describe('backend coverage of the shipped operator set', () => {
  it('leaves no operator uncovered without declaring why', () => {
    const silent = [];
    for (const opKey of SHIPPED_OPS) {
      const handle = dispatcher.findOp(opKey);
      const missing = keysWithoutKernel(handle);
      if (missing.length > 0 && handle.entry.devices === null) {
        silent.push(`${handle.name}: ${missing.join(', ')}`);
      }
    }
    expect(silent).toEqual([]);
  });

  it('gives every operator without a device declaration all four backend keys', () => {
    const gaps = [];
    for (const opKey of SHIPPED_OPS) {
      const handle = dispatcher.findOp(opKey);
      if (handle.entry.devices !== null) continue;
      const missing = keysWithoutKernel(handle);
      if (missing.length > 0) gaps.push(`${handle.name}: ${missing.join(', ')}`);
    }
    expect(gaps).toEqual([]);
  });

  it('declares exactly the operators defined as host algorithms', () => {
    const declared = [];
    for (const opKey of SHIPPED_OPS) {
      const handle = dispatcher.findOp(opKey);
      if (handle.entry.devices) declared.push(handle.name);
    }
    expect(declared.sort()).toEqual(HOST_OP_NAMES);
  });

  it('declares each host algorithm with the devices that implement it', () => {
    expect(devicesOf('qr')).toBe('cpu,wasm');
    expect(devicesOf('fft')).toBe('cpu,wasm');
    expect(devicesOf('ifft')).toBe('cpu,wasm');
    expect(devicesOf('svd')).toBe('cpu,gpu,wasm');
    expect(devicesOf('lstsq')).toBe('cpu,gpu,wasm');
    expect(devicesOf('kmeans')).toBe('cpu,wasm');
  });

  it('names the devices an operator runs on instead of a dispatch key number', () => {
    const message = dispatchError('qr', DispatchKey.GPU, zeros([4, 4]));
    expect(message).toContain("Op 'qr' has no gpu implementation");
    expect(message).toContain('it runs on cpu, wasm only');
    expect(message).not.toContain('No kernel registered');
  });

  it('names the runtime to load when a declared device has no kernels yet', () => {
    const message = dispatchError('svd', DispatchKey.GPU, zeros([4, 4]));
    expect(message).toContain("Op 'svd' runs on gpu");
    expect(message).toContain('load the gpu runtime');
    expect(message).not.toContain('No kernel registered');
  });

  it('still runs a declared operator through its public entry point', () => {
    const rows = [[12, -51], [6, 167], [-4, 24]];
    const { Q, R } = qr(tensor(rows));

    expect(Q.shape).toEqual([3, 2]);
    expect(R.shape).toEqual([2, 2]);
    expect(R.toArray()[1][0]).toBeCloseTo(0, 10);

    const reconstructed = matmul(Q, R).toArray();
    for (let i = 0; i < rows.length; i++) {
      for (let j = 0; j < rows[i].length; j++) {
        expect(reconstructed[i][j]).toBeCloseTo(rows[i][j], 4);
      }
    }

    const gram = matmul(Q.transpose(0, 1), Q).toArray();
    expect(gram[0][0]).toBeCloseTo(1, 6);
    expect(gram[1][1]).toBeCloseTo(1, 6);
    expect(gram[0][1]).toBeCloseTo(0, 6);
  });
});

describe('backend coverage of an operator defined after start-up', () => {
  it('generates working kernels for one the compiler can lower', () => {
    const handle = define('square(Tensor self) -> Tensor');
    expect(handle.entry.devices).toBeNull();
    expect(keysWithoutKernel(handle)).toEqual([]);

    const out = dispatcher.dispatch(handle, DispatchKeySet.fromKeys(DispatchKey.CPU), tensor([[1, 2], [3, 4]]));
    expect(out.shape).toEqual([2, 2]);
    expect([...out.data]).toEqual([1, 4, 9, 16]);
  });

  it('generates nothing for one the compiler cannot lower, and says so on the first call', () => {
    const handle = define('coverage_probe_unlowerable(Tensor self) -> Tensor');
    expect(keysWithoutKernel(handle)).toEqual(['cpu', 'gpu', 'wasm', 'webgpu']);

    const message = dispatchError('coverage_probe_unlowerable', DispatchKey.CPU, zeros([2, 2]));
    expect(message).toContain("Op 'coverage_probe_unlowerable' has no cpu kernel");
    expect(message).toContain('no graph-IR lowering');
    expect(message).not.toContain('No kernel registered');
    expect(message).not.toContain('Cannot infer result types');
  });

  it('generates nothing for a declared one, and names the devices it does run on', () => {
    const handle = define('coverage_probe_host(Tensor self) -> Tensor', ['cpu']);
    expect(keysWithoutKernel(handle)).toEqual(['cpu', 'gpu', 'wasm', 'webgpu']);

    const message = dispatchError('coverage_probe_host', DispatchKey.WASM, zeros([2, 2]));
    expect(message).toContain("Op 'coverage_probe_host' has no wasm implementation");
    expect(message).toContain('it runs on cpu only');
  });
});
