import { beforeEach, afterEach, describe, it, expect, vi } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { dirname, join } from 'path';
import { SEARCH_ENV } from '../../src/runtime/node/search_path.js';

const native = vi.hoisted(() => ({
  load: vi.fn(), compile: vi.fn(), invoke: vi.fn(), release: vi.fn(), lastError: vi.fn(),
}));
vi.mock('koffi', () => ({ default: { load: native.load } }));

const ext = process.platform === 'win32' ? '.dll' : process.platform === 'darwin' ? '.dylib' : '.so';
const handle = { id: 'compiled' };
let api;
let root;
let location;

function fixture(path, contents = '') {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, contents);
  return path;
}

function ffiLibrary(binding) {
  return { func(signature) {
    if (signature.includes('teraCompileFor(')) return binding.compile;
    if (signature.includes('teraInvoke(')) return binding.invoke;
    if (signature.includes('teraRelease(')) return binding.release;
    if (signature.includes('teraLastError(')) return binding.lastError;
    throw new Error('unexpected FFI signature: ' + signature);
  } };
}

beforeEach(async () => {
  vi.resetModules();
  vi.resetAllMocks();
  root = mkdtempSync(join(tmpdir(), 'mlfw-terac-test-'));
  location = { library: join(root, 'build/bin/tera-capi' + ext), llvmBin: join(root, 'llvm/bin') };
  fixture(location.library);
  fixture(join(location.llvmBin, 'mlir_c_runner_utils' + ext));
  fixture(join(location.llvmBin, 'mlir_cuda_runtime' + ext));
  for (const name of ['TERAC_LIBRARY', 'TERAC_LLVM_BIN', 'TERAC_BUILD']) vi.stubEnv(name, '');
  vi.stubEnv(SEARCH_ENV, process.env[SEARCH_ENV]);
  native.compile.mockReturnValue(handle);
  native.invoke.mockReturnValue(0);
  native.lastError.mockReturnValue('');
  native.load.mockReturnValue(ffiLibrary(native));
  api = await import('../../src/runtime/node/terac.js');
});

afterEach(() => {
  vi.unstubAllEnvs();
  rmSync(root, { recursive: true, force: true });
});

describe('native Terac library resolution and FFI contract', () => {
  it('invokes and releases each module through the library that compiled it', () => {
    const first = api.teracCompile('module {}', 'cpu', 3, location);
    const otherHandle = { id: 'second-library-module' };
    const other = {
      compile: vi.fn().mockReturnValue(otherHandle),
      invoke: vi.fn().mockReturnValue(0),
      release: vi.fn(),
      lastError: vi.fn().mockReturnValue('second-library error'),
    };
    const otherLibrary = fixture(join(root, 'other/tera-capi' + ext));
    native.load.mockReturnValueOnce(ffiLibrary(other));
    const second = api.teracCompile('module {}', 'cpu', 3, { ...location, library: otherLibrary });
    api.teracInvoke(first, 'main', [], [], [], []);
    expect(native.invoke).toHaveBeenCalledExactlyOnceWith(handle, 'main', [], [], 0, [], [], 0);
    expect(other.invoke).not.toHaveBeenCalled();
    api.teracInvoke(second, 'main', [], [], [], []);
    expect(other.invoke).toHaveBeenCalledExactlyOnceWith(otherHandle, 'main', [], [], 0, [], [], 0);
    native.invoke.mockReturnValue(1);
    native.lastError.mockReturnValue('first-library error');
    expect(() => api.teracInvoke(first, 'main', [], [], [], [])).toThrow('first-library error');
    api.teracRelease(first);
    api.teracRelease(second);
    expect(native.release).toHaveBeenCalledExactlyOnceWith(handle);
    expect(other.release).toHaveBeenCalledExactlyOnceWith(otherHandle);
  });

  it.each([
    ['cpu', ['mlir_c_runner_utils']],
    ['cuda', ['mlir_c_runner_utils', 'mlir_cuda_runtime']],
  ])('names %s as the terac target and loads the runtime paths it requires', (device, stems) => {
    api.teracCompile('module {}', device, 2, location);
    const libraries = stems.map((stem) => join(location.llvmBin, stem + ext));
    expect(native.compile).toHaveBeenCalledWith('module {}', device, '', 2, libraries, libraries.length);
  });

  it('passes the target its own options', () => {
    api.teracCompile('module {}', 'cuda', 3, location, 'chip=sm_90');
    expect(native.compile).toHaveBeenCalledWith(
      'module {}', 'cuda', 'chip=sm_90', 3, expect.anything(), 2,
    );
  });

  it('respects explicit library precedence, reuses a loaded binding, and reloads another library', () => {
    const envLibrary = fixture(join(root, 'override/tera-capi' + ext));
    vi.stubEnv('TERAC_LIBRARY', envLibrary);
    api.teracCompile('module {}', 'cpu', 3, location);
    api.teracCompile('module {}', 'cpu', 3, location);
    expect(native.load).toHaveBeenCalledExactlyOnceWith(location.library);
    api.teracCompile('module {}', 'cpu', 3, { llvmBin: location.llvmBin });
    expect(native.load).toHaveBeenCalledTimes(2);
    expect(native.load).toHaveBeenLastCalledWith(envLibrary);
  });

  it('finds a prefixed compiler library and runtime in lib when bin candidates are absent', () => {
    const build = join(root, 'lib-only');
    const library = fixture(join(build, 'lib/libtera-capi' + ext));
    const llvmBin = join(root, 'other-llvm/bin');
    const runtime = fixture(join(root, 'other-llvm/lib/libmlir_c_runner_utils' + ext));
    api.teracCompile('module {}', 'cpu', 3, { build, llvmBin });
    expect(native.load).toHaveBeenCalledWith(library);
    expect(native.compile).toHaveBeenCalledWith('module {}', 'cpu', '', 3, [runtime], 1);
  });

  it('discovers LLVM from the build config and gives an explicit location precedence', () => {
    const build = join(root, 'build');
    fixture(join(build, 'test/lit.site.cfg.py'),
      'config.llvm_tools_dir = lit_config.substitute("' + location.llvmBin.replaceAll('\\', '/') + '")');
    expect(api.teracRuntimeLibs('cpu', { build })).toEqual([join(location.llvmBin, 'mlir_c_runner_utils' + ext)]);
    const override = join(root, 'override-llvm/bin');
    const runtime = fixture(join(override, 'mlir_c_runner_utils' + ext));
    expect(api.teracRuntimeLibs('cpu', { build, llvmBin: override })).toEqual([runtime]);
  });

  it('refuses compilation when required files are missing instead of invoking the FFI', () => {
    expect(() => api.teracCompile('module {}', 'cpu', 3, { build: join(root, 'missing') }))
      .toThrow(/cannot find.*tera-capi/);
    expect(() => api.teracCompile('module {}', 'cpu', 3, { library: location.library, build: join(root, 'missing') }))
      .toThrow(/set TERAC_LLVM_BIN/);
    expect(() => api.teracCompile('module {}', 'cuda', 3, { ...location, llvmBin: join(root, 'missing') }))
      .toThrow(/cannot find mlir_c_runner_utils/);
    expect(native.compile).not.toHaveBeenCalled();
  });

  it('rejects an unsupported device before calling the native compiler', () => {
    expect(() => api.teracCompile('module {}', 'webgpu', 3, location)).toThrow(/unknown device 'webgpu'/);
    expect(native.compile).not.toHaveBeenCalled();
  });

  it.each([
    ['invalid IR', 'invalid IR'],
    ['', 'the module did not compile'],
  ])('turns a null compile handle into an error (%j)', (nativeMessage, expected) => {
    native.compile.mockReturnValue(null);
    native.lastError.mockReturnValue(nativeMessage);
    expect(() => api.teracCompile('invalid', 'cpu', 3, location)).toThrow(expected);
  });

  it('passes tensor counts rather than ranks or element counts to native invocation', () => {
    const module = api.teracCompile('module {}', 'cpu', 3, location);
    const inputs = [new Float32Array(6), new Float32Array(1)];
    const outputs = [new Float32Array(6)];
    api.teracInvoke(module, 'main', inputs, [2, 3], outputs, [3, 2]);
    expect(native.invoke).toHaveBeenCalledWith(handle, 'main', inputs, [2, 3], 2, outputs, [3, 2], 1);
  });

  it.each([
    ['shape mismatch', 'shape mismatch'],
    ['', 'main did not run'],
  ])('turns nonzero invocation status into an error (%j)', (nativeMessage, expected) => {
    const module = api.teracCompile('module {}', 'cpu', 3, location);
    native.invoke.mockReturnValue(1);
    native.lastError.mockReturnValue(nativeMessage);
    expect(() => api.teracInvoke(module, 'main', [], [], [], [])).toThrow(expected);
  });
});
