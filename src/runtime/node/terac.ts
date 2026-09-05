import koffi from 'koffi';
import { existsSync, readFileSync } from 'fs';
import { dirname, isAbsolute, join, resolve } from 'path';

import { prependSearchPath } from './search_path.js';

export type TeracHandle = Readonly<{ nativeHandle: unknown; api: TeracBinding }>;

export type TeracLocation = {
  build?: string | null;
  library?: string | null;
  llvmBin?: string | null;
};

type TeracBinding = {
  compile(
    mlir: string, target: string, targetOptions: string, optLevel: number,
    sharedLibs: string[], numSharedLibs: number,
  ): unknown;
  release(handle: unknown): void;
  lastError(): string;
  invoke(
    handle: unknown, entry: string,
    inputs: ArrayBufferView[], inputShapes: number[], numInputs: number,
    results: ArrayBufferView[], resultShapes: number[], numResults: number,
  ): number;
};

const SHLIB_EXT = process.platform === 'win32' ? '.dll' : process.platform === 'darwin' ? '.dylib' : '.so';

const LIT_SITE_CONFIG = 'test/lit.site.cfg.py';
const LLVM_TOOLS_DIR = /config\.llvm_tools_dir\s*=\s*lit_config\.substitute\("([^"]+)"\)/;

// A device is a terac target name, passed through as one. The libraries are
// here because they have to be found on disk before the module is compiled,
// which is before there is a module to ask.
const DEVICES: Readonly<Record<string, readonly string[]>> = {
  cpu: ['mlir_c_runner_utils'],
  cuda: ['mlir_c_runner_utils', 'mlir_cuda_runtime'],
};

function deviceLibs(device: string): readonly string[] {
  const libs = DEVICES[device];
  if (!libs) throw new Error(`terac: unknown device '${device}'`);
  return libs;
}

let binding: TeracBinding | null = null;
let bindingPath: string | null = null;

function defaultBuild(): string {
  return process.env.TERAC_BUILD || resolve(process.cwd(), 'terac/build');
}

function firstExisting(candidates: readonly string[], what: string): string {
  for (const candidate of candidates) if (existsSync(candidate)) return candidate;
  throw new Error(`terac: cannot find ${what}; looked in\n  ${candidates.join('\n  ')}`);
}

function libraryPath(location: TeracLocation): string {
  const given = location.library || process.env.TERAC_LIBRARY;
  if (given) return isAbsolute(given) ? given : resolve(given);
  const build = location.build || defaultBuild();
  return firstExisting(
    ['bin', 'lib'].flatMap((dir) => ['tera-capi', 'libtera-capi'].map((stem) => join(build, dir, stem + SHLIB_EXT))),
    'the tera-capi shared library (build it, or set TERAC_LIBRARY)',
  );
}

function llvmToolsDir(location: TeracLocation): string {
  const given = location.llvmBin || process.env.TERAC_LLVM_BIN;
  if (given) return resolve(given);
  const site = join(location.build || defaultBuild(), LIT_SITE_CONFIG);
  const match = existsSync(site) ? LLVM_TOOLS_DIR.exec(readFileSync(site, 'utf8')) : null;
  if (!match) throw new Error('terac: no LLVM bin directory; set TERAC_LLVM_BIN');
  return resolve(match[1]);
}

export function teracRuntimeLibs(device: string, location: TeracLocation): string[] {
  const bin = llvmToolsDir(location);
  prependSearchPath(bin);
  return deviceLibs(device).map((stem) => firstExisting(
    [bin, join(dirname(bin), 'lib')].flatMap((dir) => [stem, `lib${stem}`].map((name) => join(dir, name + SHLIB_EXT))),
    stem,
  ));
}

function load(location: TeracLocation): TeracBinding {
  const path = libraryPath(location);
  if (binding && bindingPath === path) return binding;
  prependSearchPath(dirname(path));
  const lib = koffi.load(path);
  binding = {
    compile: lib.func('void *teraCompileFor(str mlir, str target, str targetOptions, uint optLevel, const char **sharedLibs, size_t numSharedLibs)'),
    release: lib.func('void teraRelease(void *module)'),
    lastError: lib.func('str teraLastError()'),
    invoke: lib.func('int teraInvoke(void *module, str entry, void **inputs, int64 *inputShapes, int64 numInputs, void **results, int64 *resultShapes, int64 numResults)'),
  };
  bindingPath = path;
  return binding;
}

export function teracAvailable(location: TeracLocation = {}): boolean {
  try {
    load(location);
    return true;
  } catch {
    return false;
  }
}

export function teracCompile(
  mlir: string, device: string, optLevel: number, location: TeracLocation,
  targetOptions = '',
): TeracHandle {
  const api = load(location);
  const libs = teracRuntimeLibs(device, location);
  const nativeHandle = api.compile(mlir, device, targetOptions, optLevel, libs, libs.length);
  if (!nativeHandle) throw new Error(api.lastError() || 'terac: the module did not compile');
  return { nativeHandle, api };
}

export function teracInvoke(
  handle: TeracHandle, entry: string,
  inputs: ArrayBufferView[], inputShapes: number[],
  results: ArrayBufferView[], resultShapes: number[],
): void {
  const { api, nativeHandle } = handle;
  const status = api.invoke(nativeHandle, entry, inputs, inputShapes, inputs.length, results, resultShapes, results.length);
  if (status !== 0) throw new Error(api.lastError() || `terac: ${entry} did not run`);
}

export function teracRelease(handle: TeracHandle): void {
  handle.api.release(handle.nativeHandle);
}
