import { cu, nv, checkCU, readProgramLog, cudaIncludeDir } from './ffi.js';
import type { CudaHandle } from './ffi.js';
import { getDevice } from './device.js';
import { ptxCacheKey, readPtx, writePtx } from './ptx_cache.js';
import type { CudaKernel } from '../../io.js';

export type CudaProgram = { func: CudaHandle | null; module: CudaHandle | null };

const _cache = new Map<string, CudaProgram>();

const PREAMBLE =
  '#ifndef INFINITY\n#define INFINITY __int_as_float(0x7f800000)\n#endif\n' +
  '#ifndef NAN\n#define NAN __int_as_float(0x7fffffff)\n#endif\n';

const STDINT_TYPEDEFS =
  'typedef signed char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long long int64_t;\n' +
  'typedef unsigned char uint8_t;\ntypedef unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long long uint64_t;\n';

export function hashSource(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16);
}

const _ptxCache = new Map<string, Uint8Array>();

let _nvrtcVersion: string | null = null;

function nvrtcVersion(): string {
  if (_nvrtcVersion === null) {
    const major = [0], minor = [0];
    _nvrtcVersion = nv.version(major, minor) === 0 ? major[0] + '.' + minor[0] : 'unknown';
  }
  return _nvrtcVersion;
}

function wrapSource(source: string): string {
  let includes = '';
  if (source.includes('__half')) includes += '#include <cuda_fp16.h>\n';
  if (source.includes('__nv_bfloat16')) includes += '#include <cuda_bf16.h>\n';
  if (/u?int(8|16|64)_t/.test(source)) includes += STDINT_TYPEDEFS;
  if (/mma_sync|wmma::|fragment</.test(source)) includes += '#include <mma.h>\nusing namespace nvcuda::wmma;\n';
  if (source.includes('__pipeline_memcpy_async')) includes += '#include <cuda_pipeline.h>\n';
  return includes + PREAMBLE + 'extern "C" {\n' + source + '\n}\n';
}

export function compileToPTX(source: string, kernelName: string): Uint8Array {
  const { arch } = getDevice();
  const wrapped = wrapSource(source);
  const options = ['--gpu-architecture=' + arch];
  if (cudaIncludeDir) options.push('--include-path=' + cudaIncludeDir);

  const key = ptxCacheKey([wrapped, kernelName, arch, nvrtcVersion(), options.join(' ')]);
  const memHit = _ptxCache.get(key);
  if (memHit) return memHit;
  const diskHit = readPtx(key);
  if (diskHit) {
    _ptxCache.set(key, diskHit);
    return diskHit;
  }

  const prog: (CudaHandle | null)[] = [null];
  checkCU('nvrtcCreateProgram', nv.createProgram(prog, wrapped, kernelName + '.cu', 0, null, null));
  const compileCode = nv.compileProgram(prog[0], options.length, options);
  if (compileCode !== 0) {
    const log = readProgramLog(prog[0]);
    nv.destroyProgram(prog);
    throw new Error('NVRTC compile failed for kernel ' + kernelName + ':\n' + log + '\n--- source ---\n' + source);
  }
  const sz = [0n];
  checkCU('nvrtcGetPTXSize', nv.getPTXSize(prog[0], sz));
  const ptx = new Uint8Array(Number(sz[0]));
  checkCU('nvrtcGetPTX', nv.getPTX(prog[0], ptx));
  nv.destroyProgram(prog);
  _ptxCache.set(key, ptx);
  writePtx(key, ptx);
  return ptx;
}

export function clearProgramCache(): void {
  _ptxCache.clear();
  _cache.clear();
}

const _byKernel = new WeakMap<CudaKernel, CudaProgram>();

export function getProgramFor(compiledKernel: CudaKernel): CudaProgram {
  let r = _byKernel.get(compiledKernel);
  if (r) return r;
  r = getProgram(compiledKernel.source, compiledKernel.name);
  _byKernel.set(compiledKernel, r);
  return r;
}

export function getProgram(source: string, kernelName: string): CudaProgram {
  const key = hashSource(source) + ':' + kernelName;
  const cached = _cache.get(key);
  if (cached) return cached;

  const ptx = compileToPTX(source, kernelName);
  const mod: (CudaHandle | null)[] = [null];
  checkCU('cuModuleLoadData', cu.moduleLoadData(mod, ptx));
  const func: (CudaHandle | null)[] = [null];
  checkCU('cuModuleGetFunction', cu.moduleGetFunction(func, mod[0], kernelName));

  const result = { func: func[0], module: mod[0] };
  _cache.set(key, result);
  return result;
}
