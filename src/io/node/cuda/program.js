import { cu, nv, checkCU, readProgramLog, cudaIncludeDir } from './ffi.js';
import { getDevice } from './device.js';

const _cache = new Map();

const PREAMBLE =
  '#ifndef INFINITY\n#define INFINITY __int_as_float(0x7f800000)\n#endif\n' +
  '#ifndef NAN\n#define NAN __int_as_float(0x7fffffff)\n#endif\n';

const STDINT_TYPEDEFS =
  'typedef signed char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long long int64_t;\n' +
  'typedef unsigned char uint8_t;\ntypedef unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long long uint64_t;\n';

function hashSource(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16);
}

export function getProgram(source, kernelName) {
  const key = hashSource(source) + ':' + kernelName;
  const cached = _cache.get(key);
  if (cached) return cached;

  const { arch } = getDevice();
  let includes = '';
  if (source.includes('__half')) includes += '#include <cuda_fp16.h>\n';
  if (source.includes('__nv_bfloat16')) includes += '#include <cuda_bf16.h>\n';
  if (/u?int(8|16|64)_t/.test(source)) includes += STDINT_TYPEDEFS;
  const wrapped = includes + PREAMBLE + 'extern "C" {\n' + source + '\n}\n';
  const prog = [null];
  checkCU('nvrtcCreateProgram', nv.createProgram(prog, wrapped, kernelName + '.cu', 0, null, null));
  const options = ['--gpu-architecture=' + arch];
  if (cudaIncludeDir) options.push('--include-path=' + cudaIncludeDir);
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

  const mod = [null];
  checkCU('cuModuleLoadData', cu.moduleLoadData(mod, ptx));
  const func = [null];
  checkCU('cuModuleGetFunction', cu.moduleGetFunction(func, mod[0], kernelName));

  const result = { func: func[0], module: mod[0] };
  _cache.set(key, result);
  return result;
}
