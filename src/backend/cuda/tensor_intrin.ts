export type CudaIntrinInfo = { M: number; N: number; K: number; a: string; b: string; c: string; tile?: number };

export type CudaIntrinEmitTarget = {
  _blockDim: number[];
  _gridDim: number[];
  _indent: number;
  _emit(line: string): void;
};

export type CudaIntrinEmitter = (cg: CudaIntrinEmitTarget, info: CudaIntrinInfo) => void;

const CUDA_INTRINSICS = new Map<string, CudaIntrinEmitter>();

export function registerCudaIntrin(name: string, emit: CudaIntrinEmitter): void {
  CUDA_INTRINSICS.set(name, emit);
}

export function getCudaIntrin(name: string): CudaIntrinEmitter | null {
  return CUDA_INTRINSICS.get(name) || null;
}

registerCudaIntrin('wmma_16x16x16_f16f16f32', (cg, info) => {
  const { M, N, K, a, b, c } = info;
  cg._blockDim = [32, 1, 1];
  cg._gridDim = [Math.ceil(M / 16), Math.ceil(N / 16), 1];
  cg._emit('const int warpM = blockIdx.x;');
  cg._emit('const int warpN = blockIdx.y;');
  cg._emit('fragment<accumulator, 16, 16, 16, float> cf;');
  cg._emit('fill_fragment(cf, 0.0f);');
  cg._emit(`for (int kk = 0; kk < ${K}; kk += 16) {`);
  cg._indent++;
  cg._emit('fragment<matrix_a, 16, 16, 16, half, row_major> af;');
  cg._emit('fragment<matrix_b, 16, 16, 16, half, row_major> bf;');
  cg._emit(`load_matrix_sync(af, ${a} + warpM * 16 * ${K} + kk, ${K});`);
  cg._emit(`load_matrix_sync(bf, ${b} + kk * ${N} + warpN * 16, ${N});`);
  cg._emit('mma_sync(cf, af, bf, cf);');
  cg._indent--;
  cg._emit('}');
  cg._emit(`store_matrix_sync(${c} + warpM * 16 * ${N} + warpN * 16, cf, ${N}, mem_row_major);`);
});

registerCudaIntrin('gemm_pipelined_f32', (cg, info) => {
  const { M, N, K, a, b, c, tile = 16 } = info;
  cg._blockDim = [tile, tile, 1];
  cg._gridDim = [Math.ceil(N / tile), Math.ceil(M / tile), 1];
  const T = tile;
  cg._emit(`__shared__ float As[2][${T}][${T}];`);
  cg._emit(`__shared__ float Bs[2][${T}][${T}];`);
  cg._emit(`const int row = blockIdx.y * ${T} + threadIdx.y;`);
  cg._emit(`const int col = blockIdx.x * ${T} + threadIdx.x;`);
  cg._emit('float acc = 0.0f;');
  cg._emit(`const int nTiles = ${K} / ${T};`);
  cg._emit(`__pipeline_memcpy_async(&As[0][threadIdx.y][threadIdx.x], &${a}[row * ${K} + threadIdx.x], sizeof(float));`);
  cg._emit(`__pipeline_memcpy_async(&Bs[0][threadIdx.y][threadIdx.x], &${b}[threadIdx.y * ${N} + col], sizeof(float));`);
  cg._emit('__pipeline_commit();');
  cg._emit('for (int t = 0; t < nTiles; t++) {');
  cg._indent++;
  cg._emit('const int cur = t & 1, nxt = (t + 1) & 1;');
  cg._emit('if (t + 1 < nTiles) {');
  cg._indent++;
  cg._emit(`__pipeline_memcpy_async(&As[nxt][threadIdx.y][threadIdx.x], &${a}[row * ${K} + (t + 1) * ${T} + threadIdx.x], sizeof(float));`);
  cg._emit(`__pipeline_memcpy_async(&Bs[nxt][threadIdx.y][threadIdx.x], &${b}[((t + 1) * ${T} + threadIdx.y) * ${N} + col], sizeof(float));`);
  cg._emit('__pipeline_commit();');
  cg._indent--;
  cg._emit('}');
  cg._emit('__pipeline_wait_prior(t + 1 < nTiles ? 1 : 0);');
  cg._emit('__syncthreads();');
  cg._emit(`for (int kk = 0; kk < ${T}; kk++) acc += As[cur][threadIdx.y][kk] * Bs[cur][kk][threadIdx.x];`);
  cg._emit('__syncthreads();');
  cg._indent--;
  cg._emit('}');
  cg._emit(`${c}[row * ${N} + col] = acc;`);
});
