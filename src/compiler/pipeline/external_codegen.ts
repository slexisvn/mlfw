import { CublasRewritePass } from '../passes/rewrite/cublas_rewrite.js';
import { FuncAttr, CUBLAS_PROVIDER } from '../ir/func_attrs.js';
import type { ExternalKernelInfo } from '../ir/func_attrs.js';
import type { BlockNode, NodeSlots, PrimFunc, TirNode } from '../ir/tensor/nodes.js';
import type { TirModule } from '../ir/tensor/module.js';
import type { CompileTarget, CompilerConfig, GraphPass } from './pipeline_types.js';

export type { ExternalKernelInfo, ExternalCodegenAttr } from '../ir/func_attrs.js';
export type SplitKernelInfos = { cublasInfos?: ReadonlyMap<string, ExternalKernelInfo> } | null | undefined;

export type ExternalCodegenProvider = Readonly<{
  name: string;
  enabled(config: CompilerConfig, target: CompileTarget): boolean;
  suppressesEpilogueFusion?: boolean;
  graphPasses?(config: CompilerConfig, target: CompileTarget): GraphPass[];
  annotate?(tirModule: TirModule, split: SplitKernelInfos): void;
}>;

const _providers = new Map<string, ExternalCodegenProvider>();

export function registerExternalCodegenProvider(provider: ExternalCodegenProvider): void {
  _providers.set(provider.name, provider);
}

export function unregisterExternalCodegenProvider(name: string): boolean {
  return _providers.delete(name);
}

export function activeExternalCodegenProviders(config: CompilerConfig, target: CompileTarget): ExternalCodegenProvider[] {
  const active: ExternalCodegenProvider[] = [];
  for (const provider of _providers.values()) {
    if (provider.enabled(config, target)) active.push(provider);
  }
  return active;
}

export function isExternalCodegenEnabled(name: string, config: CompilerConfig, target: CompileTarget): boolean {
  const provider = _providers.get(name);
  return provider !== undefined && provider.enabled(config, target);
}

export function detectPureMatmul(primFunc: PrimFunc): ExternalKernelInfo | null {
  const blocks: BlockNode[] = [];
  const stack: TirNode[] = [primFunc.body];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n) continue;
    const slots = n as unknown as NodeSlots;
    if (n.type === 'BlockNode') { blocks.push(n as BlockNode); stack.push(slots.body as TirNode); continue; }
    if (slots.body) stack.push(slots.body as TirNode);
    if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
    if (slots.thenBody) stack.push(slots.thenBody as TirNode);
    if (slots.elseBody) stack.push(slots.elseBody as TirNode);
  }
  let matmul: BlockNode | null = null;
  for (const b of blocks) {
    if (!b.name.includes('matmul')) return null;
    if (b.reads.length >= 2 && b.writes.length >= 1) matmul = b;
  }
  if (!matmul) return null;
  const A = matmul.reads[0].buffer, B = matmul.reads[1].buffer, C = matmul.writes[0].buffer;
  if (A.dtype !== 'f32' || B.dtype !== 'f32' || C.dtype !== 'f32') return null;
  if (A.shape.length !== 2 || B.shape.length !== 2 || C.shape.length !== 2) return null;
  const M = C.shape[0] as number, N = C.shape[1] as number, K = A.shape[1] as number;
  if (![M, N, K].every(d => typeof d === 'number')) return null;
  const names: string[] = [];
  for (const [, buf] of primFunc.bufferMap) names.push(buf.name);
  const aIdx = names.indexOf(A.name), bIdx = names.indexOf(B.name), cIdx = names.indexOf(C.name);
  if (aIdx < 0 || bIdx < 0 || cIdx < 0) return null;
  return { M, N, K, aIdx, bIdx, cIdx };
}

export { CUBLAS_PROVIDER } from '../ir/func_attrs.js';

registerExternalCodegenProvider({
  name: CUBLAS_PROVIDER,
  suppressesEpilogueFusion: true,
  enabled: (config) => config.matmulBackend === CUBLAS_PROVIDER,
  graphPasses: () => [new CublasRewritePass()],
  annotate: (tirModule, split) => {
    const fromSplit = split && split.cublasInfos ? split.cublasInfos : null;
    for (const primFunc of tirModule) {
      const info = fromSplit ? fromSplit.get(primFunc.name) : detectPureMatmul(primFunc);
      if (info) primFunc.setAttr(FuncAttr.EXTERNAL_CODEGEN, { name: CUBLAS_PROVIDER, info });
    }
  },
});
