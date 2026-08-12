import { getCodegenEntry, getExternalCodegen } from './codegen_registry.js';
import { FuncAttr } from '../compiler/ir/func_attrs.js';
import type { CodegenEntry, CodegenMetadata } from './codegen_registry.js';
import type { PrimFunc } from '../compiler/ir/tensor/nodes.js';
import type { CompilerContext } from '../compiler/pipeline/compiler_context.js';
import type { ExternalCodegenAttr } from '../compiler/pipeline/external_codegen.js';
import type { TargetFeatures } from './target.js';

export type BackendPipelineOptions = { context?: CompilerContext | null };

export class CompiledKernel {
  name: string;
  source: string;
  target: TargetFeatures;
  metadata: Partial<CodegenMetadata>;

  constructor(name: string, source: string, target: TargetFeatures, metadata: Partial<CodegenMetadata> = {}) {
    this.name = name;
    this.source = source;
    this.target = target;
    this.metadata = metadata;
  }
}

export class BackendPipeline {
  target: TargetFeatures;
  context: CompilerContext | null;

  constructor(target: TargetFeatures, options: BackendPipelineOptions = {}) {
    this.target = target;
    this.context = options.context || null;
  }

  compile(primFunc: PrimFunc): CompiledKernel {
    const external = primFunc.getAttr<ExternalCodegenAttr>(FuncAttr.EXTERNAL_CODEGEN);
    if (external) {
      const entry = getExternalCodegen(external.name);
      if (entry && entry.targetKind === this.target.kind) {
        const { source, metadata } = entry.compile(primFunc, this.target, external.info);
        return new CompiledKernel(primFunc.name, source, this.target, metadata);
      }
    }
    const entry = ((this.context && this.context.getCodegenEntry(this.target.kind)) || getCodegenEntry(this.target.kind)) as CodegenEntry | null;
    if (!entry) throw new Error(`Unsupported target kind: ${this.target.kind}`);
    const { source, metadata } = entry.compile(primFunc, this.target, this);
    return new CompiledKernel(primFunc.name, source, this.target, metadata);
  }

  compileAll(primFuncs: readonly PrimFunc[]): CompiledKernel[] {
    return primFuncs.map(f => this.compile(f));
  }
}
