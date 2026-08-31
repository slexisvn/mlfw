import { getCodegenEntry, getExternalCodegen } from './codegen_registry.js';
import { FuncAttr } from '../compiler/ir/func_attrs.js';
import type { CodegenEntry, CodegenMetadata } from './codegen_registry.js';
import type { PrimFunc } from '../compiler/ir/tensor/nodes.js';

import type { ExternalCodegenAttr } from '../compiler/ir/func_attrs.js';
import type { ConstBuffer } from '../compiler/passes/lowering/lowering_registry.js';
import type { TargetFeatures } from '../compiler/support/target.js';

export type CodegenEntryLookup = { getCodegenEntry(targetKind: string): unknown };
export type BackendPipelineOptions = { context?: CodegenEntryLookup | null };
export type KernelConstBuffer = { name: string; dtype: string; data: ArrayLike<number> };

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
  context: CodegenEntryLookup | null;

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
    const constBuffers = primFunc.getAttr<ConstBuffer[]>(FuncAttr.CONST_BUFFERS);
    if (constBuffers && constBuffers.length > 0) {
      if (!this.target.supportsConstBuffers) {
        throw new Error(`Target '${this.target.name}' cannot bind constant buffers; LegalizeConstBuffersPass must expand them before codegen`);
      }
      metadata.constBuffers = constBuffers.map((cb): KernelConstBuffer => ({ name: cb.buffer.name, dtype: cb.buffer.dtype, data: cb.data }));
    }
    return new CompiledKernel(primFunc.name, source, this.target, metadata);
  }

  compileAll(primFuncs: readonly PrimFunc[]): CompiledKernel[] {
    return primFuncs.map(f => this.compile(f));
  }
}
