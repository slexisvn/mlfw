import { visitStatements, isZeroFillBody, estimateBufferSize, dynamicDimProduct } from './codegen_utils.js';
import { maxBindingExtent } from '../compiler/analysis/thread_binding.js';
import { storedBufferNames } from '../compiler/analysis/tir_queries.js';
import type { CodegenFunc, StatementVisitor } from './codegen_utils.js';
import type { TargetFeatures } from './target.js';
import type { Buffer } from '../compiler/ir/tensor/buffer.js';
import type { IRStmtNode, LIRThreadBinding } from '../compiler/ir/lir/nodes.js';

export abstract class CodegenBase {
  target: TargetFeatures;
  _indent: number;
  _lines: string[];
  declare _primFunc: CodegenFunc;

  constructor(target: TargetFeatures) {
    this.target = target;
    this._indent = 0;
    this._lines = [];
  }

  abstract _resolveShapeParam(buffer: Buffer, dimIdx: number): string;

  _emit(line: string): void {
    this._lines.push('  '.repeat(this._indent) + line);
  }

  _visitNode(node: IRStmtNode): void {
    visitStatements(this as unknown as StatementVisitor, node);
  }

  _isZeroFillBody(body: IRStmtNode): boolean {
    return isZeroFillBody(body);
  }

  _estimateBufferSize(buffer: Buffer): number {
    return estimateBufferSize(buffer);
  }

  _computeDynamicStride(buffer: Buffer, dimIdx: number): string {
    return dynamicDimProduct(buffer, dimIdx + 1, (b, j) => this._resolveShapeParam(b, j));
  }

  _dynamicNumel(buffer: Buffer): string {
    return dynamicDimProduct(buffer, 0, (b, j) => this._resolveShapeParam(b, j));
  }
}

export abstract class GpuCodegenBase extends CodegenBase {
  _threadBindings: Map<string, LIRThreadBinding[]>;
  _sharedBuffers: Buffer[];
  _storeBuffers: Set<string>;

  constructor(target: TargetFeatures) {
    super(target);
    this._threadBindings = new Map();
    this._sharedBuffers = [];
    this._storeBuffers = new Set();
  }

  _getMaxBindingExtent(tag: string | null): number {
    return maxBindingExtent(this._threadBindings, tag);
  }

  _scanStoreTargets(root: IRStmtNode): void {
    this._storeBuffers = storedBufferNames(root);
  }
}
