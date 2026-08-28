import { visitStatements, isZeroFillBody, estimateBufferSize, dynamicDimProduct } from './codegen_utils.js';
import { planCommonSubexprs } from './expr_cse.js';
import { maxBindingExtent } from '../compiler/analysis/thread_binding.js';
import { storedBufferNames } from '../compiler/analysis/tir_queries.js';
import type { CodegenFunc, StatementVisitor } from './codegen_utils.js';
import type { TargetFeatures } from './target.js';
import type { Buffer } from '../compiler/ir/tensor/buffer.js';
import type { IRStmtNode, LIRThreadBinding } from '../compiler/ir/lir/nodes.js';

const CSE_MIN_NODES = 8;

export type SourceHelper = Readonly<{ name: string; deps: readonly SourceHelper[]; code: string }>;

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
  _helpers: Map<string, SourceHelper>;
  _cseIds: ReadonlyMap<IRStmtNode, number> | null;
  _cseNames: Map<number, string>;
  _cseCounter: number;
  _cseDepth: number;

  constructor(target: TargetFeatures) {
    super(target);
    this._threadBindings = new Map();
    this._sharedBuffers = [];
    this._storeBuffers = new Set();
    this._helpers = new Map();
    this._cseIds = null;
    this._cseNames = new Map();
    this._cseCounter = 0;
    this._cseDepth = 0;
  }

  abstract _emitCseBinding(name: string, text: string): void;

  abstract _emitExprText(node: IRStmtNode): string;

  _resetSourceScope(): void {
    this._helpers = new Map();
    this._cseIds = null;
    this._cseNames = new Map();
    this._cseCounter = 0;
    this._cseDepth = 0;
  }

  _useHelper(helper: SourceHelper): string {
    if (this._helpers.has(helper.name)) return helper.name;
    for (const dep of helper.deps) this._useHelper(dep);
    this._helpers.set(helper.name, helper);
    return helper.name;
  }

  _helperPreamble(): string[] {
    return [...this._helpers.values()].map(h => h.code);
  }

  _beginExpr(root: IRStmtNode | null): boolean {
    if (this._cseDepth > 0) {
      this._cseDepth++;
      return false;
    }
    this._cseDepth = 1;
    if (root) this._scheduleCommonSubexprs(root);
    return true;
  }

  _endExpr(top: boolean): void {
    if (!top) {
      this._cseDepth--;
      return;
    }
    this._cseDepth = 0;
    this._cseIds = null;
    this._cseNames.clear();
  }

  _scheduleCommonSubexprs(root: IRStmtNode): void {
    const plan = planCommonSubexprs(root, CSE_MIN_NODES);
    if (plan.hoisted.length === 0) return;
    this._cseIds = plan.ids;
    for (const cls of plan.hoisted) {
      const text = this._emitExprText(cls.node);
      const name = `_cse${this._cseCounter++}`;
      this._emitCseBinding(name, text);
      this._cseNames.set(cls.id, name);
    }
  }

  _cseNameFor(node: IRStmtNode | null): string | null {
    if (!this._cseIds || !node) return null;
    const id = this._cseIds.get(node);
    if (id === undefined) return null;
    return this._cseNames.get(id) ?? null;
  }

  _getMaxBindingExtent(tag: string | null): number {
    return maxBindingExtent(this._threadBindings, tag);
  }

  _scanStoreTargets(root: IRStmtNode): void {
    this._storeBuffers = storedBufferNames(root);
  }
}
