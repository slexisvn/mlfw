import { MathOpNode, FloatImmNode, IntImmNode, BufferStoreNode, BufferLoadNode, BlockNode, SeqNode, CallExternNode, IfThenElseNode, CompareNode, AllocateNode } from '../../../ir/tensor/nodes.js';
import { Buffer } from '../../../ir/tensor/buffer.js';
import { registerLoweringRule, buildSpatialNest, wrapLoopsWithNodes, concatIterVars, markCommReduce, buildReduceGeometry, splitReduceDims, emitReduceMeanDiv } from '../lowering_registry.js';
import { isDtypeInt } from '../../../../util/dtype_map.js';
import type { TirNode } from '../../../ir/tensor/nodes.js';
import type { SpatialNest } from '../lowering_registry.js';

type ReduceCombiner = (a: TirNode, b: TirNode, dt: string) => TirNode;

const INT_DTYPE_MIN: Record<string, number> = { i8: -128, i16: -32768, i32: -2147483648, i64: -2147483648, ui8: 0, ui16: 0, ui32: 0, bool: 0 };
const INT_DTYPE_MAX: Record<string, number> = { i8: 127, i16: 32767, i32: 2147483647, i64: 2147483647, ui8: 255, ui16: 65535, ui32: 4294967295, bool: 1 };

function argReduceSentinel(dtype: string, isGt: boolean): TirNode {
  if (isDtypeInt(dtype)) {
    const v = isGt ? (INT_DTYPE_MIN[dtype] ?? -2147483648) : (INT_DTYPE_MAX[dtype] ?? 2147483647);
    return new IntImmNode(v);
  }
  return new FloatImmNode(isGt ? -Infinity : Infinity);
}

const REDUCE_NEST_PREFIXES = Object.freeze({ init: 'si', spatial: 'sa', reduce: 'r', reduceIter: 'rv' });

const REDUCE_COMBINERS: Record<string, ReduceCombiner> = {
  'sum':  (a, b) => new MathOpNode('+', a, b),
  'mean': (a, b) => new MathOpNode('+', a, b),
  'prod': (a, b) => new MathOpNode('*', a, b),
  'max':  (a, b, dt) => new CallExternNode('max', [a, b], dt),
  'min':  (a, b, dt) => new CallExternNode('min', [a, b], dt),
  'and':  (a, b) => new MathOpNode('*', a, b),
  'or':   (a, b, dt) => new CallExternNode('max', [a, b], dt)
};

export function getReduceCombiner(rType: string): ReduceCombiner | undefined {
  return REDUCE_COMBINERS[rType];
}

export function register(): void {
  registerLoweringRule('reduce', (ctx, op, inputs, outputs) => {
    const inBuf = inputs[0];
    const initBuf = inputs[1];
    const outBuf = outputs[0];
    const rType = op.getAttr<string>('reduce_type') || 'sum';
    const combiner = REDUCE_COMBINERS[rType];
    if (!combiner) throw new Error(`reduction lowering: unsupported reduce_type '${rType}'`);
    const { spatialDims, reduceDims } = splitReduceDims(inBuf.shape.length, op.getAttr<readonly number[]>('dimensions') || []);
    const geo = buildReduceGeometry(ctx, inBuf.shape, inBuf, spatialDims, reduceDims, REDUCE_NEST_PREFIXES);

    const initStore = new BufferStoreNode(outBuf, geo.init.indices, new BufferLoadNode(initBuf, []));
    const initBlock = new BlockNode(ctx.blockName('reduce_init'), geo.init.ivs, [{ buffer: initBuf }], [{ buffer: outBuf }], initStore);
    const initBody = spatialDims.length > 0 ? geo.init.wrap(initBlock) : initBlock;

    const loadA = new BufferLoadNode(outBuf, geo.spatial.indices);
    const loadB = new BufferLoadNode(inBuf, geo.fullIdx);
    const accStore = new BufferStoreNode(outBuf, geo.spatial.indices, combiner(loadA, loadB, outBuf.dtype));
    const accBlock = new BlockNode(ctx.blockName('reduce_acc'), geo.accIvs, [{ buffer: inBuf }], [{ buffer: outBuf }], accStore);

    const parts: TirNode[] = [initBody, geo.wrapAcc(accBlock)];
    if (rType === 'mean') {
      parts.push(emitReduceMeanDiv(ctx, outBuf, inBuf.shape, inBuf, spatialDims, reduceDims, 'sm', 'mean_div'));
    }
    return new SeqNode(parts);
  });

  function registerArgReduce(opName: string, compareFn: string): void {
    registerLoweringRule(opName, (ctx, op, inputs, outputs) => {
      const inBuf = inputs[0];
      const outBuf = outputs[0];
      const axis = op.getAttr<number>('axis') as number;
      const keepDims = op.getAttr<boolean>('keep_dims') || false;
      const dimSet = new Set([axis]);
      const spatialDims: number[] = [];
      const reduceDim = axis;
      for (let i = 0; i < inBuf.shape.length; i++) {
        if (!dimSet.has(i)) spatialDims.push(i);
      }

      const outIndicesFor = (nest: SpatialNest): TirNode[] => {
        if (!keepDims) return nest.indices;
        const idx: TirNode[] = new Array(inBuf.shape.length);
        for (let i = 0; i < spatialDims.length; i++) idx[spatialDims[i]] = nest.indices[i];
        idx[reduceDim] = new IntImmNode(0);
        return idx;
      };

      const bestValBuf = new Buffer('_argval_' + ctx.varCounter, [1], inBuf.dtype, 'local');
      ctx.varCounter++;
      const bestIdx = [new IntImmNode(0)];

      const nest = buildSpatialNest(ctx, 'ai', spatialDims, inBuf.shape, inBuf);
      const outIndices = outIndicesFor(nest);

      const initValStore = new BufferStoreNode(bestValBuf, bestIdx, argReduceSentinel(inBuf.dtype, compareFn === 'gt'));
      const initIdxStore = new BufferStoreNode(outBuf, outIndices, new IntImmNode(0));
      const initBlock = new BlockNode(ctx.blockName('arg_init'), nest.ivs, [], [{ buffer: bestValBuf }, { buffer: outBuf }], new SeqNode([initValStore, initIdxStore]));

      const rVar = ctx.allocVar('ar');
      const rBind = markCommReduce(ctx.allocBindArray('arv', [rVar]));
      const inIndices: TirNode[] = new Array(inBuf.shape.length);
      for (let i = 0; i < spatialDims.length; i++) inIndices[spatialDims[i]] = nest.ivs[i].iterVar;
      inIndices[reduceDim] = rBind[0].iterVar;

      const loadVal = new BufferLoadNode(inBuf, inIndices);
      const loadBest = new BufferLoadNode(bestValBuf, bestIdx);
      const isBetter = new CompareNode(compareFn, loadVal, loadBest);
      const newBest = new IfThenElseNode(isBetter, loadVal, loadBest);
      const loadIdx = new BufferLoadNode(outBuf, outIndices);
      const newIdx = new IfThenElseNode(isBetter, rBind[0].iterVar, loadIdx);
      const storeIdx = new BufferStoreNode(outBuf, outIndices, newIdx);
      const storeVal = new BufferStoreNode(bestValBuf, bestIdx, newBest);
      const accBlock = new BlockNode(ctx.blockName('arg_acc'), concatIterVars(nest.ivs, rBind),
        [{ buffer: inBuf }, { buffer: bestValBuf }], [{ buffer: bestValBuf }, { buffer: outBuf }],
        new SeqNode([storeIdx, storeVal]));
      const rExtent = ctx.extentNode(inBuf.shape[reduceDim], inBuf, reduceDim);
      const accBody: TirNode = wrapLoopsWithNodes(accBlock, [rVar], [rExtent]);

      const perElement = new AllocateNode(bestValBuf, bestValBuf.scope, new SeqNode([initBlock, accBody]));
      return spatialDims.length > 0 ? nest.wrap(perElement) : perElement;
    });
  }

  registerArgReduce('argmax', 'gt');
  registerArgReduce('argmin', 'lt');
}
