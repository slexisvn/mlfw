import { buildFunction } from '../compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../compiler/ir/graph/types.js';
import { compileGraph } from '../compiler/pipeline/compiler.js';
import { CPUTarget } from '../compiler/support/target.js';
import { Optimizer } from './optimizer.js';
import type { NumberTypedArray, NumberTypedArrayConstructor, OptimizerParams } from './types.js';

const F = ScalarType.F32;
const VEC = (n: number): TensorType => new TensorType([n], F);
const SCALAR = new TensorType([], F);

type IROpLike = { getResult(index: number): IRValueLike };
type IRValueLike = unknown;
type IRBuilderLike = {
  broadcast(value: IRValueLike, shape: readonly number[], dims: readonly number[]): IROpLike;
  scalarConstant(value: number, dtype: typeof F): IROpLike;
  add(lhs: IRValueLike, rhs: IRValueLike): IROpLike;
  sub(lhs: IRValueLike, rhs: IRValueLike): IROpLike;
  mul(lhs: IRValueLike, rhs: IRValueLike): IROpLike;
  div(lhs: IRValueLike, rhs: IRValueLike): IROpLike;
  sqrt(value: IRValueLike): IROpLike;
  maximum(lhs: IRValueLike | null, rhs: IRValueLike): IROpLike;
  returnOp(values: readonly (IRValueLike | null)[]): void;
};
type CompiledKernel = { run(name: string, ...args: unknown[]): void };
type TargetLike = ReturnType<typeof CPUTarget>;
type GraphFunctionLike = unknown;
type FusedDefaults = Record<string, unknown>;
type FusedSGDOptions = { lr?: number; momentum?: number; dampening?: number; weightDecay?: number; nesterov?: boolean };
type FusedAdamOptions = { lr?: number; betas?: readonly [number, number]; eps?: number; weightDecay?: number; amsgrad?: boolean };

function bcast(b: IRBuilderLike, scalarValue: IRValueLike, n: number): IRValueLike {
  return b.broadcast(scalarValue, [n], []).getResult(0);
}

export class FusedOptimizer extends Optimizer {
  protected _target: TargetLike;
  private _kernels: Map<number, CompiledKernel>;

  constructor(params: OptimizerParams, defaults: FusedDefaults, target: TargetLike | null = null) {
    super(params, defaults);
    this._target = target || CPUTarget();
    this._kernels = new Map();
  }

  _kernel(numel: number): CompiledKernel {
    let entry = this._kernels.get(numel);
    if (!entry) {
      const func = this._buildGraph(numel);
      entry = compileGraph(func as never, this._target as never, { fusion: { enabled: true } }) as unknown as CompiledKernel;
      this._kernels.set(numel, entry);
    }
    return entry;
  }

  _buildGraph(_numel: number): GraphFunctionLike {
    throw new Error(`${this.constructor.name}._buildGraph() not implemented`);
  }
}

export class FusedSGD extends FusedOptimizer {
  constructor(params: OptimizerParams, { lr = 0.01, momentum = 0, dampening = 0, weightDecay = 0, nesterov = false }: FusedSGDOptions = {}, target: TargetLike | null = null) {
    if (nesterov && (momentum === 0 || dampening !== 0)) {
      throw new Error('Nesterov momentum requires momentum > 0 and dampening = 0');
    }
    super(params, { lr, momentum, dampening, weightDecay, nesterov }, target);
  }

  _buildGraph(n: number): GraphFunctionLike {
    const momentum = this._defaults.momentum as number;
    const dampening = this._defaults.dampening as number;
    const weightDecay = this._defaults.weightDecay as number;
    const nesterov = this._defaults.nesterov as boolean;
    const useMom = momentum !== 0;
    const inTypes = useMom ? [VEC(n), VEC(n), VEC(n), SCALAR] : [VEC(n), VEC(n), SCALAR];
    const outTypes = useMom ? [VEC(n), VEC(n)] : [VEC(n)];
    const name = 'sgd_update';
    return buildFunction(name, inTypes, outTypes, (b: IRBuilderLike, a: IRValueLike[]) => {
      const w = a[0];
      const g = a[1];
      const lr = useMom ? a[3] : a[2];
      const lrV = bcast(b, lr, n);

      let gi = g;
      if (weightDecay !== 0) {
        const wd = b.scalarConstant(weightDecay, F).getResult(0);
        gi = b.add(g, b.mul(bcast(b, wd, n), w).getResult(0)).getResult(0);
      }

      if (!useMom) {
        const wNew = b.sub(w, b.mul(lrV, gi).getResult(0)).getResult(0);
        b.returnOp([wNew]);
        return;
      }

      const m = a[2];
      const momV = bcast(b, b.scalarConstant(momentum, F).getResult(0), n);
      const keep = bcast(b, b.scalarConstant(1 - dampening, F).getResult(0), n);
      const bufNew = b.add(b.mul(momV, m).getResult(0), b.mul(keep, gi).getResult(0)).getResult(0);

      let update = bufNew;
      if (nesterov) {
        update = b.add(gi, b.mul(momV, bufNew).getResult(0)).getResult(0);
      }
      const wNew = b.sub(w, b.mul(lrV, update).getResult(0)).getResult(0);
      b.returnOp([wNew, bufNew]);
    });
  }

  step(): void {
    for (const group of this._paramGroups) {
      const lr = group.lr as number;
      const momentum = group.momentum as number;
      const useMom = momentum !== 0;
      const lrArr = new Float32Array([lr]);
      for (const p of group.params) {
        if (p.grad === null) continue;
        const w = p._impl.storage.data! as NumberTypedArray;
        const g = p.grad._impl.storage.data! as NumberTypedArray;
        const n = w.length;
        const kernel = this._kernel(n);

        if (useMom) {
          const state = this._getState(p);
          let buf = state.momentumBuffer;
          if (!buf) {
            buf = new (w.constructor as NumberTypedArrayConstructor)(n);
            state.momentumBuffer = buf;
          }
          buf = buf as NumberTypedArray;
          kernel.run('sgd_update', w, g, buf, lrArr, w, buf);
        } else {
          kernel.run('sgd_update', w, g, lrArr, w);
        }
        p._impl.bumpVersion();
      }
    }
  }
}

export class FusedAdam extends FusedOptimizer {
  constructor(params: OptimizerParams, { lr = 0.001, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0, amsgrad = false }: FusedAdamOptions = {}, target: TargetLike | null = null) {
    super(params, { lr, betas: [...betas], eps, weightDecay, amsgrad }, target);
  }

  _buildGraph(n: number): GraphFunctionLike {
    const betas = this._defaults.betas as readonly [number, number];
    const eps = this._defaults.eps as number;
    const weightDecay = this._defaults.weightDecay as number;
    const amsgrad = this._defaults.amsgrad as boolean;
    const [beta1, beta2] = betas;
    const inTypes = amsgrad
      ? [VEC(n), VEC(n), VEC(n), VEC(n), VEC(n), SCALAR, SCALAR]
      : [VEC(n), VEC(n), VEC(n), VEC(n), SCALAR, SCALAR];
    const outTypes = amsgrad ? [VEC(n), VEC(n), VEC(n), VEC(n)] : [VEC(n), VEC(n), VEC(n)];
    return buildFunction('adam_update', inTypes, outTypes, (b: IRBuilderLike, a: IRValueLike[]) => {
      const w = a[0];
      const g = a[1];
      const m = a[2];
      const v = a[3];
      const vmax = amsgrad ? a[4] : null;
      const stepSize = amsgrad ? a[5] : a[4];
      const bc2sqrt = amsgrad ? a[6] : a[5];

      let gi = g;
      if (weightDecay !== 0) {
        const wd = bcast(b, b.scalarConstant(weightDecay, F).getResult(0), n);
        gi = b.add(g, b.mul(wd, w).getResult(0)).getResult(0);
      }

      const b1 = bcast(b, b.scalarConstant(beta1, F).getResult(0), n);
      const b2 = bcast(b, b.scalarConstant(beta2, F).getResult(0), n);
      const omb1 = bcast(b, b.scalarConstant(1 - beta1, F).getResult(0), n);
      const omb2 = bcast(b, b.scalarConstant(1 - beta2, F).getResult(0), n);
      const epsV = bcast(b, b.scalarConstant(eps, F).getResult(0), n);

      const mNew = b.add(b.mul(b1, m).getResult(0), b.mul(omb1, gi).getResult(0)).getResult(0);
      const g2 = b.mul(gi, gi).getResult(0);
      const vNew = b.add(b.mul(b2, v).getResult(0), b.mul(omb2, g2).getResult(0)).getResult(0);

      let vForDenom = vNew;
      let vmaxNew = null;
      if (amsgrad) {
        vmaxNew = b.maximum(vmax, vNew).getResult(0);
        vForDenom = vmaxNew;
      }

      const denom = b.add(b.div(b.sqrt(vForDenom).getResult(0), bcast(b, bc2sqrt, n)).getResult(0), epsV).getResult(0);
      const update = b.mul(bcast(b, stepSize, n), b.div(mNew, denom).getResult(0)).getResult(0);
      const wNew = b.sub(w, update).getResult(0);

      b.returnOp(amsgrad ? [wNew, mNew, vNew, vmaxNew] : [wNew, mNew, vNew]);
    });
  }

  step(): void {
    for (const group of this._paramGroups) {
      const lr = group.lr as number;
      const betas = group.betas as readonly [number, number];
      const amsgrad = group.amsgrad as boolean;
      const [beta1, beta2] = betas;
      for (const p of group.params) {
        if (p.grad === null) continue;
        const w = p._impl.storage.data! as NumberTypedArray;
        const g = p.grad._impl.storage.data! as NumberTypedArray;
        const n = w.length;
        const state = this._getState(p);

        if (state.step === undefined) {
          state.step = 0;
          state.expAvg = new (w.constructor as NumberTypedArrayConstructor)(n);
          state.expAvgSq = new (w.constructor as NumberTypedArrayConstructor)(n);
          if (amsgrad) state.maxExpAvgSq = new (w.constructor as NumberTypedArrayConstructor)(n);
        }

        state.step = (state.step as number) + 1;
        const t = state.step as number;
        const bc1 = 1 - Math.pow(beta1, t);
        const bc2 = 1 - Math.pow(beta2, t);
        const stepSize = new Float32Array([lr / bc1]);
        const bc2sqrt = new Float32Array([Math.sqrt(bc2)]);

        const kernel = this._kernel(n);
        const m = state.expAvg as NumberTypedArray;
        const v = state.expAvgSq as NumberTypedArray;
        if (amsgrad) {
          const vmax = state.maxExpAvgSq as NumberTypedArray;
          kernel.run('adam_update', w, g, m, v, vmax, stepSize, bc2sqrt, w, m, v, vmax);
        } else {
          kernel.run('adam_update', w, g, m, v, stepSize, bc2sqrt, w, m, v);
        }
        p._impl.bumpVersion();
      }
    }
  }
}
