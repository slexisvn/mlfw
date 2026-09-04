import { getActiveTracer } from './tracer.js';
import { select } from '../tensor/ops/ops.js';
import { stack } from '../tensor/ops/ops.js';
import type { TensorOutput } from './types.js';

type StepValue = TensorOutput | TensorOutput[];
type ScanFn = (carry: StepValue, xs: StepValue) => [StepValue, StepValue];

function stackSteps(steps: StepValue[]): StepValue {
  if (Array.isArray(steps[0])) {
    const first = steps[0];
    const out = new Array<TensorOutput>(first.length);
    for (let i = 0; i < out.length; i++) out[i] = stack(steps.map(s => (s as TensorOutput[])[i]), 0);
    return out;
  }
  return stack(steps as TensorOutput[], 0);
}

export function scan(fn: ScanFn, initCarry: StepValue, xs: StepValue): [StepValue, StepValue] {
  const carryIsArr = Array.isArray(initCarry);
  const xsIsArr = Array.isArray(xs);
  const xsArr = xsIsArr ? xs : [xs];
  const carryArr = carryIsArr ? initCarry : [initCarry];

  const tracer = getActiveTracer();
  if (!tracer) {
    let carry = initCarry;
    const steps = [];
    const T = xsArr[0].shape[0];
    for (let t = 0; t < T; t++) {
      const xt = xsIsArr ? xsArr.map(x => select(x, 0, t)) : select(xsArr[0], 0, t);
      const [newCarry, y] = fn(carry, xt);
      carry = newCarry;
      steps.push(y);
    }
    return [carry, stackSteps(steps)];
  }

  let yIsArr = false;
  const step = (carrySym: TensorOutput[], xtSym: TensorOutput[]) => {
    const [newCarry, y] = fn(carryIsArr ? carrySym : carrySym[0], xsIsArr ? xtSym : xtSym[0]);
    yIsArr = Array.isArray(y);
    return [
      (carryIsArr ? newCarry : [newCarry]) as TensorOutput[],
      (yIsArr ? y : [y]) as TensorOutput[],
    ];
  };
  const [finalCarryArr, ysArr] = tracer.scan(carryArr, xsArr, step as Parameters<typeof tracer.scan>[2]);
  return [carryIsArr ? finalCarryArr : finalCarryArr[0], yIsArr ? ysArr : ysArr[0]];
}
