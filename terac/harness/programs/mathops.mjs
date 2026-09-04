import { ramp } from "../values.mjs";

export default (mlfw) => ({
  inputs: [ramp([2, 4], 3), ramp([4], 5), ramp([4], 7)],
  forward: (x, weight, bias) => {
    const mean = mlfw.mean(x, -1, true);
    const centred = mlfw.sub(x, mean);
    const variance = mlfw.mean(mlfw.mul(centred, centred), -1, true);
    const normalised = mlfw.mul(centred, mlfw.rsqrt(mlfw.add(variance, mlfw.scalar(1e-5))));
    const scaled = mlfw.add(mlfw.mul(normalised, weight), bias);
    return mlfw.mul(mlfw.tanh(scaled), mlfw.sqrt(mlfw.add(mlfw.mul(x, x), mlfw.scalar(1))));
  },
});
