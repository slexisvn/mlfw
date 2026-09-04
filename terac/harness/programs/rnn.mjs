import { ramp } from "../values.mjs";

export default (mlfw) => ({
  inputs: [ramp([3, 2], 3), ramp([2], 5), ramp([2, 2], 7), ramp([2, 2], 2)],
  forward: (xs, h0, W, U) => {
    const [, ys] = mlfw.scan(
      (h, x) => {
        const next = mlfw.relu(mlfw.add(mlfw.matmul(h, W), mlfw.matmul(x, U)));
        return [next, next];
      },
      h0,
      xs,
    );
    return ys;
  },
});
