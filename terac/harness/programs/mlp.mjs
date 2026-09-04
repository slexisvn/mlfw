import { ramp } from "../values.mjs";

export default (mlfw) => ({
  inputs: [ramp([2, 4], 3), ramp([4, 6], 5), ramp([6], 7), ramp([6, 3], 2), ramp([3], 4)],
  forward: (x, w1, b1, w2, b2) =>
    mlfw.add(mlfw.matmul(mlfw.relu(mlfw.add(mlfw.matmul(x, w1), b1)), w2), b2),
});
