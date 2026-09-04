import { ramp } from "../values.mjs";

export default (mlfw) => ({
  dynamicShapes: [new Set([0]), null],
  inputs: [ramp([3, 2, 4], 3), ramp([8, 2], 5)],
  forward: (x, w) => mlfw.relu(mlfw.matmul(mlfw.reshape(x, [-1, 8]), w)),
});
