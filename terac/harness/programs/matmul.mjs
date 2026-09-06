import { ramp } from "../values.mjs";

// Shapes chosen so that terac stages the operands in shared memory: every
// extent is a multiple of the warp, which is what `-tera-tile-contraction-to-
// shared` needs before it will cut a contraction at all. 64 by 96 out of a
// contraction 128 long is six blocks of threads staging four tiles each.
//
// The second contraction reads the first one's result, so what a block wrote
// out of its registers is what the next kernel stages back in.
export default (mlfw) => ({
  inputs: [ramp([64, 128], 3), ramp([128, 96], 5), ramp([96, 64], 7)],
  forward: (x, w, v) => mlfw.matmul(mlfw.matmul(x, w), v),
});
