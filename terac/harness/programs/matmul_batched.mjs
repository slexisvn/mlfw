import { ramp } from "../values.mjs";

// The batch axis takes the third grid dimension, so a block staging the tile
// of the wrong batch is a wrong answer rather than a slow one. Three batches
// of a 64 by 96 product contracted 128 long, which is the same tile grid as
// programs/matmul.mjs with one more dimension above it.
export default (mlfw) => ({
  inputs: [ramp([3, 64, 128], 3), ramp([3, 128, 96], 5)],
  forward: (x, w) => mlfw.matmul(x, w),
});
