import { ramp, rowIndex, colIndex } from "../values.mjs";

export default (mlfw) => ({
  inputs: [ramp([4, 3], 3), ramp([3, 4], 5), ramp([4, 2], 7), rowIndex(4, 4), colIndex(4, 4)],
  forward: (q, k, v, rows, cols) => {
    const scores = mlfw.mul(mlfw.matmul(q, k), mlfw.scalar(0.5));
    const masked = mlfw.where(mlfw.ge(rows, cols), scores, mlfw.scalar(-1e9));
    return mlfw.matmul(mlfw.softmax(masked, 1), v);
  },
});
