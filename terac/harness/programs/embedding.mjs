import { ramp } from "../values.mjs";

export default (mlfw) => ({
  inputs: [
    ramp([6, 4], 3),
    mlfw.tensor([3, 0, 5, 5, 2, 3], { dtype: "i32" }),
    ramp([6, 4], 7),
    mlfw.tensor(
      [[2, 0, 3], [1, 3, 0], [3, 2, 1], [0, 1, 2], [2, 3, 0], [1, 0, 3]],
      { dtype: "i32" },
    ),
  ],
  forward: (table, rows, weight, columns) => {
    const scaled = mlfw.mul(mlfw.index_select(table, 0, rows), weight);
    const picked = mlfw.gather(scaled, 1, columns);
    return mlfw.scatter_add(scaled, 1, columns, picked);
  },
});
