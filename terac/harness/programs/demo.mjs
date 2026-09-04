export default (mlfw) => ({
  inputs: [
    [[1, 2, 3, 4], [5, 6, 7, 8]],
    [[1, 0], [0, 1], [1, 1], [0, 0]],
  ],
  forward: (x, w) => mlfw.relu(mlfw.matmul(x, w)).sum(),
});
