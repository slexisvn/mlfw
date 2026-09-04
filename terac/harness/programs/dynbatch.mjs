export default (mlfw) => ({

  dynamicShapes: [new Set([0]), null],
  inputs: [
    [[0.5, 0.25, -0.5, 0.125], [-0.25, 0.75, 0.5, -0.125], [0.125, -0.5, 0.25, 0.5]],
    [[1, 0], [0, 1], [1, 1], [0, 0]],
  ],
  forward: (x, w) => mlfw.relu(mlfw.matmul(x, w)),
});
