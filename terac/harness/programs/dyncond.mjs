export default (mlfw) => ({
  dynamicShapes: [new Set([0])],
  inputs: [[[0.5, 0.25], [0.75, -0.5], [0.125, 0.375]]],
  forward: (x) =>
    mlfw.cond(
      mlfw.gt(x.sum(), mlfw.scalar(0)),
      () => mlfw.mul(x, x),
      () => mlfw.neg(x),
    ),
});
