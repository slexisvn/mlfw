export default (mlfw) => ({
  inputs: [[1, 2, 3, 4]],
  forward: (x) =>
    mlfw.cond(
      mlfw.gt(x.sum(), mlfw.scalar(0)),
      () => mlfw.mul(x, x),
      () => mlfw.neg(x),
    ),
});
