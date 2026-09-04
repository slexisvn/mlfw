export default (mlfw) => ({
  inputs: [
    mlfw.tensor([[10, 20, 30], [40, 50, 60]], { dtype: "i32" }),
    mlfw.tensor([[4, 3, 2], [1, 1, 1]], { dtype: "i32" }),
    mlfw.tensor([[0.5, 0.25, 0.75], [-0.5, 0.125, -0.25]]),
  ],
  forward: (a, b, w) => {
    const quotient = mlfw.div(a, b);
    const larger = mlfw.maximum(quotient, b);
    const signed = mlfw.where(mlfw.lt(a, b), larger, mlfw.neg(larger));
    return mlfw.mul(signed, w);
  },
});
