import { ramp } from "../values.mjs";

export default (mlfw) => {
  const conv = new mlfw.nn.Conv2d(2, 3, 3, { bias: false, stride: 2, padding: 1 });
  const largest = new mlfw.nn.MaxPool2d(2);
  const average = new mlfw.nn.AvgPool2d(2);

  return {
    inputs: [ramp([1, 2, 8, 8], 3), ramp([3, 2, 3, 3], 5), ramp([1, 3, 2, 2], 7)],
    forward: (x, kernel, weight) => {
      conv.weight = kernel;
      const features = conv.forward(x);
      const pooled = mlfw.add(largest.forward(features), average.forward(features));
      return mlfw.mul(pooled, weight);
    },
  };
};
