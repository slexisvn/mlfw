import { ramp } from "../values.mjs";
import dynrnn from "./dynrnn.mjs";

export default (mlfw) => ({
  ...dynrnn(mlfw),
  inputs: [ramp([3, 4, 2], 3), ramp([4, 2], 5), ramp([2, 2], 7), ramp([2, 2], 2)],
});
