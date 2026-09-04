import { ramp } from "../values.mjs";
import dynreshape from "./dynreshape.mjs";

export default (mlfw) => ({
  ...dynreshape(mlfw),
  inputs: [ramp([5, 2, 4], 3), ramp([8, 2], 5)],
});
