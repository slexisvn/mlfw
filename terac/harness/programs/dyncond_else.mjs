import dyncond from "./dyncond.mjs";

export default (mlfw) => ({
  ...dyncond(mlfw),
  inputs: [[[-0.5, -0.25], [-0.75, 0.5], [-0.125, -0.375], [-0.25, -0.125], [-0.5, -0.25]]],
});
