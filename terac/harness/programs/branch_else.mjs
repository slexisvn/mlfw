import branch from "./branch.mjs";

export default (mlfw) => ({ ...branch(mlfw), inputs: [[-1, -2, -3, -4]] });
