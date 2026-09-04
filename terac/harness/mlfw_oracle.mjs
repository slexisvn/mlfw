import { createRequire } from "node:module";
import { readdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { ramp } from "./values.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const mlfw = require(resolve(here, "../../dist/index.node.js"));

const PROGRAM_SUFFIX = ".mjs";

function programNames() {
  return readdirSync(resolve(here, "programs"))
    .filter((file) => file.endsWith(PROGRAM_SUFFIX))
    .map((file) => file.slice(0, -PROGRAM_SUFFIX.length))
    .sort();
}

async function loadProgram(name) {
  if (!programNames().includes(name)) {
    console.error(
      `mlfw_oracle: unknown program '${name}'; known: ${programNames().join(", ")}`,
    );
    process.exit(1);
  }
  const module = await import(`./programs/${name}${PROGRAM_SUFFIX}`);
  return module.default(mlfw);
}

function parseArgs(argv) {
  const args = {
    name: null, out: null, json: null,
    emitMlir: false, printIr: false, grad: false, list: false,
  };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--out") args.out = argv[++i];
    else if (argv[i] === "--json") args.json = argv[++i];
    else if (argv[i] === "--emit-mlir") args.emitMlir = true;
    else if (argv[i] === "--print-ir") args.printIr = true;
    else if (argv[i] === "--grad") args.grad = true;
    else if (argv[i] === "--list") args.list = true;
    else args.name = argv[i];
  }
  return args;
}

function flatten(value, out) {
  if (Array.isArray(value)) {
    for (const element of value) flatten(element, out);
    return out;
  }
  out.push(Number(value));
  return out;
}

function describe(tensor) {
  return {
    dtype: String(tensor.dtype),
    shape: Array.from(tensor.shape),
    data: flatten(tensor.toArray(), []),
  };
}

function isFloat(tensor) {
  return String(tensor.dtype).startsWith("f");
}

function tensorOf(shape, stride) {
  const values = ramp(shape, stride);
  return shape.length === 0 ? mlfw.scalar(values) : mlfw.tensor(values);
}

function emit(text, path) {
  if (path) writeFileSync(path, text);
  else process.stdout.write(text);
}

const args = parseArgs(process.argv.slice(2));
if (args.list) {
  process.stdout.write(programNames().join("\n") + "\n");
  process.exit(0);
}

const program = await loadProgram(args.name);

const inputs = program.inputs.map(
  (values) => (values instanceof mlfw.Tensor ? values : mlfw.tensor(values)),
);

let mlir = null;
if (args.emitMlir || args.printIr) {
  const module = await mlfw.trace(program.forward, inputs, {
    name: args.name, dynamicShapes: program.dynamicShapes,
  });
  const passes = [
    new mlfw.DecompositionPass(null),
    new mlfw.ExplicitBroadcastPass(),
    new mlfw.IsolateRegionsPass(),
    new mlfw.MaterializeShapesPass(),
  ];
  for (const func of module) {
    for (const pass of passes) pass.run(func);
    if (args.grad) func.setAttr("tera.differentiable", true);
  }
  mlir = mlfw.printModule(module) + "\n";
  if (args.printIr) process.stderr.write(mlir);
}

let record;
if (args.grad) {
  const compiled = mlfw.compileWithBackward(
    { forward: program.forward }, inputs, { dynamicShapes: program.dynamicShapes });
  const output = compiled(...inputs);

  const seed = tensorOf(Array.from(output.shape), 3);
  const gradients = compiled.backward(seed);
  record = {
    entry: `${args.name}_vjp`,
    inputs: [...inputs, seed].map(describe),
    output: inputs
      .map((input, index) => [input, gradients[index]])
      .filter(([input]) => isFloat(input))
      .map(([input, gradient]) => describe(gradient ?? mlfw.zerosLike(input))),
  };
} else {
  const compiled = mlfw.compile(
    { forward: program.forward }, inputs, { dynamicShapes: program.dynamicShapes });
  const output = await compiled(...inputs);
  record = { entry: args.name, inputs: inputs.map(describe), output: describe(output) };
}

const text = JSON.stringify(record, null, 2) + "\n";
if (args.json) writeFileSync(args.json, text);
emit(args.emitMlir ? mlir : text, args.out);
