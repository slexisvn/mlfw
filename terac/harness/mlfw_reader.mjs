import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const mlfw = require(resolve(here, "../../dist/internals.node.js"));

const ARRAY_BY_DTYPE = {
  f32: Float32Array, f64: Float64Array,
  i8: Int8Array, i16: Int16Array, i32: Int32Array, i64: BigInt64Array,
  u8: Uint8Array, u16: Uint16Array, u32: Uint32Array,
  bool: Uint8Array,
};

function parseArgs(argv) {
  const args = { module: null, data: null, entry: null, tolerance: 1e-5, printIr: false };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--data") args.data = argv[++i];
    else if (argv[i] === "--entry") args.entry = argv[++i];
    else if (argv[i] === "--tolerance") args.tolerance = Number(argv[++i]);
    else if (argv[i] === "--print-ir") args.printIr = true;
    else args.module = argv[i];
  }
  return args;
}

function fail(message) {
  console.error(`mlfw_reader: ${message}`);
  process.exit(1);
}

function bufferFor(type, values) {
  const Array_ = ARRAY_BY_DTYPE[String(type.dtype)];
  if (!Array_) fail(`no typed array for dtype '${type.dtype}'`);
  const count = type.shape.reduce((a, b) => a * b, 1);
  const buffer = new Array_(count);
  if (!values) return buffer;
  if (values.length !== count) {
    fail(`a tensor of ${type} needs ${count} elements but the record holds ${values.length}`);
  }
  for (let i = 0; i < count; i++) buffer[i] = Array_ === BigInt64Array ? BigInt(values[i]) : values[i];
  return buffer;
}

const args = parseArgs(process.argv.slice(2));
if (!args.module || !args.data) fail("usage: mlfw_reader.mjs <module.mlir> --data FILE");

const record = JSON.parse(readFileSync(args.data, "utf8"));
const module_ = mlfw.parseModule(readFileSync(args.module, "utf8"));
if (args.printIr) process.stderr.write(mlfw.printModule(module_) + "\n");

const entry = args.entry ?? record.entry;
const func = module_.getFunction(entry);
if (!func) fail(`no function named ${entry}`);

if (func.inputTypes.length !== record.inputs.length) {
  fail(`${entry} takes ${func.inputTypes.length} inputs but the record holds ${record.inputs.length}`);
}

if ([...func.inputTypes, ...func.outputTypes].some((type) => type.shape.includes(-1))) {
  new mlfw.ShapeRefinementPass(record.inputs.map((input) => input.shape)).run(func);
  if (args.printIr) process.stderr.write(mlfw.printModule(module_) + "\n");
}

const inputs = func.inputTypes.map((type, i) => bufferFor(type, record.inputs[i].data));
const outputs = func.outputTypes.map((type) => bufferFor(type, null));
mlfw.compileGraph(func, mlfw.CPUTarget()).run(entry, ...inputs, ...outputs);

const expected = Array.isArray(record.output) ? record.output : [record.output];
if (expected.length !== outputs.length) {
  fail(`${entry} returns ${outputs.length} tensors but the record expects ${expected.length}`);
}

let disagreed = 0;
for (let i = 0; i < outputs.length; i++) {
  const want = expected[i].data;
  const got = outputs[i];
  if (want.length !== got.length) {
    fail(`result ${i} holds ${got.length} elements but the record expects ${want.length}`);
  }
  for (let k = 0; k < got.length; k++) {
    const a = Number(got[k]);
    const b = Number(want[k]);
    if (Math.abs(a - b) <= args.tolerance * Math.max(1, Math.abs(b))) continue;
    if (++disagreed <= 16) {
      console.error(`mlfw_reader: result ${i} element ${k} is ${a}, expected ${b}`);
    }
  }
}

if (disagreed > 0) fail(`${disagreed} elements disagree`);
process.stdout.write(JSON.stringify(
  outputs.map((buffer, i) => ({
    dtype: String(func.outputTypes[i].dtype),
    shape: Array.from(func.outputTypes[i].shape),
    data: Array.from(buffer, Number),
  })), null, 2) + "\n");
