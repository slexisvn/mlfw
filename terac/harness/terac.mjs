import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const mlfwRoot = resolve(here, "../..");

export const mlfw = await import(
  pathToFileURL(join(mlfwRoot, "dist/index.node.js")).href
);

const SHLIB_EXT = { win32: ".dll", darwin: ".dylib" }[process.platform] ?? ".so";
const EXE_EXT = process.platform === "win32" ? ".exe" : "";

function normalize(module_, differentiable) {
  const passes = [
    new mlfw.DecompositionPass(null),
    new mlfw.ExplicitBroadcastPass(),
    new mlfw.IsolateRegionsPass(),
    new mlfw.MaterializeShapesPass(),
  ];
  for (const func of module_) {
    for (const pass of passes) pass.run(func);
    if (differentiable) func.setAttr("tera.differentiable", true);
  }
  return module_;
}

function firstExisting(candidates, what) {
  const found = candidates.find((path) => existsSync(path));
  if (found) return found;
  throw new Error(
    `terac: cannot find ${what}; looked in\n  ${candidates.join("\n  ")}`,
  );
}

function discover(options) {
  const build = options.build ?? process.env.TERA_BUILD ?? join(here, "../build");
  const bin = join(build, "bin");

  let llvmBin = options.llvmBin ?? process.env.TERA_LLVM_BIN;
  if (!llvmBin) {
    const siteConfig = join(build, "test/lit.site.cfg.py");
    const match = existsSync(siteConfig)
      && /config\.llvm_tools_dir\s*=\s*lit_config\.substitute\("([^"]+)"\)/
        .exec(readFileSync(siteConfig, "utf8"));
    if (!match) {
      throw new Error(
        "terac: no LLVM bin directory; set TERA_LLVM_BIN or pass { llvmBin }",
      );
    }
    llvmBin = resolve(match[1]);
  }

  const library = (stem) =>
    firstExisting(
      [llvmBin, join(dirname(llvmBin), "lib")].flatMap((directory) =>
        [stem, `lib${stem}`].map((name) => join(directory, name + SHLIB_EXT)),
      ),
      stem,
    );

  return {
    teraOpt: options.teraOpt ?? firstExisting([join(bin, `tera-opt${EXE_EXT}`)], "tera-opt"),
    teraRunner: options.teraRunner ?? firstExisting([join(bin, `tera-runner${EXE_EXT}`)], "tera-runner"),
    llvmBin,
    library,
  };
}

function driver(tools, exe, args) {
  return execFileSync(exe, args, {
    encoding: "utf8",
    maxBuffer: 1 << 28,
    env: { ...process.env, PATH: `${tools.llvmBin}${process.platform === "win32" ? ";" : ":"}${process.env.PATH}` },
  });
}

const flatten = (value, out = []) => (
  Array.isArray(value) ? value.forEach((element) => flatten(element, out)) : out.push(Number(value)), out
);

const asTensor = (value) => (value instanceof mlfw.Tensor ? value : mlfw.tensor(value));

const describe = (tensor) => ({
  dtype: String(tensor.dtype),
  shape: Array.from(tensor.shape),
  data: flatten(tensor.toArray()),
});

const ARRAY_BY_DTYPE = {
  f16: Float32Array, bf16: Float32Array, f32: Float32Array, f64: Float64Array,
  i8: Int8Array, i16: Int16Array, i32: Int32Array, i64: BigInt64Array,
  u8: Uint8Array, u16: Uint16Array, u32: Uint32Array, bool: Uint8Array,
};

function toTensor({ dtype, shape, data }) {
  const Array_ = ARRAY_BY_DTYPE[dtype];
  if (!Array_) throw new Error(`terac: no typed array for dtype '${dtype}'`);
  const buffer = Array_ === BigInt64Array
    ? BigInt64Array.from(data, BigInt)
    : Array_.from(data);
  return mlfw.fromBuffer(buffer, shape, dtype);
}

export async function emit(forward, sample, options = {}) {
  const inputs = sample.map(asTensor);
  const name = options.name ?? forward.name ?? "main";
  const module_ = await mlfw.trace(forward, inputs, {
    name, dynamicShapes: options.dynamicShapes,
  });
  return mlfw.printModule(normalize(module_, options.differentiable)) + "\n";
}

export async function compile(forward, sample, options = {}) {
  const name = options.name ?? forward.name ?? "main";
  const mlir = await emit(forward, sample, { ...options, name, differentiable: options.grad });
  return fromMLIR(mlir, { ...options, entry: options.grad ? `${name}_vjp` : name, autodiff: options.grad });
}

export function grad(forward, sample, options = {}) {
  return compile(forward, sample, { ...options, grad: true });
}

export function fromMLIR(mlir, options = {}) {
  const tools = discover(options);
  const target = options.target ?? "cpu";
  const work = options.workdir ?? mkdtempSync(join(tmpdir(), "terac-"));
  const modulePath = join(work, "module.mlir");
  writeFileSync(modulePath, mlir.endsWith("\n") ? mlir : mlir + "\n");

  let path = modulePath;
  const passes = [...(options.passes ?? []), ...(options.autodiff ? ["--tera-autodiff"] : [])];
  if (passes.length > 0) {
    path = join(work, "opt.mlir");
    driver(tools, tools.teraOpt, [modulePath, ...passes, "-o", path]);
  } else {
    driver(tools, tools.teraOpt, [modulePath, "-o", join(work, "verified.mlir")]);
  }

  const entry = options.entry
    ?? /func\.func\s+@([A-Za-z0-9_.$-]+)/.exec(mlir)?.[1]
    ?? "main";

  const libraries = [
    tools.library("mlir_c_runner_utils"),
    ...(target === "cuda" ? [tools.library("mlir_cuda_runtime")] : []),
  ];

  const invoke = (extra, record) => {
    const args = [
      path, `--entry=${entry}`, `--target=${target}`, `-O=${options.opt ?? 3}`,
      ...libraries.map((library) => `--shared-libs=${library}`),
      ...extra,
    ];
    if (record) {
      const dataPath = join(work, "data.json");
      writeFileSync(dataPath, JSON.stringify(record));
      args.push(`--data=${dataPath}`, `--tolerance=${options.tolerance ?? 1e-5}`);
    }
    return driver(tools, tools.teraRunner, args);
  };

  const call = (...args) => {
    const record = { entry, inputs: args.map(asTensor).map(describe) };
    const answer = JSON.parse(invoke([], record));
    return Array.isArray(answer) ? answer.map(toTensor) : toTensor(answer);
  };

  call.check = (args, expected) => {
    const record = {
      entry,
      inputs: args.map(asTensor).map(describe),
      output: Array.isArray(expected)
        ? expected.map((tensor) => describe(asTensor(tensor)))
        : describe(asTensor(expected)),
    };
    const answer = JSON.parse(invoke(["--check"], record));
    return Array.isArray(answer) ? answer.map(toTensor) : toTensor(answer);
  };

  call.benchmark = ({ args = null, runs = 100, warmup = 3 } = {}) => {
    const text = invoke(
      [`--benchmark=${runs}`, `--warmup=${warmup}`],
      args && { entry, inputs: args.map(asTensor).map(describe) },
    );
    const compileMs = /compile ([\d.]+) ms/.exec(text);
    const timings = /best ([\d.]+) ms, median ([\d.]+) ms, mean ([\d.]+) ms, total ([\d.]+) ms/.exec(text);
    if (!timings) throw new Error(`terac: unreadable benchmark output:\n${text}`);
    return {
      runs,
      compileMs: compileMs ? Number(compileMs[1]) : null,
      bestMs: Number(timings[1]),
      medianMs: Number(timings[2]),
      meanMs: Number(timings[3]),
      totalMs: Number(timings[4]),
    };
  };

  call.mlir = readFileSync(path, "utf8");
  call.entry = entry;
  call.path = path;
  call.workdir = work;
  return call;
}

export async function check(forward, sample, options = {}) {
  const inputs = sample.map(asTensor);
  const fn = await compile(forward, inputs, options);
  if (!options.grad) {
    const expected = await mlfw.compile({ forward }, inputs, {
      dynamicShapes: options.dynamicShapes,
    })(...inputs);
    return fn.check(inputs, expected);
  }

  const compiled = mlfw.compileWithBackward({ forward }, inputs, {
    dynamicShapes: options.dynamicShapes,
  });
  const output = await compiled(...inputs);

  const shape = Array.from(output.shape);
  const seed = options.seed ?? toTensor({
    dtype: String(output.dtype),
    shape,
    data: Array.from(
      { length: shape.reduce((a, b) => a * b, 1) },
      (_, index) => ((index * 3) % 11) / 8 - 0.5,
    ),
  });
  const gradients = compiled.backward(seed);
  const expected = inputs
    .map((input, index) => [input, gradients[index]])
    .filter(([input]) => String(input.dtype).startsWith("f"))
    .map(([input, gradient]) => gradient ?? mlfw.zerosLike(input));
  return fn.check([...inputs, seed], expected);
}
