# mlfw pass visualizer

A teaching tool: type a model, press Run, and step through **every pass the compiler actually runs** —
the IR before and after each one, the graph animating as ops are deleted or swallowed into a fusion
region, the reasons the compiler recorded, and the kernel source that falls out the far end.

```bash
npm install
npm run dev
```

Or from the repo root: `npm run viz`.

`npm run build` here needs `dist/index.d.ts` from the repo root, so run the root `npm run build` first on a
fresh checkout. `VITE_BASE` sets the public path — leave it unset for the root of a domain, or set it to
`/<repo>/` for GitHub Pages. `npm run check:build` verifies the bundle after building; see the note under
Deploying.

## Deploying

`.github/workflows/visualizer.yml` builds the framework, then the visualizer, then publishes it to GitHub
Pages on every push to `main` that touches `src/`, `tools/visualizer/` or the build config. Pages has to be
switched to the "GitHub Actions" source once by hand, in the repository settings.

The workflow runs `check:build` between the build and the upload, and that step is not decoration. The
compiler reads every IR node's type off its class name, so a minifier that renames classes produces a
bundle that compiles nothing and fails only in production, with `no child schema for node type 'en'`. The
check fails the build instead — verified by removing `keepNames` and watching it catch all eight classes.

## What it runs

The real compiler, in the browser. `src/worker/compile.worker.ts` imports the framework straight from
`../../src` and calls the public `compile()` — no reimplementation of the pipeline, no server. Every step
you see comes from a `PassInstrument` (`src/compiler/passes/pass_instrument.ts`) attached to the graph, TIR
and LIR pass managers, which hands the recorder the live IR before and after each pass.

`dist/index.d.ts` is read at startup to give the editor real completions, so `npm run build` in the repo
root keeps the editor's types current. The compiler itself is read from source and needs no build.

## Writing a model

Every framework export is already in scope — there are no imports to write. End with `run(model, inputs)`:

```js
const model = new Sequential(new Linear(8, 16), new ReLU(), new Linear(16, 4));
const x = randn([1, 8]);

run(model, [x]);
```

`model` can be an `nn.Module` or a plain function of the inputs.

## Layout

Three panes on a wide screen: code, passes, and the stage. Below 1180px the stage drops to a full-width row
under the other two, so a graph still has room. Below 760px it becomes one pane at a time, picked with the
`Code · Passes · View` switch in the header — pressing Run jumps you to the pass list, picking a pass jumps
you to the stage. The page itself never scrolls at any size; panes scroll inside themselves.

## Reading the screen

- **Timeline** — the passes that changed something, grouped by phase, with the op count they moved.
  `show all` reveals the quiet ones too; most passes do nothing on most graphs, and that is worth seeing
  once. `⊘` on a row turns that pass off and recompiles, so you can watch what breaks downstream.
- **IR** — the printed IR before and after, as a diff.
- **Graph** — the dataflow DAG. Stepping animates it: what a pass deleted fades out, what survived slides
  to its new place, what the pass created appears last. Node identity is real — `Operation.id` is assigned
  once at construction, so a node that moved is the same op, not a lookalike. An op the pass rebuilt rather
  than kept shows up as `rewritten`, matched by name and operands, and coloured differently because that
  one is a guess.
- **Why** — what the pass is for, plus the decisions it recorded in the trace log.
- **Output** — the generated kernel, nothing else.
- **Result** — the kernel actually run. CPU, WebAssembly and WebGPU execute in the browser and are checked
  against the same model run eagerly: the verdict line says whether the numbers still match, then the
  timings, the inputs, and the compiled and eager outputs side by side. Timing samples adapt — it repeats
  the call until a batch takes long enough to measure, so a kernel that runs in microseconds does not
  report 0ms. CUDA compiles to real source here but cannot be launched without a driver, so it reports the
  kernel and says why it skipped the run.
