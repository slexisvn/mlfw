import { build } from "esbuild";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { copyFileSync, readFileSync, rmSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const entry = resolve(root, "src/index.ts");
const outdir = resolve(root, "dist");

const pkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
const external = Object.keys(pkg.dependencies ?? {});

rmSync(outdir, { recursive: true, force: true });

const NODE_STUBS = {
  webgpu: `
export const create = () => {
  throw new Error('mlfw: the "webgpu" npm package is not available in the browser; use navigator.gpu instead');
};
export const globals = undefined;
export default { create, globals };
`,
};

const browserStubPlugin = {
  name: "mlfw-browser-stubs",
  setup(b) {
    const keys = Object.keys(NODE_STUBS);
    const filter = new RegExp(
      "^(" +
        keys.map((k) => k.replace(/[/\\^$*+?.()|[\]{}]/g, "\\$&")).join("|") +
        ")$",
    );
    b.onResolve({ filter }, (a) => ({ path: a.path, namespace: "mlfw-stub" }));
    b.onLoad({ filter: /.*/, namespace: "mlfw-stub" }, (a) => ({
      contents: NODE_STUBS[a.path],
      loader: "js",
    }));
  },
};

await Promise.all([
  build({
    bundle: true,
    platform: "node",
    minify: true,
    keepNames: true,
    format: "esm",
    external,
    entryPoints: [entry],
    outfile: resolve(outdir, "index.node.js"),
  }),
  build({
    bundle: true,
    platform: "browser",
    target: ["es2022"],
    plugins: [browserStubPlugin],
    define: { "process.env.NODE_ENV": '"production"' },
    keepNames: true,
    logLevel: "info",
    format: "esm",
    minify: true,
    entryPoints: [entry],
    outfile: resolve(outdir, "index.browser.js"),
  }),
]);
