import { currentLocation } from 'mlfw/compiler/ir/loc_source.js';
import type { Location } from 'mlfw/compiler/ir/location.js';

type Forward = (...args: unknown[]) => unknown;
type Layer = {
  forward: Forward;
  modules?: () => Iterable<Layer>;
  namedModules?: () => Iterable<[string, Layer]>;
  namedChildren?: () => Iterable<[string, Layer]>;
};
type Constructor = new (...args: never[]) => object;

export type LayerOutput = { name: string; kind: string; site: Location | null; outputs: unknown[] };

const sites = new WeakMap<object, Location>();
const open: Location[] = [];

export function openedSite(): Location | null {
  return open.length === 0 ? null : open[open.length - 1];
}

export function isLayerClass(value: unknown): value is Constructor {
  return typeof value === 'function'
    && typeof (value as { prototype?: Partial<Layer> }).prototype?.forward === 'function';
}

export function siteRecording(ctor: Constructor): Constructor {
  return new Proxy(ctor, {
    construct(target, args, newTarget) {
      const site = currentLocation();
      const layer = Reflect.construct(target, args, newTarget) as object;
      if (site !== null) sites.set(layer, site);
      return layer;
    },
  });
}

function* layersOf(model: unknown): Generator<Layer> {
  const root = model as Layer | null;
  if (!root) return;
  if (typeof root.modules === 'function') yield* root.modules();
  else if (typeof root.forward === 'function') yield root;
}

function wrapForward(layer: Layer, around: (inner: Forward) => Forward): () => void {
  const owned = Object.prototype.hasOwnProperty.call(layer, 'forward');
  const inner = layer.forward;
  layer.forward = around(inner);
  return () => {
    if (owned) layer.forward = inner;
    else delete (layer as Partial<Layer>).forward;
  };
}

export function attributeLayerSites(model: unknown): () => void {
  const undo: (() => void)[] = [];

  for (const layer of layersOf(model)) {
    const site = sites.get(layer);
    if (site === undefined) continue;

    undo.push(wrapForward(layer, inner => (...args) => {
      open.push(site);
      try {
        return inner.apply(layer, args);
      } finally {
        open.pop();
      }
    }));
  }

  return () => {
    for (const restore of undo) restore();
    undo.length = 0;
    open.length = 0;
  };
}

function tensorsOf(value: unknown, into: unknown[]): unknown[] {
  if (Array.isArray(value)) {
    for (const item of value) tensorsOf(item, into);
  } else if (value !== null && typeof value === 'object' && Array.isArray((value as Layer & { shape?: unknown }).shape)) {
    into.push(value);
  }
  return into;
}

function isLeaf(layer: Layer): boolean {
  return typeof layer.namedChildren !== 'function' || [...layer.namedChildren()].length === 0;
}

function* namedLeaves(model: unknown): Generator<[string, Layer]> {
  const root = model as Layer | null;
  if (!root || typeof root.namedModules !== 'function') return;
  for (const [name, layer] of root.namedModules()) {
    if (typeof layer.forward === 'function' && isLeaf(layer)) yield [name === '' ? 'model' : name, layer];
  }
}

export function instrumentLayers(model: unknown): { rows: LayerOutput[]; stop: () => void } {
  const rows: LayerOutput[] = [];
  const undo: (() => void)[] = [];

  for (const [name, layer] of namedLeaves(model)) {
    undo.push(wrapForward(layer, inner => (...args) => {
      const output = inner.apply(layer, args);
      rows.push({
        name,
        kind: layer.constructor.name,
        site: sites.get(layer) ?? null,
        outputs: tensorsOf(output, []),
      });
      return output;
    }));
  }

  return {
    rows,
    stop: () => {
      for (const restore of undo) restore();
      undo.length = 0;
    },
  };
}
