import { currentLocation } from 'mlfw/compiler/ir/loc_source.js';
import type { Location } from 'mlfw/compiler/ir/location.js';

type Forward = (...args: unknown[]) => unknown;
type Layer = { forward: Forward; modules?: () => Iterable<Layer> };
type Constructor = new (...args: never[]) => object;

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

export function attributeLayerSites(model: unknown): () => void {
  const undo: (() => void)[] = [];

  for (const layer of layersOf(model)) {
    const site = sites.get(layer);
    if (site === undefined) continue;

    const owned = Object.prototype.hasOwnProperty.call(layer, 'forward');
    const inner = layer.forward;
    layer.forward = (...args: unknown[]) => {
      open.push(site);
      try {
        return inner.apply(layer, args);
      } finally {
        open.pop();
      }
    };
    undo.push(() => {
      if (owned) layer.forward = inner;
      else delete (layer as Partial<Layer>).forward;
    });
  }

  return () => {
    for (const restore of undo) restore();
    undo.length = 0;
    open.length = 0;
  };
}
