import { Library } from '../../dispatcher/library.js';
import { DispatchKey } from '../../dispatcher/dispatch_key.js';
import { registerJITKernels } from '../../dispatcher/jit_dispatch.js';
import { tensorOpSchemas } from '../ops/metadata.js';
import { META_KERNELS } from './meta/meta_ops.js';
import { VIEW_KERNELS } from './view/view_ops.js';
import { COMPOSITE_KERNELS } from './composite/composite_ops.js';

const META = DispatchKey.META;

let _registered = false;

const _KERNEL_DISPATCH_KEYS = [
  DispatchKey.CPU,
  DispatchKey.GPU,
  DispatchKey.WASM,
  DispatchKey.META,
  DispatchKey.CUSTOM_0,
];

export function registerNativeOps() {
  if (_registered) return;
  _registered = true;

  const defLib = new Library('mlc', 'DEF');
  for (const schema of tensorOpSchemas()) {
    defLib.def(schema);
  }

  const implLib = new Library('mlc', 'IMPL');
  for (const [name, fn] of Object.entries(META_KERNELS)) {
    implLib.impl(name, META, fn);
  }
  for (const [name, fn] of Object.entries({ ...VIEW_KERNELS, ...COMPOSITE_KERNELS })) {
    for (const key of _KERNEL_DISPATCH_KEYS) {
      implLib.impl(name, key, fn);
    }
  }

  registerJITKernels();
}
