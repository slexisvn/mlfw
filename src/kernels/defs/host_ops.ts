import { Library } from '../../dispatcher/library.js';

export type HostOpFamily = Readonly<{
  devices: readonly string[];
  schemas: readonly string[];
}>;

export function defineHostOps(family: HostOpFamily): () => void {
  let defined = false;
  return () => {
    if (defined) return;
    defined = true;
    const defLib = new Library('mlc', 'DEF');
    for (const schema of family.schemas) defLib.def(schema, family.devices);
  };
}
