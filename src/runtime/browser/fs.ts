import type { MemFs } from '../io.js';

const files = new Map<string, string | Uint8Array>();
const dirs = new Set<string>();

export const fs: MemFs = {
  readFile(path) {
    if (!files.has(path)) throw new Error('mlfw: file not found in browser memfs: ' + path);
    return files.get(path) as string;
  },
  readBinary(path) {
    if (!files.has(path)) throw new Error('mlfw: file not found in browser memfs: ' + path);
    return files.get(path) as Uint8Array;
  },
  writeFile(path, data) { files.set(path, data); },
  writeBinary(path, data) { files.set(path, data); },
  appendFile(path, data) { files.set(path, ((files.get(path) || '') as string) + data); },
  exists(path) { return files.has(path) || dirs.has(path); },
  mkdir(path) { dirs.add(path); },
  rename(from, to) {
    if (!files.has(from)) throw new Error('mlfw: file not found in browser memfs: ' + from);
    files.set(to, files.get(from) as string | Uint8Array);
    files.delete(from);
  },
  readdir(path) {
    const prefix = path.endsWith('/') ? path : path + '/';
    const names = new Set<string>();
    for (const key of [...files.keys(), ...dirs]) {
      if (key.startsWith(prefix)) {
        const rest = key.slice(prefix.length).split('/')[0];
        if (rest) names.add(rest);
      }
    }
    return [...names];
  },
  remove(path) { files.delete(path); },
};
