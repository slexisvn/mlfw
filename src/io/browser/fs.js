const files = new Map();
const dirs = new Set();

export const fs = {
  readFile(path) {
    if (!files.has(path)) throw new Error('mlfw: file not found in browser memfs: ' + path);
    return files.get(path);
  },
  readBinary(path) {
    if (!files.has(path)) throw new Error('mlfw: file not found in browser memfs: ' + path);
    return files.get(path);
  },
  writeFile(path, data) { files.set(path, data); },
  writeBinary(path, data) { files.set(path, data); },
  appendFile(path, data) { files.set(path, (files.get(path) || '') + data); },
  exists(path) { return files.has(path) || dirs.has(path); },
  mkdir(path) { dirs.add(path); },
  rename(from, to) {
    if (!files.has(from)) throw new Error('mlfw: file not found in browser memfs: ' + from);
    files.set(to, files.get(from));
    files.delete(from);
  },
  readdir(path) {
    const prefix = path.endsWith('/') ? path : path + '/';
    const names = new Set();
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
