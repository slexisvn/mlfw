const files = new Map();
const dirs = new Set();

export const fs = {
  readFile(path) {
    if (!files.has(path)) throw new Error('mlfw: file not found in browser memfs: ' + path);
    return files.get(path);
  },
  writeFile(path, data) { files.set(path, data); },
  appendFile(path, data) { files.set(path, (files.get(path) || '') + data); },
  exists(path) { return files.has(path) || dirs.has(path); },
  mkdir(path) { dirs.add(path); },
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

export const memfs = files;
