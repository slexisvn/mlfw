import { writeFileSync, appendFileSync, readFileSync, existsSync, mkdirSync, readdirSync, unlinkSync, renameSync } from 'node:fs';
import { resolve } from 'node:path';
import type { MemFs } from '../io.js';

export const fs: MemFs = {
  readFile(path) { return readFileSync(resolve(path), 'utf8'); },
  readBinary(path) { return readFileSync(resolve(path)); },
  writeFile(path, data) { writeFileSync(path, data); },
  writeBinary(path, data) { writeFileSync(resolve(path), data); },
  appendFile(path, data) { appendFileSync(path, data); },
  exists(path) { return existsSync(path); },
  mkdir(path) { mkdirSync(path, { recursive: true }); },
  readdir(path) { return readdirSync(path); },
  remove(path) { unlinkSync(path); },
  rename(from, to) { renameSync(resolve(from), resolve(to)); },
};
