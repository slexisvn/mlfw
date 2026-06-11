import { writeFileSync, appendFileSync, readFileSync, existsSync, mkdirSync, readdirSync, unlinkSync } from 'node:fs';
import { resolve } from 'node:path';

export const fs = {
  readFile(path) { return readFileSync(resolve(path), 'utf8'); },
  writeFile(path, data) { writeFileSync(path, data); },
  appendFile(path, data) { appendFileSync(path, data); },
  exists(path) { return existsSync(path); },
  mkdir(path) { mkdirSync(path, { recursive: true }); },
  readdir(path) { return readdirSync(path); },
  remove(path) { unlinkSync(path); },
};
