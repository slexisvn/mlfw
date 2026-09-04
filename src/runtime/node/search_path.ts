import { delimiter } from 'path';

export const SEARCH_ENV = process.platform === 'win32' ? 'PATH' : 'LD_LIBRARY_PATH';

export function prependSearchPath(dir: string): void {
  const cur = process.env[SEARCH_ENV] || '';
  if (!cur.split(delimiter).includes(dir)) {
    process.env[SEARCH_ENV] = cur ? dir + delimiter + cur : dir;
  }
}
