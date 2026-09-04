import { afterEach, describe, expect, it, vi } from 'vitest';
import { delimiter } from 'path';
import { SEARCH_ENV, prependSearchPath } from '../../src/runtime/node/search_path.js';

afterEach(() => vi.unstubAllEnvs());

describe('native library search path', () => {
  it.each([undefined, ''])('initializes a missing or empty path (%j)', (initial) => {
    vi.stubEnv(SEARCH_ENV, initial);
    prependSearchPath('/new/lib');
    expect(process.env[SEARCH_ENV]).toBe('/new/lib');
  });

  it('prepends once, retains order and compares complete path entries', () => {
    vi.stubEnv(SEARCH_ENV, ['/new/library', '/existing/lib'].join(delimiter));
    prependSearchPath('/new/lib');
    prependSearchPath('/new/lib');
    expect(process.env[SEARCH_ENV]).toBe(['/new/lib', '/new/library', '/existing/lib'].join(delimiter));
    prependSearchPath('/existing/lib');
    expect(process.env[SEARCH_ENV]).toBe(['/new/lib', '/new/library', '/existing/lib'].join(delimiter));
  });
});
