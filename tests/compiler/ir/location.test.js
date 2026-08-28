import { describe, it, expect } from 'vitest';
import {
  LocationKind, FusedLocation,
  fileLocation, nameLocation, callSiteLocation, fuseLocations,
  formatLocation, parseLocation, locationSites, locationNames, primarySite,
  LocationParseError,
} from '../../../src/compiler/ir/location.js';
import { parseStackFrame, stackLocationSource, installLocationSource, installStackLocations, currentLocation } from '../../../src/compiler/ir/loc_source.js';

describe('fileLocation', () => {
  it('returns the same object for the same file, line and column', () => {
    const a = fileLocation('model.js', 12, 4);
    const b = fileLocation('model.js', 12, 4);
    const c = fileLocation('model.js', 12, 5);
    expect(a).toBe(b);
    expect(a).not.toBe(c);
    expect(a.line).toBe(12);
    expect(a.column).toBe(4);
  });

  it('defaults the column to zero', () => {
    expect(fileLocation('model.js', 3).column).toBe(0);
  });
});

describe('fuseLocations', () => {
  it('returns null for an empty or all-unknown list', () => {
    expect(fuseLocations([])).toBeNull();
    expect(fuseLocations([null, undefined])).toBeNull();
  });

  it('returns the single location unchanged when untagged', () => {
    const loc = fileLocation('model.js', 7, 1);
    expect(fuseLocations([loc, null])).toBe(loc);
  });

  it('keeps a tagged single location wrapped so the tag survives', () => {
    const loc = fileLocation('model.js', 7, 1);
    const fused = fuseLocations([loc], 'fusion');
    expect(fused).toBeInstanceOf(FusedLocation);
    expect(fused.tag).toBe('fusion');
    expect(fused.locations).toEqual([loc]);
  });

  it('drops duplicate sites', () => {
    const a = fileLocation('model.js', 7, 1);
    const b = fileLocation('model.js', 9, 1);
    const fused = fuseLocations([a, b, a, b]);
    expect(fused.locations).toEqual([a, b]);
  });

  it('flattens a nested fusion carrying the same tag', () => {
    const a = fileLocation('model.js', 1, 0);
    const b = fileLocation('model.js', 2, 0);
    const c = fileLocation('model.js', 3, 0);
    const inner = fuseLocations([a, b]);
    const outer = fuseLocations([inner, c]);
    expect(outer.locations).toEqual([a, b, c]);
  });

  it('keeps a nested fusion whose tag differs', () => {
    const a = fileLocation('model.js', 1, 0);
    const b = fileLocation('model.js', 2, 0);
    const inner = fuseLocations([a, b], 'fusion');
    const outer = fuseLocations([inner, fileLocation('model.js', 3, 0)]);
    expect(outer.locations[0]).toBe(inner);
    expect(outer.locations.length).toBe(2);
  });
});

describe('locationSites', () => {
  it('is empty for an unknown location', () => {
    expect(locationSites(null)).toEqual([]);
    expect(primarySite(null)).toBeNull();
  });

  it('reaches the file location under a name wrapper', () => {
    const site = fileLocation('model.js', 5, 2);
    expect(locationSites(nameLocation('forward', site))).toEqual([site]);
    expect(primarySite(nameLocation('grad', nameLocation('forward', site)))).toBe(site);
  });

  it('collects both halves of a call site in callee-then-caller order', () => {
    const callee = fileLocation('layer.js', 4, 0);
    const caller = fileLocation('model.js', 20, 0);
    expect(locationSites(callSiteLocation(callee, caller))).toEqual([callee, caller]);
  });

  it('collects every distinct site of a fused location once', () => {
    const a = fileLocation('model.js', 1, 0);
    const b = fileLocation('model.js', 2, 0);
    const fused = new FusedLocation('fusion', [nameLocation('n', a), b, a]);
    expect(locationSites(fused)).toEqual([a, b]);
  });
});

describe('locationNames', () => {
  it('lists the names attached along the location tree', () => {
    const site = fileLocation('model.js', 5, 2);
    const loc = fuseLocations([nameLocation('grad', nameLocation('forward', site)), nameLocation('bias', site)]);
    expect(locationNames(loc)).toEqual(['grad', 'forward', 'bias']);
  });

  it('is empty when no name was ever attached', () => {
    expect(locationNames(fileLocation('model.js', 1, 0))).toEqual([]);
  });
});

describe('formatLocation', () => {
  it('prints unknown for a missing location', () => {
    expect(formatLocation(null)).toBe('unknown');
  });

  it('prints file, line and column', () => {
    expect(formatLocation(fileLocation('model.js', 12, 4))).toBe('"model.js":12:4');
  });

  it('prints a name with and without a child', () => {
    expect(formatLocation(nameLocation('forward', null))).toBe('"forward"');
    expect(formatLocation(nameLocation('forward', fileLocation('m.js', 1, 2)))).toBe('"forward"("m.js":1:2)');
  });

  it('prints a call site and a tagged fusion', () => {
    const callee = fileLocation('a.js', 1, 0);
    const caller = fileLocation('b.js', 2, 0);
    expect(formatLocation(callSiteLocation(callee, caller))).toBe('callsite("a.js":1:0 at "b.js":2:0)');
    expect(formatLocation(new FusedLocation('fusion', [callee, caller]))).toBe('fused<"fusion">["a.js":1:0, "b.js":2:0]');
    expect(formatLocation(new FusedLocation(null, [callee, caller]))).toBe('fused["a.js":1:0, "b.js":2:0]');
  });
});

describe('parseLocation', () => {
  const roundTrip = (loc) => parseLocation(formatLocation(loc));

  it('round-trips a file location', () => {
    const loc = fileLocation('src/model.js', 12, 4);
    expect(roundTrip(loc)).toBe(loc);
  });

  it('round-trips names, call sites and nested fusions', () => {
    const site = fileLocation('model.js', 3, 7);
    const named = nameLocation('grad', nameLocation('forward', site));
    const call = callSiteLocation(named, fileLocation('train.js', 40, 2));
    const fused = new FusedLocation('fusion', [call, site, nameLocation('bare', null)]);
    for (const loc of [named, call, fused]) {
      expect(formatLocation(roundTrip(loc))).toBe(formatLocation(loc));
    }
  });

  it('round-trips a file whose name contains a colon', () => {
    const loc = fileLocation('http://host:8080/model.js', 9, 1);
    expect(roundTrip(loc)).toBe(loc);
  });

  it('rejects trailing characters and unterminated strings', () => {
    expect(() => parseLocation('"model.js":1:2 junk')).toThrow(LocationParseError);
    expect(() => parseLocation('"model.js')).toThrow(LocationParseError);
  });

  it('reads an empty fusion as a fusion with no sites', () => {
    const loc = parseLocation('fused[]');
    expect(loc.kind).toBe(LocationKind.FUSED);
    expect(loc.locations).toEqual([]);
  });
});

describe('parseStackFrame', () => {
  it('reads a V8 frame that names a function', () => {
    expect(parseStackFrame('    at Linear.forward (/app/model.js:12:9)')).toEqual({
      fn: 'Linear.forward', file: '/app/model.js', line: 12, column: 9,
    });
  });

  it('reads a V8 frame with no function name', () => {
    expect(parseStackFrame('    at /app/model.js:3:1')).toEqual({
      fn: null, file: '/app/model.js', line: 3, column: 1,
    });
  });

  it('drops an anonymous function name', () => {
    expect(parseStackFrame('    at <anonymous> (/app/model.js:3:1)').fn).toBeNull();
  });

  it('reads a SpiderMonkey style frame', () => {
    expect(parseStackFrame('forward@/app/model.js:8:3')).toEqual({
      fn: 'forward', file: '/app/model.js', line: 8, column: 3,
    });
  });

  it('keeps a url with a port as part of the file', () => {
    expect(parseStackFrame('    at http://localhost:5173/model.js:4:2').file).toBe('http://localhost:5173/model.js');
  });

  it('returns null for a frame with no position', () => {
    expect(parseStackFrame('Error: something went wrong')).toBeNull();
  });
});

describe('stackLocationSource', () => {
  it('reports the frame of the caller that matches the filter', () => {
    const source = stackLocationSource({ match: file => file.includes('location.test.js') });
    const loc = source();
    expect(loc).not.toBeNull();
    const site = primarySite(loc);
    expect(site.file).toContain('location.test.js');
    expect(site.line).toBeGreaterThan(0);
  });

  it('returns null when no frame matches', () => {
    expect(stackLocationSource({ match: () => false })()).toBeNull();
  });

  it('subtracts the line offset and gives up above the first line', () => {
    const shifted = stackLocationSource({ match: file => file.includes('location.test.js'), lineOffset: 2 });
    const plain = stackLocationSource({ match: file => file.includes('location.test.js') });
    expect(primarySite(plain()).line - primarySite(shifted()).line).toBe(2);
    expect(stackLocationSource({ match: file => file.includes('location.test.js'), lineOffset: 100000 })()).toBeNull();
  });
});

describe('installLocationSource', () => {
  it('installs any source and restores the previous one', () => {
    const outer = fileLocation('outer.js', 1, 1);
    const inner = fileLocation('inner.js', 2, 1);
    const restoreOuter = installLocationSource(() => outer);
    try {
      const restoreInner = installLocationSource(() => inner);
      expect(primarySite(currentLocation()).file).toBe('inner.js');
      restoreInner();
      expect(primarySite(currentLocation()).file).toBe('outer.js');
    } finally {
      restoreOuter();
    }
    expect(currentLocation()).toBeNull();
  });

  it('raises the stack limit only while installed', () => {
    const before = Error.stackTraceLimit;
    const restore = installLocationSource(() => null, before + 25);
    expect(Error.stackTraceLimit).toBe(before + 25);
    restore();
    expect(Error.stackTraceLimit).toBe(before);
  });
});

describe('installStackLocations', () => {
  it('makes currentLocation report the caller and restores the previous source', () => {
    expect(currentLocation()).toBeNull();
    const uninstall = installStackLocations({ match: file => file.includes('location.test.js'), frameLimit: 40 });
    try {
      const site = primarySite(currentLocation());
      expect(site.file).toContain('location.test.js');
    } finally {
      uninstall();
    }
    expect(currentLocation()).toBeNull();
  });

  it('ignores a second uninstall', () => {
    const uninstall = installStackLocations({ match: () => true });
    uninstall();
    uninstall();
    expect(currentLocation()).toBeNull();
  });
});
