export const LocationKind = Object.freeze({
  FILE: 'file',
  NAME: 'name',
  CALLSITE: 'callsite',
  FUSED: 'fused',
});

export type LocationKindValue = (typeof LocationKind)[keyof typeof LocationKind];

export class FileLocation {
  readonly kind: typeof LocationKind.FILE = LocationKind.FILE;
  file: string;
  line: number;
  column: number;

  constructor(file: string, line: number, column: number) {
    this.file = file;
    this.line = line;
    this.column = column;
  }
}

export class NameLocation {
  readonly kind: typeof LocationKind.NAME = LocationKind.NAME;
  name: string;
  child: Location | null;

  constructor(name: string, child: Location | null) {
    this.name = name;
    this.child = child;
  }
}

export class CallSiteLocation {
  readonly kind: typeof LocationKind.CALLSITE = LocationKind.CALLSITE;
  callee: Location;
  caller: Location;

  constructor(callee: Location, caller: Location) {
    this.callee = callee;
    this.caller = caller;
  }
}

export class FusedLocation {
  readonly kind: typeof LocationKind.FUSED = LocationKind.FUSED;
  tag: string | null;
  locations: readonly Location[];

  constructor(tag: string | null, locations: readonly Location[]) {
    this.tag = tag;
    this.locations = locations;
  }
}

export type Location = FileLocation | NameLocation | CallSiteLocation | FusedLocation;

const _fileCache = new Map<string, FileLocation>();

export function fileLocation(file: string, line: number, column = 0): FileLocation {
  const key = `${file}:${line}:${column}`;
  const cached = _fileCache.get(key);
  if (cached) return cached;
  const loc = new FileLocation(file, line, column);
  _fileCache.set(key, loc);
  return loc;
}

export function nameLocation(name: string, child: Location | null = null): NameLocation {
  return new NameLocation(name, child);
}

export function callSiteLocation(callee: Location, caller: Location): CallSiteLocation {
  return new CallSiteLocation(callee, caller);
}

function collectFusible(loc: Location, tag: string | null, out: Location[], seen: Set<string>): void {
  if (loc.kind === LocationKind.FUSED && loc.tag === tag) {
    for (const inner of loc.locations) collectFusible(inner, tag, out, seen);
    return;
  }
  const key = formatLocation(loc);
  if (seen.has(key)) return;
  seen.add(key);
  out.push(loc);
}

export function fuseLocations(locations: Iterable<Location | null | undefined>, tag: string | null = null): Location | null {
  const flat: Location[] = [];
  const seen = new Set<string>();
  for (const loc of locations) {
    if (loc) collectFusible(loc, tag, flat, seen);
  }
  if (flat.length === 0) return null;
  if (flat.length === 1 && tag === null) return flat[0];
  return new FusedLocation(tag, flat);
}

export function formatLocation(loc: Location | null): string {
  if (!loc) return 'unknown';
  switch (loc.kind) {
    case LocationKind.FILE:
      return `${JSON.stringify(loc.file)}:${loc.line}:${loc.column}`;
    case LocationKind.NAME:
      return loc.child ? `${JSON.stringify(loc.name)}(${formatLocation(loc.child)})` : JSON.stringify(loc.name);
    case LocationKind.CALLSITE:
      return `callsite(${formatLocation(loc.callee)} at ${formatLocation(loc.caller)})`;
    default: {
      const inner = loc.locations.map(formatLocation).join(', ');
      return loc.tag === null ? `fused[${inner}]` : `fused<${JSON.stringify(loc.tag)}>[${inner}]`;
    }
  }
}

function walkLeaves(loc: Location, out: FileLocation[], seen: Set<FileLocation>): void {
  switch (loc.kind) {
    case LocationKind.FILE:
      if (!seen.has(loc)) { seen.add(loc); out.push(loc); }
      return;
    case LocationKind.NAME:
      if (loc.child) walkLeaves(loc.child, out, seen);
      return;
    case LocationKind.CALLSITE:
      walkLeaves(loc.callee, out, seen);
      walkLeaves(loc.caller, out, seen);
      return;
    default:
      for (const inner of loc.locations) walkLeaves(inner, out, seen);
  }
}

export function locationSites(loc: Location | null): readonly FileLocation[] {
  if (!loc) return [];
  const out: FileLocation[] = [];
  walkLeaves(loc, out, new Set());
  return out;
}

export function primarySite(loc: Location | null): FileLocation | null {
  const sites = locationSites(loc);
  return sites.length > 0 ? sites[0] : null;
}

export function locationNames(loc: Location | null): readonly string[] {
  if (!loc) return [];
  const out: string[] = [];
  const stack: Location[] = [loc];
  while (stack.length > 0) {
    const cur = stack.pop() as Location;
    switch (cur.kind) {
      case LocationKind.NAME:
        out.push(cur.name);
        if (cur.child) stack.push(cur.child);
        break;
      case LocationKind.CALLSITE:
        stack.push(cur.caller, cur.callee);
        break;
      case LocationKind.FUSED:
        for (let i = cur.locations.length - 1; i >= 0; i--) stack.push(cur.locations[i]);
        break;
      default:
        break;
    }
  }
  return out;
}

export class LocationParseError extends Error {}

class LocationReader {
  private _text: string;
  private _pos: number;

  constructor(text: string) {
    this._text = text;
    this._pos = 0;
  }

  get position(): number { return this._pos; }

  skipSpaces(): void {
    while (this._pos < this._text.length && this._text[this._pos] === ' ') this._pos++;
  }

  peek(): string { return this._text[this._pos] || ''; }

  eat(token: string): boolean {
    this.skipSpaces();
    if (!this._text.startsWith(token, this._pos)) return false;
    this._pos += token.length;
    return true;
  }

  expect(token: string): void {
    if (!this.eat(token)) throw new LocationParseError(`expected '${token}' at offset ${this._pos} in location`);
  }

  readQuoted(): string {
    this.skipSpaces();
    if (this.peek() !== '"') throw new LocationParseError(`expected a quoted string at offset ${this._pos} in location`);
    let end = this._pos + 1;
    while (end < this._text.length) {
      if (this._text[end] === '\\') { end += 2; continue; }
      if (this._text[end] === '"') break;
      end++;
    }
    if (end >= this._text.length) throw new LocationParseError('unterminated quoted string in location');
    const raw = this._text.slice(this._pos, end + 1);
    this._pos = end + 1;
    return JSON.parse(raw) as string;
  }

  readInt(): number {
    this.skipSpaces();
    const start = this._pos;
    while (this._pos < this._text.length && this._text[this._pos] >= '0' && this._text[this._pos] <= '9') this._pos++;
    if (this._pos === start) throw new LocationParseError(`expected an integer at offset ${start} in location`);
    return Number(this._text.slice(start, this._pos));
  }

  readLocation(): Location {
    this.skipSpaces();
    if (this.eat('callsite(')) {
      const callee = this.readLocation();
      this.expect('at');
      const caller = this.readLocation();
      this.expect(')');
      return callSiteLocation(callee, caller);
    }
    if (this.eat('fused')) {
      let tag: string | null = null;
      if (this.eat('<')) {
        tag = this.readQuoted();
        this.expect('>');
      }
      this.expect('[');
      const parts: Location[] = [];
      if (!this.eat(']')) {
        do { parts.push(this.readLocation()); } while (this.eat(','));
        this.expect(']');
      }
      return new FusedLocation(tag, parts);
    }
    const text = this.readQuoted();
    if (this.eat('(')) {
      const child = this.readLocation();
      this.expect(')');
      return nameLocation(text, child);
    }
    if (this.eat(':')) {
      const line = this.readInt();
      this.expect(':');
      return fileLocation(text, line, this.readInt());
    }
    return nameLocation(text, null);
  }
}

export function parseLocation(text: string): Location {
  const reader = new LocationReader(text);
  const loc = reader.readLocation();
  reader.skipSpaces();
  if (reader.position !== text.length) {
    throw new LocationParseError(`trailing characters in location '${text}'`);
  }
  return loc;
}
