import { fileLocation, nameLocation } from './location.js';
import type { Location } from './location.js';

export type LocationSource = () => Location | null;

export type StackFrame = { fn: string | null; file: string; line: number; column: number };

export type StackLocationOpts = Readonly<{
  match: (file: string) => boolean;
  lineOffset?: number;
  frameLimit?: number;
}>;

const ANONYMOUS = '<anonymous>';
const POSITION = /^(.*):(\d+):(\d+)$/;

let _defaultSource: LocationSource | null = null;

export function setDefaultLocationSource(source: LocationSource | null): LocationSource | null {
  const previous = _defaultSource;
  _defaultSource = source;
  return previous;
}

export function currentLocation(): Location | null {
  return _defaultSource === null ? null : _defaultSource();
}

export function parseStackFrame(text: string): StackFrame | null {
  const trimmed = text.trim();
  let fn: string | null = null;
  let position = trimmed;

  if (trimmed.startsWith('at ')) {
    const body = trimmed.slice(3);
    const open = body.lastIndexOf(' (');
    if (open >= 0 && body.endsWith(')')) {
      fn = body.slice(0, open);
      position = body.slice(open + 2, -1);
    } else {
      position = body;
    }
  } else {
    const at = trimmed.indexOf('@');
    if (at >= 0) {
      fn = trimmed.slice(0, at);
      position = trimmed.slice(at + 1);
    }
  }

  const match = POSITION.exec(position);
  if (!match) return null;
  return {
    fn: fn && fn !== ANONYMOUS ? fn : null,
    file: match[1],
    line: Number(match[2]),
    column: Number(match[3]),
  };
}

export function stackLocationSource({ match, lineOffset = 0 }: StackLocationOpts): LocationSource {
  return () => {
    const stack = new Error().stack;
    if (!stack) return null;
    const frames = stack.split('\n');
    for (let i = 1; i < frames.length; i++) {
      const frame = parseStackFrame(frames[i]);
      if (!frame || !match(frame.file)) continue;
      const line = frame.line - lineOffset;
      if (line <= 0) return null;
      const site = fileLocation(frame.file, line, frame.column);
      return frame.fn === null ? site : nameLocation(frame.fn, site);
    }
    return null;
  };
}

type StackLimited = { stackTraceLimit?: number };

export function installStackLocations(opts: StackLocationOpts): () => void {
  const limited = Error as unknown as StackLimited;
  const previousLimit = limited.stackTraceLimit;
  if (opts.frameLimit !== undefined) limited.stackTraceLimit = opts.frameLimit;
  const previousSource = setDefaultLocationSource(stackLocationSource(opts));
  let active = true;

  return () => {
    if (!active) return;
    active = false;
    setDefaultLocationSource(previousSource);
    limited.stackTraceLimit = previousLimit;
  };
}
