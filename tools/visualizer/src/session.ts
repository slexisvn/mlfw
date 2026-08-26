import { DEFAULT_OPTIONS } from './protocol.js';
import type { CompileOptions } from './protocol.js';

export type Session = { source: string; exampleId: string; options: CompileOptions };

const STORAGE_KEY = 'mlfw-visualizer-session';

function parse(text: string): Session | null {
  const raw = JSON.parse(text) as Partial<Session>;
  if (typeof raw.source !== 'string') return null;
  return {
    source: raw.source,
    exampleId: typeof raw.exampleId === 'string' ? raw.exampleId : '',
    options: { ...DEFAULT_OPTIONS, ...(raw.options ?? {}) },
  };
}

function attempt(read: () => string | null): Session | null {
  try {
    const text = read();
    return text === null ? null : parse(text);
  } catch {
    return null;
  }
}

export function readSession(): Session | null {
  return attempt(() => localStorage.getItem(STORAGE_KEY));
}

export function writeSession(session: Session): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  } catch {
    return;
  }
}
