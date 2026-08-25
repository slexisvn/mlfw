import { DEFAULT_OPTIONS } from './protocol.js';
import type { CompileOptions } from './protocol.js';

export type Session = { source: string; exampleId: string; options: CompileOptions };

const STORAGE_KEY = 'mlfw-visualizer-session';
const HASH_PREFIX = '#s=';
const CHUNK = 0x8000;

function toBase64(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = '';
  for (let at = 0; at < bytes.length; at += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(at, at + CHUNK));
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64(encoded: string): string {
  const padded = encoded.replace(/-/g, '+').replace(/_/g, '/');
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let at = 0; at < binary.length; at++) bytes[at] = binary.charCodeAt(at);
  return new TextDecoder().decode(bytes);
}

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
  const hash = location.hash.startsWith(HASH_PREFIX) ? location.hash.slice(HASH_PREFIX.length) : null;
  const shared = hash === null ? null : attempt(() => fromBase64(hash));
  if (shared) return shared;
  return attempt(() => localStorage.getItem(STORAGE_KEY));
}

export function writeSession(session: Session): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  } catch {
    return;
  }
}

export function shareUrl(session: Session): string {
  const { origin, pathname, search } = location;
  return `${origin}${pathname}${search}${HASH_PREFIX}${toBase64(JSON.stringify(session))}`;
}
