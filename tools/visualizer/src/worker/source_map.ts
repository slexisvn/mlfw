import { installLocationSource, stackLocationSource } from 'mlfw/compiler/ir/loc_source.js';
import { locationSites } from 'mlfw/compiler/ir/location.js';
import { openedSite } from './layer_sites.js';
import type { Location } from 'mlfw/compiler/ir/location.js';

export const MODEL_SOURCE_URL = 'mlfw-model.js';

const FRAME = new RegExp(`${MODEL_SOURCE_URL.replace(/[.]/g, '[.]')}:([0-9]+):`);

const STACK_DEPTH = 80;

function isModelFile(file: string): boolean {
  return file.endsWith(MODEL_SOURCE_URL);
}

export function lineFromStack(stack: string | undefined, baseLine: number): number | null {
  if (!stack) return null;
  const match = FRAME.exec(stack);
  if (!match) return null;
  const line = Number(match[1]) - baseLine;
  return line > 0 ? line : null;
}

export function modelLines(loc: Location | null): number[] {
  const lines: number[] = [];
  for (const site of locationSites(loc)) {
    if (isModelFile(site.file) && !lines.includes(site.line)) lines.push(site.line);
  }
  return lines;
}

export function recordSourceLocations(baseLine: number): () => void {
  const fromStack = stackLocationSource({ match: isModelFile, lineOffset: baseLine });
  return installLocationSource(() => fromStack() ?? openedSite(), STACK_DEPTH);
}
