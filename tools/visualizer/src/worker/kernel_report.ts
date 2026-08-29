import { audit } from 'mlfw-tests/_utils/kernel_audit.js';
import { lintKernel } from 'mlfw-tests/_utils/kernel_lint.js';
import type { Kernel, KernelReport } from '../protocol.js';

type Audit = {
  lines: number;
  loops: number;
  tempBuffers: number;
  arithmeticNoise: string[];
  extent1Loops: number;
  zeroInits: number;
  boundsChecks: number;
  modulos: number;
  truncs: number;
};

type Lint = { lang: string; issues: { kind: string; detail: string }[] };

const LONGEST_LINE_ALERT = 4000;

function longestLine(source: string): number {
  let longest = 0;
  let start = 0;
  for (let i = 0; i <= source.length; i++) {
    if (i !== source.length && source[i] !== '\n') continue;
    if (i - start > longest) longest = i - start;
    start = i + 1;
  }
  return longest;
}

export function kernelReports(kernels: readonly Kernel[]): KernelReport[] {
  return kernels.map(kernel => {
    const measured = audit(kernel.source, kernel.name) as Audit;
    const linted = lintKernel(kernel.source) as Lint;
    const widest = longestLine(kernel.source);

    return {
      name: kernel.name,
      language: linted.lang,
      bytes: kernel.source.length,
      lines: measured.lines,
      longestLine: widest,
      loops: measured.loops,
      tempBuffers: measured.tempBuffers,
      boundsChecks: measured.boundsChecks,
      modulos: measured.modulos,
      arithmeticNoise: measured.arithmeticNoise,
      extent1Loops: measured.extent1Loops,
      zeroInits: measured.zeroInits,
      issues: linted.issues,
      blownUp: widest >= LONGEST_LINE_ALERT,
    };
  });
}
