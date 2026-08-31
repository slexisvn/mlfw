import type { PassResultValue } from './pass_result.js';
import type { IRLevelValue } from '../ir/verify.js';

export type InstrumentedPass = { name: string };

export type PassInstrument = {
  runBeforePass?(pass: InstrumentedPass, target: unknown, level: IRLevelValue): void;
  runAfterPass?(pass: InstrumentedPass, target: unknown, level: IRLevelValue, result: PassResultValue | null): void;
};
