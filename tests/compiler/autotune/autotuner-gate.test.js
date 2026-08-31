import { describe, it, expect } from 'vitest';
import { Autotuner } from '../../../src/compiler/autotune/autotuner.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import { PrimFunc, SeqNode } from '../../../src/compiler/ir/tensor/nodes.js';
import { FuncAttr } from '../../../src/compiler/ir/func_attrs.js';

function makeAutotuner(onWarning) {
  const at = new Autotuner(CPUTarget(), { onWarning });
  at._funcName = 'f';
  return at;
}

const mkFunc = (name, registerBlocked = false) => {
  const f = new PrimFunc(name, [], new SeqNode([]));
  if (registerBlocked) f.setAttr(FuncAttr.GPU_REGISTER_BLOCKED, true);
  return f;
};
const PRISTINE = () => mkFunc('original');

describe('Autotuner._applyBestSchedule validity gate', () => {
  it('never adopts a tuned schedule the validator rejected, even when the fallback cannot be built', () => {
    const warnings = [];
    const at = makeAutotuner((w) => warnings.push(w));
    at._buildTunedSchedule = () => mkFunc('tuned');
    at._scheduleIsValid = () => false; 
    at._buildDefaultSchedule = () => null;

    const primFunc = PRISTINE();
    const result = at._applyBestSchedule(primFunc, new Map());

    expect(result).toBeNull();
    expect(primFunc.name).toBe('original');
    const stages = warnings.map((w) => w.stage);
    expect(stages).toContain('tuned-schedule-invalid');
    expect(stages).toContain('no-valid-schedule');
  });

  it('falls back to a valid default schedule when the tuned schedule is invalid', () => {
    const at = makeAutotuner(null);
    let adopted = null;
    at._buildTunedSchedule = () => mkFunc('tuned');
    at._buildDefaultSchedule = () => mkFunc('fallback');
    at._scheduleIsValid = (f) => f.name === 'fallback';
    at._adoptSchedule = (_target, src) => { adopted = src; };

    const result = at._applyBestSchedule(PRISTINE(), new Map());
    expect(result).not.toBeNull();
    expect(adopted.name).toBe('fallback');
  });

  it('adopts the tuned schedule when it is valid', () => {
    const at = makeAutotuner(null);
    let adopted = null;
    at._buildTunedSchedule = () => mkFunc('tuned');
    at._scheduleIsValid = () => true;
    at._adoptSchedule = (_target, src) => { adopted = src; };

    at._applyBestSchedule(PRISTINE(), new Map());
    expect(adopted.name).toBe('tuned');
  });

  it('keeps the deterministic GPU baseline over a cost-model-only tuned schedule', () => {
    const warnings = [];
    const at = makeAutotuner((w) => warnings.push(w));
    at.config.measurer = null;
    let adopted = null;
    at._buildTunedSchedule = () => mkFunc('tuned');
    at._buildDefaultSchedule = () => mkFunc('baseline', true);
    at._scheduleIsValid = () => true;
    at._adoptSchedule = (_target, src) => { adopted = src; };

    at._applyBestSchedule(PRISTINE(), new Map());
    expect(adopted.name).toBe('baseline');
    expect(warnings.map((w) => w.stage)).toContain('baseline-preferred');
  });

  it('adopts a hardware-measured register-block tuned schedule over the deterministic baseline', () => {
    const at = makeAutotuner(null);
    at.config.measurer = {};
    let adopted = null;
    at._buildTunedSchedule = () => mkFunc('tuned', true);
    at._buildDefaultSchedule = () => mkFunc('baseline', true);
    at._scheduleIsValid = () => true;
    at._adoptSchedule = (_target, src) => { adopted = src; };

    at._applyBestSchedule(PRISTINE(), new Map());
    expect(adopted.name).toBe('tuned');
  });

  it('does not let a measured generic tuned schedule displace a register-block baseline', () => {
    const at = makeAutotuner(null);
    at.config.measurer = {};
    let adopted = null;
    at._buildTunedSchedule = () => mkFunc('tuned');
    at._buildDefaultSchedule = () => mkFunc('baseline', true);
    at._scheduleIsValid = () => true;
    at._adoptSchedule = (_target, src) => { adopted = src; };

    at._applyBestSchedule(PRISTINE(), new Map());
    expect(adopted.name).toBe('baseline');
  });

  it('does not adopt the default schedule if it is also invalid', () => {
    const at = makeAutotuner(null);
    let adoptCalled = false;
    at._buildTunedSchedule = () => mkFunc('tuned');
    at._buildDefaultSchedule = () => mkFunc('fallback');
    at._scheduleIsValid = () => false;
    at._adoptSchedule = () => { adoptCalled = true; };

    const result = at._applyBestSchedule(PRISTINE(), new Map());
    expect(result).toBeNull();
    expect(adoptCalled).toBe(false);
  });
});

describe('Autotuner._warn diagnostics channel', () => {
  it('routes swallowed errors to both onWarning and trace.warn', () => {
    const warnings = [];
    const traceCalls = [];
    const at = new Autotuner(CPUTarget(), { onWarning: (w) => warnings.push(w) }, {
      warn: (phase, funcName, message) => traceCalls.push({ phase, funcName, message })
    });
    at._funcName = 'myfunc';

    at._warn('build-default-schedule', 'blk0', new Error('boom'));

    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatchObject({ stage: 'build-default-schedule', func: 'myfunc', block: 'blk0', message: 'boom' });
    expect(traceCalls).toHaveLength(1);
    expect(traceCalls[0].phase).toBe('autotune');
    expect(traceCalls[0].message).toContain('boom');
  });

  it('a throwing onWarning sink never breaks the tuner', () => {
    const at = new Autotuner(CPUTarget(), { onWarning: () => { throw new Error('sink failed'); } });
    at._funcName = 'f';
    expect(() => at._warn('apply-tuned-block', 'b', new Error('x'))).not.toThrow();
  });
});
