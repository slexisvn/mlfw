import type { Deadline } from './budget.js';

export type TuningSession = { runRound(): number; plateaued: boolean };
export type TuningTask = {
  session: TuningSession;
  weight: number;
  rounds: number;
  lastGain: number;
  gainEwma: number;
  stale: number;
  plateaued: boolean;
};
export type SchedulerPolicy = { pick(tasks: readonly TuningTask[]): TuningTask | null };
export type SchedulerConfig = Readonly<{ maxRoundsPerTask?: number; plateauPatience?: number; gainEwmaAlpha?: number }>;

export class GradientSchedulerPolicy implements SchedulerPolicy {
  pick(tasks: readonly TuningTask[]): TuningTask | null {
    const live = tasks.filter(t => !t.plateaued);
    if (live.length === 0) return null;
    const cold = live.filter(t => t.rounds === 0);
    if (cold.length > 0) return cold[0];
    let best: TuningTask | null = null;
    let bestPriority = -Infinity;
    for (const t of live) {
      const gain = t.gainEwma !== undefined ? t.gainEwma : t.lastGain;
      const priority = t.weight * gain;
      if (priority > bestPriority) {
        bestPriority = priority;
        best = t;
      }
    }
    return best;
  }
}

export class TaskScheduler {
  policy: SchedulerPolicy;

  constructor(policy: SchedulerPolicy | null = null) {
    this.policy = policy || new GradientSchedulerPolicy();
  }

  run(tasks: readonly TuningTask[], deadline: Deadline | null, config: SchedulerConfig = {}): void {
    const maxRounds = config.maxRoundsPerTask ?? 8;
    const patience = config.plateauPatience ?? 2;
    const alpha = config.gainEwmaAlpha ?? 0.5;
    for (const t of tasks) {
      t.rounds = 0;
      t.lastGain = 0;
      t.gainEwma = 0;
      t.stale = 0;
      t.plateaued = false;
    }
    while (!(deadline && deadline.expired)) {
      const task = this.policy.pick(tasks);
      if (!task) break;
      const gain = task.session.runRound();
      task.rounds++;
      task.lastGain = gain;
      task.gainEwma = alpha * gain + (1 - alpha) * task.gainEwma;
      if (gain <= 0) task.stale++;
      else task.stale = 0;
      if (task.session.plateaued || task.stale >= patience || task.rounds >= maxRounds) {
        task.plateaued = true;
      }
    }
  }
}
