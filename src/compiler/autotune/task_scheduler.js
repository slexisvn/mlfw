export class GradientSchedulerPolicy {
  pick(tasks) {
    const live = tasks.filter(t => !t.plateaued);
    if (live.length === 0) return null;
    const cold = live.filter(t => t.rounds === 0);
    if (cold.length > 0) return cold[0];
    let best = null;
    let bestPriority = -Infinity;
    for (const t of live) {
      const priority = t.weight * t.lastGain;
      if (priority > bestPriority) {
        bestPriority = priority;
        best = t;
      }
    }
    return best;
  }
}

export class TaskScheduler {
  constructor(policy = null) {
    this.policy = policy || new GradientSchedulerPolicy();
  }

  run(tasks, deadline, config = {}) {
    const maxRounds = config.maxRoundsPerTask ?? 8;
    const patience = config.plateauPatience ?? 2;
    for (const t of tasks) {
      t.rounds = 0;
      t.lastGain = 0;
      t.stale = 0;
      t.plateaued = false;
    }
    while (!(deadline && deadline.expired)) {
      const task = this.policy.pick(tasks);
      if (!task) break;
      const gain = task.session.runRound();
      task.rounds++;
      task.lastGain = gain;
      if (gain <= 0) task.stale++;
      else task.stale = 0;
      if (task.session.plateaued || task.stale >= patience || task.rounds >= maxRounds) {
        task.plateaued = true;
      }
    }
  }
}
