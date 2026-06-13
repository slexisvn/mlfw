export class Deadline {
  constructor(budgetMs = Infinity, clock = null) {
    this._clock = clock || (() => performance.now());
    this._endAt = budgetMs === Infinity || budgetMs == null ? Infinity : this._clock() + budgetMs;
  }

  get expired() {
    return this._clock() >= this._endAt;
  }

  remainingMs() {
    if (this._endAt === Infinity) return Infinity;
    return Math.max(0, this._endAt - this._clock());
  }
}
