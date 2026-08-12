export type Clock = () => number;

export class Deadline {
  private _clock: Clock;
  private _endAt: number;

  constructor(budgetMs: number | null = Infinity, clock: Clock | null = null) {
    this._clock = clock || (() => performance.now());
    this._endAt = budgetMs === Infinity || budgetMs == null ? Infinity : this._clock() + budgetMs;
  }

  get expired(): boolean {
    return this._clock() >= this._endAt;
  }

  remainingMs(): number {
    if (this._endAt === Infinity) return Infinity;
    return Math.max(0, this._endAt - this._clock());
  }
}
