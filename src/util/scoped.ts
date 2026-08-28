type Thenable = {
  then(onfulfilled: (value: unknown) => unknown, onrejected: (reason: unknown) => unknown): unknown;
};

function isThenable(value: unknown): value is Thenable {
  return typeof value === 'object' && value !== null && typeof (value as Thenable).then === 'function';
}

export function scoped<T>(fn: () => T, restore: () => void): T {
  let result: T;
  try {
    result = fn();
  } catch (error) {
    restore();
    throw error;
  }
  if (isThenable(result)) {
    return result.then(
      value => { restore(); return value; },
      error => { restore(); throw error; },
    ) as T;
  }
  restore();
  return result;
}
