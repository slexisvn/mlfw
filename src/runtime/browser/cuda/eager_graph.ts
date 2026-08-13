const unavailable = (name: string): never => { throw new Error(`mlfw: CUDA eager-graph (${name}) is not available in the browser`); };

export const beginEagerCapture = (): never => unavailable('beginEagerCapture');
export const endEagerCapture = (): never => unavailable('endEagerCapture');
export const replay = (): never => unavailable('replay');
export const syncStream = (): never => unavailable('syncStream');
