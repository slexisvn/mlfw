const unavailable = (name) => { throw new Error(`mlfw: CUDA eager-graph (${name}) is not available in the browser`); };

export const beginEagerCapture = () => unavailable('beginEagerCapture');
export const endEagerCapture = () => unavailable('endEagerCapture');
export const replay = () => unavailable('replay');
export const syncStream = () => unavailable('syncStream');
