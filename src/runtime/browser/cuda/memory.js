const unavailable = (name) => { throw new Error(`mlfw: CUDA memory (${name}) is not available in the browser`); };

export const copyDeviceToHost = () => unavailable('copyDeviceToHost');
export const copyHostToDeviceAsync = () => unavailable('copyHostToDeviceAsync');
