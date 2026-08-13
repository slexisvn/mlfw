const unavailable = (name: string): never => { throw new Error(`mlfw: CUDA memory (${name}) is not available in the browser`); };

export const copyDeviceToHost = (): never => unavailable('copyDeviceToHost');
export const copyHostToDeviceAsync = (): never => unavailable('copyHostToDeviceAsync');
