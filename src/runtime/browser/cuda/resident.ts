const unavailable = (name: string): never => { throw new Error(`mlfw: CUDA resident memory (${name}) is not available in the browser`); };

export const setEagerDeferred = (): never => unavailable('setEagerDeferred');
export const deviceBufferForInput = (): never => unavailable('deviceBufferForInput');
export const deviceBufferDptr = (): never => unavailable('deviceBufferDptr');
export const pinResident = (): never => unavailable('pinResident');
export const clearCapturePins = (): never => unavailable('clearCapturePins');
