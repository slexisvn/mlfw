const unavailable = (name) => { throw new Error(`mlfw: CUDA resident memory (${name}) is not available in the browser`); };

export const setEagerDeferred = () => unavailable('setEagerDeferred');
export const deviceBufferForInput = () => unavailable('deviceBufferForInput');
export const deviceBufferDptr = () => unavailable('deviceBufferDptr');
export const pinResident = () => unavailable('pinResident');
export const clearCapturePins = () => unavailable('clearCapturePins');
