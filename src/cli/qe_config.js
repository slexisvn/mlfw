if (typeof process !== 'undefined' && process.env && process.env.QE_WASM_MIN_CHUNK === undefined) {
  process.env.QE_WASM_MIN_CHUNK = '1';
}
