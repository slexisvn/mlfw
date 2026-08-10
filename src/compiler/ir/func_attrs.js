export const FuncAttr = Object.freeze({
  CUBLAS_INFO: 'cublas_info',
  CONV_INFO: 'conv_info',
  TENSOR_INTRIN: 'tensor_intrin',
  GPU_REGISTER_BLOCKED: 'gpu_register_blocked',
  PARTITION_TARGET: 'partition_target',
});

export function withFuncAttrs(Base) {
  return class extends Base {
    constructor(...args) {
      super(...args);
      this.attrs = new Map();
    }

    getAttr(key, fallback = null) {
      return this.attrs.has(key) ? this.attrs.get(key) : fallback;
    }

    hasAttr(key) {
      return this.attrs.has(key);
    }

    setAttr(key, value) {
      this.attrs.set(key, value);
      return this;
    }

    removeAttr(key) {
      return this.attrs.delete(key);
    }

    copyAttrsFrom(other) {
      if (other && other.attrs) {
        for (const [k, v] of other.attrs) this.attrs.set(k, v);
      }
      return this;
    }
  };
}
