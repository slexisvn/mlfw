export const FuncAttr = Object.freeze({
  CONST_BUFFERS: 'const_buffers',
  EXTERNAL_CODEGEN: 'external_codegen',
  CONV_INFO: 'conv_info',
  TENSOR_INTRIN: 'tensor_intrin',
  GPU_REGISTER_BLOCKED: 'gpu_register_blocked',
  PARTITION_TARGET: 'partition_target',
});

export type FuncAttrKey = (typeof FuncAttr)[keyof typeof FuncAttr];

export type FuncAttrSource = { attrs?: ReadonlyMap<string, unknown> } | null | undefined;

export interface FuncAttrs {
  attrs: Map<string, unknown>;
  getAttr<T = unknown>(key: string, fallback?: T | null): T | null;
  hasAttr(key: string): boolean;
  setAttr(key: string, value: unknown): this;
  removeAttr(key: string): boolean;
  copyAttrsFrom(other: FuncAttrSource): this;
}

type FuncAttrsBase = new () => object;

export function withFuncAttrs<TBase extends FuncAttrsBase>(Base: TBase): new () => InstanceType<TBase> & FuncAttrs {
  const Mixed = class extends (Base as FuncAttrsBase) implements FuncAttrs {
    attrs: Map<string, unknown> = new Map();

    getAttr<T = unknown>(key: string, fallback: T | null = null): T | null {
      return this.attrs.has(key) ? this.attrs.get(key) as T : fallback;
    }

    hasAttr(key: string): boolean {
      return this.attrs.has(key);
    }

    setAttr(key: string, value: unknown): this {
      this.attrs.set(key, value);
      return this;
    }

    removeAttr(key: string): boolean {
      return this.attrs.delete(key);
    }

    copyAttrsFrom(other: FuncAttrSource): this {
      if (other && other.attrs) {
        for (const [k, v] of other.attrs) this.attrs.set(k, v);
      }
      return this;
    }
  };
  return Mixed as unknown as new () => InstanceType<TBase> & FuncAttrs;
}
