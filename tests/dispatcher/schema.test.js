import { describe, it, expect } from 'vitest';
import { parseSchema, ArgKind } from '../../src/dispatcher/operator_schema.js';

describe('parseSchema', () => {
  it('parses simple single-tensor op', () => {
    const s = parseSchema('neg(Tensor self) -> Tensor', 'mlc');
    expect(s.name).toBe('neg');
    expect(s.namespace).toBe('mlc');
    expect(s.args.length).toBe(1);
    expect(s.args[0].kind).toBe(ArgKind.TENSOR);
    expect(s.args[0].name).toBe('self');
    expect(s.returns[0].kind).toBe(ArgKind.TENSOR);
  });

  it('parses multi-arg op with mixed types', () => {
    const s = parseSchema('slice(Tensor self, int dim, int start, int end) -> Tensor', 'mlc');
    expect(s.args.length).toBe(4);
    expect(s.args[0].kind).toBe(ArgKind.TENSOR);
    expect(s.args[1].kind).toBe(ArgKind.INT);
    expect(s.args[2].kind).toBe(ArgKind.INT);
    expect(s.args[3].kind).toBe(ArgKind.INT);
  });

  it('parses two-tensor op', () => {
    const s = parseSchema('add(Tensor self, Tensor other) -> Tensor', 'mlc');
    expect(s.args.length).toBe(2);
    expect(s.args[0].isTensor).toBe(true);
    expect(s.args[1].isTensor).toBe(true);
  });

  it('parses overload name', () => {
    const s = parseSchema('sum.dim(Tensor self, int dim) -> Tensor');
    expect(s.name).toBe('sum');
    expect(s.overload).toBe('dim');
    expect(s.key()).toBe('mlc::sum.dim');
  });

  it('parses default values', () => {
    const s = parseSchema('norm(Tensor self, float p=2.0) -> Tensor');
    expect(s.args[1].defaultValue).toBe('2.0');
  });

  it('parses no-arg op', () => {
    const s = parseSchema('rand() -> Tensor');
    expect(s.args.length).toBe(0);
  });

  it('computes tensorArgIndices correctly', () => {
    const s = parseSchema('gather(Tensor self, int dim, Tensor index) -> Tensor');
    expect(s.tensorArgIndices).toEqual([0, 2]);
    expect(s.numTensorArgs).toBe(2);
  });

  it('qualifiedName joins namespace and name', () => {
    const s = parseSchema('foo(Tensor x) -> Tensor', 'custom');
    expect(s.qualifiedName()).toBe('custom::foo');
  });

  it('key includes overload when present', () => {
    const s = parseSchema('bar.baz(Tensor x) -> Tensor', 'ns');
    expect(s.key()).toBe('ns::bar.baz');
  });

  it('key omits overload when absent', () => {
    const s = parseSchema('bar(Tensor x) -> Tensor', 'ns');
    expect(s.key()).toBe('ns::bar');
  });

  it('parses tuple return type', () => {
    const s = parseSchema('topk(Tensor self, int k) -> (Tensor, Tensor)');
    expect(s.returns.length).toBe(2);
    expect(s.returns[0].kind).toBe(ArgKind.TENSOR);
    expect(s.returns[1].kind).toBe(ArgKind.TENSOR);
  });

  it('parses out argument with ! marker', () => {
    const s = parseSchema('add.out(Tensor self, Tensor other, Tensor out!) -> Tensor');
    expect(s.args[2].isOut).toBe(true);
    expect(s.args[0].isOut).toBe(false);
  });

  it('parses Tensor[] list arg', () => {
    const s = parseSchema('cat(Tensor[] tensors, int dim) -> Tensor');
    expect(s.args[0].kind).toBe(ArgKind.TENSOR_LIST);
    expect(s.args[0].isTensor).toBe(true);
    expect(s.args[1].isTensor).toBe(false);
  });

  it('parses bool and str arg types', () => {
    const s = parseSchema('to(Tensor self, Device device, bool non_blocking, str layout) -> Tensor');
    expect(s.args[1].kind).toBe(ArgKind.DEVICE);
    expect(s.args[2].kind).toBe(ArgKind.BOOL);
    expect(s.args[3].kind).toBe(ArgKind.STRING);
  });

  it('tensorArgIndices skips non-tensor args in the middle', () => {
    const s = parseSchema('index_select(Tensor self, int dim, Tensor index, bool sorted) -> Tensor');
    expect(s.tensorArgIndices).toEqual([0, 2]);
  });

  it('key is cached after first call', () => {
    const s = parseSchema('cached_key_test(Tensor x) -> Tensor', 'ns');
    const k1 = s.key();
    const k2 = s.key();
    expect(k1).toBe(k2);
    expect(k1).toBe('ns::cached_key_test');
  });

  it('defaults namespace to mlc when not provided', () => {
    const s = parseSchema('my_op(Tensor x) -> Tensor');
    expect(s.namespace).toBe('mlc');
    expect(s.qualifiedName()).toBe('mlc::my_op');
  });
});
