import { describe, it, expect } from 'vitest';
import {
  LibrarySelector, LibraryCall,
  createGPULibrarySelector, createCPULibrarySelector
} from '../../../src/backend/library_selector.js';
import { GPUTarget, CPUTarget } from '../../../src/backend/target.js';

describe('LibrarySelector.register and select', () => {
  it('selects a registered library call matching op, dtype, and constraints', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', { dtypes: ['f32', 'f64'] });

    const call = selector.select('dot', [64, 64], 'f32');
    expect(call).toBeInstanceOf(LibraryCall);
    expect(call.opName).toBe('dot');
    expect(call.libraryName).toBe('cublas');
    expect(call.functionName).toBe('cublasSgemm');
  });

  it('returns null for unregistered op', () => {
    const selector = new LibrarySelector(GPUTarget());
    expect(selector.select('softmax', [64], 'f32')).toBeNull();
  });

  it('returns null when dtype does not match', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', { dtypes: ['f32'] });
    expect(selector.select('dot', [64, 64], 'i32')).toBeNull();
  });

  it('returns null when no library func mapping exists for dtype', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', { dtypes: ['f32', 'i8'] });
    expect(selector.select('dot', [64, 64], 'i8')).toBeNull();
  });

  it('registers multiple entries and selects first match', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', { dtypes: ['f32'] });
    selector.register('dot', 'custom_lib', { dtypes: ['f32'] });

    const call = selector.select('dot', [64, 64], 'f32');
    expect(call.libraryName).toBe('cublas');
  });
});

describe('LibrarySelector._matchesConstraints', () => {
  it('respects minElements constraint', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', { dtypes: ['f32'], minElements: 1000 });

    expect(selector.select('dot', [10, 10], 'f32')).toBeNull();
    expect(selector.select('dot', [100, 100], 'f32')).not.toBeNull();
  });

  it('computes numel as product of all shape dims', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', { dtypes: ['f32'], minElements: 100 });

    expect(selector.select('dot', [5, 5, 5], 'f32')).not.toBeNull();
    expect(selector.select('dot', [3, 3, 3], 'f32')).toBeNull();
  });

  it('respects maxRank constraint', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', { dtypes: ['f32'], maxRank: 2 });

    expect(selector.select('dot', [4, 4], 'f32')).not.toBeNull();
    expect(selector.select('dot', [2, 3, 4], 'f32')).toBeNull();
  });

  it('passes when no constraints are specified', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', {});
    expect(selector.select('dot', [4, 4], 'f32')).not.toBeNull();
  });

  it('checks all constraints together', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', { dtypes: ['f32'], minElements: 50, maxRank: 3 });

    expect(selector.select('dot', [10, 10], 'f32')).not.toBeNull();
    expect(selector.select('dot', [3, 3], 'f32')).toBeNull();
    expect(selector.select('dot', [2, 5, 5, 2], 'f32')).toBeNull();
    expect(selector.select('dot', [10, 10], 'f64')).toBeNull();
  });
});

describe('LibrarySelector.shouldUseLibrary', () => {
  it('returns true when a library call can be selected', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', { dtypes: ['f32'] });
    expect(selector.shouldUseLibrary('dot', [64, 64], 'f32')).toBe(true);
  });

  it('returns false when no library call matches', () => {
    const selector = new LibrarySelector(GPUTarget());
    selector.register('dot', 'cublas', { dtypes: ['f32'] });
    expect(selector.shouldUseLibrary('dot', [64, 64], 'i32')).toBe(false);
    expect(selector.shouldUseLibrary('conv', [64, 64], 'f32')).toBe(false);
  });
});

describe('createGPULibrarySelector', () => {
  const selector = createGPULibrarySelector(GPUTarget());

  it('selects cublasSgemm for f32 dot', () => {
    const call = selector.select('dot', [64, 64], 'f32');
    expect(call.libraryName).toBe('cublas');
    expect(call.functionName).toBe('cublasSgemm');
  });

  it('selects cublasDgemm for f64 dot', () => {
    const call = selector.select('dot', [64, 64], 'f64');
    expect(call.functionName).toBe('cublasDgemm');
  });

  it('selects cublasHgemm for f16 dot', () => {
    const call = selector.select('dot', [64, 64], 'f16');
    expect(call.functionName).toBe('cublasHgemm');
  });

  it('selects cudnnConvolutionForward for f32 conv', () => {
    const call = selector.select('conv', [1, 3, 32, 32], 'f32');
    expect(call.libraryName).toBe('cudnn');
    expect(call.functionName).toBe('cudnnConvolutionForward');
  });

  it('selects cudnnConvolutionForward for f16 conv', () => {
    const call = selector.select('conv', [1, 3, 32, 32], 'f16');
    expect(call.functionName).toBe('cudnnConvolutionForward');
  });

  it('returns null for i32 dot (no cublas mapping)', () => {
    expect(selector.select('dot', [64, 64], 'i32')).toBeNull();
  });

  it('returns null for f64 conv (no cudnn mapping)', () => {
    expect(selector.select('conv', [1, 3, 32, 32], 'f64')).toBeNull();
  });

  it('returns null for unregistered ops', () => {
    expect(selector.select('relu', [64], 'f32')).toBeNull();
    expect(selector.select('softmax', [64], 'f32')).toBeNull();
  });
});

describe('createCPULibrarySelector', () => {
  const selector = createCPULibrarySelector(CPUTarget());

  it('selects sgemm for f32 dot with enough elements', () => {
    const call = selector.select('dot', [64, 64], 'f32');
    expect(call.libraryName).toBe('blas');
    expect(call.functionName).toBe('sgemm');
  });

  it('selects dgemm for f64 dot', () => {
    const call = selector.select('dot', [64, 64], 'f64');
    expect(call.functionName).toBe('dgemm');
  });

  it('returns null for small dot (below minElements=64)', () => {
    expect(selector.select('dot', [7, 7], 'f32')).toBeNull();
  });

  it('selects for exactly 64 elements', () => {
    expect(selector.select('dot', [8, 8], 'f32')).not.toBeNull();
  });

  it('returns null for f16 dot (no blas mapping)', () => {
    expect(selector.select('dot', [64, 64], 'f16')).toBeNull();
  });
});

describe('LibraryCall properties', () => {
  it('stores all provided fields', () => {
    const constraints = { dtypes: ['f32'], minElements: 100 };
    const call = new LibraryCall('dot', 'cublas', 'cublasSgemm', constraints);
    expect(call.opName).toBe('dot');
    expect(call.libraryName).toBe('cublas');
    expect(call.functionName).toBe('cublasSgemm');
    expect(call.constraints).toBe(constraints);
  });
});

describe('GPU vs CPU library selection differences', () => {
  it('GPU has no minElements for dot, CPU has minElements=64', () => {
    const gpuSel = createGPULibrarySelector(GPUTarget());
    const cpuSel = createCPULibrarySelector(CPUTarget());

    expect(gpuSel.select('dot', [2, 2], 'f32')).not.toBeNull();
    expect(cpuSel.select('dot', [2, 2], 'f32')).toBeNull();
  });

  it('GPU supports f16 dot, CPU does not', () => {
    const gpuSel = createGPULibrarySelector(GPUTarget());
    const cpuSel = createCPULibrarySelector(CPUTarget());

    expect(gpuSel.select('dot', [64, 64], 'f16')).not.toBeNull();
    expect(cpuSel.select('dot', [64, 64], 'f16')).toBeNull();
  });

  it('GPU supports conv, CPU does not', () => {
    const gpuSel = createGPULibrarySelector(GPUTarget());
    const cpuSel = createCPULibrarySelector(CPUTarget());

    expect(gpuSel.select('conv', [1, 3, 32, 32], 'f32')).not.toBeNull();
    expect(cpuSel.select('conv', [1, 3, 32, 32], 'f32')).toBeNull();
  });
});
