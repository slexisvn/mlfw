import { Module } from '../module.js';
import { interpolate } from '../functional/upsample.js';
import type { UpsampleMode } from '../functional/upsample.js';
import type { NNTensor } from '../types.js';

type UpsampleOptions = {
  size?: readonly number[] | null;
  scaleFactor?: number | readonly number[] | null;
  mode?: UpsampleMode;
};

export class Upsample extends Module {
  size: readonly number[] | null;
  scaleFactor: number | readonly number[] | null;
  mode: UpsampleMode;

  constructor({ size = null, scaleFactor = null, mode = 'nearest' }: UpsampleOptions = {}) {
    super();
    this.size = size;
    this.scaleFactor = scaleFactor;
    this.mode = mode;
  }

  forward(input: NNTensor): NNTensor {
    return interpolate(input, { size: this.size, scaleFactor: this.scaleFactor, mode: this.mode });
  }
}
