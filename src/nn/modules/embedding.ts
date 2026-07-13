import { Module } from '../module.js';
import { Parameter } from '../parameter.js';
import { embedding } from '../functional/embedding.js';
import { normal_ } from '../init.js';
import { empty } from '../../tensor/factory/creation_ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';

export class Embedding extends Module {
  numEmbeddings: number;
  embeddingDim: number;
  weight: Parameter;

  constructor(numEmbeddings: number, embeddingDim: number) {
    super();
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.weight = new Parameter(empty([numEmbeddings, embeddingDim]));
    normal_(this.weight);
  }

  forward(indices: Tensor): Tensor {
    return embedding(this.weight, indices);
  }
}
