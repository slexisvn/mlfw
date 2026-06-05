import { Module } from '../module.js';
import { Parameter } from '../parameter.js';
import { embedding } from '../functional/embedding.js';
import { normal_ } from '../init.js';
import { empty } from '../../tensor/factory/creation_ops.js';

export class Embedding extends Module {
  constructor(numEmbeddings, embeddingDim) {
    super();
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.weight = new Parameter(empty([numEmbeddings, embeddingDim]));
    normal_(this.weight);
  }

  forward(indices) {
    return embedding(this.weight, indices);
  }
}
