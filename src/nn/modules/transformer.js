import { Module } from '../module.js';
import { Linear } from './linear.js';
import { LayerNorm } from './normalization.js';
import { Dropout } from './dropout.js';
import { ModuleList } from './container.js';
import { scaled_dot_product_attention } from '../functional/attention.js';
import * as ops from '../../tensor/ops/ops.js';
import { full } from '../../tensor/factory/creation_ops.js';
import { empty } from '../../tensor/factory/creation_ops.js';
import { relu, gelu } from '../functional/activation.js';

function _getActivation(name) {
  if (name === 'gelu') return gelu;
  return relu;
}

export class MultiheadAttention extends Module {
  constructor(embedDim, numHeads, dropout = 0, bias = true, kdim = null, vdim = null, batchFirst = true) {
    super();
    this.embedDim = embedDim;
    this.numHeads = numHeads;
    this.headDim = Math.floor(embedDim / numHeads);
    this.batchFirst = batchFirst;
    this.dropout = dropout;
    this.qProj = new Linear(embedDim, embedDim, bias);
    this.kProj = new Linear(kdim ?? embedDim, embedDim, bias);
    this.vProj = new Linear(vdim ?? embedDim, embedDim, bias);
    this.outProj = new Linear(embedDim, embedDim, bias);
  }

  forward(query, key, value, attnMask = null, keyPaddingMask = null, isCausal = false) {
    if (!this.batchFirst) {
      query = query.transpose(0, 1);
      key = key.transpose(0, 1);
      value = value.transpose(0, 1);
    }

    const B = query.shape[0];
    const L = query.shape[1];
    const S = key.shape[1];

    let q = this.qProj.forward(query);
    let k = this.kProj.forward(key);
    let v = this.vProj.forward(value);

    q = q.reshape([B, L, this.numHeads, this.headDim]).permute([0, 2, 1, 3]);
    k = k.reshape([B, S, this.numHeads, this.headDim]).permute([0, 2, 1, 3]);
    v = v.reshape([B, S, this.numHeads, this.headDim]).permute([0, 2, 1, 3]);

    if (keyPaddingMask) {
      const negInf = full(keyPaddingMask.shape, -Infinity);
      const zero = full(keyPaddingMask.shape, 0);
      let paddingMask = ops.where(keyPaddingMask, negInf, zero);
      paddingMask = paddingMask.unsqueeze(1).unsqueeze(2);
      attnMask = attnMask ? ops.add(attnMask, paddingMask) : paddingMask;
    }

    let attnOut = scaled_dot_product_attention(q, k, v, attnMask, this.dropout, isCausal, this.training);

    attnOut = attnOut.permute([0, 2, 1, 3]).reshape([B, L, this.embedDim]);
    let output = this.outProj.forward(attnOut);

    if (!this.batchFirst) {
      output = output.transpose(0, 1);
    }

    return output;
  }
}

export class TransformerEncoderLayer extends Module {
  constructor(dModel, nhead, dimFeedforward = 2048, dropout = 0.1, activation = 'relu', layerNormEps = 1e-5, batchFirst = true, normFirst = false) {
    super();
    this.selfAttn = new MultiheadAttention(dModel, nhead, dropout, true, null, null, batchFirst);
    this.linear1 = new Linear(dModel, dimFeedforward);
    this.linear2 = new Linear(dimFeedforward, dModel);
    this.norm1 = new LayerNorm(dModel, layerNormEps);
    this.norm2 = new LayerNorm(dModel, layerNormEps);
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
    this.dropoutFFN = new Dropout(dropout);
    this._activation = _getActivation(activation);
    this._activationName = activation;
    this.normFirst = normFirst;
    this._dModel = dModel;
    this._nhead = nhead;
    this._dimFeedforward = dimFeedforward;
    this._dropout = dropout;
    this._layerNormEps = layerNormEps;
    this._batchFirst = batchFirst;
  }

  forward(src, srcMask = null, srcKeyPaddingMask = null, isCausal = false) {
    if (this.normFirst) {
      return this._forwardPreNorm(src, srcMask, srcKeyPaddingMask, isCausal);
    }
    return this._forwardPostNorm(src, srcMask, srcKeyPaddingMask, isCausal);
  }

  _forwardPostNorm(src, srcMask, srcKeyPaddingMask, isCausal) {
    let x = this.selfAttn.forward(src, src, src, srcMask, srcKeyPaddingMask, isCausal);
    x = this.norm1.forward(ops.add(src, this.dropout1.forward(x)));
    let ff = this._activation(this.linear1.forward(x));
    ff = this.linear2.forward(this.dropoutFFN.forward(ff));
    x = this.norm2.forward(ops.add(x, this.dropout2.forward(ff)));
    return x;
  }

  _forwardPreNorm(src, srcMask, srcKeyPaddingMask, isCausal) {
    let normed = this.norm1.forward(src);
    let x = this.selfAttn.forward(normed, normed, normed, srcMask, srcKeyPaddingMask, isCausal);
    x = ops.add(src, this.dropout1.forward(x));
    let ff = this._activation(this.linear1.forward(this.norm2.forward(x)));
    ff = this.linear2.forward(this.dropoutFFN.forward(ff));
    x = ops.add(x, this.dropout2.forward(ff));
    return x;
  }
}

export class TransformerDecoderLayer extends Module {
  constructor(dModel, nhead, dimFeedforward = 2048, dropout = 0.1, activation = 'relu', layerNormEps = 1e-5, batchFirst = true, normFirst = false) {
    super();
    this.selfAttn = new MultiheadAttention(dModel, nhead, dropout, true, null, null, batchFirst);
    this.crossAttn = new MultiheadAttention(dModel, nhead, dropout, true, null, null, batchFirst);
    this.linear1 = new Linear(dModel, dimFeedforward);
    this.linear2 = new Linear(dimFeedforward, dModel);
    this.norm1 = new LayerNorm(dModel, layerNormEps);
    this.norm2 = new LayerNorm(dModel, layerNormEps);
    this.norm3 = new LayerNorm(dModel, layerNormEps);
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
    this.dropout3 = new Dropout(dropout);
    this.dropoutFFN = new Dropout(dropout);
    this._activation = _getActivation(activation);
    this._activationName = activation;
    this.normFirst = normFirst;
    this._dModel = dModel;
    this._nhead = nhead;
    this._dimFeedforward = dimFeedforward;
    this._dropout = dropout;
    this._layerNormEps = layerNormEps;
    this._batchFirst = batchFirst;
  }

  forward(tgt, memory, tgtMask = null, memoryMask = null, tgtKeyPaddingMask = null, memoryKeyPaddingMask = null, isCausal = false) {
    if (this.normFirst) {
      return this._forwardPreNorm(tgt, memory, tgtMask, memoryMask, tgtKeyPaddingMask, memoryKeyPaddingMask, isCausal);
    }
    return this._forwardPostNorm(tgt, memory, tgtMask, memoryMask, tgtKeyPaddingMask, memoryKeyPaddingMask, isCausal);
  }

  _forwardPostNorm(tgt, memory, tgtMask, memoryMask, tgtKeyPaddingMask, memoryKeyPaddingMask, isCausal) {
    let x = this.selfAttn.forward(tgt, tgt, tgt, tgtMask, tgtKeyPaddingMask, isCausal);
    x = this.norm1.forward(ops.add(tgt, this.dropout1.forward(x)));
    let x2 = this.crossAttn.forward(x, memory, memory, memoryMask, memoryKeyPaddingMask);
    x = this.norm2.forward(ops.add(x, this.dropout2.forward(x2)));
    let ff = this._activation(this.linear1.forward(x));
    ff = this.linear2.forward(this.dropoutFFN.forward(ff));
    x = this.norm3.forward(ops.add(x, this.dropout3.forward(ff)));
    return x;
  }

  _forwardPreNorm(tgt, memory, tgtMask, memoryMask, tgtKeyPaddingMask, memoryKeyPaddingMask, isCausal) {
    let normed = this.norm1.forward(tgt);
    let x = this.selfAttn.forward(normed, normed, normed, tgtMask, tgtKeyPaddingMask, isCausal);
    x = ops.add(tgt, this.dropout1.forward(x));
    let normed2 = this.norm2.forward(x);
    let x2 = this.crossAttn.forward(normed2, memory, memory, memoryMask, memoryKeyPaddingMask);
    x = ops.add(x, this.dropout2.forward(x2));
    let ff = this._activation(this.linear1.forward(this.norm3.forward(x)));
    ff = this.linear2.forward(this.dropoutFFN.forward(ff));
    x = ops.add(x, this.dropout3.forward(ff));
    return x;
  }
}

function _cloneEncoderLayer(layer) {
  return new TransformerEncoderLayer(
    layer._dModel, layer._nhead, layer._dimFeedforward,
    layer._dropout, layer._activationName, layer._layerNormEps,
    layer._batchFirst, layer.normFirst
  );
}

function _cloneDecoderLayer(layer) {
  return new TransformerDecoderLayer(
    layer._dModel, layer._nhead, layer._dimFeedforward,
    layer._dropout, layer._activationName, layer._layerNormEps,
    layer._batchFirst, layer.normFirst
  );
}

export class TransformerEncoder extends Module {
  constructor(encoderLayer, numLayers, norm = null) {
    super();
    this.layers = new ModuleList(
      Array.from({ length: numLayers }, () => _cloneEncoderLayer(encoderLayer))
    );
    this.norm = norm;
  }

  forward(src, mask = null, srcKeyPaddingMask = null, isCausal = false) {
    let output = src;
    for (const layer of this.layers) {
      output = layer.forward(output, mask, srcKeyPaddingMask, isCausal);
    }
    if (this.norm) output = this.norm.forward(output);
    return output;
  }
}

export class TransformerDecoder extends Module {
  constructor(decoderLayer, numLayers, norm = null) {
    super();
    this.layers = new ModuleList(
      Array.from({ length: numLayers }, () => _cloneDecoderLayer(decoderLayer))
    );
    this.norm = norm;
  }

  forward(tgt, memory, tgtMask = null, memoryMask = null, tgtKeyPaddingMask = null, memoryKeyPaddingMask = null, isCausal = false) {
    let output = tgt;
    for (const layer of this.layers) {
      output = layer.forward(output, memory, tgtMask, memoryMask, tgtKeyPaddingMask, memoryKeyPaddingMask, isCausal);
    }
    if (this.norm) output = this.norm.forward(output);
    return output;
  }
}

export class Transformer extends Module {
  constructor({
    dModel = 512,
    nhead = 8,
    numEncoderLayers = 6,
    numDecoderLayers = 6,
    dimFeedforward = 2048,
    dropout = 0.1,
    activation = 'relu',
    batchFirst = true,
    normFirst = false,
    layerNormEps = 1e-5,
  } = {}) {
    super();
    const encLayer = new TransformerEncoderLayer(dModel, nhead, dimFeedforward, dropout, activation, layerNormEps, batchFirst, normFirst);
    const decLayer = new TransformerDecoderLayer(dModel, nhead, dimFeedforward, dropout, activation, layerNormEps, batchFirst, normFirst);
    this.encoder = new TransformerEncoder(encLayer, numEncoderLayers);
    this.decoder = new TransformerDecoder(decLayer, numDecoderLayers);
    this.dModel = dModel;
  }

  forward(src, tgt, srcMask = null, tgtMask = null, memoryMask = null, srcKeyPaddingMask = null, tgtKeyPaddingMask = null, memoryKeyPaddingMask = null) {
    const memory = this.encoder.forward(src, srcMask, srcKeyPaddingMask);
    return this.decoder.forward(tgt, memory, tgtMask, memoryMask, tgtKeyPaddingMask, memoryKeyPaddingMask);
  }

  static generateSquareSubsequentMask(sz) {
    const mask = empty([sz, sz]);
    const data = mask._impl.storage.data;
    for (let i = 0; i < sz; i++) {
      for (let j = 0; j < sz; j++) {
        data[i * sz + j] = j <= i ? 0 : -Infinity;
      }
    }
    return mask;
  }
}
