let _nodeIdCounter = 0;

export class Edge {
  constructor(node, inputNr) {
    this.node = node;
    this.inputNr = inputNr;
  }
}

export class AutogradNode {
  constructor(numInputs) {
    this._id = _nodeIdCounter++;
    this._numInputs = numInputs || 0;
    this._nextEdges = [];
    this._savedTensors = [];
    this._inputMetadata = [];
  }

  get id() {
    return this._id;
  }

  get numInputs() {
    return this._numInputs;
  }

  get nextEdges() {
    return this._nextEdges;
  }

  addNextEdge(node, inputNr) {
    this._nextEdges.push(new Edge(node, inputNr));
  }

  setNextEdge(index, node, inputNr) {
    while (this._nextEdges.length <= index) this._nextEdges.push(null);
    this._nextEdges[index] = new Edge(node, inputNr);
  }

  saveTensor(tensor) {
    this._savedTensors.push(tensor);
  }

  savedTensors() {
    return this._savedTensors;
  }

  saveInputMetadata(index, shape, dtype) {
    this._inputMetadata[index] = { shape, dtype };
  }

  inputMetadata(index) {
    return this._inputMetadata[index] || null;
  }

  apply(gradOutputs) {
    throw new Error(`${this.name()}.apply() not implemented`);
  }

  name() {
    return this.constructor.name;
  }

  releaseVariables() {
    this._savedTensors = [];
    this._inputMetadata = [];
  }
}
