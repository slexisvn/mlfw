export class Vocab {
  constructor(specials = []) {
    this._tokenToId = new Map();
    this._idToToken = [];
    for (const token of specials) this.add(token);
  }

  add(token) {
    let id = this._tokenToId.get(token);
    if (id === undefined) {
      id = this._idToToken.length;
      this._tokenToId.set(token, id);
      this._idToToken.push(token);
    }
    return id;
  }

  getId(token, fallback = -1) {
    const id = this._tokenToId.get(token);
    return id === undefined ? fallback : id;
  }

  getToken(id) {
    return id >= 0 && id < this._idToToken.length ? this._idToToken[id] : undefined;
  }

  has(token) {
    return this._tokenToId.has(token);
  }

  get size() {
    return this._idToToken.length;
  }

  tokens() {
    return this._idToToken.slice();
  }
}
