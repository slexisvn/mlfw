const PAIR_SEP = String.fromCharCode(31);

function pairKey(a, b) {
  return a + PAIR_SEP + b;
}

export class BpeStrategy {
  constructor({ numMerges = 1000, lowercase = false, endOfWord = '</w>' } = {}) {
    this._numMerges = numMerges;
    this._lowercase = lowercase;
    this._eow = endOfWord;
    this._ranks = new Map();
    this._encodeCache = new Map();
  }

  _pretokenize(text) {
    const normalized = this._lowercase ? String(text).toLowerCase() : String(text);
    return normalized.split(/\s+/).filter(Boolean);
  }

  _baseSymbols(word) {
    const symbols = Array.from(word);
    symbols.push(this._eow);
    return symbols;
  }

  fit(texts) {
    this._encodeCache = new Map();

    const wordFreq = new Map();
    for (const text of texts) {
      for (const word of this._pretokenize(text)) {
        wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
      }
    }

    const symbols = new Map();
    for (const word of wordFreq.keys()) symbols.set(word, this._baseSymbols(word));

    const pairCounts = new Map();
    const pairWords = new Map();

    const heap = [];
    const higher = (a, b) => a[0] > b[0] || (a[0] === b[0] && a[1] < b[1]);
    const pushHeap = (count, key) => {
      heap.push([count, key]);
      let i = heap.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (higher(heap[i], heap[p])) { const t = heap[p]; heap[p] = heap[i]; heap[i] = t; i = p; }
        else break;
      }
    };
    const popHeap = () => {
      const top = heap[0];
      const last = heap.pop();
      if (heap.length > 0) {
        heap[0] = last;
        let i = 0;
        const n = heap.length;
        while (true) {
          let pick = i;
          const l = 2 * i + 1, r = 2 * i + 2;
          if (l < n && higher(heap[l], heap[pick])) pick = l;
          if (r < n && higher(heap[r], heap[pick])) pick = r;
          if (pick === i) break;
          const t = heap[pick]; heap[pick] = heap[i]; heap[i] = t; i = pick;
        }
      }
      return top;
    };

    const addWordPairs = (word) => {
      const seq = symbols.get(word);
      const freq = wordFreq.get(word);
      for (let i = 0; i + 1 < seq.length; i++) {
        const key = pairKey(seq[i], seq[i + 1]);
        const count = (pairCounts.get(key) || 0) + freq;
        pairCounts.set(key, count);
        pushHeap(count, key);
        let words = pairWords.get(key);
        if (!words) { words = new Set(); pairWords.set(key, words); }
        words.add(word);
      }
    };

    const removeWordPairs = (word) => {
      const seq = symbols.get(word);
      const freq = wordFreq.get(word);
      for (let i = 0; i + 1 < seq.length; i++) {
        const key = pairKey(seq[i], seq[i + 1]);
        const remaining = (pairCounts.get(key) || 0) - freq;
        if (remaining <= 0) {
          pairCounts.delete(key);
        } else {
          pairCounts.set(key, remaining);
          pushHeap(remaining, key);
        }
        const words = pairWords.get(key);
        if (words) words.delete(word);
      }
    };

    for (const word of wordFreq.keys()) addWordPairs(word);

    this._ranks = new Map();
    for (let merge = 0; merge < this._numMerges; merge++) {
      let best = null;
      let bestCount = 0;
      while (heap.length > 0) {
        const [count, key] = popHeap();
        const current = pairCounts.get(key);
        if (current === count && current > 0) { best = key; bestCount = current; break; }
      }
      if (best === null || bestCount <= 0) break;

      const sep = best.indexOf(PAIR_SEP);
      const left = best.slice(0, sep);
      const right = best.slice(sep + 1);
      const merged = left + right;
      this._ranks.set(best, merge);

      const affected = pairWords.get(best);
      if (!affected || affected.size === 0) break;

      for (const word of [...affected]) {
        removeWordPairs(word);
        const seq = symbols.get(word);
        const next = [];
        for (let i = 0; i < seq.length; i++) {
          if (i + 1 < seq.length && seq[i] === left && seq[i + 1] === right) {
            next.push(merged);
            i++;
          } else {
            next.push(seq[i]);
          }
        }
        symbols.set(word, next);
        addWordPairs(word);
      }
    }
  }

  _encodeWord(word) {
    const cached = this._encodeCache.get(word);
    if (cached !== undefined) return cached;

    let seq = this._baseSymbols(word);
    while (seq.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i + 1 < seq.length; i++) {
        const rank = this._ranks.get(pairKey(seq[i], seq[i + 1]));
        if (rank !== undefined && rank < bestRank) { bestRank = rank; bestIdx = i; }
      }
      if (bestIdx < 0) break;
      seq = seq.slice(0, bestIdx).concat(seq[bestIdx] + seq[bestIdx + 1], seq.slice(bestIdx + 2));
    }
    this._encodeCache.set(word, seq);
    return seq;
  }

  segment(text) {
    const pieces = [];
    for (const word of this._pretokenize(text)) {
      for (const piece of this._encodeWord(word)) pieces.push(piece);
    }
    return pieces;
  }

  detokenize(tokens) {
    let joined = '';
    for (const token of tokens) joined += token;
    return joined.split(this._eow).join(' ').trim();
  }

  toJSON() {
    const merges = [...this._ranks.entries()]
      .map(([key]) => {
        const sep = key.indexOf(PAIR_SEP);
        return [key.slice(0, sep), key.slice(sep + 1)];
      });
    return {
      numMerges: this._numMerges,
      lowercase: this._lowercase,
      endOfWord: this._eow,
      merges,
    };
  }

  static fromJSON(data = {}) {
    if (!Array.isArray(data.merges)) throw new Error('mlfw tokenizer: bpe strategy merges must be an array');
    const strategy = new BpeStrategy({
      numMerges: data.numMerges ?? data.merges.length,
      lowercase: data.lowercase ?? false,
      endOfWord: data.endOfWord ?? '</w>',
    });
    strategy._ranks = new Map();
    for (let rank = 0; rank < data.merges.length; rank++) {
      const pair = data.merges[rank];
      if (!Array.isArray(pair) || pair.length !== 2 || pair.some(x => typeof x !== 'string')) {
        throw new Error('mlfw tokenizer: bpe merges must be string pairs');
      }
      strategy._ranks.set(pairKey(pair[0], pair[1]), rank);
    }
    return strategy;
  }
}
