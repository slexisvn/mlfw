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

    const addWordPairs = (word) => {
      const seq = symbols.get(word);
      const freq = wordFreq.get(word);
      for (let i = 0; i + 1 < seq.length; i++) {
        const key = pairKey(seq[i], seq[i + 1]);
        pairCounts.set(key, (pairCounts.get(key) || 0) + freq);
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
        if (remaining <= 0) pairCounts.delete(key);
        else pairCounts.set(key, remaining);
        const words = pairWords.get(key);
        if (words) words.delete(word);
      }
    };

    for (const word of wordFreq.keys()) addWordPairs(word);

    this._ranks = new Map();
    for (let merge = 0; merge < this._numMerges; merge++) {
      let best = null;
      let bestCount = 0;
      for (const [key, count] of pairCounts) {
        if (count > bestCount) { bestCount = count; best = key; }
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
}
