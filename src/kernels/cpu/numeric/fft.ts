function bitReversePermute(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
}

function fftRadix2(re: Float64Array, im: Float64Array, invert: boolean): void {
  const n = re.length;
  bitReversePermute(re, im);
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((invert ? 1 : -1) * 2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      const half = len >> 1;
      for (let j = 0; j < half; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + half] * curRe - im[i + j + half] * curIm;
        const vIm = re[i + j + half] * curIm + im[i + j + half] * curRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + half] = uRe - vRe;
        im[i + j + half] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
  if (invert) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function bluestein(re: Float64Array, im: Float64Array, invert: boolean): void {
  const n = re.length;
  const m = nextPow2(2 * n - 1);
  const sign = invert ? 1 : -1;
  const cosTab = new Float64Array(n);
  const sinTab = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const j = (i * i) % (2 * n);
    const ang = (sign * Math.PI * j) / n;
    cosTab[i] = Math.cos(ang);
    sinTab[i] = Math.sin(ang);
  }
  const aRe = new Float64Array(m);
  const aIm = new Float64Array(m);
  for (let i = 0; i < n; i++) {
    aRe[i] = re[i] * cosTab[i] - im[i] * sinTab[i];
    aIm[i] = re[i] * sinTab[i] + im[i] * cosTab[i];
  }
  const bRe = new Float64Array(m);
  const bIm = new Float64Array(m);
  bRe[0] = cosTab[0];
  bIm[0] = -sinTab[0];
  for (let i = 1; i < n; i++) {
    bRe[i] = cosTab[i];
    bIm[i] = -sinTab[i];
    bRe[m - i] = cosTab[i];
    bIm[m - i] = -sinTab[i];
  }
  fftRadix2(aRe, aIm, false);
  fftRadix2(bRe, bIm, false);
  for (let i = 0; i < m; i++) {
    const r = aRe[i] * bRe[i] - aIm[i] * bIm[i];
    aIm[i] = aRe[i] * bIm[i] + aIm[i] * bRe[i];
    aRe[i] = r;
  }
  fftRadix2(aRe, aIm, true);
  for (let i = 0; i < n; i++) {
    re[i] = aRe[i] * cosTab[i] - aIm[i] * sinTab[i];
    im[i] = aRe[i] * sinTab[i] + aIm[i] * cosTab[i];
  }
  if (invert) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

export function fftHost(re: Float64Array, im: Float64Array, invert: boolean): void {
  const n = re.length;
  if (n === 0) return;
  if ((n & (n - 1)) === 0) fftRadix2(re, im, invert);
  else bluestein(re, im, invert);
}
