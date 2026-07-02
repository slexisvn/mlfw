import { vectorOf } from './_util.js';

export function mean_squared_error(yTrue, yPred) {
  const a = vectorOf(yTrue);
  const b = vectorOf(yPred);
  let s = 0;
  for (let i = 0; i < a.n; i++) {
    const d = a.data[i] - b.data[i];
    s += d * d;
  }
  return s / a.n;
}

export function mean_absolute_error(yTrue, yPred) {
  const a = vectorOf(yTrue);
  const b = vectorOf(yPred);
  let s = 0;
  for (let i = 0; i < a.n; i++) s += Math.abs(a.data[i] - b.data[i]);
  return s / a.n;
}

export function r2_score(yTrue, yPred) {
  const a = vectorOf(yTrue);
  const b = vectorOf(yPred);
  let mean = 0;
  for (let i = 0; i < a.n; i++) mean += a.data[i];
  mean /= a.n;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < a.n; i++) {
    const dr = a.data[i] - b.data[i];
    const dt = a.data[i] - mean;
    ssRes += dr * dr;
    ssTot += dt * dt;
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

export function accuracy_score(yTrue, yPred) {
  const a = vectorOf(yTrue);
  const b = vectorOf(yPred);
  let correct = 0;
  for (let i = 0; i < a.n; i++) if (a.data[i] === b.data[i]) correct++;
  return correct / a.n;
}

export function confusion_matrix(yTrue, yPred) {
  const a = vectorOf(yTrue);
  const b = vectorOf(yPred);
  let maxLabel = 0;
  for (let i = 0; i < a.n; i++) maxLabel = Math.max(maxLabel, a.data[i], b.data[i]);
  const k = maxLabel + 1;
  const cm = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < a.n; i++) cm[a.data[i]][b.data[i]]++;
  return cm;
}
