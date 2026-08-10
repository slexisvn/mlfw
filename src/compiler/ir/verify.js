import { verifyModule, verifyFunction } from './graph/verifier.js';
import { TensorVerifier } from './tensor/verifier.js';
import { verifyLIR } from './lir/verifier.js';

export const IRLevel = Object.freeze({
  GRAPH_MODULE: 'graph-module',
  GRAPH_FUNC: 'graph-func',
  TIR: 'tir',
  LIR: 'lir',
});

const _verifiers = new Map();

export function registerIRVerifier(level, verify) {
  _verifiers.set(level, verify);
  return verify;
}

export function unregisterIRVerifier(level) {
  return _verifiers.delete(level);
}

export function getIRVerifier(level) {
  return _verifiers.get(level) || null;
}

export function verifyIR(level, target) {
  const verify = _verifiers.get(level);
  if (!verify) throw new Error(`No IR verifier registered for level '${level}'`);
  return verify(target);
}

export function irLevels() {
  return [..._verifiers.keys()];
}

function toMessages(errors) {
  const out = [];
  for (const e of errors) out.push(typeof e === 'string' ? e : e.toString());
  return out;
}

registerIRVerifier(IRLevel.GRAPH_MODULE, (module) => toMessages(verifyModule(module)));
registerIRVerifier(IRLevel.GRAPH_FUNC, (func) => toMessages(verifyFunction(func)));
registerIRVerifier(IRLevel.TIR, (primFunc) => toMessages(new TensorVerifier().verify(primFunc)));
registerIRVerifier(IRLevel.LIR, (lirFunc) => toMessages(verifyLIR(lirFunc)));
