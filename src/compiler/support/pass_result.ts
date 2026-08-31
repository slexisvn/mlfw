export const PassResult = Object.freeze({
  UNCHANGED: 0,
  CHANGED: 1,
  FAILED: 2
});

export type PassResultValue = (typeof PassResult)[keyof typeof PassResult];
