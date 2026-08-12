export const DEFAULT_TOL = 1e-12;
export const MAX_JACOBI_SWEEPS = 100;
export const DEFAULT_RCOND = 1e-12;
export const JACOBI_ZERO = 1e-300;

export type LinalgOpts = { tol?: number; maxSweeps?: number; rcond?: number };
