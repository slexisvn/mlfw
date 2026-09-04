export type TeracHandle = unknown;

export type TeracLocation = {
  build?: string | null;
  library?: string | null;
  llvmBin?: string | null;
};

function unavailable(): never {
  throw new Error('terac is a native compiler and is not available in the browser');
}

export function teracAvailable(): boolean {
  return false;
}

export function teracRuntimeLibs(): string[] {
  return unavailable();
}

export function teracCompile(): TeracHandle {
  return unavailable();
}

export function teracInvoke(): void {
  return unavailable();
}

export function teracRelease(): void {
}
