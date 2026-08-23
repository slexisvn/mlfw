export function onProcessExit(handler: () => void): void {
  process.on('exit', handler);
}
