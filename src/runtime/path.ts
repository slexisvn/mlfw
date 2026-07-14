export function joinPath(...parts: Array<string | null | undefined>): string {
  return parts
    .filter((p) => p != null && p !== '')
    .join('/')
    .replace(/\/+/g, '/');
}
