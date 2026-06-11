export function joinPath(...parts) {
  return parts
    .filter((p) => p != null && p !== '')
    .join('/')
    .replace(/\/+/g, '/');
}
