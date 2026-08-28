const LISTED = 3;

export function lineNote(lines: readonly number[]): string {
  if (lines.length === 0) return '';
  if (lines.length === 1) return `line ${lines[0]}`;
  const listed = lines.slice(0, LISTED).join(', ');
  return lines.length > LISTED ? `lines ${listed} +${lines.length - LISTED}` : `lines ${listed}`;
}
