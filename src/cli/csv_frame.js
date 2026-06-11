import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

export class CsvFrame {
  constructor(headers, columns, numRows) {
    this._headers = headers;
    this._columns = columns;
    this._numRows = numRows;
    this._indexMap = new Map();
    for (let i = 0; i < headers.length; i++) {
      this._indexMap.set(headers[i], i);
    }
  }

  get headers() { return this._headers; }

  get shape() { return [this._numRows, this._headers.length]; }

  get numRows() { return this._numRows; }

  get numCols() { return this._headers.length; }

  _resolveIndex(nameOrIndex) {
    if (typeof nameOrIndex === 'number') {
      if (nameOrIndex < 0 || nameOrIndex >= this._columns.length) {
        throw new Error(`Column index ${nameOrIndex} out of range [0, ${this._columns.length})`);
      }
      return nameOrIndex;
    }
    const idx = this._indexMap.get(nameOrIndex);
    if (idx === undefined) {
      throw new Error(`Column '${nameOrIndex}' not found. Available: ${this._headers.join(', ')}`);
    }
    return idx;
  }

  column(nameOrIndex) {
    return this._columns[this._resolveIndex(nameOrIndex)];
  }

  select(...names) {
    if (names.length === 0) return this;
    const indices = names.map(n => this._resolveIndex(n));
    const headers = indices.map(i => this._headers[i]);
    const columns = indices.map(i => this._columns[i]);
    return new CsvFrame(headers, columns, this._numRows);
  }

  drop(...names) {
    const dropSet = new Set(names.map(n => this._resolveIndex(n)));
    const headers = [];
    const columns = [];
    for (let i = 0; i < this._headers.length; i++) {
      if (!dropSet.has(i)) {
        headers.push(this._headers[i]);
        columns.push(this._columns[i]);
      }
    }
    return new CsvFrame(headers, columns, this._numRows);
  }

  toArray() {
    const rows = new Array(this._numRows);
    for (let r = 0; r < this._numRows; r++) {
      const row = new Array(this._columns.length);
      for (let c = 0; c < this._columns.length; c++) {
        row[c] = this._columns[c][r];
      }
      rows[r] = row;
    }
    return rows;
  }

  toTensor(tensorFn, ...colNames) {
    const frame = colNames.length > 0 ? this.select(...colNames) : this;
    for (let c = 0; c < frame._columns.length; c++) {
      const col = frame._columns[c];
      for (let r = 0; r < col.length; r++) {
        if (typeof col[r] !== 'number') {
          throw new Error(
            `Column '${frame._headers[c]}' contains non-numeric value '${col[r]}' at row ${r}. Use encode() for categorical columns.`
          );
        }
      }
    }
    const flat = new Float32Array(frame._numRows * frame._columns.length);
    let idx = 0;
    for (let r = 0; r < frame._numRows; r++) {
      for (let c = 0; c < frame._columns.length; c++) {
        flat[idx++] = frame._columns[c][r];
      }
    }
    return tensorFn(flat, { shape: [frame._numRows, frame._columns.length] });
  }

  encode(nameOrIndex, knownClasses = null) {
    const col = this.column(nameOrIndex);
    const classMap = new Map();
    const classes = knownClasses ? [...knownClasses] : [];
    for (let i = 0; i < classes.length; i++) {
      const key = typeof classes[i] === 'number' ? String(classes[i]) : classes[i];
      classMap.set(key, i);
    }
    const encoded = new Float32Array(col.length);
    for (let i = 0; i < col.length; i++) {
      const val = col[i];
      const key = typeof val === 'number' ? String(val) : val;
      let idx = classMap.get(key);
      if (idx === undefined) {
        if (knownClasses) throw new Error(`Unknown class '${val}' not present in fitted classes`);
        idx = classes.length;
        classMap.set(key, idx);
        classes.push(val);
      }
      encoded[i] = idx;
    }
    return { encoded, classes };
  }

  head(n = 5) {
    const rows = Math.min(n, this._numRows);
    const result = [this._headers.join(',')];
    for (let r = 0; r < rows; r++) {
      const row = [];
      for (let c = 0; c < this._columns.length; c++) {
        row.push(this._columns[c][r]);
      }
      result.push(row.join(','));
    }
    return result.join('\n');
  }

  shuffle() {
    const indices = Array.from({ length: this._numRows }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }
    const columns = this._columns.map(col => {
      const shuffled = new Array(this._numRows);
      for (let i = 0; i < this._numRows; i++) {
        shuffled[i] = col[indices[i]];
      }
      return shuffled;
    });
    return new CsvFrame([...this._headers], columns, this._numRows);
  }

  slice(start, end) {
    const s = Math.max(0, start);
    const e = Math.min(this._numRows, end ?? this._numRows);
    const len = e - s;
    if (len <= 0) return new CsvFrame([...this._headers], this._headers.map(() => []), 0);
    const columns = this._columns.map(col => col.slice(s, e));
    return new CsvFrame([...this._headers], columns, len);
  }

  toString() {
    return `CsvFrame(${this._numRows} rows x ${this._headers.length} cols: [${this._headers.join(', ')}])`;
  }
}

export function parseCsv(content, separator = ',') {
  const lines = splitLines(content);
  if (lines.length === 0) throw new Error('CSV file is empty');

  const headerFields = parseRow(lines[0], separator);
  const numCols = headerFields.length;
  const columns = new Array(numCols);
  for (let c = 0; c < numCols; c++) columns[c] = [];

  for (let i = 1; i < lines.length; i++) {
    const fields = parseRow(lines[i], separator);
    if (fields.length !== numCols) {
      throw new Error(`Row ${i + 1} has ${fields.length} fields, expected ${numCols}`);
    }
    for (let c = 0; c < numCols; c++) {
      columns[c].push(coerceValue(fields[c]));
    }
  }

  return new CsvFrame(headerFields, columns, lines.length - 1);
}

function splitLines(text) {
  const lines = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
      current += ch;
    } else if ((ch === '\n' || ch === '\r') && !inQuotes) {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      const trimmed = current.trim();
      if (trimmed.length > 0) lines.push(trimmed);
      current = '';
    } else {
      current += ch;
    }
  }
  const trimmed = current.trim();
  if (trimmed.length > 0) lines.push(trimmed);
  return lines;
}

function parseRow(line, separator) {
  const fields = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === separator) {
      fields.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  fields.push(current.trim());
  return fields;
}

function coerceValue(raw) {
  if (raw === '' || raw === 'null' || raw === 'NULL' || raw === 'NA' || raw === 'NaN') return null;
  if (raw === 'true' || raw === 'True' || raw === 'TRUE') return 1;
  if (raw === 'false' || raw === 'False' || raw === 'FALSE') return 0;
  const num = Number(raw);
  if (!Number.isNaN(num)) return num;
  return raw;
}

export function loadCsv(filePath, separator = ',') {
  const absPath = resolve(filePath);
  const content = readFileSync(absPath, 'utf-8');
  return parseCsv(content, separator);
}
