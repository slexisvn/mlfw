import { CsvStreamParser } from './dist/csv.esm.js';

self.onmessage = async (e) => {
  const { file, separator } = e.data;
  const parser = new CsvStreamParser(separator || ',');
  const reader = file.stream().getReader();
  const decoder = new TextDecoder();
  let read = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      read += value.byteLength;
      parser.feed(decoder.decode(value, { stream: true }));
      self.postMessage({ type: 'progress', read });
    }
    parser.feed(decoder.decode());
    const { columns, rowCount, headers } = parser.finish();
    self.postMessage({ type: 'done', columns, rowCount, headers });
  } catch (err) {
    self.postMessage({ type: 'error', message: String((err && err.message) || err) });
  }
};
