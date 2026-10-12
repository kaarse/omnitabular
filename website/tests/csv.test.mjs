import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import ts from 'typescript';

// Compile this isolated parser in memory so tests also run on supported Node 20.
const source = fs.readFileSync(new URL('../src/lib/csv.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } });
const { parseCsvHeader } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);

test('reads ordinary and BOM-prefixed CRLF CSV headers', () => {
  assert.deepEqual(parseCsvHeader('\uFEFFnumber,category,target\r\n1,a,0'), ['number', 'category', 'target']);
});
test('preserves quoted commas, escaped quotes, multiline names and apostrophes', () => {
  assert.deepEqual(parseCsvHeader('"last, first","say ""hello""","two\nlines",owner\'s,target\n'), ['last, first', 'say "hello"', 'two\nlines', "owner's", 'target']);
});
test('preserves spaces in column names to match pandas', () => {
  assert.deepEqual(parseCsvHeader(' amount ,target\n'), [' amount ', 'target']);
});
test('rejects missing, duplicate and malformed column names', () => {
  for (const input of ['', 'target', 'a,,target', 'a,a,target', '"a,target', 'a,"b"oops,target', 'a,b"c,target']) {
    assert.throws(() => parseCsvHeader(input), undefined, input);
  }
});
