/** Parse the first CSV record, including quoted commas, escaped quotes and newlines. */
export function parseCsvHeader(text: string): string[] {
  const input = text.replace(/^\uFEFF/, "");
  const columns: string[] = [];
  let value = "", quoted = false, closed = false;
  for (let i = 0; i < input.length; i++) {
    const char = input[i];
    if (quoted) {
      if (char === '"') {
        if (input[i + 1] === '"') { value += '"'; i++; }
        else { quoted = false; closed = true; }
      } else value += char;
    } else if (char === '"') {
      if (value || closed) throw new Error("Invalid quote in the CSV header.");
      quoted = true;
    } else if (char === ",") {
      columns.push(value); value = ""; closed = false;
    } else if (char === "\n" || char === "\r") {
      break;
    } else {
      if (closed) throw new Error("Unexpected text after a quoted CSV column name.");
      value += char;
    }
  }
  if (quoted) throw new Error("The CSV header has an unclosed quote or is too long.");
  columns.push(value);
  if (columns.length < 2) throw new Error("Include at least one feature column and one target column, separated by commas.");
  if (columns.some(column => !column.trim())) throw new Error("Every CSV column needs a name.");
  if (new Set(columns).size !== columns.length) throw new Error("CSV column names must be unique.");
  return columns;
}
