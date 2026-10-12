"use client";

import { useEffect, useRef, useState } from "react";

export function CodeBlock({ code, label = "Python" }: { code: string; label?: string }) {
  const [feedback, setFeedback] = useState("Copy");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  async function copy() {
    try {
      if (!navigator.clipboard) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(code);
      setFeedback("Copied");
    } catch {
      setFeedback("Select code to copy");
    }
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setFeedback("Copy"), 2500);
  }

  return (
    <div className="code-terminal glass">
      <div className="terminal-header"><span>{label}</span><button type="button" onClick={copy} aria-label={`Copy ${label} example`}>{feedback}</button></div>
      <pre tabIndex={0} aria-label={label}><code>{code}</code></pre>
      <span className="sr-only" role="status">{feedback === "Copy" ? "" : feedback}</span>
    </div>
  );
}
