"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { parseCsvHeader } from "@/lib/csv";

type Result = { task: "classification" | "regression"; ensemble_oof_score: number; weights: Record<string, number> };
type Task = "auto" | "classification" | "regression";
const apiUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080").replace(/\/$/, "");

export default function Dashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [target, setTarget] = useState("");
  const [task, setTask] = useState<Task>("auto");
  const [status, setStatus] = useState<"idle" | "reading" | "training" | "success" | "error">("idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const selection = useRef(0);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => { selection.current++; request.current?.abort(); }, []);
  const busy = status === "reading" || status === "training";

  async function selectFile(selected: File) {
    if (busy) return;
    const version = ++selection.current;
    setResult(null); setError(""); setColumns([]); setTarget(""); setFile(null);
    try {
      if (!selected.name.toLowerCase().endsWith(".csv")) throw new Error("Choose a .csv file.");
      if (!selected.size) throw new Error("The selected CSV is empty.");
      setStatus("reading");
      const header = parseCsvHeader(await selected.slice(0, 65536).text());
      if (version !== selection.current) return;
      setFile(selected); setColumns(header); setTarget(header[header.length - 1]); setStatus("idle");
    } catch (err) {
      if (version !== selection.current) return;
      setError(err instanceof Error ? err.message : "Could not read this file."); setStatus("error");
    }
  }

  async function train(event: React.FormEvent) {
    event.preventDefault();
    if (!file || !target || busy) return;
    setStatus("training"); setResult(null); setError("");
    const controller = new AbortController();
    request.current = controller;
    const form = new FormData();
    form.append("file", file); form.append("target_col", target); form.append("task", task);
    try {
      const response = await fetch(`${apiUrl}/train_upload`, { method: "POST", body: form, signal: controller.signal });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = data?.detail;
        throw new Error(typeof detail === "string" ? detail : `Training request failed (HTTP ${response.status}). Check the API configuration and selected task.`);
      }
      if (!data || !["classification", "regression"].includes(data.task) || typeof data.ensemble_oof_score !== "number" || !Number.isFinite(data.ensemble_oof_score) || !data.weights || typeof data.weights !== "object" || Array.isArray(data.weights) || !Object.values(data.weights).every(value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1)) {
        throw new Error("The server returned an unexpected training response.");
      }
      setResult(data); setStatus("success");
    } catch (err) {
      if (controller.signal.aborted) return;
      setError(err instanceof TypeError ? "Could not reach the training service. Check that the API is running and allows this website's origin." : err instanceof Error ? err.message : "Training failed.");
      setStatus("error");
    } finally { if (request.current === controller) request.current = null; }
  }

  return (
    <div className="dashboard-container">
      <header className="page-heading"><p className="eyebrow">Your data, one ensemble</p><h1>Training dashboard</h1><p>Upload a CSV, choose what to predict, and compare the ensemble&apos;s validation score.</p></header>
      <div className="notice"><p>The training API must be running. No login is required. <Link href="/api-reference#local-setup">Setup instructions →</Link></p><p>Train LightGBM, random forest, and linear / logistic regression using 3 validation folds.</p></div>
      <form className="main-panel glass" onSubmit={train} aria-busy={busy}>
        <fieldset disabled={busy}>
          <legend>1. Choose a dataset</legend>
          <input id="csv-file" aria-label="CSV dataset" type="file" accept=".csv,text/csv" className="sr-only" ref={input} onChange={event => { const selected = event.target.files?.[0]; if (selected) void selectFile(selected); event.target.value = ""; }} />
          <div className={`drag-drop-zone ${dragging ? "active" : ""}`} onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); const selected = event.dataTransfer.files[0]; if (selected) void selectFile(selected); }}>
            <p className="upload-symbol" aria-hidden="true">↑</p><p>{file ? file.name : "Drop a CSV file here"}</p>
            {file && <p className="hint">{(file.size / 1024).toFixed(1)} KB · {columns.length} columns</p>}
            <button className="secondary-btn" type="button" onClick={() => input.current?.click()}>{file ? "Change file" : "Browse files"}</button>
            <p className="hint">Comma-separated CSV with unique column names and a target column.</p>
          </div>
        </fieldset>
        {columns.length > 0 && <fieldset disabled={busy}><legend>2. Choose what to predict</legend><div className="form-grid"><div className="form-group"><label htmlFor="target">Target column</label><select id="target" value={target} onChange={event => { setTarget(event.target.value); setResult(null); setStatus("idle"); setError(""); }}>{columns.map(column => <option key={column}>{column}</option>)}</select></div><div className="form-group"><label htmlFor="task">Prediction task</label><select id="task" value={task} onChange={event => { setTask(event.target.value as Task); setResult(null); setStatus("idle"); setError(""); }}><option value="auto">Auto-detect</option><option value="classification">Classification · categories</option><option value="regression">Regression · numeric values</option></select></div></div><p className="hint">Choose regression explicitly for prices, counts, or other numeric quantities. Classification needs at least 3 rows per class.</p><p className="hint">Features ({columns.length - 1}): {columns.filter(column => column !== target).join(", ")}</p></fieldset>}
        <button className="btn-primary" type="submit" disabled={!file || !target || busy}>{status === "training" ? "Training ensemble…" : status === "reading" ? "Reading CSV…" : "Train ensemble"}</button>
        <div role="status" aria-live="polite">{status === "training" && <p className="training-status"><span className="spinner" aria-hidden="true" />Training and cross-validation are running. Large datasets may take several minutes.</p>}{status === "success" && <p className="training-status">Training completed. Results are shown below.</p>}</div>
        {error && <div role="alert" className="error-box"><strong>Could not complete training</strong><p>{error}</p><Link href="/api-reference#troubleshooting">Troubleshooting →</Link></div>}
      </form>
      {result && <section className="results-container" aria-labelledby="results-title"><h2 id="results-title">Ensemble results</h2><div className="metric-card glass"><p>{result.task === "classification" ? "Out-of-fold log loss" : "Out-of-fold RMSE"}</p><strong className="metric-value">{result.ensemble_oof_score.toFixed(5)}</strong><p className="hint">Lower is better. Evaluate on separate test data before assessing performance on new rows.</p></div><h3>Model weights</h3><div className="weights-grid">{Object.entries(result.weights).sort(([, a], [, b]) => b - a).map(([model, weight]) => <article className="weight-card" key={model}><div className="weight-heading"><h4>{model}</h4><span>{(weight * 100).toFixed(1)}%</span></div><div className="weight-bar-container" aria-hidden="true"><div className="weight-bar" style={{ width: `${weight * 100}%` }} /></div></article>)}</div></section>}
    </div>
  );
}
