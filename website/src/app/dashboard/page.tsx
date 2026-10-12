"use client";
import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, FileType, AlertCircle, Play, ChevronRight, Activity, Cpu } from 'lucide-react';

type TrainingResult = {
  task: string;
  ensemble_oof_score: number;
  weights: Record<string, number>;
};

export default function Dashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [targetCol, setTargetCol] = useState<string>('');
  
  const [status, setStatus] = useState<'idle' | 'uploading' | 'training' | 'success' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');
  
  const [result, setResult] = useState<TrainingResult | null>(null);
  
  // Terminal logs state
  const [logs, setLogs] = useState<{time: string, msg: string, type: 'info'|'success'|'warn'}[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addLog = (msg: string, type: 'info'|'success'|'warn' = 'info') => {
    const time = new Date().toISOString().split('T')[1].substring(0, 12);
    setLogs(prev => [...prev, { time, msg, type }]);
  };

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.add('active');
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.remove('active');
  };

  const processFile = (selected: File) => {
    setStatus('idle');
    setErrorMsg('');
    setResult(null);
    setColumns([]);
    setTargetCol('');
    if (!selected.name.toLowerCase().endsWith('.csv')) {
      setErrorMsg('Please select a CSV file.');
      setStatus('error');
      return;
    }
    setFile(selected);
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const firstLine = text.replace(/^\uFEFF/, '').split(/\r?\n/)[0];
      if (firstLine) {
        const cols: string[] = [];
        let value = '', quoted = false;
        for (let i = 0; i < firstLine.length; i++) {
          const char = firstLine[i];
          if (char === '"') {
            if (quoted && firstLine[i + 1] === '"') { value += '"'; i++; }
            else quoted = !quoted;
          } else if (char === ',' && !quoted) { cols.push(value); value = ''; }
          else value += char;
        }
        cols.push(value);
        setColumns(cols);
        setTargetCol(cols[cols.length - 1]);
      }
    };
    reader.onerror = () => {
      setErrorMsg('Could not read the selected file.');
      setStatus('error');
    };
    reader.readAsText(selected.slice(0, 65536));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.remove('active');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024, dm = 2, sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  const handleTrain = async () => {
    if (!file || !targetCol) return;
    setStatus('training');
    setResult(null);
    setErrorMsg('');
    setLogs([]);
    addLog('Uploading dataset and waiting for cross-validation and ensemble training...');
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_col', targetCol);
    formData.append('task', 'auto');

    try {
      const res = await fetch(`${(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080').replace(/\/$/, '')}/train_upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || 'Training failed on server.');
      }
      
      setResult(data);
      addLog('Training complete. Results received from the server.', 'success');
      setStatus('success');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Training failed.';
      addLog(message, 'warn');
      setErrorMsg(message);
      setStatus('error');
    }
  };

  return (
    <div className="dashboard-container">
      <h1 className="dashboard-title stagger-1">OmniTabular Control Center</h1>
      <p className="dashboard-subtitle stagger-2">Upload a CSV dataset, train an ensemble, and review its cross-validation score.</p>
      
      <div className="glass-panel main-panel stagger-3">
        {/* Step 1: Upload */}
        <div className="form-group">
          <label className="flex items-center gap-2 mb-2 text-[var(--accent-1)] font-semibold">
            <ChevronRight size={18} /> 1. Ingest Data (CSV)
          </label>
          
          {!file ? (
            <div 
              className="drag-drop-zone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <UploadCloud className="upload-icon mx-auto" />
              <h3 style={{fontSize: '1.2rem', marginBottom: '0.5rem', fontWeight: 600}}>Drag & Drop Dataset</h3>
              <p style={{color: 'var(--text-muted)'}}>or click to browse local files</p>
              <input 
                type="file" 
                accept=".csv" 
                onChange={handleFileChange} 
                ref={fileInputRef} 
                hidden 
              />
            </div>
          ) : (
            <div className="file-info-card">
              <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
                <FileType size={32} color="var(--accent-2)" />
                <div>
                  <h4 style={{fontWeight: 600, fontSize: '1.1rem'}}>{file.name}</h4>
                  <span style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>{formatSize(file.size)}</span>
                </div>
              </div>
              <button 
                disabled={status === 'training'}
                onClick={() => { setFile(null); setColumns([]); setResult(null); setStatus('idle'); }} 
                style={{background: 'rgba(255,50,50,0.1)', color: '#ff7b72', border: '1px solid rgba(255,50,50,0.3)', padding: '0.5rem 1rem', borderRadius: '8px', cursor: 'pointer', transition: 'all 0.2s', fontWeight: 600}}
              >
                Change File
              </button>
            </div>
          )}
        </div>
        
        {/* Step 2: Target Column */}
        {columns.length > 0 && (
          <div className="form-group animation-fade" style={{marginTop: '2rem'}}>
            <label className="flex items-center gap-2 mb-2 text-[var(--accent-1)] font-semibold">
               <ChevronRight size={18} /> 2. Define Target Objective
            </label>
            <div className="glass-select-wrapper">
              <select value={targetCol} onChange={(e) => setTargetCol(e.target.value)} className="glass-select">
                {columns.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
            
            <div style={{marginTop: '1rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center'}}>
               <span style={{color: 'var(--text-muted)', fontSize: '0.85rem', marginRight: '0.5rem'}}>Detected Features ({columns.length - 1}):</span>
               {columns.filter(c => c !== targetCol).slice(0, 8).map(c => (
                 <span key={c} style={{background: 'rgba(255,255,255,0.05)', padding: '3px 10px', borderRadius: '6px', fontSize: '0.8rem', border: '1px solid rgba(255,255,255,0.08)'}}>{c}</span>
               ))}
               {columns.length > 9 && <span style={{fontSize: '0.8rem', color: 'var(--text-muted)'}}>+{columns.length - 9} more</span>}
            </div>
          </div>
        )}
        
        {/* Step 3: Trigger Action */}
        {file && status !== 'training' && status !== 'success' && (
          <button 
            className="btn-primary animation-fade" 
            onClick={handleTrain}
            disabled={!targetCol}
            style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginTop: '3rem', padding: '1.2rem'}}
          >
            <Play size={20} fill="currentColor" /> Train Ensemble
          </button>
        )}
        
        {/* Step 4: Loading & Live Logs */}
        {status === 'training' && (
          <div className="animation-fade" style={{marginTop: '3rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '2rem'}}>
            <div style={{display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem'}}>
              <div className="spinner" style={{margin: 0, width: '28px', height: '28px', borderWidth: '3px'}}></div>
              <h3 style={{color: 'var(--accent-2)', margin: 0, fontWeight: 700}}>Optimization in Progress...</h3>
            </div>
            
            <div className="terminal-simulator">
              {logs.map((log, idx) => (
                <div key={idx} className="terminal-log">
                  <span className="term-time">[{log.time}]</span>
                  <span className={`term-msg ${log.type === 'success' ? 'term-success' : log.type === 'warn' ? 'term-warn' : ''}`}>
                    {log.msg}
                  </span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        )}
        
        {status === 'error' && (
          <div className="error-box animation-fade" style={{display: 'flex', alignItems: 'flex-start', gap: '1rem', marginTop: '2rem'}}>
            <AlertCircle size={24} color="#ff3232" style={{flexShrink: 0}} />
            <div>
              <strong style={{display: 'block', fontSize: '1.1rem', marginBottom: '0.2rem'}}>System Failure</strong>
              <span>{errorMsg}</span>
            </div>
          </div>
        )}
      </div>

      {/* Results View */}
      {status === 'success' && result && (
        <div className="results-container stagger-4">
          <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1rem', marginBottom: '3rem'}}>
             <Activity size={36} color="var(--accent-3)" />
             <h2 className="results-title" style={{margin: 0}}>Optimization Achieved</h2>
          </div>
          
          <div className="metric-card" style={{maxWidth: '400px', margin: '0 auto 3rem auto'}}>
            <div className="metric-content" style={{padding: '3rem', textAlign: 'center'}}>
              <div style={{textTransform: 'uppercase', letterSpacing: '1px', fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '0.5rem'}}>
                Ensemble Cross-Validation Score ({result.task})
              </div>
              <div className="metric-value" style={{fontSize: '4rem', background: 'linear-gradient(to right, var(--accent-3), #2dd4bf)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', lineHeight: '1.1'}}>
                {result.ensemble_oof_score?.toFixed(5)}
              </div>
            </div>
          </div>
          
          <div style={{display: 'flex', alignItems: 'center', gap: '0.8rem', marginBottom: '2rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem'}}>
             <Cpu size={24} color="var(--accent-1)" />
             <h3 className="weights-title" style={{margin: 0}}>Ensemble Weights</h3>
          </div>
          
          <div className="weights-grid">
            {Object.entries(result.weights || {})
              .sort(([,a], [,b]) => b - a)
              .map(([model, weight], idx) => (
                <div key={model} className="weight-card" style={{animation: `slideUp 0.5s ease-out ${0.2 + (idx*0.1)}s both`}}>
                  <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '1rem'}}>
                    <div className="weight-name" style={{margin: 0, textTransform: 'uppercase', letterSpacing: '0.5px', fontSize: '0.9rem'}}>{model}</div>
                    <div className="weight-val" style={{fontWeight: 800, color: 'var(--text-main)', fontSize: '1.1rem'}}>{(weight * 100).toFixed(1)}%</div>
                  </div>
                  <div className="weight-bar-container">
                    <div className="weight-bar" style={{ width: `${(weight * 100).toFixed(1)}%` }}></div>
                  </div>
                </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
