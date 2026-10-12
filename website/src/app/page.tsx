'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function Home() {
  const [copyText, setCopyText] = useState('Copy');
  const [copyColor, setCopyColor] = useState('var(--text-main)');

  useEffect(() => {
    // Scroll Reveal Animation via Intersection Observer
    function reveal() {
      const reveals = document.querySelectorAll('.reveal');
      for (let i = 0; i < reveals.length; i++) {
        const windowHeight = window.innerHeight;
        const elementTop = reveals[i].getBoundingClientRect().top;
        const elementVisible = 100;
        if (elementTop < windowHeight - elementVisible) {
          reveals[i].classList.add('active');
        }
      }
    }
    window.addEventListener('scroll', reveal);
    reveal();

    return () => window.removeEventListener('scroll', reveal);
  }, []);

  const handleCopy = () => {
    const codeBlock = document.querySelector('code')?.innerText || '';
    navigator.clipboard.writeText(codeBlock).then(() => {
      setCopyText('Copied!');
      setCopyColor('var(--accent-3)');
      setTimeout(() => {
        setCopyText('Copy');
        setCopyColor('var(--text-main)');
      }, 2000);
    }).catch(err => {
      console.error('Failed to copy: ', err);
    });
  };

  return (
    <>
      <div className="blob-bg">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        <div className="blob blob-3"></div>
      </div>

      <nav className="navbar glass">
        <div className="logo">OmniTabular</div>
        <ul className="nav-links">
          <li><Link href="#features">Features</Link></li>
          <li><Link href="#architectures">Architectures</Link></li>
          <li><Link href="/dashboard">Dashboard</Link></li>
          <li><Link href="/api-reference">Documentation & API</Link></li>
          <li><a href="https://github.com/kaarse/omnitabular" target="_blank" rel="noreferrer" className="btn primary-btn">GitHub</a></li>
        </ul>
      </nav>

      <header className="hero">
        <h1 className="hero-title">The Ultimate <br /><span className="gradient-text">Tabular ML</span> Framework.</h1>
        <p className="hero-subtitle">Seamlessly ensemble 32 state-of-the-art architectures using stochastic Hill Climbing optimization. Zero manual tuning required.</p>
        
        <div className="code-terminal glass">
          <div className="terminal-header">
            <span className="dot red"></span>
            <span className="dot yellow"></span>
            <span className="dot green"></span>
          </div>
          <pre><code><span className="keyword">from</span> omnitabular <span className="keyword">import</span> OmniTabular
<span className="keyword">from</span> sklearn.datasets <span className="keyword">import</span> load_breast_cancer

X, y = load_breast_cancer(return_X_y=<span className="boolean">True</span>)

<span className="comment"># Train 32 models & optimize weights</span>
model = OmniTabular(task=<span className="string">&quot;classification&quot;</span>, verbose=<span className="boolean">True</span>)
model.fit(X, y)

<span className="function">print</span>(model.leaderboard())</code></pre>
          <button 
            className="copy-btn" 
            id="copy-btn" 
            onClick={handleCopy} 
            style={{ color: copyColor }}
          >
            {copyText}
          </button>
        </div>
      </header>

      <section id="features" className="features-section">
        <h2 className="section-title">Why OmniTabular?</h2>
        <div className="features-grid">
          <div className="feature-card glass reveal">
            <div className="feature-icon">🧩</div>
            <h3>32 Architectures</h3>
            <p>Native support for Gradient Boosting, Deep NNs, Transformers, CNNs, LSTMs, Graph Nets, and Rule-based models.</p>
          </div>
          <div className="feature-card glass reveal">
            <div className="feature-icon">⛰️</div>
            <h3>Hill Climbing</h3>
            <p>Stochastic weight search on OOF predictions with thermodynamic annealing to find the perfect ensemble blend.</p>
          </div>
          <div className="feature-card glass reveal">
            <div className="feature-icon">🚀</div>
            <h3>Zero-Config API</h3>
            <p>Call .fit() and automatically evaluate, rank, and blend all compatible models mathematically.</p>
          </div>
        </div>
      </section>

      <section id="architectures" className="architecture-section">
        <h2 className="section-title">Supported Architectures</h2>
        <div className="arch-container">
          <div className="arch-group glass reveal">
            <h3>🌲 Tree Ensembles & Classics</h3>
            <ul>
              <li>LightGBM, XGBoost, CatBoost</li>
              <li>Random Forest, Extra Trees</li>
              <li>SVM, Linear Regression, KNN</li>
            </ul>
          </div>
          <div className="arch-group glass reveal">
            <h3>🧠 Deep Neural Networks</h3>
            <ul>
              <li>TabNet, NODE, TreeNN</li>
              <li>Tabular Factorization Machines</li>
              <li>Multi-Layer Perceptrons</li>
            </ul>
          </div>
          <div className="arch-group glass reveal">
            <h3>🧬 Evolutionary & Rule-Based</h3>
            <ul>
              <li>Simulated Annealing, Genetic Algo</li>
              <li>Symbolic Regression (GP)</li>
              <li>RuleFit, RIPPER, PRE</li>
            </ul>
          </div>
          <div className="arch-group glass reveal">
            <h3>🕸️ Sequences & Graphs</h3>
            <ul>
              <li>FT-Transformer, SAINT</li>
              <li>LSTMs, GRUs, 1D/2D CNNs</li>
              <li>Graph Neural Networks</li>
            </ul>
          </div>
        </div>
      </section>
    </>
  );
}
