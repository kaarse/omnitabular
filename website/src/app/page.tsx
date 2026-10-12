import Link from "next/link";
import { CodeBlock } from "@/components/code-block";
import { quickStart } from "@/lib/examples";

const families = [
  ["Trees & classic ML", "LightGBM, XGBoost, CatBoost, random forest, extra trees, linear / logistic regression, SVM, KNN, and Naive Bayes."],
  ["Deep tabular models", "MLP, TabNet, NODE, TreeNN, and tabular factorization machines."],
  ["Transformers, sequences & graphs", "FT-Transformer, TabTransformer, SAINT, temporal transformer, LSTM, BiLSTM, GRU, BiGRU, 1D / 2D CNNs, and GNN."],
  ["Rules & evolutionary models", "RuleFit, RIPPER, PRE, simulated annealing, genetic algorithms, learning classifier systems, and symbolic regression."],
];

export default function Home() {
  return (
    <>
      <header className="hero">
        <p className="eyebrow">One API. 32 model options.</p>
        <h1 className="hero-title">Compare models.<br /><span className="gradient-text">Combine their strengths.</span></h1>
        <p className="hero-subtitle">Train on spreadsheet-style data, compare cross-validation scores, and blend predictions with optimized ensemble weights.</p>
        <div className="actions"><Link className="btn primary-btn" href="/api-reference#quick-start">Get started</Link><Link className="secondary-btn" href="/dashboard">Open dashboard →</Link></div>
        <CodeBlock code={quickStart} />
        <p className="hint">Start with three models. Choose more when your dataset and compute budget call for them.</p>
      </header>
      <section id="features" className="features-section">
        <h2 className="section-title">From data to an ensemble</h2>
        <div className="features-grid">
          {[["1. Prepare", "Impute missing values, scale numbers, and encode categories separately within each validation fold."], ["2. Compare", "Evaluate selected models on out-of-fold predictions and inspect their scores in one leaderboard."], ["3. Combine", "Search for nonnegative blend weights, then retrain on the full training set to predict new rows."]].map(([title, text]) => <article className="feature-card glass" key={title}><h3>{title}</h3><p>{text}</p></article>)}
        </div>
      </section>
      <section id="architectures" className="architecture-section">
        <h2 className="section-title">Choose the models that fit your task</h2>
        <div className="arch-container">{families.map(([title, text]) => <article className="arch-group glass" key={title}><h3>{title}</h3><p>{text}</p></article>)}</div>
        <p className="hint">32 options for classification; 31 for regression. Naive Bayes supports classification only.</p>
      </section>
      <section id="documentation" className="docs-section">
        <div className="docs-container glass"><h2>Run it your way</h2><p>Use the Python library, upload a CSV in the dashboard, or call the REST API. No account is required for the local dashboard.</p><div className="actions"><Link className="btn primary-btn" href="/api-reference">Read the documentation</Link><a className="secondary-btn" href="https://github.com/kaarse/omnitabular">View source ↗</a></div></div>
      </section>
    </>
  );
}
