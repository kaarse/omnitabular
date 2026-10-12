'use client';

import { useEffect } from 'react';
import Link from 'next/link';

export default function ApiReference() {
  useEffect(() => {
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

  return (
    <>
      <div className='blob-bg'>
        <div className='blob blob-1'></div>
        <div className='blob blob-2'></div>
        <div className='blob blob-3'></div>
      </div>

      <nav className='navbar glass'>
        <div className='logo'>OmniTabular</div>
        <ul className='nav-links'>
          <li><Link href='/'>Home</Link></li>
          <li><Link href='/#documentation'>Documentation</Link></li>
          <li><a href='https://github.com/kaarse/omnitabular' target='_blank' rel='noreferrer' className='btn primary-btn'>GitHub</a></li>
        </ul>
      </nav>

      <main style={{ padding: '150px 10%' }}>
        <h1 style={{ textAlign: 'center', fontSize: '4rem', marginBottom: '4rem' }}>
          API <span className='gradient-text'>Reference</span>
        </h1>
<div className='api-card glass reveal' id='OmniTabular'>
<h2>OmniTabular</h2>
<div className='docstring'>
High-level tabular ML class combining **KNN**, **LightGBM**,<br />
    **Linear / Logistic Regression**, **SVM**, **Naive Bayes**,<br />
    **Random Forest**, **Extra Trees**, **XGBoost**, and **CatBoost**<br />
    with **Hill Climbing ensemble weight optimisation**.<br />
<br />
<h4 className='doc-section-title'>Workflow</h4>
    1. ``fit(X, y)``<br />
       a. Detect task type (classification / regression).<br />
       b. Select applicable models (Naive Bayes is skipped for regression).<br />
       c. Generate cross-validated OOF predictions per model.<br />
       d. Run Hill Climbing to find optimal blending weights.<br />
       e. Re-train each model on the full dataset.<br />
    2. ``predict(X)`` / ``predict_proba(X)``<br />
       Returns blended predictions using optimal weights.<br />
    3. ``leaderboard()``<br />
       Returns a DataFrame comparing individual OOF scores vs. ensemble.<br />
<br />
<h4 className='doc-section-title'>Parameters</h4>
<strong>task</strong>: &#123;&quot;auto&quot;, &quot;classification&quot;, &quot;regression&quot;&#125;<br />
<strong>knn_params</strong>: dict, optional<br />
<strong>lgb_params</strong>: dict, optional<br />
<strong>linear_params</strong>: dict, optional<br />
<strong>svm_params</strong>: dict, optional<br />
<strong>naive_bayes_params</strong>: dict, optional<br />
<strong>rf_params</strong>: dict, optional<br />
<strong>et_params</strong>: dict, optional<br />
<strong>xgb_params</strong>: dict, optional<br />
<strong>catboost_params</strong>: dict, optional<br />
<strong>hc_params</strong>: dict, optional<br />
<strong>Extra parameters for</strong>: class:`HillClimbingEnsemble`.<br />
<strong>metric</strong>: str or None<br />
        Evaluation metric.  Defaults to ``&quot;log_loss&quot;`` (classification) or<br />
        ``&quot;rmse&quot;`` (regression).<br />
<strong>cv_folds</strong>: int<br />
        Number of cross-validation folds for OOF generation.<br />
<strong>random_state</strong>: int<br />
<strong>verbose</strong>: bool<br />
<br />
<h4 className='doc-section-title'>Attributes</h4>
<strong>task_</strong>: str<br />
        Detected or specified task type.<br />
<strong>model_names_</strong>: list[str]<br />
        Names of the models that participated in the ensemble (task-compatible).<br />
<strong>models_</strong>: dict[str, BaseModel]<br />
        Fitted base models keyed by name.<br />
<strong>weights_</strong>: np.ndarray<br />
        Optimal blending weights (same order as ``model_names_``).<br />
<strong>ensemble_</strong>: WeightedEnsemble<br />
        The final blending ensemble.<br />
<strong>oof_scores_</strong>: dict[str, float]<br />
        Per-model OOF metric scores.<br />
<strong>ensemble_oof_score_</strong>: float<br />
        OOF metric score of the blended ensemble.<br />
<br />
<h4 className='doc-section-title'>Examples</h4>
<pre><code>    &gt;&gt;&gt; from omnitabular import OmniTabular
    &gt;&gt;&gt; from sklearn.datasets import load_iris
    &gt;&gt;&gt; X, y = load_iris(return_X_y=True)
    &gt;&gt;&gt; model = OmniTabular(task=&quot;classification&quot;, verbose=True)
    &gt;&gt;&gt; model.fit(X, y)
    &gt;&gt;&gt; print(model.leaderboard())
    &gt;&gt;&gt; preds = model.predict(X[:10])
</code></pre>

</div>
<h3>__init__<span className='signature'>(self, task: &#x27;str&#x27; = &#x27;auto&#x27;, knn_params: &#x27;Optional[Dict]&#x27; = None, lgb_params: &#x27;Optional[Dict]&#x27; = None, linear_params: &#x27;Optional[Dict]&#x27; = None, svm_params: &#x27;Optional[Dict]&#x27; = None, naive_bayes_params: &#x27;Optional[Dict]&#x27; = None, rf_params: &#x27;Optional[Dict]&#x27; = None, et_params: &#x27;Optional[Dict]&#x27; = None, xgb_params: &#x27;Optional[Dict]&#x27; = None, catboost_params: &#x27;Optional[Dict]&#x27; = None, mlp_params: &#x27;Optional[Dict]&#x27; = None, ft_transformer_params: &#x27;Optional[Dict]&#x27; = None, tab_transformer_params: &#x27;Optional[Dict]&#x27; = None, saint_params: &#x27;Optional[Dict]&#x27; = None, tree_nn_params: &#x27;Optional[Dict]&#x27; = None, tabnet_params: &#x27;Optional[Dict]&#x27; = None, node_params: &#x27;Optional[Dict]&#x27; = None, tabfm_params: &#x27;Optional[Dict]&#x27; = None, gru_params: &#x27;Optional[Dict]&#x27; = None, bigru_params: &#x27;Optional[Dict]&#x27; = None, lstm_params: &#x27;Optional[Dict]&#x27; = None, bilstm_params: &#x27;Optional[Dict]&#x27; = None, cnn1d_params: &#x27;Optional[Dict]&#x27; = None, cnn2d_params: &#x27;Optional[Dict]&#x27; = None, gnn_params: &#x27;Optional[Dict]&#x27; = None, temporal_transformer_params: &#x27;Optional[Dict]&#x27; = None, sa_params: &#x27;Optional[Dict]&#x27; = None, ga_params: &#x27;Optional[Dict]&#x27; = None, symbolic_regression_params: &#x27;Optional[Dict]&#x27; = None, lcs_params: &#x27;Optional[Dict]&#x27; = None, rulefit_params: &#x27;Optional[Dict]&#x27; = None, ripper_params: &#x27;Optional[Dict]&#x27; = None, pre_params: &#x27;Optional[Dict]&#x27; = None, hc_params: &#x27;Optional[Dict]&#x27; = None, metric: &#x27;Optional[str]&#x27; = None, cv_folds: &#x27;int&#x27; = 5, random_state: &#x27;int&#x27; = 42, verbose: &#x27;bool&#x27; = True) -&gt; &#x27;None&#x27;</span></h3>
<div className='methods-container'>
<div className='method-block'>
<h4>feature_importances<span className='signature'>(self) -&gt; &#x27;pd.DataFrame&#x27;</span></h4>
<div className='method-doc'>
Return LGB feature importances (gain-based).<br />
<br />
        Returns<br />
        pd.DataFrame with columns: [feature_index, importance]<br />

</div>
</div>
<div className='method-block'>
<h4>fit<span className='signature'>(self, X, y) -&gt; &quot;&#x27;OmniTabular&#x27;&quot;</span></h4>
<div className='method-doc'>
Fit OmniTabular on (X, y).<br />
<br />
        Steps<br />
        1. Detect task + choose metric.<br />
        2. OOF prediction for every compatible base model.<br />
        3. Hill Climbing on OOF predictions to find weights.<br />
        4. Re-train on full dataset.<br />

</div>
</div>
<div className='method-block'>
<h4>leaderboard<span className='signature'>(self) -&gt; &#x27;pd.DataFrame&#x27;</span></h4>
<div className='method-doc'>
Return a comparison DataFrame of individual model OOF scores and the ensemble.<br />
<br />
        Returns<br />
        pd.DataFrame with columns: [model, oof_&lt;metric&gt;, weight]<br />

</div>
</div>
<div className='method-block'>
<h4>predict<span className='signature'>(self, X) -&gt; &#x27;np.ndarray&#x27;</span></h4>
<div className='method-doc'>
Return final blended predictions.<br />

</div>
</div>
<div className='method-block'>
<h4>predict_proba<span className='signature'>(self, X) -&gt; &#x27;np.ndarray&#x27;</span></h4>
<div className='method-doc'>
Return final blended probability predictions (classification only).<br />

</div>
</div>
</div>
</div>

<div className='api-card glass reveal' id='HillClimbingEnsemble'>
<h2>HillClimbingEnsemble</h2>
<div className='docstring'>
Stochastic Hill Climbing optimiser for ensemble blending weights.<br />
<br />
    Given a collection of OOF predictions from multiple base models, this<br />
    optimiser searches for a weight vector **w** (summing to 1) that maximises<br />
    (or minimises, depending on the metric) the blended prediction score.<br />
<br />
    Algorithm<br />
    1. Start from a uniform weight vector ``w = [1/K, ..., 1/K]``.<br />
    2. At each step, sample a random perturbation ``Δw`` and compute<br />
       ``w&#x27; = normalise(w + step_size * Δw)``.<br />
    3. Accept ``w&#x27;`` if it improves the objective.<br />
    4. Gradually shrink ``step_size`` (simulated annealing schedule).<br />
    5. Repeat for ``n_iterations`` steps.<br />
<br />
<h4 className='doc-section-title'>Parameters</h4>
<strong>metric</strong>: str or None<br />
        Metric name from ``METRIC_REGISTRY``. If None, defaults to<br />
        ``&quot;log_loss&quot;`` (classification) or ``&quot;rmse&quot;`` (regression).<br />
<strong>n_iterations</strong>: int<br />
        Number of hill climbing iterations.<br />
<strong>step_size</strong>: float<br />
        Initial perturbation magnitude.<br />
<strong>step_decay</strong>: float<br />
        Multiplicative decay applied to ``step_size`` every ``decay_every`` steps.<br />
<strong>decay_every</strong>: int<br />
        How often (steps) to apply step decay.<br />
<strong>random_state</strong>: int or None<br />
<strong>verbose</strong>: bool<br />
        Print progress if True.<br />
<br />
<h4 className='doc-section-title'>Attributes</h4>
<strong>weights_</strong>: np.ndarray of shape (n_models,)<br />
        Optimal blending weights found after ``fit``.<br />
<strong>best_score_</strong>: float<br />
        Best metric value achieved.<br />
<strong>score_history_</strong>: list of float<br />
        Score at each accepted improvement.<br />
<br />
<h4 className='doc-section-title'>Examples</h4>
<pre><code>    &gt;&gt;&gt; import numpy as np
    &gt;&gt;&gt; from omnitabular import HillClimbingEnsemble
    &gt;&gt;&gt; rng = np.random.default_rng(0)
    &gt;&gt;&gt; y_true = rng.integers(0, 2, 200)
    &gt;&gt;&gt; # Simulate OOF probabilities from 3 models
    &gt;&gt;&gt; oof_probas = [rng.dirichlet([1, 1], size=200) for _ in range(3)]
    &gt;&gt;&gt; hc = HillClimbingEnsemble(metric=&quot;log_loss&quot;, n_iterations=500)
    &gt;&gt;&gt; hc.fit(y_true, oof_preds=None, oof_probas=oof_probas, task=&quot;classification&quot;)
    &gt;&gt;&gt; hc.weights_
</code></pre>

</div>
<h3>__init__<span className='signature'>(self, metric: &#x27;Optional[str]&#x27; = None, n_iterations: &#x27;int&#x27; = 3000, step_size: &#x27;float&#x27; = 0.1, step_decay: &#x27;float&#x27; = 0.99, decay_every: &#x27;int&#x27; = 50, random_state: &#x27;Optional[int]&#x27; = 42, verbose: &#x27;bool&#x27; = False) -&gt; &#x27;None&#x27;</span></h3>
<div className='methods-container'>
<div className='method-block'>
<h4>blend_predictions<span className='signature'>(self, predictions: &#x27;List[np.ndarray]&#x27;) -&gt; &#x27;np.ndarray&#x27;</span></h4>
<div className='method-doc'>
Apply ``weights_`` to a list of prediction arrays.<br />
<br />
<h4 className='doc-section-title'>Parameters</h4>
<strong>predictions</strong>: list of np.ndarray<br />
            Same order as during ``fit``.<br />
<br />
        Returns<br />
<strong>blended</strong>: np.ndarray<br />

</div>
</div>
<div className='method-block'>
<h4>fit<span className='signature'>(self, y_true: &#x27;np.ndarray&#x27;, task: &#x27;str&#x27;, oof_preds: &#x27;Optional[List[np.ndarray]]&#x27; = None, oof_probas: &#x27;Optional[List[np.ndarray]]&#x27; = None) -&gt; &quot;&#x27;HillClimbingEnsemble&#x27;&quot;</span></h4>
<div className='method-doc'>
Find optimal blending weights.<br />
<br />
<h4 className='doc-section-title'>Parameters</h4>
<strong>y_true</strong>: array-like of shape (n_samples,)<br />
<strong>task</strong>: &#123;&quot;classification&quot;, &quot;regression&quot;&#125;<br />
<strong>oof_preds</strong>: list of np.ndarray of shape (n_samples,), optional<br />
            Hard OOF predictions per model (used for regression or hard-vote blending).<br />
<strong>oof_probas</strong>: list of np.ndarray of shape (n_samples, n_classes), optional<br />
            Soft OOF probabilities per model (classification only).<br />

</div>
</div>
</div>
</div>

<div className='api-card glass reveal' id='WeightedEnsemble'>
<h2>WeightedEnsemble</h2>
<div className='docstring'>
Combines predictions from a list of fitted base models using fixed weights.<br />
<br />
    For **classification**, predictions are blended as a weighted sum of<br />
    ``predict_proba`` outputs.  For **regression**, a weighted sum of<br />
    ``predict`` outputs is used.<br />
<br />
<h4 className='doc-section-title'>Parameters</h4>
<strong>models</strong>: list of BaseModel<br />
        Fitted base learners.<br />
<strong>weights</strong>: array-like of shape (n_models,)<br />
        Blending weights (need not sum to 1 — they will be normalised).<br />
<strong>task</strong>: &#123;&quot;classification&quot;, &quot;regression&quot;&#125;<br />
<br />
<h4 className='doc-section-title'>Attributes</h4>
<strong>weights_</strong>: np.ndarray<br />
        Normalised weight vector.<br />
<br />
<h4 className='doc-section-title'>Examples</h4>
<pre><code>    &gt;&gt;&gt; from omnitabular import KNNModel, LGBModel, WeightedEnsemble
    &gt;&gt;&gt; import numpy as np
    &gt;&gt;&gt; X, y = np.random.rand(100, 4), np.random.randint(0, 3, 100)
    &gt;&gt;&gt; knn = KNNModel(task=&quot;classification&quot;).fit(X, y)
    &gt;&gt;&gt; lgb = LGBModel(task=&quot;classification&quot;).fit(X, y)
    &gt;&gt;&gt; ens = WeightedEnsemble([knn, lgb], weights=[0.4, 0.6], task=&quot;classification&quot;)
    &gt;&gt;&gt; ens.predict_proba(X[:5])
</code></pre>

</div>
<h3>__init__<span className='signature'>(self, models: &#x27;List[BaseModel]&#x27;, weights: &#x27;np.ndarray&#x27;, task: &#x27;str&#x27;) -&gt; &#x27;None&#x27;</span></h3>
<div className='methods-container'>
<div className='method-block'>
<h4>predict<span className='signature'>(self, X: &#x27;np.ndarray&#x27;) -&gt; &#x27;np.ndarray&#x27;</span></h4>
<div className='method-doc'>
For classification: argmax over blended probabilities.<br />
        For regression: weighted average of predictions.<br />

</div>
</div>
<div className='method-block'>
<h4>predict_proba<span className='signature'>(self, X: &#x27;np.ndarray&#x27;) -&gt; &#x27;np.ndarray&#x27;</span></h4>
<div className='method-doc'>
Weighted average of predict_proba outputs (classification only).<br />

</div>
</div>
</div>
</div>

      </main>
    </>
  );
}