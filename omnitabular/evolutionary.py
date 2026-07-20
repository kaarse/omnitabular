"""Evolutionary and Metaheuristic Models."""

import numpy as np
from typing import Optional, List, Dict, Callable
import copy

from .base import BaseModel

# ── 1. Simulated Annealing ───────────────────────────────────────────

class SAModel(BaseModel):
    """
    Simulated Annealing model for learning linear weights.
    """
    def __init__(self, task="auto", max_iter=1000, init_temp=1.0, cooling_rate=0.99, random_state=42, **kwargs):
        super().__init__(task)
        self.max_iter = max_iter
        self.init_temp = init_temp
        self.cooling_rate = cooling_rate
        self.random_state = random_state
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        X = self._to_numpy(X)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        
        self.w = rng.randn(n_features) * 0.01
        self.b = rng.randn() * 0.01
        
        if self.task == "classification":
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            if n_classes > 2:
                self.w = rng.randn(n_features, n_classes) * 0.01
                self.b = rng.randn(n_classes) * 0.01

        temp = self.init_temp
        best_w, best_b = np.copy(self.w), np.copy(self.b)
        best_loss = self._loss(X, y)

        for _ in range(self.max_iter):
            # Proposal
            prop_w = self.w + rng.randn(*self.w.shape) * temp * 0.1
            prop_b = self.b + rng.randn(*self.b.shape) * temp * 0.1
            
            old_w, old_b = self.w, self.b
            self.w, self.b = prop_w, prop_b
            loss = self._loss(X, y)
            
            if loss < best_loss:
                best_w, best_b = np.copy(self.w), np.copy(self.b)
                best_loss = loss
            elif rng.rand() > np.exp(-(loss - best_loss) / (temp + 1e-9)):
                # Rejected
                self.w, self.b = old_w, old_b
                
            temp *= self.cooling_rate
            
        self.w, self.b = best_w, best_b
        self._fitted = True
        return self

    def _loss(self, X, y):
        pred = X @ self.w + self.b
        if self.task == "classification":
            if len(self.classes_) > 2: # multiclass
                exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
                probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
                return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9))
            else: # binary
                probs = 1 / (1 + np.exp(-pred))
                return -np.mean(y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9))
        else:
            return np.mean((y - pred)**2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._to_numpy(X)
        pred = X @ self.w + self.b
        if self.task == "classification":
            if len(self.classes_) > 2:
                return np.argmax(pred, axis=1)
            else:
                return (pred >= 0).astype(int)
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError
        X = self._to_numpy(X)
        pred = X @ self.w + self.b
        if len(self.classes_) > 2:
            exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        else:
            p = 1 / (1 + np.exp(-pred))
            return np.column_stack([1 - p, p])


# ── 2. Genetic Algorithm ─────────────────────────────────────────────

class GAModel(BaseModel):
    """
    Genetic Algorithm model for learning linear weights.
    """
    def __init__(self, task="auto", pop_size=50, generations=100, mutation_rate=0.1, random_state=42, **kwargs):
        super().__init__(task)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.w = None
        self.b = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        X = self._to_numpy(X)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        
        self.is_multi = False
        if self.task == "classification":
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            if n_classes > 2:
                self.is_multi = True
        
        shape = (n_features, n_classes) if self.is_multi else (n_features,)
        b_shape = (n_classes,) if self.is_multi else ()
        
        pop = [(rng.randn(*shape), rng.randn(*b_shape)) for _ in range(self.pop_size)]
        
        def loss_fn(w, b):
            pred = X @ w + b
            if self.task == "classification":
                if self.is_multi:
                    exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
                    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
                    return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9))
                else:
                    probs = 1 / (1 + np.exp(-pred))
                    return -np.mean(y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9))
            else:
                return np.mean((y - pred)**2)

        for _ in range(self.generations):
            fitness = np.array([loss_fn(w, b) for w, b in pop])
            best_idx = np.argsort(fitness)
            pop = [pop[i] for i in best_idx]
            
            # Elitism: keep top 20%
            elite_count = max(1, int(self.pop_size * 0.2))
            new_pop = pop[:elite_count]
            
            # Crossover and mutate
            while len(new_pop) < self.pop_size:
                p1 = pop[rng.choice(elite_count)]
                p2 = pop[rng.choice(elite_count)]
                
                # uniform crossover
                mask_w = rng.rand(*shape) < 0.5
                child_w = np.where(mask_w, p1[0], p2[0])
                mask_b = rng.rand(*b_shape) < 0.5
                child_b = np.where(mask_b, p1[1], p2[1])
                
                # mutate
                if rng.rand() < self.mutation_rate:
                    child_w += rng.randn(*shape) * 0.1
                    child_b += rng.randn(*b_shape) * 0.1
                    
                new_pop.append((child_w, child_b))
            pop = new_pop
            
        final_fitness = [loss_fn(w, b) for w, b in pop]
        best = pop[np.argmin(final_fitness)]
        self.w, self.b = best
        self._fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._to_numpy(X)
        pred = X @ self.w + self.b
        if self.task == "classification":
            if self.is_multi:
                return np.argmax(pred, axis=1)
            else:
                return (pred >= 0).astype(int)
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError
        X = self._to_numpy(X)
        pred = X @ self.w + self.b
        if self.is_multi:
            exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        else:
            p = 1 / (1 + np.exp(-pred))
            return np.column_stack([1 - p, p])


# ── 3. Symbolic Regression (GP) ──────────────────────────────────────

class Node:
    def __init__(self, op=None, left=None, right=None, val=None, feature_idx=None):
        self.op = op
        self.left = left
        self.right = right
        self.val = val
        self.feature_idx = feature_idx
        
    def eval(self, X):
        if self.op is not None:
            l_val = self.left.eval(X)
            r_val = self.right.eval(X)
            if self.op == '+': return l_val + r_val
            if self.op == '-': return l_val - r_val
            if self.op == '*': return l_val * r_val
            if self.op == '/': 
                # protected div
                return np.where(np.abs(r_val) < 1e-3, 1.0, l_val / (r_val + 1e-9))
        elif self.feature_idx is not None:
            return X[:, self.feature_idx]
        else:
            return np.full(X.shape[0], self.val)

class SymbolicRegressionModel(BaseModel):
    def __init__(self, task="auto", pop_size=30, generations=20, max_depth=3, random_state=42, **kwargs):
        super().__init__(task)
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = None
        self.is_multi = False
        
    def _random_tree(self, depth, rng, n_features):
        if depth == 0 or rng.rand() < 0.2:
            if rng.rand() < 0.5:
                return Node(feature_idx=rng.randint(n_features))
            else:
                return Node(val=rng.randn())
        else:
            op = rng.choice(['+', '-', '*', '/'])
            return Node(op=op, 
                        left=self._random_tree(depth - 1, rng, n_features),
                        right=self._random_tree(depth - 1, rng, n_features))
                        
    def _mutate(self, node, rng, n_features, depth=0):
        if depth > self.max_depth:
            return self._random_tree(0, rng, n_features)
        if rng.rand() < 0.1:
            return self._random_tree(self.max_depth - depth, rng, n_features)
        new_node = copy.copy(node)
        if new_node.op is not None:
            new_node.left = self._mutate(new_node.left, rng, n_features, depth + 1)
            new_node.right = self._mutate(new_node.right, rng, n_features, depth + 1)
        return new_node

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        X = self._to_numpy(X)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        
        self.is_multi = False
        if self.task == "classification":
            self.classes_ = np.unique(y)
            if len(self.classes_) > 2:
                self.is_multi = True
        
        # 1 tree for regression/binary, N trees for multiclass
        num_trees = len(self.classes_) if self.is_multi else 1
        
        pop = [[self._random_tree(self.max_depth, rng, n_features) for _ in range(num_trees)] for _ in range(self.pop_size)]
        
        def loss_fn(trees):
            if self.is_multi:
                preds = np.column_stack([t.eval(X) for t in trees])
                preds = np.clip(preds, -50, 50)
                exp_pred = np.exp(preds - np.max(preds, axis=1, keepdims=True))
                probs = exp_pred / (np.sum(exp_pred, axis=1, keepdims=True) + 1e-9)
                return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9))
            else:
                pred = trees[0].eval(X)
                pred = np.clip(pred, -50, 50)
                if self.task == "classification":
                    probs = 1 / (1 + np.exp(-pred))
                    return -np.mean(y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9))
                else:
                    return np.mean((y - pred)**2)

        valid_pop = pop
        for _ in range(self.generations):
            fitness = [loss_fn(ind) for ind in valid_pop]
            fitness = np.nan_to_num(fitness, nan=1e9)
            best_idx = np.argsort(fitness)
            valid_pop = [valid_pop[i] for i in best_idx]
            elite = valid_pop[:max(1, int(self.pop_size * 0.2))]
            
            new_pop = list(elite)
            while len(new_pop) < self.pop_size:
                parent = valid_pop[rng.choice(len(elite))]
                child = [self._mutate(t, rng, n_features) for t in parent]
                new_pop.append(child)
            valid_pop = new_pop
            
        final_fitness = [loss_fn(ind) for ind in valid_pop]
        self.trees_ = valid_pop[np.argmin(final_fitness)]
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._to_numpy(X)
        if self.is_multi:
            preds = np.column_stack([t.eval(X) for t in self.trees_])
            return np.argmax(preds, axis=1)
        else:
            pred = self.trees_[0].eval(X)
            if self.task == "classification":
                return (pred >= 0).astype(int)
            return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError
        X = self._to_numpy(X)
        if self.is_multi:
            preds = np.column_stack([t.eval(X) for t in self.trees_])
            preds = np.clip(preds, -50, 50)
            exp_pred = np.exp(preds - np.max(preds, axis=1, keepdims=True))
            return exp_pred / (np.sum(exp_pred, axis=1, keepdims=True) + 1e-9)
        else:
            pred = self.trees_[0].eval(X)
            pred = np.clip(pred, -50, 50)
            p = 1 / (1 + np.exp(-pred))
            return np.column_stack([1 - p, p])


# ── 4. Learning Classifier Systems ───────────────────────────────────

class LCSRule:
    def __init__(self, condition_bounds, action, prediction=None):
        self.condition_bounds = condition_bounds # [(min, max), ...]
        self.action = action
        self.prediction = prediction # for regression
        self.fitness = 1.0
        self.experience = 0

class LCSModel(BaseModel):
    def __init__(self, task="auto", pop_size=100, epochs=10, random_state=42, **kwargs):
        super().__init__(task)
        self.pop_size = pop_size
        self.epochs = epochs
        self.random_state = random_state
        self.rules = []

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        X = self._to_numpy(X)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y) if self.task == "classification" else None
        
        self.rules = []
        # basic Michigan style loop
        for _ in range(self.epochs):
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                
                # Match set
                match_set = []
                for r in self.rules:
                    match = True
                    for j in range(n_features):
                        if not (r.condition_bounds[j][0] <= xi[j] <= r.condition_bounds[j][1]):
                            match = False
                            break
                    if match:
                        match_set.append(r)
                        
                # Covering
                if len(match_set) == 0:
                    bounds = []
                    for j in range(n_features):
                        span = rng.rand() * 2 * np.std(X[:, j]) if np.std(X[:, j]) > 0 else 0.1
                        bounds.append((xi[j] - span, xi[j] + span))
                    
                    action = yi if self.task == "classification" else None
                    prediction = yi if self.task == "regression" else None
                    new_rule = LCSRule(bounds, action, prediction)
                    self.rules.append(new_rule)
                    match_set.append(new_rule)
                    
                    if len(self.rules) > self.pop_size:
                        self.rules.sort(key=lambda r: r.fitness, reverse=True)
                        self.rules = self.rules[:self.pop_size]

                # Update fitness
                for r in match_set:
                    r.experience += 1
                    if self.task == "classification":
                        if r.action == yi:
                            r.fitness += 0.1 * (1 - r.fitness)
                        else:
                            r.fitness += 0.1 * (0 - r.fitness)
                    else:
                        error = abs(r.prediction - yi)
                        accuracy = np.exp(-error)
                        r.fitness += 0.1 * (accuracy - r.fitness)
                        # update prediction
                        r.prediction += 0.1 * (yi - r.prediction)
                        
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._to_numpy(X)
        preds = []
        
        for i in range(X.shape[0]):
            xi = X[i]
            match_set = []
            for r in self.rules:
                match = True
                for j in range(X.shape[1]):
                    if not (r.condition_bounds[j][0] <= xi[j] <= r.condition_bounds[j][1]):
                        match = False
                        break
                if match:
                    match_set.append(r)
            
            if len(match_set) == 0:
                if self.task == "classification":
                    preds.append(self.classes_[0])
                else:
                    preds.append(0.0)
            else:
                if self.task == "classification":
                    votes = {c: 0.0 for c in self.classes_}
                    for r in match_set:
                        votes[r.action] += r.fitness
                    preds.append(max(votes, key=votes.get))
                else:
                    total_fit = sum(r.fitness for r in match_set)
                    if total_fit > 0:
                        preds.append(sum(r.prediction * r.fitness for r in match_set) / total_fit)
                    else:
                        preds.append(sum(r.prediction for r in match_set) / len(match_set))
                        
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError
        X = self._to_numpy(X)
        probs = []
        for i in range(X.shape[0]):
            xi = X[i]
            match_set = []
            for r in self.rules:
                match = True
                for j in range(X.shape[1]):
                    if not (r.condition_bounds[j][0] <= xi[j] <= r.condition_bounds[j][1]):
                        match = False
                        break
                if match:
                    match_set.append(r)
            
            row = np.zeros(len(self.classes_))
            if len(match_set) > 0:
                total_fit = sum(r.fitness for r in match_set)
                for r in match_set:
                    idx = np.where(self.classes_ == r.action)[0][0]
                    row[idx] += r.fitness
                if total_fit > 0:
                    row /= total_fit
                else:
                    row = np.ones(len(self.classes_)) / len(self.classes_)
            else:
                row = np.ones(len(self.classes_)) / len(self.classes_)
            probs.append(row)
        return np.array(probs)
