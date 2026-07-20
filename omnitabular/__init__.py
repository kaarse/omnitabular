"""
omnitabular — Ensemble tabular ML framework with Hill Climbing optimisation.

Models: KNN, LightGBM, Linear/Logistic Regression, SVM, Naive Bayes.
"""

from .knn import KNNModel
from .lgb import LGBModel
from .linear import LinearModel
from .svm import SVMModel
from .naive_bayes import NaiveBayesModel
from .rf import RFModel
from .et import ETModel
from .xgb import XGBModel
from .cat import CatBoostModel
from .mlp import MLPModel
from .ft_transformer import FTTransformerModel
from .tab_transformer import TabTransformerModel
from .saint import SAINTModel
from .tree_nn import TreeNNModel
from .tabnet import TabNetModel
from .node import NODEModel
from .tabfm import TabFMModel
from .rnn import GRUModel, BiGRUModel, LSTMModel, BiLSTMModel
from .cnn import CNN1DModel, CNN2DModel
from .gnn import GNNModel
from .temporal_transformer import TemporalTransformerModel
from .evolutionary import SAModel, GAModel, SymbolicRegressionModel, LCSModel
from .rules import RuleFitModel, RipperModel, PREModel
from .hill_climbing import HillClimbingEnsemble
from .ensemble import WeightedEnsemble
from .omnitabular import OmniTabular

__all__ = [
    "KNNModel",
    "LGBModel",
    "LinearModel",
    "SVMModel",
    "NaiveBayesModel",
    "RFModel",
    "ETModel",
    "XGBModel",
    "CatBoostModel",
    "MLPModel",
    "FTTransformerModel",
    "TabTransformerModel",
    "SAINTModel",
    "TreeNNModel",
    "TabNetModel",
    "NODEModel",
    "TabFMModel",
    "GRUModel",
    "BiGRUModel",
    "LSTMModel",
    "BiLSTMModel",
    "CNN1DModel",
    "CNN2DModel",
    "GNNModel",
    "TemporalTransformerModel",
    "SAModel",
    "GAModel",
    "SymbolicRegressionModel",
    "LCSModel",
    "RuleFitModel",
    "RipperModel",
    "PREModel",
    "HillClimbingEnsemble",
    "WeightedEnsemble",
    "OmniTabular",
]

__version__ = "0.2.0"
