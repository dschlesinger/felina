"""Model registry for the OpenADMET CYP challenge.

To add a new model: create a module in this directory, subclass CYPModel,
set a unique `name` class variable, and add it here.
"""

from models.base import CYPModel
from models.decision_tree import DecisionTree
from models.ridge import Ridge
from models.random_forest import RandomForest
from models.gbm import LightGBM
from models.XGBoost import XGBoost
from models.TabPFN import TabPFN

REGISTRY: dict[str, type[CYPModel]] = {
    DecisionTree.name: DecisionTree,
    Ridge.name: Ridge,
    RandomForest.name: RandomForest,
    LightGBM.name: LightGBM,
    XGBoost.name: XGBoost,
    TabPFN.name: TabPFN,
}

__all__ = ["CYPModel", "REGISTRY"]
