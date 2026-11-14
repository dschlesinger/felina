"""TabPFN regressor for the CYP challenge.

TabPFN-3.5 accepts at most 20,000 features and 1,000,000 rows, so wide
representations may need reducing before they fit: pass --dimred to evaluate-models
(e.g. --dimred pca[256], which matches the PCA this model used to apply itself).
"""

from typing import ClassVar, Literal

import numpy as np

from models.base import CYPModel

# Above this many features evaluate-models warns when no --dimred is given.
WIDE_FEATURES = 256


class TabPFN(CYPModel):
    """In-context tabular regressor; no gradient training happens in fit().

    Callers filter missing labels with models.utils.drop_nan_rows.
    """

    name: ClassVar[str] = "tabpfn"

    def __init__(
        self,
        n_estimators: int | Literal["auto"] = "auto",
        softmax_temperature: float | Literal["auto"] = "auto",
        ignore_pretraining_limits: bool = False,
        device: str = "auto",
        model_path: str = "auto",
        random_state: int = 42,
    ) -> None:
        from tabpfn import TabPFNRegressor

        self._model = TabPFNRegressor(
            n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            ignore_pretraining_limits=ignore_pretraining_limits,
            device=device,
            model_path=model_path,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
