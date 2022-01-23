import numpy as np

from haystack.optimal_sampling_classifier import OptimalSamplingClassifier


class ThresholdTuner(OptimalSamplingClassifier):

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.base_estimator.fit(X=X, y=y)
        self._threshold = None
        super().fit(X=X, y=y)

    def _fit_parameters(self, X: np.ndarray, y: np.ndarray) -> None:
        predicted_proba = np.unique(self.predict_proba(X=X)[:, 1])
        loss = []
        for threshold in predicted_proba:
            self._threshold = threshold
            loss.append(self._compute_loss(X=X, y=y).mean())
        return predicted_proba[np.argmin(loss)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.predict_proba(X=X)[:, 1] > self.threshold, self.minority_class, self.majority_class)

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (
            (self.predict(X=X) == y)
            * np.argwhere(y == self.minority_class, self.false_negative_cost, self.false_positive_cost)
        )

    @property
    def threshold(self) -> float:
        return self._threshold
