import logging
from typing import Tuple, Optional, Union, Type

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV


class OptimalSamplingClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
            self,
            base_estimator: Type[ClassifierMixin, BaseEstimator],
            false_positive_cost: float,
            false_negative_cost: float,
            max_iter: int = 20,
            max_step: float = 0.1,
            cv: int = 5,
            tol: float = 1e-3,
            verbose: bool = True,
            random_state: Optional[int] = None
    ):
        self.base_estimator = base_estimator
        self.false_positive_cost = false_positive_cost
        self.false_negative_cost = false_negative_cost
        self.max_iter = max_iter
        self.max_step = max_step
        self.cv = cv
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _validate_data(
            self,
            X: np.ndarray,
            y: np.ndarray = None,
            reset: bool = True,
            validate_separately: bool = False,
            **check_params
    ) -> None:
        super()._validate_data(X=X, y=y, validate_separately=validate_separately, **check_params)
        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise NotImplementedError(
                f"Optimal sampling has only been implemented for binary classification problems. "
                f"The provided data contains {n_classes} classes."
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._validate_data(X=X, y=y)
        self._initialize_sampler(y=y)
        for i in range(self.max_iter):
            if self.verbose:
                logging.info(
                    f"Iter: {str(i + 1)} \t"
                    f"Sampling probability: {'{:.2e}'.format(self.sampling_proba)} \t"
                )
            self._update_sampling_proba(X=X, y=y)
            if i:
                if abs(self.sampling_proba - self._sampling_proba_history[-1]) < self.tol:
                    if self.verbose:
                        logging.info(f"Convergence tolerance reached. Terminating search.")
                    break
                elif i > 1:
                    if abs(self.sampling_proba - self._sampling_proba_history[-2]) < self.tol:
                        self.max_step *= 0.5
                        if self.verbose:
                            logging.info(
                                f"Oscillation in sampling probability detected. "
                                f"Reducing max step size to {'{:.2e}'.format(self.max_step)}"
                            )

        self._fit_parameters(*self._resample(X=X, y=y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base_estimator.predict(X=X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_estimator.predict_proba(X=X)

    def _initialize_sampler(self, y: np.ndarray):
        classes, counts = np.unique(y, return_counts=True)
        self._minority_class = classes[counts.argmin()]
        self._majority_class = classes[counts.argmax()]
        self._nominal_proba = counts.argmin() / len(y)
        self._sampling_proba = 0.5
        self._sampling_proba_history = []
        self._rng = np.random.RandomState(self.random_state)

    def _update_sampling_proba(self, X: np.ndarray, y: np.ndarray) -> None:
        self._sampling_proba_history.append(self._sampling_proba)
        X_resampled, y_resampled = self._resample(X, y)
        self._sampling_proba = float(
            np.clip(
                np.mean([
                    self._estimate_optimal_sampling_proba_from_fold(
                        X=X_resampled,
                        y=y_resampled,
                        train=train,
                        val=val
                    )
                    for train, val in StratifiedKFold(n_splits=self.cv).split(X=X, y=y)
                ]),
                a_max=max(self.sampling_proba - self.max_step, 0),
                a_min=min(self.sampling_proba + self.max_step, 1)
            )
        )

    def _resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # If sampling probability matches nominal class probability, return data without resampling
        if np.isclose(self.sampling_proba, self.nominal_proba):
            return X, y

        # Select classes to over/under-sample
        undersampled_class, oversampled_class = (
            (self.majority_class, self.minority_class)
            if self.sampling_proba > self.nominal_proba
            else (self.minority_class, self.majority_class)
        )

        # Select samples to include in resampled data
        oversampled_mask = y == oversampled_class
        n_samples = y.shape[0]
        n_negative_samples = max(
            int(np.ceil(n_samples * np.mean(oversampled_mask) / self.sampling_proba * (1 - self.sampling_proba))),
            2 * self.cv
        )
        undersampled_mask = np.zeros(n_samples, dtype=bool)
        undersampled_mask[self._rng.permutation(np.argwhere(~oversampled_mask))[:n_negative_samples]] = True
        mask = undersampled_mask | oversampled_mask

        return X[mask], y[mask]

    def _estimate_optimal_sampling_proba_from_fold(
            self,
            X: np.ndarray,
            y: np.ndarray,
            train: np.ndarray,
            val: np.ndarray
    ) -> float:
        self._fit_parameters(X=X[train], y=y[train])
        loss = self._compute_loss(X=X[val], y=y[val])
        minority_mask = y[val] == self.minority_class
        return float(
            np.clip(
                1 - (1 - self._nominal_proba) * np.sqrt((loss[~minority_mask] ** 2).mean()) / loss.mean()
                if self.sampling_proba >= self.nominal_proba
                else self._nominal_proba * np.sqrt((loss[minority_mask] ** 2).mean()) / loss.mean(),
                a_min=self.tol,
                a_max=1 - self.tol
            )
        )

    def _fit_parameters(self, X: np.ndarray, y: np.ndarray) -> None:

        # Calibrate class weights used by base estimator
        class_weight = {
            self.minority_class: self.false_negative_cost * self.nominal_proba / self.sampling_proba,
            self.majority_class: self.false_positive_cost * (1 - self.nominal_proba) / (1 - self.sampling_proba),
        }
        if isinstance(self.base_estimator, GridSearchCV):
            self.base_estimator.estimator.class_weight = class_weight
        else:
            self.base_estimator.class_weight = class_weight

        # Refit base estimator
        self.base_estimator.fit(X=X, y=y)

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        minority_proba = self.predict_proba(X=X)[:, 1]
        minority_mask = y == self.minority_class
        return (
            (np.log(minority_proba) * minority_mask + np.log(1 - minority_proba) * (~minority_mask))
            * np.where(minority_mask, self.false_negative_cost, self.false_positive_cost)
        )

    @property
    def sampling_proba(self) -> float:
        return self._sampling_proba

    @property
    def nominal_proba(self) -> float:
        return self._nominal_proba

    @property
    def majority_class(self) -> Union[int, bool, str]:
        return self._majority_class

    @property
    def minority_class(self) -> Union[int, bool, str]:
        return self._minority_class