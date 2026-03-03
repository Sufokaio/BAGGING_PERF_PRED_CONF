# ensemble.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import clone

class HomogeneousEnsemble:
    def __init__(
        self,
        base_model,
        base_scaler,
        # n_estimators,
        strategy,
        combiner="mean",
        feature_fraction=None,
        random_state=25
    ):
        self.base_model = base_model
        self.base_scaler = base_scaler
        # self.n_estimators = n_estimators
        self.strategy = strategy
        self.combiner = combiner
        self.feature_fraction = feature_fraction
        self.random_state = random_state

        self.models = []
        self.feature_indices = []
        self.sampled_data = []

    def _sample_features(self, X, rng):
        if self.feature_fraction is None:
            return X, None

        n_features = X.shape[1]
        if self.feature_fraction == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        else:
            k = max(1, int(n_features * self.feature_fraction))

        idx = rng.choice(n_features, k, replace=False)
        return X[:, idx], idx

    def fit_incremental(self, X, y, max_estimators):
        """
        Train max_estimators once, store everything.
        """
        self.models = []
        self.feature_indices = []
        self.sampled_data = []

        rng = np.random.RandomState(self.random_state)

        for _ in range(max_estimators):
            # ---- Sample rows ----
            if self.strategy in ["bagging", "rp"]:
                idx = rng.choice(len(X), len(X), replace=True)
                X_i, y_i = X[idx], y[idx]
            else:
                X_i, y_i = X, y

            self.sampled_data.append((X_i, y_i))

            # ---- Sample features ----
            if self.strategy in ["rs", "rp"]:
                X_i, feat_idx = self._sample_features(X_i, rng)
            else:
                feat_idx = None

            model = clone(self.base_model)
            scaler = clone(self.base_scaler)

            pipe = Pipeline([
                ("scaler", scaler),
                ("model", model)
            ])

            pipe.fit(X_i, y_i)

            self.models.append(pipe)
            self.feature_indices.append(feat_idx)

    def predict_subset(self, X, k, return_all_preds=False):
        """
        Predict using only first k estimators.
        """
        preds = []

        for model, feat_idx in zip(self.models[:k], self.feature_indices[:k]):
            X_i = X[:, feat_idx] if feat_idx is not None else X
            preds.append(model.predict(X_i))

        preds = np.vstack(preds)

        if return_all_preds:
            return preds

        return (
            np.mean(preds, axis=0)
            if self.combiner == "mean"
            else np.median(preds, axis=0)
        )
