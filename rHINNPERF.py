import os
import pprint
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from base import Base
from HINNPERF.runHINNPerf import get_HINNPerf_MRE_and_predictions, get_HINNPerf_MRE
import json


class HINNPerf(Base):
    def __init__(self, dataset_name, data_path="data", split=0.8, split_mode=True, num_runs=30):
        super().__init__({}, "HINNPerf", dataset_name, data_path, split, split_mode, num_runs)
        self.init_seed = 25
        np.random.seed(self.init_seed)
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.file_path)
        self.whole_data = np.genfromtxt(self.file_path, delimiter=',', skip_header=1)
        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values

    def get_top_1(self):
        top_10 = self.get_top_10()
        return top_10[0][0]  # return the config dict only

    def get_top_10(self):
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train

        config_errors = {}

        for run in tqdm(range(self.num_runs)):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            training_index = indices[:num_train]
            testing_index = indices[num_train:num_train + num_test]

            # Get all configs and their errors for this run
            _, all_configs = get_HINNPerf_MRE_and_predictions([self.whole_data, training_index, testing_index, False, []])
            for config, error in all_configs:
                config_key = json.dumps(config, sort_keys=True)  # use JSON string as key for consistency
                if config_key not in config_errors:
                    config_errors[config_key] = {"config": config, "errors": []}
                config_errors[config_key]["errors"].append(error)

        # Average errors for each config
        averaged_configs = []
        for entry in config_errors.values():
            avg_error = sum(entry["errors"]) / len(entry["errors"])
            averaged_configs.append((entry["config"], avg_error))

        # sort by error ascending and take top 10
        top_10_configs = sorted(averaged_configs, key=lambda x: x[1])[:10]

        return top_10_configs

    def run_experiment(self):
        """
        Runs experiment only on top-1 config from get_top_1().
        Returns a list with a single dict, matching Base.py's run_experiment output.
        """
        top_1 = self.get_top_1()
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train

        param_metrics = {
            "Rank": 1,
            "Params": top_1,
            "Metrics": {
                "MRE": [],
                "MAE": [],
                "SA": [],
                "SA_5": [],
                "D": [],
                "MBRE": [],
                "MIBRE": [],
                "LSD": []
            },
            "Runs": []
        }

        # Load baseline once
        mae_p0, Sp0, Sa_5 = self.load_baseline()

        for run in range(self.num_runs):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            training_index = indices[:num_train]
            testing_index = indices[num_train:num_train + num_test]

            config_with_lists = {k: [v] for k, v in top_1.items()}
            rel_error, y_pred, y_test, X_test, y_train_pred = get_HINNPerf_MRE(
                [self.whole_data, training_index, testing_index, False, config_with_lists]
            )

            mre = rel_error
            mae = mean_absolute_error(y_test, y_pred)
            sa = self.standardized_accuracy(mae, mae_p0)
            d = self.effect_size_test(mae, mae_p0, Sp0)
            mibre, mbre, lsd = self.compute_metrics(y_test, y_pred)

            param_metrics["Metrics"]["MRE"].append(mre)
            param_metrics["Metrics"]["MAE"].append(mae)
            param_metrics["Metrics"]["SA"].append(sa)
            param_metrics["Metrics"]["SA_5"].append(Sa_5)
            param_metrics["Metrics"]["D"].append(d)
            param_metrics["Metrics"]["MBRE"].append(mbre)
            param_metrics["Metrics"]["MIBRE"].append(mibre)
            param_metrics["Metrics"]["LSD"].append(lsd)

            param_metrics["Runs"].append({
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "y_train": self.y[training_index].tolist(),
                "y_train_pred": y_train_pred.tolist()
            })

        # Add mean metric summaries for convenience
        param_metrics["Mean_MAE"] = np.mean(param_metrics["Metrics"]["MAE"])
        param_metrics["Mean_MRE"] = np.mean(param_metrics["Metrics"]["MRE"])
        param_metrics["Mean_MBRE"] = np.mean(param_metrics["Metrics"]["MBRE"])
        param_metrics["Mean_MIBRE"] = np.mean(param_metrics["Metrics"]["MIBRE"])

        return [param_metrics]

    def _run_single_config(self, config, num_train, num_test):
        """
        Runs a single configuration for all runs.
        """
        results = []
        for run in range(self.num_runs):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            training_index = indices[:num_train]
            testing_index = indices[num_train:num_train + num_test]

            X_train, y_train = self.X[training_index], self.y[training_index]

            config_with_lists = {k: [v] for k, v in config.items()}
            _, y_pred, y_test, X_test, y_train_pred = get_HINNPerf_MRE(
                [self.whole_data, training_index, testing_index, False, config_with_lists]
            )

            mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            mae = mean_absolute_error(y_test, y_pred)
            results.append((X_test, y_test, y_pred, mae, mre, X_train, y_train, y_train_pred))
        return results

    def run_single_config_experiment(self, config):
        """
        Runs experiment for a single config set.
        """
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train
        return self._run_single_config(config, num_train, num_test) 
    
    def run_ensemble_experiment(self, params, ensemble_cfg):
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        results = []

        requested_ns = ensemble_cfg["n_estimators_list"]
        max_n = max(requested_ns)

        for run in range(self.num_runs):
            np.random.seed(self.init_seed + run)
            rng = np.random.RandomState(self.init_seed + run)

            indices = np.random.permutation(len(self.y))
            training_index = indices[:num_train]
            testing_index = indices[num_train:]

            train_index_samples = []
            feature_sets = []

            for _ in range(max_n):
                # Row sampling
                if ensemble_cfg["strategy"] in ["bagging", "rp"]:
                    sampled_train_idx = rng.choice(training_index, size=len(training_index), replace=True)
                else:
                    sampled_train_idx = training_index

                train_index_samples.append(sampled_train_idx)

                # Feature sampling
                if ensemble_cfg["strategy"] in ["rs", "rp"] and ensemble_cfg["feature_fraction"] is not None:
                    n_features = self.X.shape[1]
                    if ensemble_cfg["feature_fraction"] == "sqrt":
                        kf = max(1, int(np.sqrt(n_features)))
                    else:
                        kf = max(1, int(n_features * ensemble_cfg["feature_fraction"]))
                    feat_idx = rng.choice(n_features, kf, replace=False)
                else:
                    feat_idx = np.arange(self.X.shape[1])

                feature_sets.append(feat_idx)

            estimator_preds = []
            x_train_samples = []
            y_train_samples = []

            for est_idx in range(max_n):
                feat_idx = feature_sets[est_idx]
                train_idx = train_index_samples[est_idx]

                x_train_samples.append(self.X[train_idx])
                y_train_samples.append(self.y[train_idx])

                config_with_lists = {k: [v] for k, v in params.items()}
                config_with_lists["input_dim"] = [len(feat_idx)]

                data_with_selected_features = self.whole_data[:, list(feat_idx) + [-1]]

                _, y_pred, y_test, _, _ = get_HINNPerf_MRE([
                    data_with_selected_features,
                    train_idx,
                    testing_index,
                    False,
                    config_with_lists
                ])

                estimator_preds.append(np.array(y_pred))

            estimator_preds = np.vstack(estimator_preds)

            # Predict only for requested n_estimators sizes
            for k in requested_ns:
                for combiner in ["mean", "median"]:
                    if combiner == "mean":
                        final_pred = np.mean(estimator_preds[:k], axis=0)
                    else:
                        final_pred = np.median(estimator_preds[:k], axis=0)

                    results.append({
                        "n_estimators": k,
                        "combiner": combiner,
                        # "y_train": [y.tolist() for y in y_train_samples[:k]],
                        "y_test": y_test.tolist(),
                        "y_pred": final_pred.tolist(),
                        # "X_train": [x.tolist() for x in x_train_samples[:k]],
                        # "X_test": self.X[testing_index].tolist(),
                        # "feature_sets": [f.tolist() for f in feature_sets[:k]]
                    })

        return results
