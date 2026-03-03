import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from DEEPPERF.mlp_sparse_model_tf2 import MLPSparseModel
from DEEPPERF.mlp_plain_model_tf2 import MLPPlainModel
import json
from base import Base

def nn_l1_val(X_train1, Y_train1, X_train2, Y_train2, n_layer, lambd, lr_initial):
    config = dict()
    config['num_input'] = X_train1.shape[1]
    config['num_layer'] = n_layer
    config['num_neuron'] = 128
    config['lambda'] = lambd
    config['verbose'] = 0

    model = MLPSparseModel(config)
    model.build_train()
    model.train(X_train1, Y_train1, lr_initial)

    Y_pred_val = model.predict(X_train2)
    abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
    rel_error = np.mean(np.abs(np.divide(Y_train2 - Y_pred_val, Y_train2)))
    return abs_error, rel_error




 
def flatten(a):
    return [sublist[0] for sublist in a]


class DeepPerf(Base):
    def __init__(self, dataset_name, data_path="data", split=0.8, split_mode=True, num_runs=30):
        super().__init__({}, "DeepPerf", dataset_name, data_path, split, split_mode, num_runs)
        self.init_seed = 25
        np.random.seed(self.init_seed)
        self.load_data()

    def get_top_10(self):
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train
        all_scores = {}

        for run in tqdm(range(self.num_runs)):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            X_shuffled, y_shuffled = self.X[indices], self.y[indices]

            X_train, Y_train = X_shuffled[:num_train], y_shuffled[:num_train]
            X_test, y_test = X_shuffled[num_train:num_train + num_test], y_shuffled[num_train:num_train + num_test]
            Y_train = Y_train[:, np.newaxis]

            (N, n) = X_train.shape
            max_X = np.amax(X_train, axis=0)
            max_X[max_X == 0] = 1
            X_train = np.divide(X_train, max_X)
            max_Y = np.max(Y_train) / 100
            if max_Y == 0:
                max_Y = 1
            Y_train = np.divide(Y_train, max_Y)

            N_train = num_train
            N_cross = int(np.ceil(N_train * 2 / 3))
            X_train1 = X_train[0:N_cross, :]
            Y_train1 = Y_train[0:N_cross]
            X_train2 = X_train[N_cross:N_train, :]
            Y_train2 = Y_train[N_cross:N_train]

            config = dict()
            config['num_input'] = n
            config['num_neuron'] = 128
            config['lambda'] = 'NA'
            config['decay'] = 'NA'
            config['verbose'] = 0
            abs_error_all = np.zeros((15, 4))
            abs_error_all_train = np.zeros((15, 4))
            abs_error_layer_lr = np.zeros((15, 2))
            abs_err_layer_lr_min = 100
            count = 0
            layer_range = range(2, 15)
            lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 4)
            for n_layer in layer_range:
                config['num_layer'] = n_layer
                for lr_index, lr_initial in enumerate(lr_range):
                    model = MLPPlainModel(config)
                    model.build_train()
                    model.train(X_train1, Y_train1, lr_initial)

                    Y_pred_train = model.predict(X_train1)
                    abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
                    abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                    Y_pred_val = model.predict(X_train2)
                    abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
                    abs_error_all[int(n_layer), lr_index] = abs_error

                temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
                temp_idx = np.where(abs(temp) < 0.0001)[0]
                if len(temp_idx) > 0:
                    lr_best = lr_range[np.max(temp_idx)]
                    err_val_best = abs_error_all[int(n_layer), np.max(temp_idx)]
                else:
                    lr_best = lr_range[np.argmin(temp)]
                    err_val_best = abs_error_all[int(n_layer), np.argmin(temp)]

                abs_error_layer_lr[int(n_layer), 0] = err_val_best
                abs_error_layer_lr[int(n_layer), 1] = lr_best

                if abs_err_layer_lr_min >= abs_error_all[int(n_layer), np.argmin(temp)]:
                    abs_err_layer_lr_min = abs_error_all[int(n_layer), np.argmin(temp)]
                    count = 0
                else:
                    count += 1

                if count >= 2:
                    break
            abs_error_layer_lr = abs_error_layer_lr[abs_error_layer_lr[:, 1] != 0]
            n_layer_opt = layer_range[np.argmin(abs_error_layer_lr[:, 0])] + 5

            config['num_layer'] = n_layer_opt
            for lr_index, lr_initial in enumerate(lr_range):
                model = MLPPlainModel(config)
                model.build_train()
                model.train(X_train1, Y_train1, lr_initial)

                Y_pred_train = model.predict(X_train1)
                abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
                abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                Y_pred_val = model.predict(X_train2)
                abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
                abs_error_all[int(n_layer), lr_index] = abs_error

            temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
            temp_idx = np.where(abs(temp) < 0.0001)[0]
            if len(temp_idx) > 0:
                lr_best = lr_range[np.max(temp_idx)]
            else:
                lr_best = lr_range[np.argmin(temp)]
            lr_opt = lr_best

            lambda_range = np.logspace(-2, np.log10(1000), 30)
            error_min = np.zeros((1, len(lambda_range)))
            for idx, lambd in enumerate(lambda_range):
                val_abserror, _ = nn_l1_val(X_train1, Y_train1, X_train2, Y_train2, n_layer_opt, lambd, lr_opt)
                error_min[0, idx] = val_abserror

            for idx, lambd in enumerate(lambda_range):
                config = {
                    'num_input': n,
                    'num_neuron': 128,
                    'lambda': float(lambd),
                    'num_layer': int(n_layer_opt),
                    'lr': float(lr_opt)
                }
                error = float(error_min[0, idx])
                # key = json.dumps({k: str(v) for k, v in config.items()}, sort_keys=True)
                key = json.dumps(config, sort_keys=True)

                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(error)

        avg_errors = []
        for key, errors in all_scores.items():
            avg_errors.append((key, np.mean(errors)))
        avg_errors.sort(key=lambda x: x[1])
        top_10_configs_overall = avg_errors[:10]

        return [(json.loads(key), avg_error) for key, avg_error in top_10_configs_overall]

    def run_experiment(self):
        top_10 = self.get_top_10()
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train

        metrics_results = []

        for rank, (params, _) in enumerate(top_10, 1):
            param_metrics = {
                "Rank": rank,
                "Params": params,
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
                "Runs": []  # all predictions 
            }

            for run in range(self.num_runs):
                np.random.seed(self.init_seed + run)
                indices = np.random.permutation(len(self.y))
                X_shuffled, y_shuffled = self.X[indices], self.y[indices]

                X_train, Y_train = X_shuffled[:num_train], y_shuffled[:num_train]
                X_test, y_test = X_shuffled[num_train:num_train + num_test], y_shuffled[num_train:num_train + num_test]
                Y_train = Y_train[:, np.newaxis]
                y_test = y_test[:, np.newaxis]

                max_X = np.amax(X_train, axis=0)
                max_X[max_X == 0] = 1
                X_train_norm = np.divide(X_train, max_X)
                X_test_norm = np.divide(X_test, max_X)
                max_Y = np.max(Y_train) / 100
                if max_Y == 0:
                    max_Y = 1
                Y_train_norm = np.divide(Y_train, max_Y)

                config = params.copy()
                config['decay'] = 'NA'
                config['verbose'] = 0
                model = MLPSparseModel(config)
                model.build_train()
                model.train(X_train_norm, Y_train_norm, config.get('lr', 0.001))

                y_pred_test = model.predict(X_test_norm)
                y_pred = max_Y * y_pred_test

                y_pred_train = model.predict(X_train_norm)
                y_train_pred = max_Y * y_pred_train

                mae = mean_absolute_error(y_test, y_pred)
                mae_p0, Sp0, Sa_5 = self.load_baseline()
                sa = self.standardized_accuracy(mae, mae_p0)
                d = self.effect_size_test(mae, mae_p0, Sp0)
                mibre, mbre, lsd = self.compute_metrics(y_test, y_pred)

                param_metrics["Metrics"]["MRE"].append(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
                param_metrics["Metrics"]["MAE"].append(mae)
                param_metrics["Metrics"]["SA"].append(sa)
                param_metrics["Metrics"]["SA_5"].append(Sa_5)
                param_metrics["Metrics"]["D"].append(d)
                param_metrics["Metrics"]["MBRE"].append(mbre)
                param_metrics["Metrics"]["MIBRE"].append(mibre)
                param_metrics["Metrics"]["LSD"].append(lsd)

                # Store run data for aggregation
                param_metrics["Runs"].append({
                    # "X_test": X_test.tolist(),
                    "y_test": y_test.ravel().tolist(),
                    "y_pred": y_pred.ravel().tolist(),
                    # "mae": mae,
                    # "mre": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
                    # "X_train": X_train.tolist(),
                    "y_train": Y_train.ravel().tolist(),
                    "y_train_pred": y_train_pred.ravel().tolist()
                })

            metrics_results.append(param_metrics)


        return metrics_results
    

    def _run_single_config(self, params, num_train, num_test):
        """
        Runs a single configuration for all runs.
        """
        results = []
        for run in range(self.num_runs):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            X_shuffled, y_shuffled = self.X[indices], self.y[indices]

            X_train, Y_train = X_shuffled[:num_train], y_shuffled[:num_train]
            X_test, y_test = X_shuffled[num_train:num_train + num_test], y_shuffled[num_train:num_train + num_test]
            Y_train = Y_train[:, np.newaxis]
            y_test = y_test[:, np.newaxis]

            max_X = np.amax(X_train, axis=0)
            max_X[max_X == 0] = 1
            X_train_norm = np.divide(X_train, max_X)
            X_test_norm = np.divide(X_test, max_X)
            max_Y = np.max(Y_train) / 100
            if max_Y == 0:
                max_Y = 1
            Y_train_norm = np.divide(Y_train, max_Y)

            config = params.copy()
            config['decay'] = 'NA'
            config['verbose'] = 0
            model = MLPSparseModel(config)
            model.build_train()
            model.train(X_train_norm, Y_train_norm, config.get('lr', 0.001))

            y_pred_test = model.predict(X_test_norm)
            y_pred = max_Y * y_pred_test

            y_pred_train = model.predict(X_train_norm)
            y_train_pred = max_Y * y_pred_train


            mae = mean_absolute_error(y_test, y_pred)

            mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            

            results.append((X_test, y_test.ravel(), y_pred.ravel(), mae, mre, X_train, Y_train.ravel(), y_train_pred.ravel()))
        return results

    def run_single_config_experiment(self, params):
        """
        Runs experiment for a single config set (params).
        """
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train
        return self._run_single_config(params, num_train, num_test)