import os
import pprint
import pandas as pd

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import time


import numpy as np
import json

from ensemble import HomogeneousEnsemble

# Base clase for each model type
class Base:
    def __init__(self, param_grid, model_name, dataset_name, data_path="data", split=0.8, split_mode=True, num_runs=30):
        self.dataset_name = dataset_name
        self.file_path = f"{data_path}/{dataset_name}.csv"
        self.split = split
        self.split_mode = split_mode
        self.num_runs = num_runs
        self.param_grid = param_grid
        self.model_name = model_name
        
        self.init_seed = 25
        np.random.seed(self.init_seed)
        self.load_data()
    
    def load_data(self):
        df = pd.read_csv(self.file_path)
        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values
    
    def load_baseline(self):
        metrics = {}
        with open(f"results/{self.dataset_name}/{self.split}/baseline.txt", "r") as file:
            for line in file:
                if ':' in line:
                    key, value = line.strip().split(':')
                    metrics[key.strip()] = float(value.strip())
        return metrics['MAEp0'], metrics["Sp0"], metrics["SA_5"]
    
    def standardized_accuracy(self, MAEpi, MAEp0):
        return 1 - (MAEpi / MAEp0)
    
    def effect_size_test(self, MAEpi, MAEp0, Sp0):
        return (MAEpi - MAEp0) / Sp0
    

    def compute_metrics(self, actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        n = len(actual)

        # MBRE
        mbre = np.mean(np.abs(actual - predicted) / np.minimum(actual, predicted))
        
        # MIBRE
        mibre = np.mean(np.abs(actual - predicted) / np.maximum(actual, predicted))
        
        # LSD
        lambda_i = np.log(actual) - np.log(predicted)
        s_squared = np.var(lambda_i, ddof=1)
        lsd = np.sqrt(np.sum((lambda_i + (s_squared / 2))**2) / (n - 1))

        return mibre, mbre, lsd
    
    # Construct top 10 single variants
    def get_top_10(self):
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train

        all_scores = {}

        for run in tqdm(range(self.num_runs)):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            X_shuffled, y_shuffled = self.X[indices], self.y[indices]

            X_train, y_train = X_shuffled[:num_train], y_shuffled[:num_train]
            X_test, y_test = X_shuffled[num_train:num_train + num_test], y_shuffled[num_train:num_train + num_test]

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])

            grid_search = GridSearchCV(pipeline, self.param_grid, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            results = pd.DataFrame(grid_search.cv_results_)

            ranked_results = results.sort_values(by='mean_test_score', ascending=False)
                
            for _, row in ranked_results.iterrows():
                params_clean = {}
                for k, v in row['params'].items():

                    # Store model name consistently instead of object repr
                    if k == 'model':
                        params_clean[k] = type(v).__name__   # e.g., "SVR"
                    elif k == 'scaler':
                        params_clean[k] = type(v).__name__   # e.g., "StandardScaler"
                    # Keep real hyperparams as-is
                    else:
                        params_clean[k] = v

                # Unique param set key for cross-run aggregation
                params_key = json.dumps(params_clean, sort_keys=True)

                if params_key not in all_scores:
                    all_scores[params_key] = []
                all_scores[params_key].append(row['mean_test_score'])

        avg_scores = [(json.loads(k), np.mean(v)) for k, v in all_scores.items()]
        top_10_overall = sorted(avg_scores, key=lambda x: x[1], reverse=True)[:10]

    
        return top_10_overall


    def get_top_1(self):
        top_10 = self.get_top_10()
        return top_10[0][0]   # params dict only



    def run_experiment(self):
  

        top_1_params = self.get_top_1()
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train

        param_metrics = {
            "Rank": 1,
            "Params": top_1_params,
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

        for run in range(self.num_runs):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            X_shuffled, y_shuffled = self.X[indices], self.y[indices]

            X_train, y_train = X_shuffled[:num_train], y_shuffled[:num_train]
            X_test, y_test = X_shuffled[num_train:num_train + num_test], y_shuffled[num_train:num_train + num_test]

            params_copy = top_1_params.copy()
            model_name = params_copy.pop('model')
            scaler_name = params_copy.pop('scaler', 'StandardScaler()')

            model = eval(model_name)()
            scaler = eval(scaler_name)()

            formatted_params = {}
            for k, v in params_copy.items():
                param_name = k.split('__')[1]
                if v == 'None':
                    formatted_params[param_name] = None
                elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
                    formatted_params[param_name] = eval(v)
                else:
                    formatted_params[param_name] = v

            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', model.set_params(**formatted_params))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_train_pred = pipeline.predict(X_train)

            mae = mean_absolute_error(y_test, y_pred)
            mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            mibre, mbre, lsd = self.compute_metrics(y_test, y_pred)

            mae_p0, Sp0, Sa_5 = self.load_baseline()
            sa = self.standardized_accuracy(mae, mae_p0)
            d = self.effect_size_test(mae, mae_p0, Sp0)

            param_metrics["Metrics"]["MRE"].append(mre)
            param_metrics["Metrics"]["MAE"].append(mae)
            param_metrics["Metrics"]["MBRE"].append(mbre)
            param_metrics["Metrics"]["MIBRE"].append(mibre)
            param_metrics["Metrics"]["LSD"].append(lsd)

            # Keep SA, SA_5, D empty lists to keep json keys same
            param_metrics["Metrics"]["SA"].append(sa)
            param_metrics["Metrics"]["SA_5"].append(Sa_5)
            param_metrics["Metrics"]["D"].append(d)
            param_metrics["Mean_MAE"] = np.mean(param_metrics["Metrics"]["MAE"])
            param_metrics["Mean_MRE"] = np.mean(param_metrics["Metrics"]["MRE"])
            param_metrics["Mean_MBRE"] = np.mean(param_metrics["Metrics"]["MBRE"])
            param_metrics["Mean_MIBRE"] = np.mean(param_metrics["Metrics"]["MIBRE"])

            param_metrics["Runs"].append({
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "y_train": y_train.tolist(),
                "y_train_pred": y_train_pred.tolist()
            })

        return [param_metrics] 

    # Run only one variant
    def _run_single_config(self, params, num_train, num_test):
        """
        Runs a single configuration for all runs.
        """
        results = []
        for run in range(self.num_runs):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            X_shuffled, y_shuffled = self.X[indices], self.y[indices]

            X_train, y_train = X_shuffled[:num_train], y_shuffled[:num_train]


            
            X_test, y_test = X_shuffled[num_train:num_train + num_test], y_shuffled[num_train:num_train + num_test]

            params_copy = params.copy()
            model_name = params_copy.pop('model')
            scaler_name = params_copy.pop('scaler', 'StandardScaler()')

            model = eval(model_name)()
            scaler = eval(scaler_name)()

            formatted_params = {}
            for k, v in params_copy.items():
                param_name = k.split('__')[1]
                # Convert string 'None' to Python None
                if v == 'None':
                    formatted_params[param_name] = None
                elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
                    formatted_params[param_name] = eval(v)
                else:
                    formatted_params[param_name] = v

            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', model.set_params(**formatted_params))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_train_pred = pipeline.predict(X_train)


            mae = mean_absolute_error(y_test, y_pred)

            mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            results.append((X_test, y_test, y_pred, mae , mre, X_train, y_train, y_train_pred))
        return results

    def run_single_config_experiment(self, params):
        """
        Runs experiment for a single config set (params).
        """
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train
        return self._run_single_config(params, num_train, num_test)


    def run_ensemble_experiment(self, params, ensemble_cfg):
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        results = []

        requested_ns = ensemble_cfg["n_estimators_list"]
        max_n = max(requested_ns)

        for run in range(self.num_runs):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            X_train, y_train = self.X[indices[:num_train]], self.y[indices[:num_train]]
            X_test, y_test = self.X[indices[num_train:]], self.y[indices[num_train:]]

            params_copy = params.copy()
            model = eval(params_copy.pop("model"))()
            scaler = eval(params_copy.pop("scaler", "StandardScaler()"))()

            for k, v in params_copy.items():
                model.set_params(**{k.split("__")[1]: None if v == "None" else v})

            ensemble = HomogeneousEnsemble(
                base_model=model,
                base_scaler=scaler,
                **{k: v for k, v in ensemble_cfg.items() if k != "n_estimators_list"},
                random_state=self.init_seed + run
            )

            # Train ONCE with max_n estimators
            ensemble.fit_incremental(X_train, y_train, max_n)

            # Predict only for requested n_estimators sizes
            for k in requested_ns:
                for combiner in ["mean", "median"]:
                    ensemble.combiner = combiner
                    y_pred = ensemble.predict_subset(X_test, k)

                    feature_sets = [
                        list(range(self.X.shape[1])) if f is None else f.tolist()
                        for f in ensemble.feature_indices[:k]
                    ]

                    results.append({
                        "n_estimators": k,
                        "combiner": combiner,
                        # "y_train": [y.tolist() for _, y in ensemble.sampled_data[:k]],
                        "y_test": y_test.tolist(),
                        "y_pred": y_pred.tolist(),
                        # "X_train": [x.tolist() for x, _ in ensemble.sampled_data[:k]],
                        # "X_test": X_test.tolist(),
                        # "feature_sets": feature_sets
                    })

        return results
