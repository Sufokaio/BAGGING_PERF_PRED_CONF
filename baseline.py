import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import os


class Baseline:
    def __init__(self, dataset_name, data_path="data", split=0.8, split_mode=True, num_runs=30):
        self.dataset_name = dataset_name
        self.file_path = f"{data_path}/{dataset_name}.csv"
        self.split = split
        self.split_mode = split_mode
        self.num_runs = num_runs
        # ignored
        self.param_grid = {
            'model': [Ridge(), Lasso(max_iter=7000)],
            'scaler': [StandardScaler(), MinMaxScaler(), None],
            'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        }
        self.init_seed = 25
        np.random.seed(self.init_seed)
        self.load_data()
    
    def load_data(self):
        df = pd.read_csv(self.file_path)
        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values
    
    def random_guessing_mae(self, target_values, num_trials=200):
        target_values = np.array(target_values)
        n = len(target_values)
        mae_values = []
        for _ in range(num_trials):
            predictions = [np.random.choice(np.delete(target_values, i)) for i in range(n)] 
            #np.random.choice(target_values, size=n, replace=False)
            mae_values.append(mean_absolute_error(target_values, predictions))
        return np.mean(mae_values), np.std(mae_values, ddof=1), np.percentile(mae_values, 5)
    


    
    def run_experiment(self):
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train
        
        metrics = { "MAEp0": [], "Sp0": [], "Q5p0": [] }

        for run in tqdm(range(self.num_runs)):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            X_shuffled, y_shuffled = self.X[indices], self.y[indices]

            X_train, y_train = X_shuffled[:num_train], y_shuffled[:num_train]
            X_test, y_test = X_shuffled[num_train:num_train + num_test], y_shuffled[num_train:num_train + num_test]

        
            mae_p0, Sp0, quantile_5 = self.random_guessing_mae(y_test)

            metrics["MAEp0"].append(mae_p0)
            metrics["Sp0"].append(Sp0)
            metrics["Q5p0"].append(quantile_5)
        
        for key, values in metrics.items():
                metrics[key] = np.mean(values)
        
        metrics["SA_5"] = 1 - ( metrics["Q5p0"] / metrics["MAEp0"])




        os.makedirs(f"results/{self.dataset_name}/{self.split}", exist_ok=True)
        with open(f"results/{self.dataset_name}/{self.split}/baseline.txt", "a") as file:
            for key, values in metrics.items():
                file.write(f"{key}: {values:.4f}\n")

        return metrics
