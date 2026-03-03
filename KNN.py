import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

from base import Base
class KNN(Base):
    def __init__(self, dataset_name, data_path="data", split=0.8, split_mode=True, num_runs=30):
        super().__init__({
            'model': [KNeighborsRegressor()],
            'scaler': [StandardScaler()],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['auto', 'kd_tree', 'ball_tree'],
        }, "KNN", dataset_name, data_path, split, split_mode, num_runs)

    def get_top_10(self):
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split

        max_k = max(1, int(num_train * 0.1))

        if max_k <= 6:
            neighbors = [1,2,3,4,5]  
        else:
            neighbors = np.linspace(1, max_k, 6, dtype=int).tolist()
            neighbors = sorted(set(neighbors))
            while len(neighbors) < 6:
                neighbors.append(min(max_k, neighbors[-1] + 1))
                neighbors = sorted(set(neighbors))

        self.param_grid['model__n_neighbors'] = neighbors

        return super().get_top_10()

