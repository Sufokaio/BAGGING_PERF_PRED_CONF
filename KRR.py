import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error

from base import Base

class KRR(Base):
    def __init__(self, dataset_name, data_path="data", split=0.8, split_mode=True, num_runs=30):
        super().__init__( {
            'model': [KernelRidge()],
            'scaler': [StandardScaler()], 
            'model__alpha': [0.001, 0.01, 0.1, 1],
            'model__kernel': ['rbf', 'polynomial'],
            'model__degree': [2, 3, 4, 5],

            'model__gamma': [0.001, 0.01, 0.1],
            'model__coef0': [ 0, 2, 4, 6]

                                 }, "KRR",dataset_name, data_path, split, split_mode, num_runs)
        