import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


from base import Base

class RF(Base):
    def __init__(self, dataset_name, data_path="data", split=0.8, split_mode=True, num_runs=30):
        super().__init__( {
            'model': [RandomForestRegressor(random_state=0)],
            'scaler': [StandardScaler()], 
            'model__criterion': ['squared_error'],
            'model__max_depth': [3, 5,  10, 20, 25],

            'model__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'model__min_samples_leaf': [2, 6, 10, 20, 30],
            'model__max_features': [None, "sqrt" ]
                                 }, "RF",dataset_name, data_path, split, split_mode, num_runs)
    