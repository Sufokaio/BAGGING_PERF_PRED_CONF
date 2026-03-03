import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error

from base import Base

class SVRWrapper(Base):
    def __init__(self, dataset_name, data_path="data", split=0.8, split_mode=True, num_runs=30):
        super().__init__( {
            'model': [SVR()],
            'scaler': [StandardScaler()],
            'model__C': [  1, 10, 50, 100],
            'model__kernel': ['rbf'], 

            'model__gamma': ['scale', 'auto'], 
            'model__epsilon': [0.001, 0.01, 0.1, 1]

                                 } , "SVR",dataset_name, data_path, split, split_mode, num_runs)
        

