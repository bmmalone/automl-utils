import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.constants import *

import automlutils.automl_utils as automl_utils
from automlutils.asl_fancyimpute.base import AutoSklearnImputationAlgorithm

class BiScaler(AutoSklearnImputationAlgorithm):
    
    def __init__(self, center_rows, center_columns, scale_rows, scale_columns,
            random_state=None):

        self.center_rows = center_rows
        self.center_columns = center_columns
        self.scale_rows = scale_rows
        self.scale_columns = scale_columns
        self.random_state = random_state

    def fit(self, X, y=None):
        import fancyimpute

        self.tolerance = 0.001

        self.imputer = fancyimpute.BiScaler(
            center_rows=self.center_rows,
            center_columns=self.center_columns,
            scale_rows=self.scale_rows,
            scale_columns=self.scale_columns,
            tolerance=self.tolerance,
            min_value=None,
            max_value=None,
            max_iters=100,
            verbose=False
        )

        return self

    def transform(self, X):
        if self.imputer is None:
            raise NotImplementedError("The BiScaler imputer was not fit")

        # check if we are already complete
        m_missing = np.isnan(X)
        if not m_missing.any():
            return X
        
        X_filled = self.imputer.complete(X)
        
        return X_filled
        
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'BiScaler Fill',
            'name': 'BiScaler Fill',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            # TODO document that we have to be very careful
            'is_deterministic': False,
            'input': (DENSE, UNSIGNED_DATA),
            'output': (DENSE, UNSIGNED_DATA)
        }
        
     
    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):

        center_rows = automl_utils.create_binary_hyperparameter("center_rows")
        center_columns = automl_utils.create_binary_hyperparameter("center_columns")
        scale_rows = automl_utils.create_binary_hyperparameter("scale_rows")
        scale_columns = automl_utils.create_binary_hyperparameter("scale_columns")
        
        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            center_rows,
            center_columns,
            scale_rows,
            scale_columns
        ])
        return cs   
