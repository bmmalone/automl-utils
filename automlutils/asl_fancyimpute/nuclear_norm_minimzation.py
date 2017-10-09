import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.constants import *

import automlutils.automl_utils as automl_utils
from automlutils.asl_fancyimpute.base import AutoSklearnImputationAlgorithm

class NuclearNormMinimization(AutoSklearnImputationAlgorithm):
    
    def __init__(self, require_symmetric_solution, fast_but_approximate,
            random_state=None):

        self.require_symmetric_solution = require_symmetric_solution
        self.init_fill_method = init_fill_method
        self.random_state = random_state

    def fit(self, X, y=None):
        import fancyimpute

        self.error_tolerance = 0.0001

        self.imputer = fancyimpute.NuclearNormMinimization(
            require_symmetric_solution=self.require_symmetric_solution,
            error_tolerance=self.error_tolerance,
            fast_but_approximate=self.fast_but_approximate,
            min_value=None,
            max_value=None,
            verbose=False
        )

        return self

    def transform(self, X):
        if self.imputer is None:
            msg = "The NuclearNormMinimization imputer was not fit"
            raise NotImplementedError(msg)

        # check if we are already complete
        m_missing = np.isnan(X)
        if not m_missing.any():
            return X
        
        X_filled = self.imputer.complete(X)
        
        return X_filled
        
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'NNM Fill',
            'name': 'NuclearNormMinimization Fill',
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

        require_symmetric_solution = automl_utils.create_binary_hyperparameter("require_symmetric_solution")
        fast_but_approximate = automl_utils.create_binary_hyperparameter("fast_but_approximate")        

        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            require_symmetric_solution,
            fast_but_approximate
        ])
        return cs   
