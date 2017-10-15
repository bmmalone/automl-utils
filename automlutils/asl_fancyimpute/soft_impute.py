import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.constants import *

import automlutils.automl_utils as automl_utils
from automlutils.asl_fancyimpute.base import AutoSklearnImputationAlgorithm

class SoftImpute(AutoSklearnImputationAlgorithm):
    
    def __init__(self,
            max_rank=10,
            n_power_iterations=1,
            init_fill_method='zero',
            random_state=None):
    
        self.max_rank = max_rank
        self.n_power_iterations = n_power_iterations
        self.init_fill_method = init_fill_method
        self.random_state = random_state

    def fit(self, X, y=None):
        import fancyimpute

        self.convergence_threshold = 0.001
        self.imputer = fancyimpute.SoftImpute(
            shrinkage_value=None,
            convergence_threshold=self.convergence_threshold,
            max_rank=self.max_rank,
            power_iterations=n_power_iterations,
            init_fill_method=self.init_fill_method,
            normalizer=None,
            min_value=None,
            max_value=None,
            max_iters=100,
            verbose=False
        )

        return self

    def transform(self, X):
        if self.imputer is None:
            raise NotImplementedError("The SoftImpute imputer was not fit")

        # check if we are already complete
        m_missing = np.isnan(X)
        if not m_missing.any():
            return X
        
        X_filled = self.imputer.complete(X)
        
        return X_filled
        
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'SI Fill',
            'name': 'Soft Impute Fill',
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

        max_rank = automl_utils.create_modest_integer_hyperparameter("max_rank")
        n_power_iterations = automl_utils.create_modest_integer_hyperparameter(
            "n_power_iterations",
            default=1,
            upper=5
        )
        init_fill_method =  CategoricalHyperparameter(
            "init_fill_method",
            ["zero", "mean", "median", "min", "random"],
            default="zero"
        )
        

        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            max_rank,
            n_power_iterations,
            init_fill_method
        ])
        return cs   
