import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.constants import *

import automlutils.automl_utils as automl_utils
from automlutils.asl_fancyimpute.base import AutoSklearnImputationAlgorithm

class IterativeSVD(AutoSklearnImputationAlgorithm):
    
    def __init__(self,
            rank=10,
            gradual_rank_increase=True,
            init_fill_method='zero',
            random_state=None):

        self.rank = rank
        self.gradual_rank_increase = gradual_rank_increase
        self.init_fill_method = init_fill_method
        self.random_state = random_state

    def fit(self, X, y=None):
        import fancyimpute

        self.convergence_threshold = 0.001

        self.imputer = fancyimpute.IterativeSVD(
            rank=self.rank,
            convergence_threshold=self.convergence_threshold,
            gradual_rank_increase=self.gradual_rank_increase,
            init_fill_method=self.init_fill_method,
            svd_algorithm='arpack', 
            min_value=None,
            max_value=None,
            max_iters=200,
            verbose=False
        )

        return self

    def transform(self, X):
        if self.imputer is None:
            raise NotImplementedError("The IterativeSVD imputer was not fit")

        # check if we are already complete
        m_missing = np.isnan(X)
        if not m_missing.any():
            return X
        
        X_filled = self.imputer.complete(X)
        
        return X_filled
        
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'IterativeSVD Fill',
            'name': 'IterativeSVD Fill',
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

        rank = automl_utils.create_modest_integer_hyperparameter("rank")
        gradual_rank_increase = automl_utils.create_binary_hyperparameter("gradual_rank_increase")
        init_fill_method =  CategoricalHyperparameter(
            "init_fill_method",
            ["zero", "mean", "median", "min", "random"],
            default="zero"
        )
        

        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            rank,
            gradual_rank_increase,
            init_fill_method
        ])
        return cs   
