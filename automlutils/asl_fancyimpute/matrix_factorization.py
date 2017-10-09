import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from autosklearn.pipeline.constants import *

import automlutils.automl_utils as automl_utils
from automlutils.asl_fancyimpute.base import AutoSklearnImputationAlgorithm

class MatrixFactorization(AutoSklearnImputationAlgorithm):
    def __init__(self, rank, learning_rate, l1_penalty, l2_penalty,
            random_state=None):

        self.rank = rank
        self.learning_rate = learning_rate
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.random_state = random_state
        
    def fit(self, X, y=None):
        import fancyimpute

        self.min_improvement = 0.005
        self.patience = 5
        self.max_gradient_norm = 5
        
        self.imputer = fancyimpute.MatrixFactorization(
            rank=self.rank,
            learning_rate=self.learning_rate,
            patience=self.patience,
            l1_penalty=self.l1_penalty,
            l2_penalty=self.l2_penalty,
            min_improvement=self.min_improvement,
            max_gradient_norm=self.max_gradient_norm,
            optimization_algorithm='adam',
            min_value=None,
            max_value=None,
            verbose=False
        )
        
        return self
    
    def transform(self, X):
        if self.imputer is None:
            raise NotImplementedError("The MatrixFactorization imputer was not fit")

        # check if we are already complete
        m_missing = np.isnan(X)
        if not m_missing.any():
            return X
        
        X_filled = self.imputer.complete(X)
        
        return X_filled
        
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'MF Fill',
            'name': 'Matrix Factorization Fill',
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

        rank = UniformIntegerHyperparameter(
            name="rank", lower=1, upper=100, default=10
        )

        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=1e-6, upper=1, default=0.001
        )

        l1_penalty = UniformFloatHyperparameter(
            name="l1_penalty", lower=1e-6, upper=1, default=0.001
        )
        
        l2_penalty = UniformFloatHyperparameter(
            name="l2_penalty", lower=1e-6, upper=1, default=0.001
        )
       
        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            rank,
            learning_rate,
            l1_penalty,
            l2_penalty
        ])
        return cs   
