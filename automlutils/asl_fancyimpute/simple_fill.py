import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.constants import *

from automlutils.asl_fancyimpute.base import AutoSklearnImputationAlgorithm

class SimpleFill(AutoSklearnImputationAlgorithm):
    def __init__(self,
            fill_method='zero',
            random_state=None):

        self.fill_method = fill_method
        self.random_state = random_state
        
    def fit(self, X, y=None):
        import fancyimpute
        
        self.imputer = fancyimpute.SimpleFill(fill_method=self.fill_method)
        
        return self
    
    def transform(self, X):
        if self.imputer is None:
            raise NotImplementedError("The SimpleFill imputer was not fit")

        ###
        # make sure we have numpy floats. we might not in cases where
        # the original data matrix contains mixed types (categorical and
        # numeric types)
        #
        # See this SO comment for more details:
        #   https://stackoverflow.com/questions/18557337/numpy-attributeerror-float-object-has-no-attribute-exp#comment27300354_18557337
        ###
        X = X.astype(float)

        # check if we are already complete
        m_missing = np.isnan(X)
        if not m_missing.any():
            return X

        # in case we are given a single column, make sure we still work
        reshape = False
        if len(X.shape) == 1:
            X = np.atleast_2d(X)
            X = X.transpose()
            reshape = True
        
        X_filled = self.imputer.complete(X)

        if reshape:
            X_filled = X_filled.ravel()

        return X_filled
        
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Simple Fill',
            'name': 'Simple Fill',
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
        fill_method = CategoricalHyperparameter(
            "fill_method",
            ["zero", "mean", "median", "min", "random"],
            default="zero"
        )
        
        cs = ConfigurationSpace()
        cs.add_hyperparameters([fill_method])
        return cs   
