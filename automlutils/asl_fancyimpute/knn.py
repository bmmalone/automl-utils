import random
import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from autosklearn.pipeline.constants import *

import automlutils.automl_utils as automl_utils
from automlutils.asl_fancyimpute.base import AutoSklearnImputationAlgorithm

class KNN(AutoSklearnImputationAlgorithm):
    """ Wrap the fancyimpute.KNN imputer for asl

    Parameters
    ----------
    k: int
        The number of nearest neighbors to consider

    random_state: int or random state object
        The random state. fancyimpute does not directly use random seeds, but
        random and np.random are still seeded before fitting the imputer to
        attempt to improve reproducibility.

    Fixed parameters
    ----------------
    None. All of the parameters for fancyimpute.KNN are included in the
    ConfigSpace.
    """
    def __init__(self, k=1, random_state=None):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y=None):
        """ Create the KNN imputer object
        """
        import fancyimpute
        
        self.imputer = fancyimpute.KNN(k=self.k, verbose=False)
        
        return self
    
    def transform(self, X):
        """ Use the KNN imputer to complete the data

        Parameters
        ----------
        X: data matrix
            Missing values are indicated with np.nan
        """
        if self.imputer is None:
            raise NotImplementedError("The KNN imputer was not fit")

        # check if we are already complete
        m_missing = np.isnan(X)
        if not m_missing.any():
            return X
        
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        X_filled = self.imputer.complete(X)
        
        return X_filled
        
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'kNN Fill',
            'name': 'kNN Fill',
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

        n_neighbors = UniformIntegerHyperparameter(
            name="k", lower=1, upper=100, log=True, default=1)
        
        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_neighbors])
        return cs   
