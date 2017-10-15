import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.constants import *

import automlutils.automl_utils as automl_utils
from automlutils.asl_fancyimpute.base import AutoSklearnImputationAlgorithm

class MICE(AutoSklearnImputationAlgorithm):
    
    def __init__(self,
            visit_sequence='monotone',
            n_imputations=10,
            n_burn_in=10,
            impute_type='col',
            n_pmm_neighbors=10,
            init_fill_method='mean',
            random_state=None):
    
        self.visit_sequence = visit_sequence
        self.n_imputations = n_imputations
        self.n_burn = n_burn
        self.impute_type = impute_type
        self.n_pmm_neighbors = n_pmm_neighbors
        self.init_fill_method = init_fill_method
        self.random_state = random_state

    def fit(self, X, y=None):
        import fancyimpute

        self.n_nearest_columns = np.inf

        #self.model = just use the default

        self.imputer = fancyimpute.MICE(
            visit_sequence=self.visit_sequence,
            n_imputations=self.n_imputations,
            n_burn_in=self.n_burn_in,
            impute_type=self.impute_type,
            n_pmm_neighbors=self.n_pmm_neighbors,
            n_nearest_columns=self.n_nearest_columns,
            init_fill_method=self.init_fill_method,
            min_value=None,
            max_value=None,
            verbose=False
        )

        return self

    def transform(self, X):
        if self.imputer is None:
            raise NotImplementedError("The MICE imputer was not fit")

        # check if we are already complete
        m_missing = np.isnan(X)
        if not m_missing.any():
            return X
        
        X_filled = self.imputer.complete(X)
        
        return X_filled
        
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'MICE Fill',
            'name': 'MICE Fill',
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

        visit_sequence = CategoricalHyperparameter(
            "visit_sequence",
            ["monotone", "roman", "arabic", "revmonotone"],
            default="monotone"
        )
        
        n_imputations = automl_utils.create_modest_integer_hyperparameter("n_imputations")
        n_burn_in = automl_utils.create_modest_integer_hyperparameter("n_burn_in")

        impute_type = CategoricalHyperparameter(
            "impute_type",
            ["pmm", "col"],
            default="col"
        )
        
        n_pmm_neighbors = automl_utils.create_modest_integer_hyperparameter("n_pmm_neighbors")

        init_fill_method =  CategoricalHyperparameter(
            "init_fill_method",
            ["mean", "median", "random"],
            default="mean"
        )
        
        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            visit_sequence,
            n_imputations,
            n_burn_in,
            impute_type,
            n_pmm_neighbors,
            init_fill_method
        ])
        return cs   
