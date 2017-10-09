import os
import numpy as np

from collections import OrderedDict

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.components.base import ThirdPartyComponents
from autosklearn.pipeline.components.base import AutoSklearnChoice
from autosklearn.pipeline.components.base import find_components

from automlutils.asl_fancyimpute.base import AutoSklearnImputationAlgorithm

imputer_directory = os.path.split(__file__)[0]
_imputers = find_components(
    __package__,
    imputer_directory,
    AutoSklearnImputationAlgorithm
)

_addons = ThirdPartyComponents(AutoSklearnImputationAlgorithm)

def add_imputer(imputer):
    _addons.add_component(imputer)


class ImputerChoice(AutoSklearnChoice):

    def get_components(self):
        components = OrderedDict()
        components.update(_imputers)
        components.update(_addons.components)
        return components

    def get_available_components(self, dataset_properties=None,
                                 include=None,
                                 exclude=None):

        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)

        # TODO remaining logic to check properties of dataset against
        # properties of imputer

        return available_comp
    
    
    def get_hyperparameter_search_space(self, dataset_properties=None,
                                        default=None,
                                        include=None,
                                        exclude=None):
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal preprocessors for this problem
        available_imputers = self.get_available_components(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        if len(available_imputers) == 0:
            raise ValueError(
                "No imputers found, please add SimpleFill")
            
        
        if default is None:
            defaults = ['simple_fill', 'knn']
            for default_ in defaults:
                if default_ in available_imputers:
                    default = default_
                    break

        imputer = CategoricalHyperparameter(
            '__choice__',
            list(available_imputers.keys()),
            default=default
        )
        
        cs.add_hyperparameter(imputer)
        
        for name in available_imputers:
            
            imputer_cs = available_imputers[name].get_hyperparameter_search_space(dataset_properties)
            parent_hyperparameter = {'parent': imputer, 'value': name}
            
            cs.add_configuration_space(
                name,
                imputer_cs,
                parent_hyperparameter=parent_hyperparameter
            )

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
        return cs

    def transform(self, X):
        m_missing = np.isnan(X)
        if not m_missing.any():
            return X
        return self.choice.transform(X)
