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
    
    @staticmethod
    def get_imputer(imputer_name, dataset_properties=None):
        """ Retrieve an object for use in an autosklearn pipeline by name

        N.B. The imputer will be created with default parameters passed to
        the constructor. If desired, hese can be updated later using, for
        example, `set_params`.

        Parameters
        ----------
        imputer_name: string
            Either `choice` or the short name of one of the registered components

        dataset_properties: dict or None
            The dataset properties. This is used to determine which of the
            concrete imputers are valid.

        Returns
        -------
        imputer: either an ImputerChoice or autoSklearnImputationAlgorithm
            The specified imputer
        """
        imputer_choice = ImputerChoice(dataset_properties=dataset_properties)

        if imputer_name == "choice":
            imputer = imputer_choice
        else:
            components = imputer_choice.get_components()

            # make sure we know it
            if imputer_name not in components:
                choices = ' '.join(components.keys())
                msg = ("[ImputerChoice]: the specified imputer name is not "
                    "recognized: {}. choices: {}".format(imputer_name, choices))
                raise ValueError(msg)

            # then get the pointer to the class
            imputer = components[imputer_name]

            # and create one of them
            imputer = imputer()

        return imputer


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

    @staticmethod
    def get_regression_pipeline(imputer_name, init_params=None):
        """ Construct a regression pipeline suitable for use with `autosklearn` 
        using the specified imputer

        Parameters
        ----------
        imputer_name: string
            The name of the imputer. The valid choices are the short names, such
            as "knn" or "matrix_factorization", of the imputer components, or
            "choice", in which case an `ImputerChoice` is used.

        init_params: dictionary
            Initialization parameters for the various choice objects. Currently,
            the supported parameters are:

                * `init_params['one_hot_encoding']['categorical_features']`
                    The features which will one hot encoded. Please see the
                    documentation for `sklearn.preprocessing.OneHotEncoder` for
                    more details.

        Returns
        -------
        steps: list of `autosklearn` compatible steps
            The steps of the regression pipeline

        Example::
                
            custom_pipeline = functools.partial(
                asl_fancyimpute.get_imputer_regression_pipeline,
                imputer_name=imputer_name
            )

            asl_regressor = AutoSklearnWrapper(
                args=args,
                custom_pipeline=custom_pipeline,
                estimator_named_step='regressor'
            )
        """
        
        from autosklearn.pipeline.components.regression import RegressorChoice
        from autosklearn.pipeline.components.data_preprocessing.rescaling import RescalingChoice
        from autosklearn.pipeline.components.data_preprocessing.one_hot_encoding.one_hot_encoding import OneHotEncoder
        from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
            
        from automlutils.asl_fancyimpute import ImputerChoice
        
        default_dataset_properties = {'target_type': 'regression'}

        # Add the always active preprocessing components
        if init_params is not None and 'one_hot_encoding' in init_params:
            ohe_init_params = init_params['one_hot_encoding']
            if 'categorical_features' in ohe_init_params:
                categorical_features = ohe_init_params['categorical_features']
        else:
            categorical_features = None
            
        ohe = OneHotEncoder(categorical_features=categorical_features)
        imputer = ImputerChoice.get_imputer("knn", dataset_properties=default_dataset_properties)
        rescaling = RescalingChoice(default_dataset_properties)
        preprocessor = FeaturePreprocessorChoice(default_dataset_properties)
        regressor = RegressorChoice(default_dataset_properties)

        steps = [
            ["one_hot_encoding", ohe],
            ["imputation", imputer],
            ["rescaling", rescaling],
            ["preprocessor", preprocessor],
            ["regressor", regressor]
        ]

        return steps

    @staticmethod
    def get_classification_pipeline(imputer_name, init_params=None):
        """ Construct a classification pipeline suitable for use with `autosklearn` 
        using the specified imputer

        Parameters
        ----------
        imputer_name: string
            The name of the imputer. The valid choices are the short names, such
            as "knn" or "matrix_factorization", of the imputer components, or
            "choice", in which case an `ImputerChoice` is used.

        init_params: dictionary
            Initialization parameters for the various choice objects. Currently,
            the supported parameters are:

                * `init_params['one_hot_encoding']['categorical_features']`
                    The features which will one hot encoded. Please see the
                    documentation for `sklearn.preprocessing.OneHotEncoder` for
                    more details.

        Returns
        -------
        steps: list of `autosklearn` compatible steps
            The steps of the regression pipeline

        Example::
                
            custom_pipeline = functools.partial(
                ImputerChoice.get_classification_pipeline,
                imputer_name=imputer_name
            )

            asl_classifier = AutoSklearnWrapper(
                args=args,
                custom_pipeline=custom_pipeline,
                estimator_named_step='classifier'
            )
        """
        
        from autosklearn.pipeline.components.data_preprocessing.one_hot_encoding.one_hot_encoding import OneHotEncoder
        from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import Balancing
        from autosklearn.pipeline.components.data_preprocessing.rescaling import RescalingChoice
        from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
        from autosklearn.pipeline.components.classification import ClassifierChoice
            
            
        from automlutils.asl_fancyimpute import ImputerChoice
        
        default_dataset_properties = {'target_type': 'classification'}

        # Add the always active preprocessing components
        if init_params is not None and 'one_hot_encoding' in init_params:
            ohe_init_params = init_params['one_hot_encoding']
            if 'categorical_features' in ohe_init_params:
                categorical_features = ohe_init_params['categorical_features']
        else:
            categorical_features = None
            
        ohe = OneHotEncoder(categorical_features=categorical_features)
        imputer = ImputerChoice.get_imputer("knn", dataset_properties=default_dataset_properties)
        rescaling = RescalingChoice(default_dataset_properties)
        balancing = Balancing()
        preprocessor = FeaturePreprocessorChoice(default_dataset_properties)
        classifier = ClassifierChoice(default_dataset_properties)

        steps = [
            ["one_hot_encoding", ohe],
            ["imputation", imputer],
            ["rescaling", rescaling],
            ["balancing", balancing],
            ["preprocessor", preprocessor],
            ["classifier", classifier]
        ]

        return steps
