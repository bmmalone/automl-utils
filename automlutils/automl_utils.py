###
#   These are helpers related to auto-sklearn, AutoFolio, ASlib, etc.
###

import logging
logger = logging.getLogger(__name__)

# system imports
import collections
import os

# pydata imports
import joblib
import numpy as np
import sklearn.preprocessing

# widely-used package imports
import networkx as nx
import statsmodels.stats.weightstats

# less-used imports
from aslib_scenario.aslib_scenario import ASlibScenario
import misc.utils as utils

import ConfigSpace.hyperparameters

imputer_strategies = [
    'mean', 
    'median', 
    'most_frequent',
    'zero_fill'
]

def check_imputer_strategy(imputer_strategy, raise_error=True, error_prefix=""):
    """ Ensure that the imputer strategy is a valid selection. 

    Parameters
    ----------
    imputer_strategy: str
        The name of the strategy to check

    raise_error: bool
        Whether to raise a ValueError (or print a warning message) if the
        strategy is not in the valid list.

    error_prefix: str
        A string to prepend to the error/warning message

    Returns
    -------
    is_valid: bool
        True if the imputer strategy is in the allowed list (and an error is
        not raised)    
    """
    if imputer_strategy not in imputer_strategies:
        imputer_strategies_str = ','.join(imputer_strategies)
        msg = ("{}The imputer strategy is not allowed. given: {}. allowed: {}".
            format(error_prefix, imputer_strategy, imputer_strategies_str))

        if raise_error:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    return True

def get_imputer(imputer_strategy, verify=False):
    """ Get the imputer for use in an sklearn pipeline

    Parameters
    ----------
    imputer_strategy: str
        The name of the strategy to use. This must be one of the imputer
        strategies. check_imputer_strategy can be used to verify the string.

    verify: bool
        If true, check_imputer_strategy will be called before creating the
        imputer

    Returns
    -------
    imputer: sklearn.transformer
        A transformer with the specified strategy
    """

    if verify:
        check_imputer_strategy(imputer_strategy)

    imputer = None
    if imputer_strategy == 'zero_fill':
        imputer = sklearn.preprocessing.FunctionTransformer(
            np.nan_to_num,
            validate=False
        )

    else:
        imputer = sklearn.preprocessing.Imputer(strategy=imputer_strategy)

    return imputer

import autosklearn.pipeline.components.classification as c
import autosklearn.pipeline.components.feature_preprocessing as fp
import autosklearn.pipeline.components.regression as r

all_classifiers = c._classifiers.keys()
all_preprocessors = fp._preprocessors.keys()
all_regressors = r._regressors.keys()

from autosklearn.regression import AutoSklearnRegressor
from autosklearn.classification import AutoSklearnClassifier

ASL_REGRESSION_PIPELINE_TYPES = {
    utils.get_type("autosklearn.pipeline.regression.SimpleRegressionPipeline")
}

ASL_CLASSIFICATION_PIPELINE_TYPES = {
    utils.get_type("autosklearn.pipeline.classification.SimpleClassificationPipeline")
}

ASL_PIPELINE_TYPES = ASL_REGRESSION_PIPELINE_TYPES | ASL_CLASSIFICATION_PIPELINE_TYPES

DUMMY_CLASSIFIER_TYPES = {
    utils.get_type("autosklearn.evaluation.abstract_evaluator.MyDummyClassifier"),
    utils.get_type("autosklearn.evaluation.abstract_evaluator.DummyClassifier"),
}

DUMMY_REGRESSOR_TYPES = {
    utils.get_type("autosklearn.evaluation.abstract_evaluator.MyDummyRegressor"),   
    utils.get_type("autosklearn.evaluation.abstract_evaluator.DummyRegressor"),  
}

DUMMY_PIPELINE_TYPES= DUMMY_CLASSIFIER_TYPES | DUMMY_REGRESSOR_TYPES

CLASSIFIER_CHOICE_TYPES = {
    utils.get_type("autosklearn.pipeline.components.classification.ClassifierChoice"),
}
REGRESSOR_CHOICE_TYPES = {
    utils.get_type("autosklearn.pipeline.components.regression.RegressorChoice"),
}

ESTIMATOR_CHOICE_TYPES = REGRESSOR_CHOICE_TYPES | CLASSIFIER_CHOICE_TYPES

ESTIMATOR_NAMED_STEPS = {
    'regressor',
    'classifier'
}

regression_pipeline_steps = [
    'one_hot_encoding', 
    'imputation', 
    'rescaling', 
    'preprocessor', 
    'regressor'
]

classificaation_pipeline_steps = [
    'one_hot_encoding',
    'imputation',
    'rescaling',
    'balancing',
    'preprocessor',
    'classifier'
]

pipeline_step_names_map = {
    # preprocessors
    "<class 'sklearn.feature_selection.from_model.SelectFromModel'>": "Model-based",
    "<class 'sklearn.feature_selection.univariate_selection.SelectPercentile'>": "Percentile-based",
    "<class 'sklearn.cluster.hierarchical.FeatureAgglomeration'>": "Feature agglomeration",
    "<class 'int'>": "None",
    "<class 'sklearn.preprocessing.data.PolynomialFeatures'>": "Polynomial expansion",
    "<class 'sklearn.decomposition.pca.PCA'>": "PCA",
    "<class 'sklearn.ensemble.forest.RandomTreesEmbedding'>": "Random tree embedding",
    "<class 'sklearn.decomposition.fastica_.FastICA'>": "ICA",
    "<class 'sklearn.kernel_approximation.RBFSampler'>": "RBF Sampler",
    "<class 'sklearn.decomposition.kernel_pca.KernelPCA'>": "Kernel PCA",
    "<class 'sklearn.kernel_approximation.Nystroem'>": "Nystroem approximation",
    
    # regression models
    "<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>": "Linear model trained with stochastic gradient descent",
    "<class 'sklearn.ensemble.weight_boosting.AdaBoostRegressor'>": "AdaBoost",
    "<class 'sklearn.tree.tree.DecisionTreeRegressor'>": "Regression trees",
    "<class 'sklearn.ensemble.forest.ExtraTreesRegressor'>": "Extremely randomized trees",
    "<class 'sklearn.svm.classes.SVR'>": "episilon-support vector regression",
    "<class 'sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>": "Gradient-boosted regression trees",
    "<class 'xgboost.sklearn.XGBRegressor'>": "Extreme gradient-boosted regression trees",
    "<class 'sklearn.gaussian_process.gaussian_process.GaussianProcess'>": "Gaussian process regression",
    "<class 'sklearn.ensemble.forest.RandomForestRegressor'>": "Random forest regression",
    "<class 'sklearn.linear_model.ridge.Ridge'>": "Ridge regression",
    "<class 'sklearn.linear_model.bayes.ARDRegression'>": "Bayesian ridge regression",
    "<class 'sklearn.svm.classes.LinearSVR'>": "Linear support vector regression",

    # classification models
    "<class 'autosklearn.evaluation.abstract_evaluator.MyDummyClassifier'>": "Dummy classifier",
    "<class 'sklearn.ensemble.forest.RandomForestClassifier'>": "Random forest with optimal splits classifier",
    "<class 'sklearn.svm.classes.LinearSVC'>": "Support vector classification (liblinear)",
    "<class 'sklearn.svm.classes.SVC'>": "Support vector classification (libsvm)",
    "<class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>": "Linear discriminant analysis",
    "<class 'sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'>": "Gradient-boosted regression trees for classification",
    "<class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'>": "Linear model trained with stochastic gradient descent",
    "<class 'sklearn.ensemble.weight_boosting.AdaBoostClassifier'>": "AdaBoost (SAMME), default base: DecisionTree",
    "<class 'sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier'>": "Margin-based Passive-Aggressive classifier",
    "<class 'sklearn.ensemble.forest.ExtraTreesClassifier'>": "Random forest with random splits classifier",
    "<class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>": "Quadratic discriminant analysis",
    "<class 'sklearn.naive_bayes.GaussianNB'>": "Naive Bayes with Gaussian observations",
    "<class 'sklearn.tree.tree.DecisionTreeClassifier'>": "Basic decision tree",
    "<class 'sklearn.naive_bayes.BernoulliNB'>": "Naive Bayes with Bernoulli observations",
    "<class 'sklearn.naive_bayes.MultinomialNB'>": "Naive Bayes with multinomial observations",
    "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>": "k-nearest neighbors classifier"
    
}

PRETTY_NAME_TYPE_MAP = {
    utils.get_type(k.split("'")[1]): v for k,v in pipeline_step_names_map.items()
}

# the names of the actual "element" (estimator, etc.) for steps in
# autosklearn pipelines
pipeline_step_element_names = {
    "one_hot_encoding": None,
    "imputation": None,
    "rescaling": 'preprocessor',
    "preprocessor": 'preprocessor',
    "regressor": 'estimator',
    "classifier": 'estimator',
    "balancing": None
}

# pipeline types which do not do anything and can be ignored
passthrough_types = {
    utils.get_type("autosklearn.pipeline.components.data_preprocessing.rescaling.none.NoRescalingComponent"),
    int
}

def get_simple_pipeline(asl_pipeline, as_array=False):
    """ Extract the "meat" elements from an auto-sklearn pipeline to create
    a "normal" sklearn pipeline.

    N.B.
    
    * The new pipeline is based on clones.

    * In case the pipeline is actually one of the DUMMY_PIPELINE_TYPES, it will
      be returned without a pipeline wrapper.

    * If asl_pipeline is not one of ASL_PIPELINE_TYPES, then a simple clone
      is created and returned.

    Parameters
    ----------
    asl_pipeline: autosklearn.Pipeline
        The pipeline learned by auto-sklearn

    as_array: bool
        Whether to return the selected steps as a list of 2-tuples of an
        sklearn.pipeline.Pipeline

    Returns
    -------
    sklearn_pipeline: sklearn.pipeline.Pipeline
        A pipeline containing clones of the elements from the auto-sklearn
        pipeline, but in a "normal" sklearn pipeline
    """
    if type(asl_pipeline) not in ASL_PIPELINE_TYPES:
        msg = "[automl_utils]: asl_pipeline is not an auto-sklearn pipeline"
        logger.debug(msg)

        pipeline = sklearn.base.clone(asl_pipeline)
        return pipeline

    simple_pipeline = []

    # if it is one of the dummy classifiers, then just wrap it and return it
    if type(asl_pipeline) in DUMMY_PIPELINE_TYPES:
        return asl_pipeline

        dummy_estimator_name = 'regressor'
        if type(asl_pipeline) in DUMMY_CLASSIFIER_TYPES:
            dummy_estimator_name = 'classifier'

        simple_pipeline = [(dummy_estimator_name, asl_pipeline)] # list of 2-tuples

        if not as_array:
            simple_pipeline = sklearn.pipeline.Pipeline(simple_pipeline)
        return simple_pipeline

    for step, element in asl_pipeline.steps:
        attr = pipeline_step_element_names[step]

        if (attr is not None): # and (type(element) not in base_types):

            choice = element.choice
            
            if type(choice) in passthrough_types:
                continue
            
            choice = choice.__getattribute__(attr)

            if type(choice) in passthrough_types:
                continue

            choice = sklearn.base.clone(choice)

            # check if this has a "prefit" attribute
            if hasattr(choice, 'prefit'):
                # if so and it is true
                choice.prefit = False                    
        else:
            choice = element


        simple_pipeline.append([step, choice])
        
    if not as_array:
        simple_pipeline = sklearn.pipeline.Pipeline(simple_pipeline)
    return simple_pipeline

def retrain_asl_wrapper(asl_wrapper, X_train, y_train):
    """ Retrain the ensemble in the asl_wrapper using the new training data

    The relative weights of the members of the ensemble will remaining
    unchanged. If some member cannot be retrained for some reason, it is
    discarded (and a warning message is logged).

    Parameters
    ----------
    asl_wrapper: AutoSklearnWrapper 
        The (fit) wrapper

    {X,y}_train: data matrices
        Data matrics of the same type used to originally train the wrapper

    Returns
    -------
    weights: np.array of floats
        The weights of members of the ensemble

    pipelines: np.array of sklearn pipelines
        The updated pipelines which could be retrained
    """

    new_weights = []
    new_ensemble = []

    it = zip(asl_wrapper.ensemble_[0], asl_wrapper.ensemble_[1])
    for (weight, asl_pipeline) in it:
        p = get_simple_pipeline(asl_pipeline)

        try:
            p.fit(X_train, y_train)
            new_weights.append(weight)
            new_ensemble.append(p)
        except Exception as ex:
            msg = "Failed to fit a pipeline. Error: {}".format(str(ex))
            logger.warning(msg)

    new_weights = np.array(new_weights) / np.sum(new_weights)
    new_ensemble = [new_weights, new_ensemble]
    
    return new_ensemble



def filter_model_types(aml, model_types):
    """ Remove all models of the specified type from the ensemble.

    Parameters
    ----------
    aml : tuple of (weights, pipelines)
        A tuple representing a trained ensemble, read in with read_automl
        (or similarly constructed)

    model_types : set-like of types
        A set containing the types to remove. N.B. This *should not* be a set
        of strings; it should include the actual type objects.

    Returns
    -------
    filtered_aml : (weights, pipelines) tuple
        The weights and pipelines from the original aml, with the types
        specified by model_types removed.

        The weights are renormalized.
    """

    (weights, pipelines) = (np.array(aml[0]), np.array(aml[1]))
    estimator_types = [
        type(get_aml_estimator(p)) for p in pipelines
    ]
    
    to_filter = [
            i for i, et in enumerate(estimator_types) if not et in model_types
    ]
    
    weights = weights[to_filter]
    weights = weights / np.sum(weights)
    
    return (weights, pipelines[to_filter])

def create_binary_hyperparameter(name, default="True",
        allowed_values=set(["True", "False"])):
    """ Create a binary ConfigSpace.hyperparameter

    Parameters
    ----------
    name: string
        The name of the hyperparameter

    default: string in ["True", "False"]
        The default value for the hyperparameters

    allowed_values: set-like
        The valid values for "default". This should probably not be changed. 

    Returns
    -------
    binary_hyperparameter: ConfigSpace.hyperparameter.CategoricalHyperparameter
        The hyperparameter, suitable to add to a ConfigurationSpace
    """

    if default not in allowed_values:
        msg = ("[automl_utils.create_binary_hyperparameter]: default must be "
            "one of: {}. Found: {}".format(allowed_values, default))
        raise ValueError(msg)

    p = ConfigSpace.hyperparameters.CategoricalHyperparameter(name,
        ["True", "False"], default=default)
    return p

def create_small_float_hyperparameter(name, default=0.001, lower=1e-6, upper=1):
    """ Create a uniform float hyperparameter with the specified range

    The defaults for this function are designed with "small" hyperparameters
    in mind, such as error tolerances and regularization hyperparameters.
    """

    p = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name=name, default=default, lower=lower, upper=upper
    )
    return p

def create_modest_integer_hyperparameter(name, default=10, lower=1, upper=100):
    """ Create a uniform integer hyperparameter with the specified range

    The defaults for this function are designed with "modest" integer
    hyperparameters in mind, such as rank for low-rank approximations or k
    for kNN.
    """
    p = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name=name, default=default, lower=lower, upper=upper
    )
    return p
