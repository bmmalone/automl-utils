""" Utilities for working with ConfigSpace for specifying a hyperparameter
search space:

    https://github.com/automl/ConfigSpace
"""

import itertools
import collections
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction
from autosklearn.pipeline.constants import SPARSE



def forbid_nonlinear_estimators_with_feature_transformers(cs,
            possible_default_estimators,
            estimator_choice_key='classifier:__choice__',
            transformer_choice_key='preprocessor:__choice__',
            allowed_estimators=None,
            allowed_transformers=None,
            dataset_properties=None):
    
    """ Add clauses to the ConfigSpace to forbid using nonlinear estimators
    with "feature learning".
    
    The original autosklearn authors suggest that combining non-linear models
    with some feature learning algorithms take too long, so skip these.
    
    See Line ~215 in /autosklearn/pipeline/classification.py
    See Line ~221 in /autosklearn/pipeline/regression.py
    
    Parameters
    ----------
    cs: ConfigSpace
    
    possible_default_estimators: queue-like
    
    estimator_choice_key: string
        Valid values:
        * 'classifier:__choice__'
        * 'regressor:__choice__'
        
    transformer_choice_key: string
        Valid values:
        * 'preprocessor:__choice__'
    
    allowed_estimators: set-like
    
    allowed_transformers: set-like
    
    dataset_properties: dict-like
    
    Returns
    -------
    None, but the ConfigSpace is updated with the appropriate forbidden clauses
    """

    # the original autosklearn authors suggest that combining
    # non-linear models with some feature learning algorithm takes
    # too long, so skip these combinations
    nonlinear_estimators = [
        "adaboost",
        "decision_tree",
        "extra_trees",
        "gaussian_process",
        "gradient_boosting",
        "k_nearest_neighbors",
        "libsvm_svc",
        "random_forest",
        "gaussian_nb",
        "decision_tree",
        "xgradient_boosting"
    ]
    
    feature_learning_transformers = [
        "kitchen_sinks",
        "nystroem_sampler",
        "kernel_pca"
    ]

    for e, t in itertools.product(nonlinear_estimators, feature_learning_transformers):
        # if we already excluded these, then skip this combination
        if (allowed_estimators is not None) and (e not in allowed_estimators):
            continue
        if (allowed_transformers is not None) and (t not in allowed_transformers):
            continue
            
        while True:
            try:
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        estimator_choice_key), e),
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        transformer_choice_key), t)
                ))
                break
            except KeyError:
                break
            except ValueError as e:
                # we get a ValueError if the default setting would become
                # forbidden due to the new clause.

                # So change the default and try again
                try:
                    default = possible_default_classifiers.pop()
                except IndexError:
                    raise ValueError(
                        "Cannot find a legal default configuration.")
                cs.get_hyperparameter(
                    estimator_choice_key).default = default

def forbid_discrete_estimators_with_continuous_transformers(cs,
            possible_default_estimators,
            estimator_choice_key='classifier:__choice__',
            transformer_choice_key='preprocessor:__choice__',
            allowed_estimators=None,
            allowed_transformers=None,
            dataset_properties=None):
    """ Add clauses to the ConfigSpace to forbid using estimators which
    assume discrete data, such as multnomial naive Bayes, with preprocessors which
    transform the input into a continuous space, such as PCA.
    
    Parameters
    ----------
    cs: ConfigSpace
    
    possible_default_estimators: queue-like
    
    estimator_choice_key: string
        Valid values:
        * 'classifier:__choice__'
        * 'regressor:__choice__'
        
    transformer_choice_key: string
        Valid values:
        * 'preprocessor:__choice__'
    
    allowed_estimators: set-like
    
    allowed_transformers: set-like
    
    dataset_properties: dict-like
    
    Returns
    -------
    None, but the ConfigSpace is updated with the appropriate forbidden clauses
    """
       
    discrete_classifiers = ["multinomial_nb"]
    
    continuous_transformers = [
        "kitchen_sinks",
        "pca",
        "truncatedSVD",
        "fast_ica",
        "kernel_pca",
        "nystroem_sampler"
    ]

    for e, t in itertools.product(discrete_classifiers, continuous_transformers):
        
        # we may have already forbid this combination for some other reason
        if (allowed_estimators is not None) and (e not in allowed_estimators):
            continue
        if (allowed_transformers is not None) and (t not in allowed_transformers):
            continue
            
        while True:
            try:
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        transformer_choice_key), t),
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        estimator_choice_key), e)
                ))
                break
            except KeyError:
                break
            except ValueError:
                # we get a ValueError if the default setting would become
                # forbidden due to the new clause.

                # So change the default and try again
                try:
                    default = possible_default_classifiers.pop()
                
                except IndexError:
                    raise ValueError(
                        "Cannot find a legal default configuration.")
                cs.get_hyperparameter(
                    estimator_choice_key).default = default

def forbid_sparse_estimators_with_densifier(cs,
            possible_default_estimators,
            all_estimator_properties,
            estimator_choice_key='classifier:__choice__',
            transformer_choice_key='preprocessor:__choice__',
            allowed_estimators=None,
            allowed_transformers=None,
            dataset_properties=None):
    """ Add clauses to the ConfigSpace to forbid using sparse estimators 
    with the "densifier" preprocessor.
    
    
    The original autosklearn authors suggest that combining sparse models
    with the "densifier" causes too much RAM usage, so skip these.
    
    See Line ~188 in /autosklearn/pipeline/classification.py
    See Line ~195 in /autosklearn/pipeline/regression.py
    
    Parameters
    ----------
    cs: ConfigSpace
    
    possible_default_estimators: queue-like
    
    all_estimator_properties: dict-like
    
    estimator_choice_key: string
        Valid values:
        * 'classifier:__choice__'
        * 'regressor:__choice__'
        
    transformer_choice_key: string
        Valid values:
        * 'preprocessor:__choice__'
    
    allowed_estimators: set-like
    
    allowed_transformers: set-like
    
    dataset_properties: dict-like
    
    Returns
    -------
    None, but the ConfigSpace is updated with the appropriate forbidden clauses
    """

    for key in allowed_estimators:
        estimator_properties = all_estimator_properties[key].get_properties(
            dataset_properties=dataset_properties
        )
        
        if SPARSE in estimator_properties['input']:
            if 'densifier' in allowed_transformers:
                while True:
                    try:
                        cs.add_forbidden_clause(ForbiddenAndConjunction(
                            ForbiddenEqualsClause(
                                cs.get_hyperparameter(
                                    estimator_choice_key), key),
                            ForbiddenEqualsClause(
                                cs.get_hyperparameter(
                                    transformer_choice_key), 'densifier')
                        ))
                        # Success
                        break
                    except ValueError:
                        # we get a ValueError if the default setting would become
                        # forbidden due to the new clause.
                        
                        # So change the default and try again
                        try:
                            default = possible_default_estimators.pop()
                        except IndexError:
                            msg = "Cannot find a legal default configuration."
                            raise ValueError(msg)
                            
                        cs.get_hyperparameter(
                            estimator_choice_key).default = default

feature_types_fields = ' '.join([
    "categorical_features",
    "numerical_features"
])

feature_types_tuple = collections.namedtuple(
    "feature_types",
    feature_types_fields
)

def get_feature_types(init_params):

    categorical_features = None
    numerical_features = None

    if init_params is not None and 'one_hot_encoding' in init_params:
        ohe_init_params = init_params['one_hot_encoding']
        if 'categorical_features' in ohe_init_params:
            categorical_features = ohe_init_params['categorical_features']

    elif init_params is not None and 'categorical_features' in init_params:
        categorical_features = init_params['categorical_features']


    if init_params is not None and 'numerical_features' in init_params:
        numerical_features = init_params['numerical_features']
        
    ret = feature_types_tuple(categorical_features, numerical_features)
    return ret

classification_target_types = {'classification'}
regression_target_types = {'regression'}

def get_target_type(target_type):
    
    if target_type in classification_target_types:
        target_type = 'classification'
    elif target_type in regression_target_types:
        target_type = 'regression'
    else:
        msg = ("[get_target_type] invalid target type: {}".format(target_type))
        raise ValueError(msg)
        
    return target_type
