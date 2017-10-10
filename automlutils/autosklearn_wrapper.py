###
#   A wrapper for autosklearn optimizers.
###
import logging
logger = logging.getLogger(__name__)

import collections

import numpy as np
import joblib
import sklearn.preprocessing

import statsmodels.stats.weightstats


from automlutils import automl_utils

def _validate_fit_asl_wrapper(asl_wrapper):
    """ Check that the AutoSklearnWrapper contains a valid ensemble
    """
    if len(asl_wrapper.ensemble_) != 2:
        msg = ("[asl_wrapper]: the ensemble_ must be a list-like with two "
            "elements. Presumably, it is either read from a pickle file using "
            "read_asl_wrapper or extracted from a fit autosklearn object.")
        raise ValueError(msg)

    if len(asl_wrapper.ensemble_[0]) != len(asl_wrapper.ensemble_[1]):
        msg = ("[asl_wrapper]: the ensemble_[0] and ensemble_[1] list-likes "
            "must be the same length.")
        raise ValueError(msg)

    if asl_wrapper.estimator_named_step not in automl_utils.ESTIMATOR_NAMED_STEPS:
        s = sorted([e for e in automl_utils.ESTIMATOR_NAMED_STEPS])
        msg = ("[asl_wrapper]: the estimator named step must be one of: {}".
            format(s))
        raise ValueError(msg)

    if asl_wrapper.autosklearn_optimizer is not None:
        msg = ("[asl_wrapper]: the fit wrapper should not include the "
            "autosklearn optimizer")
        raise ValueError(msg)

def _get_asl_estimator(asl_pipeline, pipeline_step='regressor'):
    """ Extract the concrete estimator (RandomForest, etc.) from the asl
    pipeline.

    In case the pipeline is one of the DUMMY_PIPELINE_TYPES, then the asl
    pipeline is actually the estimator, and it is returned as-is.
    """
    asl_pipeline_type = type(asl_pipeline)
    if asl_pipeline_type in automl_utils.DUMMY_PIPELINE_TYPES:
        return asl_pipeline

    asl_model = asl_pipeline.named_steps[pipeline_step]
    
    # this is from the old development branch
    #aml_model_estimator = aml_model_model.estimator

    # for the 0.1.3 branch, grab the "choice" estimator

    # this may be either a "choice" type or an actual model
    if type(asl_model) in automl_utils.ESTIMATOR_CHOICE_TYPES:
        asl_model = asl_model.choice.estimator

    return asl_model

def _get_asl_pipeline(aml_model):
    """ Extract the pipeline object from an autosklearn_optimizer model.
    """
    # this is from the old development branch
    # the model *contained* a pipeline
    #aml_model_pipeline = aml_model.pipeline_

    # this is the updated 0.1.3 branch

    # that is, the model simply *is* a pipeline now
    asl_pipeline = aml_model
    return asl_pipeline

def _extract_autosklearn_ensemble(autosklearn_optimizer,
        estimtor_named_step='regressor'):
    """ Extract the nonzero weights, associated pipelines and estimators from
    a fit autosklearn optimizer (i.e., an AutoSklearnRegressor or an
    AutoSklearnClassifier).
    """
    import numpy as np

    aml = autosklearn_optimizer._automl._automl
    models = aml.models_

    e = aml.ensemble_
    weights = e.weights_
    model_identifiers = np.array(e.identifiers_)

    nonzero_weight_indices = np.nonzero(weights)[0]
    nonzero_weights = weights[nonzero_weight_indices]
    nonzero_model_identifiers = model_identifiers[nonzero_weight_indices]
    
    asl_models = [
        models[tuple(m)] for m in nonzero_model_identifiers
    ]

    asl_pipelines = [
        _get_asl_pipeline(m) for m in asl_models
    ]
    
    asl_estimators = [
        _get_asl_estimator(p, estimtor_named_step) for p in asl_pipelines
    ]

    return (nonzero_weights, asl_pipelines, asl_estimators)

class AutoSklearnWrapper(object):
    """ A wrapper for an autosklearn optimizer to easily integrate it within a
    larger sklearn.Pipeline. The purpose of this class is largely to minimize
    the amount of time the autosklearn.AutoSklearnXXX objects are present. They
    include many functions related to the Bayesian optimization which are not
    relevant after the parameters of the ensemble have been fit. Thus, outside
    of the fit method, there is no need to keep it around.

    Parameters
    ----------
    ensemble_: an asl_wrapper ensemble (two-tupe of weights and pipelines)
        In case a desired ensemble already exists, it can be passed in.

    autosklearn_optimizer: an AutoSklearnClassifier or AutoSklearnRegressor
        In case an optimizer is created outside this class, it can be directly
        passed in.

    estimator_named_step: "classifier" or "regressor"
        Whether this wrapper will be for classification or regression.
        Specifically, this value is used to determine which optimizer to
        create and the name of the estimator step in the learned pipelines.

    args: argparse.Namespace
        A namespace containing the options from
        `automlutils.automl_command_line_utils`. If present, these are used
        as defaults for the respect asl optimizer.

    le_: label encoder
        An `sklearn.preprocessing.LabelEncoder` for the target variable, if one
        has been fit externally.

    metric: `autosklearn.metrics.Scorer`
        An optional scorer. By default, accuracy is used for classification and
        RMSE for regression.

    custom_pipeline: function pointer with a single parameter, `self`
        Optionally, the respective `_get_pipeline` function can be overwritten
        when creating the asl optimizer. This should return something
        compatible with `autosklearn`. In particular, the steps of the pipeline
        should implement `get_properties` and
        `get_hyperparameter_search_space`. If this is provided, then no
        `initial_configurations_via_metalearning` will be used.

        N.B. The `custom_pipeline` is fairly brittle as it involves monkey
        patching. It should be considered highly experimental.

    """

    def __init__(self,
            ensemble_=None,
            autosklearn_optimizer=None,
            estimator_named_step=None,
            args=None,
            le_=None,
            metric=None,
            custom_pipeline=None):

        msg = ("[asl_wrapper]: initializing a wrapper. ensemble: {}. "
            "autosklearn: {}" .format(ensemble_, autosklearn_optimizer))
        logger.debug(msg)

        self.args = args
        self.ensemble_ = ensemble_
        self.autosklearn_optimizer = autosklearn_optimizer
        self.estimator_named_step = estimator_named_step
        self.metric = metric
        self.custom_pipeline = custom_pipeline
        self.le_ = le_

    def create_classification_optimizer(self, args, **kwargs):        
        """ Create an AutoSklearnClassifier and use it as the autosklearn
        optimizer.

        The parameters can either be passed via an argparse.Namespace or using
        keyword arguments. The keyword arguments take precedence over args. The
        following keywords are used:

            * total_training_time
            * iteration_time_limit
            * ensemble_size
            * ensemble_nbest
            * seed
            * estimators
            * tmp

        Parameters
        ----------
        args: Namespace
            An argparse.Namepsace-like object which  presumably comes from
            parsing the add_automl_options arguments.

        kwargs: key=value pairs
            Additional options for creating the autosklearn classifier

        Returns
        -------
        self
        """
        from autosklearn.classification import AutoSklearnClassifier
        from autosklearn.pipeline.classification import SimpleClassificationPipeline

        args_dict = args.__dict__
        args_dict.update(kwargs)

        # check if we have a new pipeline
        initial_configurations_via_metalearning = 25 # the standard asl default
        if self.custom_pipeline is not None:
            msg = "[asl_wrapper]: attempting to change classification pipeline"
            logger.debug(msg)

            initial_configurations_via_metalearning = 0
            SimpleClassificationPipeline._get_pipeline = self.custom_pipeline


        asl_classification_optimizer = AutoSklearnClassifier(
            time_left_for_this_task=args_dict.get('total_training_time', None),
            per_run_time_limit=args_dict.get('iteration_time_limit', None),
            ensemble_size=args_dict.get('ensemble_size', None),
            ensemble_nbest=args_dict.get('ensemble_nbest', None),
            seed=args_dict.get('seed', None),
            include_estimators=args_dict.get('estimators', None),
            tmp_folder=args_dict.get('tmp', None),
            initial_configurations_via_metalearning=initial_configurations_via_metalearning
        )

        self.autosklearn_optimizer = asl_classification_optimizer
        self.estimator_named_step = "classifier"
        return self


    def create_regression_optimizer(self, args, **kwargs):        
        """ Create an AutoSklearnRegressor and use it as the autosklearn
        optimizer.

        The parameters can either be passed via an argparse.Namespace or using
        keyword arguments. The keyword arguments take precedence over args. The
        following keywords are used:

            * total_training_time
            * iteration_time_limit
            * ensemble_size
            * ensemble_nbest
            * seed
            * estimators
            * tmp

        Parameters
        ----------
        args: Namespace
            An argparse.Namepsace-like object which  presumably comes from
            parsing the add_automl_options arguments.

        kwargs: key=value pairs
            Additional options for creating the autosklearn regressor

        Returns
        -------
        self
        """
        from autosklearn.regression import AutoSklearnRegressor
        from autosklearn.pipeline.regression import SimpleRegressionPipeline

        args_dict = args.__dict__
        args_dict.update(kwargs)

        # check if we have a new pipeline
        initial_configurations_via_metalearning = None
        if self.custom_pipeline is not None:
            msg = "[asl_wrapper]: attempting to change regression pipeline"
            logger.debug(msg)
            
            initial_configurations_via_metalearning = 0
            SimpleRegressionPipeline._get_pipeline = self.custom_pipeline

        args_dict = args.__dict__
        args_dict.update(kwargs)

        asl_regression_optimizer = AutoSklearnRegressor(
            time_left_for_this_task=args_dict.get('total_training_time', None),
            per_run_time_limit=args_dict.get('iteration_time_limit', None),
            ensemble_size=args_dict.get('ensemble_size', None),
            ensemble_nbest=args_dict.get('ensemble_nbest', None),
            seed=args_dict.get('seed', None),
            include_estimators=args_dict.get('estimators', None),
            tmp_folder=args_dict.get('tmp', None),
            initial_configurations_via_metalearning=initial_configurations_via_metalearning
        )

        self.autosklearn_optimizer = asl_regression_optimizer
        self.estimator_named_step = "regressor"
        return self

    def fit(self, X_train, y, metric=None, encode_y=True):
        """ Optimize the ensemble parameters with autosklearn """

        # overwrite our metric, if one was given
        if metric is not None:
            self.metric = metric

        # check if we have either args or a learner
        if self.args is not None:
            if self.autosklearn_optimizer is not None:
                msg = ("[asl_wrapper]: have both args and an autosklearn "
                    "optimizer. Please set only one or the other.")
                raise ValueError(msg)

            if self.estimator_named_step == 'regressor':
                msg = ("[asl_wrapper]: creating an autosklearn regression "
                    "optimizer")
                logger.debug(msg)

                self.create_regression_optimizer(self.args)
            elif self.estimator_named_step == 'classifier':
                msg = ("[asl_wrapper]: creating an autosklearn classification "
                    "optimizer")
                logger.debug(msg)

                self.create_classification_optimizer(self.args)

                # sometimes, it seems we need to encode labels to keep them
                # around.
                if encode_y:
                    # In some cases, members of the ensemble drop some of the
                    # labels. Make sure we can reconstruct those
                    self.le = sklearn.preprocessing.LabelEncoder()
                    self.le_ = self.le.fit(y)
                    y = self.le_.transform(y)
                else:
                    # we still need to keep around the number of classes
                    # we assume y is a data frame which gives this value.
                    self.le = sklearn.preprocessing.LabelEncoder()
                    self.le_ = self.le
                    self.le_.classes = np.unique(y.columns)

            else:
                msg = ("[asl_wrapper]: the valid `estimator_named_step`s are: "
                    "[\"regressor\", \"classifier\"], but found: {}".format(
                    self.estimator_named_step))
                raise ValueError(msg)

        elif self.autosklearn_optimizer is None:
            msg = ("[asl_wrapper]: have neither args nor an autosklearn "
                "optimizer. Please set one or the other (but not both).")
            raise ValueError(msg)

        msg = ("[asl_wrapper]: fitting a wrapper with metric: {}".format(
            self.metric))
        logger.debug(msg)

        self.autosklearn_optimizer.fit(X_train, y, metric=self.metric)
        
        vals = _extract_autosklearn_ensemble(
            self.autosklearn_optimizer,
            self.estimator_named_step
        )

        (weights, pipelines, estimators) = vals
        self.ensemble_ = (weights, pipelines)
        
        # since we have the ensemble, we can get rid of the Bayesian optimizer
        self.autosklearn_optimizer = None

        # also, print the unique classes:
        #for e in estimators:
        #    print("estimator:", e, "unique class labels:", e.classes_)

        return self


    def _predict_regression(self, X_test):
        """ Predict the values using the fitted ensemble
        """
        _validate_fit_asl_wrapper(self)
        (weights, pipelines) = self.ensemble_

        y_pred = np.array([w*p.predict(X_test) 
                                for w,p in zip(weights, pipelines)])

        y_pred = y_pred.sum(axis=0)
        return y_pred

    def _predict_classification(self, X_test):
        """ Predict the class using the fitted ensemble and weighted majority
        voting
        """

        # first, get the weighted predictions from each member of the ensemble
        y_pred = self.predict_proba(X_test)

        # now take the majority vote
        y_pred = y_pred.argmax(axis=1)

        return y_pred        

    def predict(self, X_test):
        """ Use the fit ensemble to predict on the given test set.
        """
        _validate_fit_asl_wrapper(self)
        
        predict_f = self._predict_classification
        if self.estimator_named_step == "regressor":
            predict_f = self._predict_regression        
        
        return predict_f(X_test)

    def predict_proba(self, X_test):
        """ Use the automl ensemble to estimate class probabilities
        """
        if self.estimator_named_step == "regressor":
            msg = ("[asl_wrapper]: cannot use predict_proba for regression "
                "problems")
            raise ValueError(msg)

        _validate_fit_asl_wrapper(self)
        (weights, pipelines) = self.ensemble_

        res_shape = (X_test.shape[0], len(self.le_.classes_))
        res = np.zeros(shape=res_shape)

        for w,p in zip(weights, pipelines):

            # get the weighted class probabilities
            y_pred = w*p.predict_proba(X_test)
            
            # and make sure we are looking in the correct columns
            e = _get_asl_estimator(p, pipeline_step='classifier')
            p_classes = e.classes_
            res[:,p_classes] += y_pred

        return res

    def predict_dist(self, X_test, weighted=True):
        """ Use the ensemble to predict a normal distribution for each instance

        This method uses statsmodels.stats.weightstats.DescrStatsW to handle
        weights.

        Parameters
        ----------
        X_test: np.array
            The observations

        weighted: bool
            Whether to weight the predictions of the ensemble members

        Returns
        -------
        y_pred_mean: np.array
            The (weighted) predicted mean estimate

        y_pred_std: np.array
            The (weighted) standard deviation of the estimates
        """

        if self.estimator_named_step != "regressor":
            msg = ("[asl_wrapper]: predict_dist can only be used for "
                "regression problems")
            raise ValueError(msg)

        _validate_fit_asl_wrapper(self)
        (weights, pipelines) = self.ensemble_

        # make the predictions
        y_pred = np.array([p.predict(X_test) for p in pipelines])

        # extract summary statistics

        # the weighting does not work properly with single-member "ensembles"
        if weighted and len(weights) > 1:
            s = statsmodels.stats.weightstats.DescrStatsW(y_pred, weights=weights)
        else:
            s = statsmodels.stats.weightstats.DescrStatsW(y_pred)

        return (s.mean, s.std)

    def get_estimators(self):
        """ Extract the concrete estimators from the ensemble (RandomForest,
        etc.) as a list

        N.B. Predictions use the entire pipelines in the ensemble
        (preprocessors, etc.). This is primarily a convenience method for
        inspecting the learned estimators.
        """
        _validate_fit_asl_wrapper(self)
        pipelines = self.ensemble_[1]
        estimators = [
            _get_asl_estimator(p, self.estimator_named_step) for p in pipelines 
        ]
        return estimators

    def get_ensemble_model_summary(self):
        """ Create a mapping from pretty model names to their total weight in
        the ensemble
        
        Parameters
        ----------
        asl_wrapper: a fit AutoSklearnWrapper
        
        Returns
        -------
        ensemble_summary: dict of string -> float
            A mapping from model names to weights
        """
        weights = self.ensemble_[0]    
        estimators = self.get_estimators()
        
        summary = collections.defaultdict(float)
        
        for w, e in zip(weights, estimators):
            type_str = automl_utils.PRETTY_NAME_TYPE_MAP[type(e)]
            summary[type_str] += w
            
        return summary

    def get_params(self, deep=True):
        params = {
            'autosklearn_optimizer': self.autosklearn_optimizer,
            'ensemble_': self.ensemble_,
            'estimator_named_step': self.estimator_named_step,
            'args': self.args,
            "le_": self.le_,
            'metric': self.metric
        }
        return params

    def set_params(self, **parameters):
        if 'args' in parameters:
            self.args = parameters.pop('args')

        if 'autosklearn_optimizer' in parameters:
            self.autosklearn_optimizer = parameters.pop('autosklearn_optimizer')

        if 'ensemble_' in parameters:
            self.ensemble_ = parameters.pop('ensemble_')

        if 'estimator_named_step' in parameters:
            self.estimator_named_step = parameters.pop('estimator_named_step')

        if 'le_' in parameters:
            self.le_ = parameters.pop('le_')

        if 'metric' in parameters:
            self.metric = parameters.pop('metric')

        return self


    def __getstate__(self):
        """ Returns everything to be pickled """
        state = {}
        state['args'] = self.args
        state['autosklearn_optimizer'] = self.autosklearn_optimizer
        state['ensemble_'] = self.ensemble_
        state['estimator_named_step'] = self.estimator_named_step
        state['le_'] = self.le_
        state['metric'] = self.metric

        return state

    def __setstate__(self, state):
        """ Re-creates the object after pickling """
        self.args = state['args']
        self.autosklearn_optimizer = state['autosklearn_optimizer']
        self.estimator_named_step = state['estimator_named_step']
        self.ensemble_ = state['ensemble_']
        self.le_ = state['le_']
        self.metric = state['metric']

    def write(self, out_file):
        """ Validate that the wrapper has been fit and then write it to disk
        """
        _validate_fit_asl_wrapper(self)
        joblib.dump(self, out_file)

    @classmethod
    def read(cls, in_file):
        """ Read back in an asl_wrapper which was written to disk """
        asl_wrapper = joblib.load(in_file)
        return asl_wrapper


