import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

import copy

import numpy as np
from sklearn.base import ClassifierMixin
import sklearn.base

from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import Balancing
from autosklearn.pipeline.components.classification import ClassifierChoice

from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.base import BasePipeline
from automlutils.autosklearn_wrapper import AutoSklearnWrapper
import automlutils.configspace_utils as configspace_utils



class AslClassifierOnly(ClassifierMixin, BasePipeline):
    """ `auto-sklearn` for a classifier only

    That is, no preprocessing is considered at all. In principle, this should
    make optimization of the classifier more efficient.
    """

    def __init__(self, config=None, pipeline=None, dataset_properties=None,
                 include=None, exclude=None, random_state=None,
                 init_params=None, use_balancing=False):

        self._output_dtype = np.int32
        self.use_balancing = use_balancing
        super(AslClassifierOnly, self).__init__(
            config,
            pipeline,
            dataset_properties,
            include,
            exclude,
            random_state,
            init_params
        )

    def fit_transformer(self, X, y, fit_params=None):

        self.num_targets = 1 if len(y.shape) == 1 else y.shape[1]

        if fit_params is None:
            fit_params = {}

        # we have to handle balancing differently depending on the type of
        # classifier/regressor we have.
        #
        # see /autosklearn/pipeline/components/data_preprocessing/balancing/balancing.py
        if self.use_balancing:
            if self.configuration['balancing:strategy'] == 'weighting':
                balancing = Balancing(strategy='weighting')

                _init_params, _fit_params = balancing.get_weights(
                    y,
                    self.configuration['classifier:__choice__'],
                    None, #self.configuration['preprocessor:__choice__'],
                    init_params={},
                    fit_params={}
                )

                self.set_hyperparameters(
                    configuration=self.configuration,
                    init_params=_init_params
                )

                if _fit_params is not None:
                    fit_params.update(_fit_params)


        X, fit_params = super(AslClassifierOnly, self).fit_transformer(
            X, y, fit_params=fit_params
        )

        return X, fit_params

    def predict_proba(self, X, batch_size=None):
        """predict_proba.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        batch_size: int or None, defaults to None
            batch_size controls whether the pipeline will be
            called on small chunks of the data. Useful when calling the
            predict method on the whole array X results in a MemoryError.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        if batch_size is None:
            Xt = X

            # and predict_proba on the classifier
            return self.steps[-1][-1].predict_proba(Xt)

        else:
            if type(batch_size) is not int or batch_size <= 0:
                msg =("[AslClassifierOnly] batch_size must be a positive integer")
                raise ValueError(msg)

            else:
                # Probe for the target array dimensions
                target = self.predict_proba(X[0:2].copy())

                y = np.zeros((X.shape[0], target.shape[1]),
                             dtype=np.float32)

                # now just predict_proba one batch at a time
                for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                        batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    y[batch_from:batch_to] = \
                        self.predict_proba(X[batch_from:batch_to],
                                           batch_size=None).\
                            astype(np.float32)

                return y

    def _get_pipeline(self, init_params=None):

        msg = ("*** GETTING ASL CLASSIFIER-ONLY BASELINE PIPELINE ***")
        logger.debug(msg)
        
        default_dataset_properties = {'target_type': 'classification'}

        # check which features have which types
        feature_types = configspace_utils.get_feature_types(init_params)
                      

        steps = []

        if self.use_balancing:
            balancer = Balancing() 
            steps.append(["balancing", balancer])
            
        estimator = ClassifierChoice(default_dataset_properties)
        steps.append(["classifier", estimator])

        return steps


    def _get_hyperparameter_search_space(self, include=None, exclude=None,
                                         dataset_properties=None):
        """Create the hyperparameter configuration space.

        Parameters
        ----------
        include : dict (optional, default=None)

        Returns
        -------
        """


        estimator_choice_key = 'classifier:__choice__'
        #transformer_choice_key = 'preprocessor:__choice__'

        msg = ("*** CREATING ASL CLASSIFIER-ONLY PIPELINE HS SEARCH SPACE ***")
        logger.debug(msg)

        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()
        if not 'target_type' in dataset_properties:
            dataset_properties['target_type'] = 'classification'

        if dataset_properties['target_type'] != 'classification':
            msg = ("[asl_classifier_only._get_hss]: expected "
                "'classification' target type, but found '{}'".format(
                dataset_properties['target_type'])
            )
            raise ValueError(msg)

        pipeline = self.steps

        # construct the initial search space
        cs = self._get_base_search_space(
            cs=cs,
            dataset_properties=dataset_properties,
            exclude=exclude,
            include=include,
            pipeline=pipeline
        )

        # and pull out the list of available and allowed classifiers
        # and transformers
        estimators = cs.get_hyperparameter(estimator_choice_key).choices

        estimator_properties = pipeline[-1][1].get_available_components(
            dataset_properties
        )

        # keep track of what we can possibly use as a default
        possible_default_estimators = copy.copy(list(
            estimator_properties.keys())
        )

        # remove the original default since it is already set
        default = cs.get_hyperparameter(estimator_choice_key).default
        del possible_default_estimators[possible_default_estimators.index(default)]

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties

        return cs
    
    @classmethod
    def get_pipeline(klass,
            args=None,
            dataset_manager=None,
            dask_client=None,
            fold=None,
            **kwargs):
        """ Get a pipeline which includes only the classifier component
        """

        # try to fix tmp so we do not write everything there
        args = copy.copy(args)

        if args.tmp is not None:
            tmp = "autosklearn-{}.fold-{}".format(args.result_type, fold)
            args.tmp = os.path.join(args.tmp, tmp)

        asl_classifier = AutoSklearnWrapper(
            args=args,
            custom_pipeline=klass,
            estimator_named_step='classifier',
            dask_client=dask_client,
            custom_pipeline_attributes=kwargs
        )

        ###
        # Finally, put everything together into a single sklearn pipeline
        ###

        pipeline = sklearn.pipeline.Pipeline([
            ("classifier", asl_classifier)
        ])

        return pipeline

