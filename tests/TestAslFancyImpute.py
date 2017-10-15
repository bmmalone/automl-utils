"""
Unit and integration tests for the various automl-utils

This structure of this file is taken from an old blog post here:
    http://blog.jameskyle.org/2010/10/nose-unit-testing-quick-start/
"""


class TestAslFancyImpute(object):

    @classmethod
    def init_data_structures(self):
        """ Initialize some relevant data structures """
        from argparse import Namespace
        import automlutils.automl_command_line_utils as clu

        self.args = Namespace()
        clu.add_automl_values_to_args(args, total_training_time=30)


    @classmethod
    def teardown_class(klass):
        """This method is run once for each class _after_ all tests are run"""
        pass

    def setUp(self):
        """This method is run once before _each_ test method is executed"""
        pass

    def teardown(self):
        """This method is run once after _each_ test method is executed"""
        pass

    def test_mcar_classification(self):
        """ Test on a simple classification problem with MCAR data

        TODO: this does not actually test anything; it just runs through the
        steps of loading the datasets, adding missing values, selecting an
        imputation algorithm, and fitting an asl_wrapper. It will raise
        exceptions if anything fails, though.
        """     
        import sklearn.datasets

        from automlutils.asl_fancyimput import ImputerChoice
        import misc.missing_data_utils

        X_complete, y = sklearn.datasets.load_iris(return_X_y=True)

        # mcar
        missing_likelihood = 0.2
        X_mcar_incomplete = missing_data_utils.get_mcar_incomplete_data(
            X_complete,
            missing_likelihood
        )

        mcar_data = missing_data_utils.get_incomplete_data_splits(
            X_complete,
            X_mcar_incomplete,
            y
        )

        # get a list of all possible imputers
        imputer_choice = ImputerChoice(None)
        imputer_choices = imputer_choice.get_available_components()

        # now, try them all out
        for imputer_name in imputer_choices.keys():
            print("=== {}: classification ===".format(imputer_name))
            
            custom_pipeline = functools.partial(
                ImputerChoice.get_classification_pipeline,
                imputer_name=imputer_name
            )

            asl_classifier = AutoSklearnWrapper(
                args=args,
                custom_pipeline=custom_pipeline,
                estimator_named_step='classifier'    
            )
            
            training_results = missing_data_utils.train_on_incomplete_data(
                asl_classifier,
                mcar_data
            )

    def test_mcar_regression(self):
        """ Test on a simple regression problem with MCAR data

        TODO: this does not actually test anything; it just runs through the
        steps of loading the datasets, adding missing values, selecting an
        imputation algorithm, and fitting an asl_wrapper. It will raise
        exceptions if anything fails, though.
        """     
        import sklearn.datasets

        from automlutils.asl_fancyimput import ImputerChoice
        import misc.missing_data_utils

        X_complete, y = sklearn.datasets.load_diabetes(return_X_y=True)

        # mcar
        missing_likelihood = 0.2
        X_mcar_incomplete = missing_data_utils.get_mcar_incomplete_data(
            X_complete,
            missing_likelihood
        )

        mcar_data = missing_data_utils.get_incomplete_data_splits(
            X_complete,
            X_mcar_incomplete,
            y
        )

        # get a list of all possible imputers
        imputer_choice = ImputerChoice(None)
        imputer_choices = imputer_choice.get_available_components()

        # now, try them all out
        for imputer_name in imputer_choices.keys():
            print("=== {}: regression ===".format(imputer_name))
            
            custom_pipeline = functools.partial(
                ImputerChoice.get_classification_pipeline,
                imputer_name=imputer_name
            )

            asl_regressor = AutoSklearnWrapper(
                args=args,
                custom_pipeline=custom_pipeline,
                estimator_named_step='regressor'    
            )
            
            training_results = missing_data_utils.train_on_incomplete_data(
                asl_regressor,
                mcar_data
            )

