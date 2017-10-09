from autosklearn.pipeline.components.base import AutoSklearnComponent


class AutoSklearnImputationAlgorithm(AutoSklearnComponent):
    """Provide an abstract interface for imputation algorithms in
    auto-sklearn.
    
    This interface is based on fancyimpute.Solver

    See :ref:`extending` for more information."""

    def __init__(self):
        self.imputer = None

    def transform(self, X):
        """Call the `complete` method of the underlying imputer
        
        N.B. We call the method `transform` to keep consistent with the
        sklearn pipeline contract.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        X : array
            Return the completed training data

        Notes
        -----
        Please see the `fancyimpute.Solver` docs for further information."""
        raise NotImplementedError()

    def get_imputer(self):
        """Return the underlying imputer object.

        Returns
        -------
        imputer : the underlying imputer object
        """
        return self.imputer
