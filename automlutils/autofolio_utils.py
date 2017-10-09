###
#   Utilities to help with the autofolio package:
#
#       https://github.com/mlindauer/AutoFolio
###

def load_autofolio(fn:str):
    """ Read a pickled autofolio model.

    Parameters
    ----------
    fn: string
        The path to the file

    Returns
    -------
    A namedtuple with the following fields:
        - scenario: ASlibScenario
            The aslib scenario information used to learn the model

        - preprocessing: list of autofolio.feature_preprocessing objects
            All of the preprocessing objects

        - pre_solver: autofolio.pre_solving.aspeed_schedule.Aspeed
            Presolving schedule

        - selector: autofolio.selector
            The trained pairwise selection model

        - config: ConfigSpace.configuration_space.Configuration
            The dict-like configuration information
    """
    import collections
    import pickle

    af = collections.namedtuple("af",
        "scenario,preprocessing,pre_solver,selector,config"
    )

    with open(fn, "br") as fp:
        autofolio_model = pickle.load(fp)

    autofolio_model = af(
        scenario = autofolio_model[0],
        preprocessing = autofolio_model[1],
        pre_solver = autofolio_model[2],
        selector = autofolio_model[3],
        config = autofolio_model[4]
    )

    return autofolio_model

