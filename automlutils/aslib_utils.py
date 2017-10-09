###
#   Utilities to help with the ASlib package:
#
#       https://github.com/mlindauer/ASlibScenario
###

def _get_feature_step_dependencies(scenario, feature_step):
    fg = scenario.feature_group_dict[feature_step]
    return set(fg.get('requires', []))
    
def check_all_dependencies(scenario, feature_steps):
    """ Ensure all dependencies all included for all feature sets
    
    Parameters
    ----------
    scenario: ASlibScenario
        The ASlib scenario
        
    feature_steps: list-like of strings
        A list of feature sets
        
    Returns
    -------
    dependencies_met: bool
        True if all dependencies for all feature sets are included,
        False otherwise
    """
    feature_steps = set(feature_steps)
    for feature_step in feature_steps:
        dependencies = _get_feature_step_dependencies(scenario, feature_step)
        if not dependencies.issubset(feature_steps):
            return False
    return True

def extract_feature_step_dependency_graph(scenario):
    """ Create a graph encoding dependencies among the feature steps

    Specifically, an edge in the graph from A to B implies B requires A. Root
    nodes have no dependencies.

    Parameters
    ----------
    scenario: ASlibScenario
        The ASlib scenario

    Returns
    -------
    dependency_graph: networkx.DiGraph
        A directed graph in which each node corresponds to a feature step and
        edges encode the dependencies among the steps
    """
    # build the feature_group graph
    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(scenario.feature_steps)

    # an edge from A -> B implies B requires A
    for feature_step in scenario.feature_steps:
        dependencies = _get_feature_step_dependencies(scenario, feature_step)
        
        for dependency in dependencies:
            dependency_graph.add_edge(dependency, feature_step)

    return dependency_graph

def extract_feature_names(scenario, feature_steps):
    """ Extract the names of features for the specified feature steps

    Parameters
    ----------
    scenario: ASlibScenario
        The ASlib scenario

    feature_steps: list-like of strings
        The names of the feature steps

    Returns
    -------
    feature_names: list of strings
        The names of all features in the given steps
    """
    feature_names = [
        scenario.feature_group_dict[f]['provides']
            for f in feature_steps
    ]
    feature_names = utils.flatten_lists(feature_names)
    return feature_names

def load_scenario(scenario_path, return_name=True):
    """ Load the ASlibScenario in the given path

    This is a convenience function that can be more easily used in list
    comprehensions, etc.

    Parameters
    ----------
    scenario_path: path-like
        The location of the scenario directory

    return_name: bool
        Whether to return the name of the scenario
    Returns
    -------
    scenario_name: string
        The name of the scenario (namely, `scenario.scenario`). This is only
        present if return_name is True

    scenario: ASlibScenario
        The actual scenario
    """
    scenario = ASlibScenario()
    scenario.read_scenario(scenario_path)

    if return_name:
        return scenario.scenario, scenario
    else:
        return scenario


def load_all_scenarios(scenarios_dir):
    """ Load all scenarios in scenarios_dir into a dictionary

    In particular, this function assumes all subdirectories within
    scenarios_dir are ASlibScenarios

    Parameters
    ----------
    scenarios_dir: path-like
        The location of the scenarios

    Returns
    -------
    scenarios: dictionary of string -> ASlibScenario
        A dictionary where the key is the name of the scenario and the value
        is the corresponding ASlibScenario
    """
        
    # first, just grab everything in the directory
    scenarios = [
        os.path.join(scenarios_dir, o) for o in os.listdir(scenarios_dir)
    ]

    # only keep the subdirectories
    scenarios = sorted([
        t for t in scenarios if os.path.isdir(t)
    ])

    # load the scenarios
    scenarios = [
        load_scenario(s) for s in scenarios
    ]
    
    # the list was already (key,value) pairs, so create the dictionary
    scenarios = dict(scenarios)

    return scenarios

def create_cv_splits(scenario):
    """ Create cross-validation splits for the scenario and save to file

    In particular, this is useful if a scenario does not already include cv
    splits, and cv splits will be used in multiple locations.

    Parameters
    ----------
    scenario: ASlibScenario
        The scenario

    Returns
    -------
    None, but new cv splits will be assigned to the instances and the file
    scenario.dir_.cv.arff will be (over)written.
    """
    import arff

    # first, make sure the cv splits exist
    if scenario.cv_data is None:
        scenario.create_cv_splits()

    # format the cv splits for the arff file
    scenario.cv_data.index.name = 'instance_id'
    cv_data = scenario.cv_data.reset_index()
    cv_data['repetition'] = 1
    cv_data['fold'] = cv_data['fold'].astype(int)

    # and put the fields in the correct order
    fields = ['instance_id', 'repetition', 'fold']
    cv_data_np = cv_data[fields].values

    # create the yaml-like dict for writing
    cv_dataset = {
        'description': "CV_{}".format(scenario.scenario),
        'relation': "CV_{}".format(scenario.scenario),
        'attributes': [
            ('instance_id', 'STRING'),
            ('repetition', 'NUMERIC'),
            ('fold', 'NUMERIC')
        ],
        'data': cv_data_np
    }

    # and write the file
    cv_loc = os.path.join(scenario.dir_, "cv.arff")
    with open(cv_loc, 'w') as f:
        arff.dump(cv_dataset, f)
