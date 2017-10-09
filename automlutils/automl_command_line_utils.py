def add_automl_options(parser,
    default_out = ".",
    default_tmp = None,
    default_estimators = None,
    default_seed = 8675309,
    default_total_training_time = 3600,
    default_iteration_time_limit = 360,
    default_ensemble_size = 50,
    default_ensemble_nbest = 50):

    """ This function adds standard automl command line options to the parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the argparse parser

    all others
        the default options to use in the parser

    Returns
    -------
    None, but the automl options are added to the parser
    """
    automl_options = parser.add_argument_group("auto-sklearn options")

    automl_options.add_argument('--out', help="The output folder", 
        default=default_out)
    automl_options.add_argument('--tmp', help="If specified, this will be used "
        "as the temp directory rather than the default", default=default_tmp)

    automl_options.add_argument('--estimators', help="The names of the estimators "
        "to use for learning. If not specified, all available estimators will "
        "be used.", nargs='*', default=default_estimators)

    automl_options.add_argument('--seed', help="The random seed", type=int,
        default=default_seed)

    automl_options.add_argument('--total-training-time', help="The total training "
        "time for auto-sklearn.\n\nN.B. This appears to be more of a "
        "\"suggestion\".", type=int, default=default_total_training_time)

    automl_options.add_argument('--iteration-time-limit', help="The maximum "
        "training time for a single model during the search.\n\nN.B. This also "
        "appears to be more of a \"suggestion\".", type=int,
        default=default_iteration_time_limit)

    automl_options.add_argument('--ensemble-size', help="The number of models to keep "
        "in the learned ensemble.", type=int, default=default_ensemble_size)
    automl_options.add_argument('--ensemble-nbest', help="The number of models to use "
        "for prediction.", type=int, default=default_ensemble_nbest)

def add_automl_values_to_args(args,
        out = ".",
        tmp = None,
        estimators = None,
        seed = 8675309,
        total_training_time = 3600,
        iteration_time_limit = 360,
        ensemble_size = 50,
        ensemble_nbest = 50):
        
    """ Add the automl options to the given argparse namespace

    This function is mostly intended as a helper for use in ipython notebooks.
    """
    args.out = out
    args.tmp = tmp
    args.estimators = estimators
    args.seed = seed
    args.total_training_time = total_training_time
    args.iteration_time_limit = iteration_time_limit
    args.ensemble_size = ensemble_size
    args.ensemble_nbest = ensemble_nbest


def get_automl_options_string(args):
    """ This function creates a string suitable for passing to another script
        of the automl command line options.

    The expected use case for this function is that a "driver" script is given
    the automl command line options (added to its parser with add_automl_options),
    and it calls multiple other scripts which also accept the automl options.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace containing all of the options from add_automl_options

    Returns
    -------
    string:
        A string containing all of the automl options
    """

    args_dict = vars(args)

    # first, pull out the text arguments
    automl_options = ['out', 'tmp', 'seed', 'total_training_time', 
        'iteration_time_limit', 'ensemble_size', 'ensemble_nbest']

    # create a new dictionary mapping from the flag to the value
    automl_flags_and_vals = {'--{}'.format(o.replace('_', '-')) : args_dict[o] 
        for o in automl_options if args_dict[o] is not None}

    estimators = ""
    if args_dict['estimators'] is not None:
        estimators = " ".join(mt for mt in args_dict['estimators'])
        estimators = "--estimators {}".format(estimators)

    s = ' '.join("{} {}".format(k,v) for k,v in automl_flags_and_vals.items())
    s = "{} {}".format(estimators, s)

    return s

###
#   Utilities to help with OpenBLAS
###
def add_blas_options(parser, default_num_blas_cpus=1):
    """ Add options to the parser to control the number of BLAS threads
    """
    blas_options = parser.add_argument_group("blas options")

    blas_options.add_argument('--num-blas-threads', help="The number of threads to "
        "use for parallelizing BLAS. The total number of CPUs will be "
        "\"num_cpus * num_blas_cpus\". Currently, this flag only affects "
        "OpenBLAS and MKL.", type=int, default=default_num_blas_cpus)

    blas_options.add_argument('--do-not-update-env', help="By default, num-blas-threads "
        "requires that relevant environment variables are updated. Likewise, "
        "if num-cpus is greater than one, it is necessary to turn off python "
        "assertions due to an issue with multiprocessing. If this flag is "
        "present, then the script assumes those updates are already handled. "
        "Otherwise, the relevant environment variables are set, and a new "
        "processes is spawned with this flag and otherwise the same "
        "arguments. This flag is not inended for external users.",
        action='store_true')

def get_blas_options_string(args):
    """  Create a string suitable for passing to another script of the BLAS
    command line options
    """
    s = "--num-blas-threads {}".format(args.num_blas_threads)
    return s


def spawn_for_blas(args):
    """ Based on the BLAS command line arguments, update the environment and
    spawn a new version of the process

    Parameters
    ----------
    args: argparse.Namespace
        A namespace containing num_cpus, num_blas_threads and do_not_update_env

    Returns
    -------
    spawned: bool
        A flag indicating whether a new process was spawned. Presumably, if it
        was, the calling context should quit
    """
    import os
    import sys
    import shlex

    import misc.shell_utils as shell_utils

    spawned = False
    if not args.do_not_update_env:

        ###
        #
        # There is a lot going on with settings these environment variables.
        # please see the following references:
        #
        #   Turning off assertions so we can parallelize sklearn across
        #   multiple CPUs for different solvers/folds
        #       https://github.com/celery/celery/issues/1709
        #
        #   Controlling OpenBLAS threads
        #       https://github.com/automl/auto-sklearn/issues/166
        #
        #   Other environment variables controlling thread usage
        #       http://stackoverflow.com/questions/30791550
        #
        ###
        
        # we only need to turn off the assertions if we parallelize across cpus
        if args.num_cpus > 1:
            os.environ['PYTHONOPTIMIZE'] = "1"

        # openblas
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_blas_threads)
        
        # mkl blas
        os.environ['MKL_NUM_THREADS'] = str(args.num_blas_threads)

        # other stuff from the SO post
        os.environ['OMP_NUM_THREADS'] = str(args.num_blas_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_blas_threads)

        cmd = ' '.join(shlex.quote(a) for a in sys.argv)
        cmd += " --do-not-update-env"
        shell_utils.check_call(cmd)
        spawned = True

    return spawned


