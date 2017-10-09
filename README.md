# `automl-utils`

This repo contains various utilities for performing AutoML in python3. It mostly
contains wrappers for working with [`auto-sklearn`](https://github.com/automl/auto-sklearn)
and related projects, including the algorithm selection framework provided by
the [`ASlib`](https://github.com/mlindauer/ASlibScenario) project.

**This project is under "alpha" development and even `master` should not be
considered stable.**

### Installation

This project is written in `python3` and can be installed with `pip`. 
The project indirectly depends on the [`pyrfr`](https://github.com/automl/random_forest_run)
and, thus, also requires [`SWIG-3`](http://www.swig.org/).

```
pip3 install -r requirements.txt
pip3 install -r requirements.2.txt
```


**N.B.** If installing under anaconda, please use `pip` rather than `pip3`.

Please see the [installation docs](docs/installation.md) for more details,
including detailed prerequisite descriptions.(Locally, viewing
[the html version](docs/installation.html) may be easier.)

