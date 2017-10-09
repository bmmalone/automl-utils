# Installation for `automl-utils`

This project is written in `python3` and can be installed with `pip`. 
The project indirectly depends on the [`pyrfr`](https://github.com/automl/random_forest_run)
and, thus, also requires [`SWIG-3`](http://www.swig.org/).

```
pip3 install -r requirements.txt
pip3 install -r requirements.2.txt
```

However, the framework uses [`autosklearn`](http://automl.github.io/auto-sklearn/stable/),
which in turn depends on [`scikit-learn`](http://scikit-learn.org/stable/). We
recommend to install an efficient BLAS implementation, such as OpenBLAS. Some
(unofficial) installation instructions can be found [here](https://gist.github.com/bmmalone/1b5f9ff72754c7d4b313c0b044c42684).

**More installation description to come.**
