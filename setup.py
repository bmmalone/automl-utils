from setuptools import find_packages, setup

console_scripts = []

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='automlutils',
        version='0.0.1',
        description="This package includes utilities for AutoML in python3.",
        long_description=readme(),
        keywords="automl autosklearn utilities",
        url="https://github.com/bmmalone/automl-utils",
        author="Brandon Malone",
        author_email="bmmalone@gmail.com",
        license='MIT',
        packages=find_packages(),
        install_requires = [
            'cython',
            'numpy',
            'scipy',
            'scikit-learn',
            'statsmodels',
            'matplotlib',
            'pandas',
            'networkx',
            'docopt',
            'tqdm',
            'joblib',
            'graphviz',
            'misc', # handled by requirements.txt 
            'auto-sklearn', # handled by requirements.txt
            'aslib_scenario' # handled by requirements.txt

        ],
        include_package_data=True,
        test_suite='nose.collector',
        tests_require=['nose'],
        entry_points = {
            'console_scripts': console_scripts
        },
        zip_safe=False
        )
