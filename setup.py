# Module: lwMCMC
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Release: lwMCMC 0.2
# Last modified : May 11 2021
# Github: https://github.com/daniel-furman/lwMCMC

import os
import sys
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open('README.md') as f:
    readme = f.read()

setup(
    name="lwMCMC",
    version="0.2",
    author="Daniel Ryan Furman",
    author_email="dryanfurman@gmail.com",
    description=("Bayesian inference with MCMC parameter space sampling, with options to compare a from-scratch random walk NumPy implementation with the PyMC3 Hamiltonian approach."),
    long_description="See documentation at https://github.com/daniel-furman/lwMCMC",
    license="MIT",
    keywords="bayesian-inference machine-learning statistical-modeling",
    url="https://github.com/daniel-furman/lwMCMC",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib", "pymc3"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
        ],

)
