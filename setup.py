#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:17:41 2021

@author: danielfurman
"""

import os
import sys
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open('docs/long_description.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="lwMCMC",
    version="0.1.0",
    author="Daniel Ryan Furman",
    author_email="dryanfurman@gmail.com",
    description=("Class for MCMC serach in Python"),
    long_description=readme,
    long_description_content_type='text/markdown',
    license=license,
    keywords="bayesian montecarlo machinelearning deeplearning",
    url="https://github.com/daniel-furman/lwMCMC",
    packages=find_packages(),
    classifiers=[
        "Topic :: MCMC for predictive modeling",
        "License :: MIT",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bayesian Statistics"
        ],

)
