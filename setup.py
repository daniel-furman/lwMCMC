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

with open('README.md') as f:
    readme = f.read()

setup(
    name="lwMCMC",
    version="0.1",
    author="Daniel Ryan Furman",
    author_email="dryanfurman@gmail.com",
    description=("Parameter space sampling with lightweight MCMC powered by NumPy and Metropolis Hastings"),
    long_description="See documentation at https://github.com/daniel-furman/lwMCMC",
    license="MIT",
    keywords="bayesian-inference machine-learning statistical-modeling",
    url="https://github.com/daniel-furman/lwMCMC",
    packages=find_packages(),
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
