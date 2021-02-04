#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:17:41 2021

@author: danielfurman
"""

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        errno = pytest.main(self.test_args)
        sys.exit(errno)


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


setup(
    name="lwMCMC",
    version="0.1.0",
    author="Daniel Ryan Furman",
    author_email="dryanfurman@gmail.com",
    description=("Class for MCMC serach in Python"),
    license=license,
    keywords="bayesian montecarlo machinelearning deeplearning",
    url="https://github.com/daniel-furman/lw-MCMC",
    packages=find_packages(exclude=('tests', 'docs', 'data', 'script')),                           
    #package_dir={"": "src"},
    #packages=["lwMCMC"],
    long_description="See documentation at GitHub",
    install_requires=["numpy", "matplotlib"],
    tests_require=["pytest", "coverage"],
    cmdclass={"test": PyTest},
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