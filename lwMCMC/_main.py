# -*- coding: utf-8 -*-
# Module: lwMCMC
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Release: lwMCMC 0.1.0
# Last modified : 25/10/2020
# Github: https://github.com/daniel-furman/lwMCMC

import numpy as np
import matplotlib.pyplot as plt

class lwMCMC(object):
    
    """ MCMC chain using a lightweight implementation of the Metropolis
    Hastings search algorithm, an object-oriented class with the 
    following inputs:
        
    * log_likelihood - function returning the log of the likelihood
       p(data|theta), pre-defined (see examples). The function should
       take two variables (data, theta) and return a single value
       log(p(data | theta)).

    * data is the fixed input data in the log_likelihood form

    * theta is a list or array with the starting parameter values for the
        Marcov chain.

    * step_size is a list or array with the step size in each dimension of
        theta."""
    
    def __init__(self, log_likelihood, data, theta, step_size, names=None,
                 seed=2145):
        self.log_likelihood = log_likelihood
        self.data = data
        self.theta = np.array(theta)
        self.nparams = len(theta)
        self.step_size = np.array(step_size)
        self.rng = np.random.RandomState(seed)
        self.naccept = 0
        self.current_loglike = log_likelihood(self.data, self.theta)
        self.samples = []
        if names is None:
            names = ["Paramter {:d}".format(k+1) for k in range(self.nparams)]
        self.names = names            

    def step(self, save=True):
        """Take a step in the MCMC chain (a single step)"""
        new_theta = self.theta + self.step_size * self.rng.normal(
            size=len(self.step_size))
        new_loglike = self.log_likelihood(self.data, new_theta)
        diff = new_loglike - self.current_loglike

        if diff >= 0:
            take_step = True
        else:
            take_step = self.rng.uniform() < np.exp(diff)

        if take_step:
            self.current_loglike = new_loglike
            self.theta = new_theta

        if save:
            self.samples.append(self.theta)
            if take_step: 
                self.naccept += 1

    def burn(self, nburn):
        """Number of burns (results not saved)"""
        for i in range(nburn):
            self.step(save=False)

    def run(self, nsteps):
        """Take nsteps steps (results saved)"""
        for i in range(nsteps):
            self.step()

    def accept_fraction(self):
        """Returns the fraction of candidate steps that were accpeted
        so far."""
        if len(self.samples) > 0:
            return float(self.naccept) / len(self.samples)
        else:
            return 0.
        
    def clear(self, step_size=None, theta=None):
        """Clear the list of stored samples from any runs so far.
        Optional change step_size or theta here."""
        if step_size is not None:
            assert len(step_size) == self.nparams
            self.step_size = np.array(step_size)
        if theta is not None:
            assert len(theta) == self.nparams
            self.theta = np.array(theta)
            self.current_loglike = self.log_likelihood(self.data, self.theta)
        self.samples = []
        self.naccept = 0
        
    def get_samples(self):
        """Return the sampled theta values at each step in the chain as a
        2d numpy array."""
        return np.array(self.samples)
        
    def plot_hist(self):
        """Plot a histogram of the sample values for each parameter in the
        theta vector."""
        all_samples = self.get_samples()
        for k in range(self.nparams):
            theta_k = all_samples[:,k]
            plt.hist(theta_k, bins=100)
            plt.xlabel(self.names[k])
            plt.ylabel("N Samples")
            plt.show()
        
    def plot_samples(self):
        """Plot the sample values over the course of the chain so far."""
        all_samples = self.get_samples()
        for k in range(self.nparams):
            theta_k = all_samples[:,k]
            plt.plot(range(len(theta_k)), theta_k)
            plt.xlabel("Step in chain")
            plt.ylabel(self.names[k])
            plt.show()

    def calculate_mean(self, weight=None):
        """Calculate the mean of each parameter according to the samples
        taken so far. Optionally, provide a weight array to weight
        the samples."""
        return np.average(self.get_samples(), axis=0, weights=weight)
    
    def calculate_cov(self, weight=None):
        """Calculate the covariance matrix of the parameters according to
        the samples taken so far. Optionally, provide a weight array to 
        weight the samples."""
        return np.cov(self.get_samples(), rowvar=False, aweights=weight)
