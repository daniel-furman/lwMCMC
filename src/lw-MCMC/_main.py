#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 08:29:05 2021

@author: danielfurman
"""
# CLASS: Metropolis Hastings Markov Chain Monte Carlo
# Author: Daniel Furman <dryanfurman@gmail.com>
# Last modified : Feb. 3, 2021
# MIT License

import numpy as np
import matplotlib.pyplot as plt

class lwMCMC(object):
    """Class that can run an MCMC chain using the Metropolis Hastings
    algorithm for parameter search. 
    
    lwMCMC = lwMCMC(log_likelihood, data, theta, step_size)
        
    * log_likelihood - function returning the log of the likelihood
       p(data|theta), which needs to be pre-defined (see example).
                
       The function should take two variables (data, theta) and 
       return a single value log(p(data | theta)).

    * data is the input data in whatever form the log_likelihood function
        is expecting it. This is fixed over the course of running an
        MCMC chain.

    * theta is a list or array with the starting parameter values for the
        Marcov chain.

    * step_size is a list or array with the step size in each dimension of
        theta.
    """
    
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
        """See appendix for logic and math"""
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
        """Take nburn steps, but don't save the results"""
        for i in range(nburn):
            self.step(save=False)

    def run(self, nsteps):
        """Take nsteps steps"""
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
        
        You can also change the step_size to a new value at this time by
        giving a step_size as an optional parameter value.
        
        In addition, you can reset theta to a new starting value if theta is
        not None.
        """
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
        taken so far.
        
        Optionally, provide a weight array to weight the samples.
        
        Returns the mean values as a numpy array.
        """
        return np.average(self.get_samples(), axis=0, weights=weight)
    
    def calculate_cov(self, weight=None):
        """Calculate the covariance matrix of the parameters according to
        the samples taken so far.

        Optionally, provide a weight array to weight the samples.
        
        Returns the covariance matrix as a 2d numpy array.
        """
        return np.cov(self.get_samples(), rowvar=False, aweights=weight)


    """
    Appendix

    Context:
        
    The motivation of MCMC is a need to understand a complex probability 
    distribution, p(x). In applications to Bayesian statistics, this distr.
    is usually a posterior from a Bayesian analysis. A good way to understand 
    intractable distributions is to simulate realisations from it and study those.
    However, this often isn’t easy, either. The idea behind MCMC is to simulate a
    Markov chain whose equilibrium distribution is p(x). Metropolis-Hastings (M-H)
    provides a way to correct a fairly arbitrary transition kernel q(x’|x) (which
    typically won’t have p(x) as its equilibrium) to give a chain which does have
    the desired target. In M-H, the transition kernel is used to generate a
    proposed new value, x’ for the chain, which is then accepted as the new state
    at random with a particular probability a(x’|x)=min(1,A), where 
    A = p(x’)q(x|x’)/[p(x)q(x’|x)].

    For more see:
        https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/

    Step Function:
        
    We use log(likelihood) rather than likelihood here.
        
    Convert the code that was shown in class for doing a single step of the MCMC chain
    to use the log_likelihood function, rather than the normal likelihood.

    * Calculate the new theta value.
    * Calculate the log(likelihood) for that theta.       
    * Decide whether or not to take the step.
    * If taking the step, update self.current_loglike and self.theta.
    * If save==True, add the sample to self.samples and maybe add 1 to self.naccept.
    

"""
