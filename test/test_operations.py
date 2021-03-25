#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:36:32 2021

@author: danielfurman
"""

import numpy as np
from pytest import fixture

@fixture
def op():
    from lwMCMC._main import lwMCMC as MCMC
    return MCMC(gauss_log_likelihood, data, theta, step_size,
                names=('$x$','$y$','$z$'))


# Unit tests for general purpose utility, MCMC.
# Let's test this with a 3D Gaussian.

def gauss_log_likelihood(data, theta):
    """Comput a Gaussian log(likelihood) for a trial value theta = (x,y,z)

    Here, the "data" are just the fixed values (x0,y0,z0) where the Gaussian
    is centered.

    The theta parameters are the MCMC's trial of x,y,z.  When they match
    (x0,y0,z0), that will correspond to the maximum likelihood.  For other
    values of (x,y,z), the likelihood is smaller.
    """
    x, y, z = theta
    x0, y0, z0, sigma = data
    # likelihood = exp(-((x-x0)^2 + (y-y0)^2 + (z-z0)^2) / (2 sigma^2))
    # log(likelihood) = -((x-x0)^2 + (y-y0)^2 + (z-z0)^2) / (2 sigma^2))
    return -((x-x0)**2 + (y-y0)**2 + (z-z0)**2) / (2 * sigma**2)

x0, y0, z0 = 12, 21, 30    # The peak location is at (12, 21, 30)
sigma = 0.5                # Uncertainty on each is 0.5
data = (x0, y0, z0, sigma) # The "data" for this toy model

assert gauss_log_likelihood(data, (x0,y0,z0)) == 0.  # Peak has a value of 0
assert gauss_log_likelihood(data, (x0-0.01, y0+0.01, z0-0.01)) < 0.
  # Any other point, no matter how close, is < 0.

# Use an initial guess of (0,0,0)
theta = np.zeros(3)
#print('log likelihood at (0,0,0) = ',gauss_log_likelihood(data, theta))

# Note that the likelihood underflows to 0.
#print('likelihood at (0,0,0) = ',np.exp(gauss_log_likelihood(data, theta)))
# But the log likelihood is fine.

assert np.isfinite(gauss_log_likelihood(data, theta))

# Make the mcmc object with this likelihood function
step_size = (0.5,0.5,0.5)  # 1 sigma in each direction

def testers(op):

    # Run for 1000 steps
    op.run(1000)

    # Plot what we have from that.
    #print('First 1000 steps:')
    #op.plot_samples()

    # The initial samples are clearly biased.
    # But by 300 or so, it seems to have settled into a steady state behavior.
    # This means that around 500 steps is a safe value for the burn in.

    # Start over from our initial guess and burn 1000 before running 100000 steps
    op.clear(theta=theta)
    op.burn(1000)
    op.run(100000)

    # Now we shouldn't see any evidence of the initial ramp.
    #print('After burn of 1000, next 100,000 steps:')
    #op.plot_samples()

    # We'd like an acceptance rate near 50%.
    #print('Acceptance rate is ', op.accept_fraction())
    #print()
    assert 0.2 < op.accept_fraction() < 0.8

    # The mean should be pretty close to the actual maximum likelihood.
    mean = op.calculate_mean()
    #print('Mean values of (x,y,z) = ', mean)
    #print('loglike at mean = ', gauss_log_likelihood(data, mean))
    #print()

    assert np.allclose(mean, (x0,y0,z0), rtol=1.e-3)

    # The covariance matrix is less well estimated, but still well witin 10%
        # with only 10,000 steps
    cov = op.calculate_cov()
    #print('Uncertainties of (x,y,z) = ', np.sqrt(cov.diagonal()))
    #print('Full covariance matrix = \n', cov)
    #print()
    assert np.allclose(cov.diagonal(), sigma**2, rtol=0.1)


    # Show the one-d distributions for each of (x,y,z)
    # These should look fairly smooth.
    #op.plot_hist()

    # Finally, check the weighted average options
    # Let's say you learn that y - x > 10.
    samples = op.get_samples()
    x_samples, y_samples, z_samples = samples.T  # extract by columns.
    weight = np.ones(len(samples))
    weight[y_samples - x_samples <= 10] = 0.

    prior_mean = op.calculate_mean(weight)
    prior_cov = op.calculate_cov(weight)
    #print('After applying prior that y-x > 10:')
    #print('Mean values of (x,y,z) = ', prior_mean)
    #print('Uncertainties of (x,y,z) = ', np.sqrt(prior_cov.diagonal()))
    #print('Full covariance matrix = \n', prior_cov)

    # Note that the means shifted.  y is pushed up and x is pushed down.
    # Also, x and y are now correlated.
    # And all uncertainties moved down, even z (slightly)

    assert prior_mean[0] < x0
    assert prior_mean[1] > y0
    assert prior_cov[0,1] > 0
    assert all(np.sqrt(prior_cov.diagonal()[:2]) < sigma)
