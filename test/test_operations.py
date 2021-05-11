# Module: lwMCMC
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Release: lwMCMC 0.2
# Last modified : May 11 2021
# Github: https://github.com/daniel-furman/lwMCMC

import numpy as np
from pytest import fixture

# Unit tests for MCMC on toy Gaussian model (with four-dimensions).

def gauss_log_likelihood(data, theta):
    """Computes log(likelihood) for a set of hypothetical modeling parameters.

    Here, the "data" are just the fixed values (beta_0, beta_1, beta_2, beta_3)
    where the Gaussian is centered.
    The theta parameters are the MCMC's trial of x,y,z.  When they match
    (x0,y0,z0), that will correspond to the maximum likelihood.  For other
    values of (x,y,z), the likelihood is smaller.
    """
    x, y, z, a = theta
    beta_0, beta_1, beta_2, beta_3, sigma = data
    # likelihood = exp(-((x-beta_0)^2 + (y-beta_1)^2 + (z-beta_2)^2) / (2 sigma^2))
    # log(likelihood) = -((x-beta_0)^2 + (y-beta_1)^2 + (z-beta_2)^2) / (2 sigma^2))
    return -((x-beta_0)**2 + (y-beta_1)**2 + (z-beta_2)**2 + (a-beta_3)**2) / (2 * sigma**2)

beta_0, beta_1, beta_2, beta_3 = 12, 21, 30, 45   # The peak location is at (12, 21, 30, 45)
sigma = 0.5 # Uncertainty on each is 0.5
data = (beta_0, beta_1, beta_2, beta_3, sigma) # The toy model
step_size = (0.5,0.5,0.5,0.5)

# Test the log likelihood function before calling the class
theta = np.zeros(4)
assert gauss_log_likelihood(data, (beta_0,beta_1,beta_2,beta_3)) == 0.  # Peak has a value of 0
assert gauss_log_likelihood(data, (beta_0-0.01, beta_1+0.01, beta_2-0.01, beta_3+0.1)) < 0
assert np.isfinite(gauss_log_likelihood(data, theta))

# Call the MCMC object so PyTest can run.
@fixture
def op():
    from lwMCMC import MCMC
    return MCMC(gauss_log_likelihood, data, theta, step_size,
                names=('$beta_0$','$beta_1$','$beta_2$', '$beta_3$'))

def tests(op):
    op.run_chain(nsteps=2000)
    # Start over from our initial guess and create chain
    op.clear_chain(theta=theta)
    op.burnout(nburn=2000)
    op.run_chain(nsteps=100000)
    assert 0.3 < op.ratio_accepted() < 0.7
    mean = op.parameter_means()
    # low rtol for the mean
    assert np.allclose(mean, (beta_0,beta_1,beta_2,beta_3), rtol=1.e-2)
    cov = op.parameter_cov()
    #not ass accurate for cov
    assert np.allclose(cov.diagonal(), sigma**2, rtol=0.1)

    # Finally, check the prior and weight functions
    # Let's say you learn that beta_1 - beta_0 > 10.
    samples = op.chain_samples()
    x_samples, y_samples, z_samples, a_samples = samples.T  # extract by columns.
    print(samples.T)
    weight = np.ones(len(samples))
    weight[y_samples - x_samples <= 10] = 0.
    print(weight)

    prior_mean = op.parameter_means(weight)
    prior_cov = op.parameter_cov(weight)
    print('After applying prior that y-x > 10:')
    print('Mean values of (x,y,z) = ', prior_mean)

    # Note that the means shifted and all uncertainties moved
    # down some amount
    assert prior_mean[0] < beta_0
    assert prior_mean[1] > beta_1
    assert prior_cov[0,1] > 0
    assert all(np.sqrt(prior_cov.diagonal()[:2]) < sigma)
    print('done')
