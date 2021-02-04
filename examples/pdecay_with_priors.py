#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:36:32 2021

@author: danielfurman
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import corner
from lwMCMC.src.lwMCMC._main import lwMCMC as MCMC

"""

The file ParticleDecay.dat contains (simulated) data of a particle decay.  The
    columns are time t_i, count rate R_i, and uncertainty in the count rate
    sigma_i.

     t_i      R_i  sigma_i
    0.00    16.93     4.47
    0.10    15.66     4.39
    0.20    25.69     4.31
    0.30    15.48     4.23
    0.40    21.85     4.16
    0.50    20.46     4.08
    
    ...
    
We will model this data as a constant background plus an exponential
decay term:
    $$
    R(t) = A + B e^{-\lambda t}
    $$
    where lambda is the decay constant.

Requires a relatively good starting guess. So start by trying to
    eyeball a rough estimate of a reasonable model.

Use the MCMC class to sample the likelihood function
    $p(\mathcal{D}|\boldsymbol{\theta})$ and obtain the mean and variance of
    $\lambda$.  Starting at the above guess.

Suppose via an independent method tof measurement that the
    background of the instrument is $A = 24.38 \pm 0.44$.  Use this as a
    gaussian prior in your MCMC above to obtain the new posterior
    (mean and variance) for $\lambda$.

Use corner to plot the chains from both parts (b) and (c). 

"""

def decay_model(t, theta):
    """Returning the model values of R(t) at the times t,
    given theta = (A, B, lam).
    """
    a, b, lam = theta
    return a + b * np.exp(-lam*t)

def decay_start():
    """Return a good starting point for the chain as a tuple (A, B, lam)

    This needs to be a bit better than for question 1,
    so try to have a reasonable guess.
    """
    # Basically anything that kind of goes through the data points is fine.
    return (30, 50, 1)

def plot_data_with_model(data, theta):
    """Plot the data (with error bars) along with the given model,
    parametrized by theta.
    """
    t, r, sigma = data
    plt.errorbar(t, r, sigma, fmt='o', ms=5)
    model = decay_model(t, theta)
    plt.plot(t, model)
    plt.show()
    
# Check that the initial guess is in the right ballpark.
data = np.loadtxt('data/ParticleDecay.dat').T
plot_data_with_model(data, decay_start())

def decay_loglike(data, theta):
    """Return the natural logarithm of the likelihood P(data | theta) for our
    model of the decay data.
    
    data is expected to be a tuple of numpy arrays = (t, R, sigma)
    theta is expected to be an array of parameters = (A, B, lam)
    """

    t, r, sigma = data
    n = len(t)
    model = decay_model(t, theta)
    lnlike = -0.5 * (n*np.log(2.*np.pi) + 
                     np.sum(2.*np.log(sigma) + (r-model)**2 / sigma**2))
    return lnlike

def decay_step_size():
    """Return a good step size to use as a 3-element tuple giving the steps
    in each of (A, B, lam).
    """
    return (0.5, 1, 0.08)

def decay_nburn():
    """Return how many steps to use for the burn in.
    """
    return 1000

def decay_nsteps():
    """Return how many steps to use in the chain
    """
    return 100000

# Initial guess:
start = decay_start()
print('Log(like) at {:s} = {:.1f}'.format(str(start),
                                          decay_loglike(data, start)))

# Make the object
decay_mcmc = MCMC(decay_loglike, data, start, decay_step_size(), names=(
    '$A$', '$B$', r'$\lambda$'))

# Run the burn-in
decay_mcmc.burn(decay_nburn())

# Run for the specified number of steps
decay_mcmc.run(decay_nsteps())
print('After running for {} steps:'.format(decay_nsteps()))
decay_mcmc.plot_samples()
print('Acceptance rate is ', decay_mcmc.accept_fraction())
assert 0.3 < decay_mcmc.accept_fraction() < 0.7 
    #Adjust step sizes if this fails.

print('Mean value of params = ', decay_mcmc.calculate_mean())
print()

# Run for 2N more steps
decay_mcmc.run(2*decay_nsteps())
print('After running for {} steps:'.format(3*decay_nsteps()))
mean = decay_mcmc.calculate_mean()
print('Mean value of params = ', mean)

cov = decay_mcmc.calculate_cov()
print('Uncertainties of params = ', np.sqrt(cov.diagonal()))

# Report the result
print()
print('The inferred decay rate is {:.2f} +- {:.2f} / sec'.format(mean[2],
                                                        np.sqrt(cov[2,2])))

# Double check that everything worked by looking at the best fit model with
# the data.
plot_data_with_model(data, mean)

# And here are the one-d distributions, which should be fairly smooth.
decay_mcmc.plot_hist()

def calculate_prior_weights(mcmc):
    """Calculate appropriate weights for the mcmc samples, given the prior
    A = 24.38 +- 0.44.
 
    Returns the weights as a numpy array.
    """
    samples = mcmc.get_samples()
    a_prior = 24.38
    sigma_prior = 0.44
    a, b, lam = samples.T
    weight = np.exp(-0.5 * (a-a_prior)**2/sigma_prior**2) 
    return weight
    
def calculate_lambda_with_prior(mcmc):
    """Calculate the mean and variance of lambda, given the prior on 
    A = 24.38 +- 0.44.
    
    Returns (mean, variance) as a tuple.
    """
    weight = calculate_prior_weights(mcmc)    
    mean = mcmc.calculate_mean(weight)
    cov = mcmc.calculate_cov(weight)
    
    return mean[2], cov[2,2]

mean1, var = calculate_lambda_with_prior(decay_mcmc)

print('With the prior on A, the inferred decay rate is {:.2f} +- {:.2f} / sec'.
      format(mean1, np.sqrt(var)))

def plot_corner_decay(mcmc):
    """Make a corner plot for the parameters of the decay model.
    Include contours at 68% and 95% confidence levels.
    """
    samples = mcmc.get_samples()
    matplotlib.rc('font', size=16)
    levels=(0.68, 0.95)
    fig = corner.corner(samples, labels=mcmc.names, levels=levels, bins=50)
    fig.set_size_inches(10,10)
    plt.show()
    
def plot_corner_decay_prior(mcmc):
    """Make a corner plot for the parameters of the decay model, this time
        with the prior on A.
    
    Include contours at 68% and 95% confidence levels.

    """
    weight = calculate_prior_weights(mcmc)    
    samples = mcmc.get_samples()
    matplotlib.rc('font', size=16)
    levels=(0.68, 0.95)
    fig = corner.corner(samples, labels=mcmc.names, levels=levels, bins=50,
                        weights=weight)
    fig.set_size_inches(10,10)
    plt.show()
    
plot_corner_decay(decay_mcmc)
plot_corner_decay_prior(decay_mcmc)


# Double check that everything worked by looking at the best fit model with
# the data after fitting with the prior
mean[2] = mean1
print(mean)
plot_data_with_model(data, mean)

