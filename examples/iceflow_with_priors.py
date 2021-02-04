#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:36:32 2021

@author: danielfurman
"""

# Required Libraries:

import numpy as np
import matplotlib.pylab as plt
import matplotlib
import pandas as pd #
import seaborn as sns
import corner
from scipy import stats

from lwMCMC.lwMCMC._main import lwMCMC as MCMC

table = pd.read_csv('data/iceflow_rates.csv', delimiter=',',
                          header = 'infer')
table.columns = ['id', 'grain_radius', 'mean_pr',
                       'stress', 'densification_rate']
table['rate_uncertainty'] = table['densification_rate']/3

"""

The file iceflow_rates.csv contains laboratory data of a ice densification and
    assoicated flow rate (power law creep, to be precise).  The
    columns are 'id', 'grain_radius', 'mean_pr','stress', 'densification_rate'.
    We are going to sample for uncertainty below 'rate_uncertainty'.

    id  grain_radius  mean_pr  stress  densification_rate  rate_uncertainty
     1      187       0.8097    0.7743     3.932e-09         dens_rate/3
     4        5       0.8310    0.7181     6.698e-08         dens_rate/3

     (rate_uncertainites were experimentally determined from rough upper
     limit of measurement error)
    ...

Notice already how the finer grained sample is faster, even at less stress.
This is the grain-size-sensitivity governing parts of our data.

We will model this data as a log-log linear model capturing the power law
between the densification_rate and stress. The parameter distribution of
interest is the power law "stress" exponent n, which in log-log space becomes
linear slope
    $$
    densification_rate = intercept + slope * stress
    $$
    slope is the n stress exponent in the power law:

    densification_rate = exp(intercept) * stress^slope

We also have priors on n for all three of our regression experiments (from
solid ice creep experiments)
n = 1.8 +/- 0.225 for the green and blue series, and
n = 4.0 +/- 0.5 for the red series

Use the MCMC class to sample the likelihood function and obtain the mean and
    variance of $\n$.  Starting at the above guess.

Then use the above knowledge as gaussian priors in your MCMC to obtain the new
    posterior(mean and variance) for $\n$.

I also intent to compare our MCMC class's trivial mh algorithm to at least
    one other - sgld perhaps.

Lastly, use these distributions as the priors on the final model, to sample
    the distribution of the semi empirical param in the flow law.

"""

# log-log linear regression of power law relationship for green series
y = np.array(table['densification_rate'][10:15])
X = np.array(table['stress'][10:15])
errs = np.array(table['rate_uncertainty'][10:15])
# log transforms
y = np.log(y)
X = np.log(X)
# least squares fitting
slope, intercept, r_value, p_value, std_err = stats.linregress(X,y)
#reg_conf = 1.96*std_err # 95 percent confidence interval
print(stats.linregress(X,y))

y = np.array(table['densification_rate'][10:15])
X = np.array(table['stress'][10:15])
errs = np.array(table['rate_uncertainty'][10:15])
y_hat = np.exp(intercept) * X ** slope

# Create figure in log log space
fig, ax1 = plt.subplots(1,1)
ax1.loglog(X, y_hat)
ax1.errorbar(X, y, yerr = errs, fmt='*', color='tab:orange', ecolor='grey',
                            elinewidth=1.35, capsize=1.5,
                            markersize=10) # some noisy data
# set plotting params
ax1.set_ylabel('$log$  $\.\epsilon$  (dp/pdt)')
ax1.set_xlabel('$log$ $\sigma$ (Mpa)')
ax1.set_title('Experimental Densification Rates, 233 K', fontweight = 'bold')
ax1.grid(axis = 'y')
#ax1.set_xlim([1e-1,10])
#ax1.set_ylim([1e-9,1e-6])

def powerlaw_model(X, theta):
    """Returning the model values of R(stress)
    given theta = (intercept, slope).
    """
    intercept, slope = theta
    return np.exp(intercept) * X ** slope

def powerlaw_start():
    """Return a good starting point for the chain as a tuple (intercept, slope)
    """
    return (-15, 1.5)

def plot_data_with_model(data, theta):
    """Plot the data (with error bars) along with the given model,
    parametrized by theta.
    """
    plt.figure()
    x, y, sigma = data
    plt.errorbar(x, y, sigma, fmt='o', ms=5, label='$lab$ $data$')
    model = powerlaw_model(x, theta)
    plt.plot(x, model, label = '$MCMC$ $model$')
    plt.xlabel('$stress$', labelpad=15)
    plt.ylabel('$dens$ $rate$ $R(stress)$', labelpad=15)
    plt.legend()
    #plt.savefig('data/mcmc-decay.png', dpi = 144)
    plt.show()

# Check that the initial guess is in the right ballpark.
data = np.array([table['stress'][10:15],
                 table['densification_rate'][10:15],
                 table['rate_uncertainty'][10:15]])
plot_data_with_model(data, powerlaw_start())

def powerlaw_loglike(data, theta):
    """Return the natural logarithm of the likelihood P(data | theta) for our
    model of the decay data.

    data is expected to be a tuple of numpy arrays = (x, y, sigma)
    theta is expected to be an array of parameters = (intercept, slope)
    """

    x, y, sigma = data
    n = len(x)
    model = powerlaw_model(x, theta)
    lnlike = -0.5 * (n*np.log(2.*np.pi) +
                     np.sum(2.*np.log(errs) + (y-model)**2 / sigma**2))
    return lnlike

def powerlaw_step_size():
    """Return a good step size to use as a 3-element tuple giving the steps
    in each of (A, B, lam).
    """
    return (0.3, 0.08)

def powerlaw_nburn():
    """Return how many steps to use for the burn in.
    """
    return 1000

def powerlaw_nsteps():
    """Return how many steps to use in the chain
    """
    return 17500

# Initial guess:
start = powerlaw_start()
print('Log(like) at {:s} = {:.1f}'.format(str(start),
                                          powerlaw_loglike(data, start)))

# Make the object
powerlaw_mcmc = MCMC(powerlaw_loglike, data, start, powerlaw_step_size(),
                     names=('$intercept$', '$slope$'), seed=21451)

# Run the burn-in
powerlaw_mcmc.burn(powerlaw_nburn())

# Run for the specified number of steps
powerlaw_mcmc.run(powerlaw_nsteps())
print('After running for {} steps:'.format(powerlaw_nsteps()))
powerlaw_mcmc.plot_samples()
print('Acceptance rate is ', powerlaw_mcmc.accept_fraction())
assert 0.3 < powerlaw_mcmc.accept_fraction() < 0.7
    #Adjust step sizes if this fails.

print('Mean value of params = ', powerlaw_mcmc.calculate_mean())
print()

# Run for 2N more steps
powerlaw_mcmc.run(2*powerlaw_nsteps())
print('After running for {} steps:'.format(3*powerlaw_nsteps()))
mean = powerlaw_mcmc.calculate_mean()
print('Mean value of params = ', mean)

cov = powerlaw_mcmc.calculate_cov()
print('Uncertainties of params = ', np.sqrt(cov.diagonal()))

# Report the result
print()
print('The inferred slope is {:.2f} +- {:.2f}'.format(mean[1],
                                                        np.sqrt(cov[1,1])))

# Double check that everything worked by looking at the best fit model with
# the data.
plot_data_with_model(data, mean)

# And here are the one-d distributions, which should be fairly smooth.
powerlaw_mcmc.plot_hist()
all_samples = powerlaw_mcmc.get_samples()
for k in range(powerlaw_mcmc.nparams):
    theta_k = all_samples[:,k]
    sns.distplot(theta_k)
    plt.figure()

def calculate_prior_weights(mcmc):
    """Calculate appropriate weights for the mcmc samples, given the prior
    n = 1.8 +- 0.225.

    Returns the weights as a numpy array.
    """
    samples = mcmc.get_samples()
    a_prior = 1.8
    sigma_prior = 0.225
    intercept, slope = samples.T
    weight = np.exp(-0.5 * (slope-a_prior)**2/sigma_prior**2)
    return weight

def calculate_slope_with_prior(mcmc):
    """Calculate the mean and variance of lambda, given the prior on
    n = 1.8 +- 0.225.

    Returns (mean, variance) as a tuple.
    """
    weight = calculate_prior_weights(mcmc)
    mean = mcmc.calculate_mean(weight)
    cov = mcmc.calculate_cov(weight)

    return mean[1], cov[1,1]

mean1, var = calculate_slope_with_prior(powerlaw_mcmc)

print('With the prior on A, the inferred decay rate is {:.2f} +- {:.2f}'.
      format(mean1, np.sqrt(var)))


def plot_corner_powerlaw(mcmc):
    """Make a corner plot for the parameters of the decay model.
    Include contours at 68% and 95% confidence levels."""

    samples = mcmc.get_samples()
    matplotlib.rc('font', size=16)
    levels=(0.68, 0.95)
    fig = corner.corner(samples, labels=mcmc.names, levels=levels, bins=50)
    fig.set_size_inches(10,10)
    plt.show()

def plot_corner_powerlaw_prior(mcmc):
    """Make a corner plot for the parameters of the decay model, this time
        with the prior on A."""


    weight = calculate_prior_weights(mcmc)
    samples = mcmc.get_samples()
    matplotlib.rc('font', size=16)
    levels=(0.68, 0.95)
    fig = corner.corner(samples, labels=mcmc.names, levels=levels, bins=50,
                        weights=weight)
    fig.set_size_inches(10,10)
    plt.show()

plot_corner_powerlaw(powerlaw_mcmc)
plot_corner_powerlaw_prior(powerlaw_mcmc)

# Double check that everything worked by looking at the best fit model with
# the data after fitting the prior on decay rate.

plot_data_with_model(data, np.array([-16.01358366, mean1]))

# log-log linear regression of power law relationship for green series
y = np.array(table['densification_rate'][10:15])
X = np.array(table['stress'][10:15])
x = np.array([1e-1,5])
errs = np.array(table['rate_uncertainty'][10:15])
intercept = -16.01358366
slope = mean1
y_hat = np.exp(intercept) * x ** slope

# Create figure in log log space
fig, ax1 = plt.subplots(1,1)
ax1.loglog(x, y_hat, label = 'MCMC $\.\epsilon$ model', color='tab:orange')
ax1.errorbar(X, y, yerr = errs, fmt='.', color='tab:blue', ecolor='grey',
                            elinewidth=1.35, capsize=1.5, label = 'Laboratory rates',
                            markersize=10) # some noisy data
# set plotting params
ax1.set_ylabel('$Flow$ $rate$ $(log$ $\.\epsilon)$', labelpad = 15)
ax1.set_xlabel('$Applied$ $stress$ $(log$ $\sigma)$', labelpad = 15)
#ax1.set_title('Experimental Densification Rates, 233 K', fontweight = 'bold')
#ax1.grid(axis = 'y')
ax1.set_xlim([1e-1,5])
ax1.set_ylim([5e-9,1e-6])
ax1.legend(loc = 'lower right')
