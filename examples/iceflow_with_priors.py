# Required Libraries:
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import pandas as pd #
import seaborn as sns
import corner
from scipy import stats

from lwMCMC import MCMC

table = pd.read_csv('data/iceflow_rates.csv', delimiter=',', header = 'infer')
table.columns = ['id', 'grain_radius', 'mean_pr', 'stress',
    'densification_rate']
table['rate_uncertainty'] = table['densification_rate']/3

"""
The file iceflow_rates.csv contains laboratory data of a ice densification and
associated flow rate (power law creep, to be precise). The columns are 'id',
'grain_radius', 'mean_pr','stress', 'densification_rate'. We are going to
sample for uncertainty from the 'rate_uncertainty'.

    id  grain_radius  mean_pr  stress  densification_rate  rate_uncertainty
    1      187       0.8097    0.7743     3.932e-09         dens_rate/3
    4        5       0.8310    0.7181     6.698e-08         dens_rate/3

    ...

Notice already how the finer grained sample is faster, even at lower applied
stress. This is the grain-size-sensitivity we were testing for.

We will model this data as a log-log linear model capturing the power law
between the densification_rate and stress. The parameter distribution of
interest is the power law "stress" exponent n, which in log-log space becomes
linear slope.

We have: densification_rate = intercept + slope * stress.

And, slope is the n stress exponent in the power law.

So, we have: densification_rate = exp(intercept) * stress^slope

We also have priors on n for all three of our regression experiments (from
solid ice creep experiments).

* n = 1.8 +/- 0.225 for the green and blue series, and
* n = 4.0 +/- 0.5 for the red series

We will use the MCMC class to sample the likelihood function and obtain the
mean and variances of the parameters.  Starting at the above guess.

Then use the above knowledge as gaussian priors in your MCMC to obtain the
Bayesian inferred posteriors (mean and variance) for the parameters.
"""

# log-log regression
y = np.array(table['densification_rate'][10:15])
X = np.array(table['stress'][10:15])
errs = np.array(table['rate_uncertainty'][10:15])
y = np.log(y)
X = np.log(X)
slope, intercept, r_value, p_value, std_err = stats.linregress(X,y)
print(stats.linregress(X,y))

y = np.array(table['densification_rate'][10:15])
X = np.array(table['stress'][10:15])
errs = np.array(table['rate_uncertainty'][10:15])
y_hat = np.exp(intercept) * X ** slope

def powerlaw_model(X, theta):
    """Returning the model values of R(stress) given theta = (intercept, slope).
    """
    intercept, slope = theta
    return np.exp(intercept) * X ** slope

def powerlaw_start():
    """Return a good starting point for the chain as a tuple (intercept, slope).
    """
    return (-16, 1.6)

def plot_data_with_model(data, theta):
    """Plot the data (with error bars) along with the given model, parametrized
    by theta.
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
    model of the ice flow.

    data is expected to be a tuple of numpy arrays = (x, y, sigma)
    theta is expected to be an array of parameters = (intercept, slope)
    """

    x, y, sigma = data
    n = len(x)
    model = powerlaw_model(x, theta)
    lnlike = -0.5 * (n*np.log(2.*np.pi) + np.sum(2.*np.log(errs) + (
        y-model)**2 / sigma**2))
    return lnlike

def powerlaw_step_size():
    """Return a good step size to use as a 2-element tuple giving the steps
    in each of (intercept, slope).
    """
    return (0.3, 0.08)

def powerlaw_nburn():
    """Return how many steps to use for the burn in.
    """
    return 1000

def powerlaw_nsteps():
    """Return how many steps to use in the chain
    """
    return 17000

# Initial guess:
start = powerlaw_start()
print('Log(like) at {:s} = {:.1f}'.format(str(start),
    powerlaw_loglike(data, start)))

# Make the object
powerlaw_mcmc = MCMC(powerlaw_loglike, data, start, powerlaw_step_size(),
    names=('$intercept$', '$slope$'), seed=42)

# Run the burn-in
powerlaw_mcmc.burnout(powerlaw_nburn())

# Run for the specified number of steps
powerlaw_mcmc.run_chain(powerlaw_nsteps())
print('After running for {} steps:'.format(powerlaw_nsteps()))
powerlaw_mcmc.sample_plotter()
print('Acceptance rate is ', powerlaw_mcmc.ratio_accepted())
assert 0.3 < powerlaw_mcmc.ratio_accepted() < 0.7
print('Mean value of params = ', powerlaw_mcmc.parameter_means())
print()

# Run for 2N more steps
powerlaw_mcmc.run_chain(2*powerlaw_nsteps())
print('After running for {} steps:'.format(3*powerlaw_nsteps()))
mean = powerlaw_mcmc.parameter_means()
print('Mean value of params = ', mean)
cov = powerlaw_mcmc.parameter_cov()
print('Uncertainties of params = ', np.sqrt(cov.diagonal()))

# Report the result
print('The inferred slope is {:.2f} +- {:.2f}'.format(mean[1],
    np.sqrt(cov[1,1])))

# Double check that everything worked by looking at the best fit model with
# the data.
plot_data_with_model(data, mean)

# And here are the one-d distributions, which should be fairly smooth.
powerlaw_mcmc.hist_plotter()
all_samples = powerlaw_mcmc.chain_samples()
for k in range(powerlaw_mcmc.nparams):
    theta_k = all_samples[:,k]
    sns.distplot(theta_k)
    plt.figure()

def calculate_prior_weights(mcmc):
    """Calculate appropriate weights for the mcmc samples, given the prior
    n = 1.8 +- 0.225.

    Returns the weights as a numpy array.
    """
    samples = mcmc.chain_samples()
    a_prior = 1.8
    sigma_prior = 0.225
    intercept, slope = samples.T
    weight = np.exp(-0.5 * (slope-a_prior)**2/sigma_prior**2)
    return weight

def calculate_slope_with_prior(mcmc):
    """Calculate the mean and variance of the slope, given the prior n = 1.8
    +- 0.225. Returns (mean, variance) as a tuple.
    """
    weight = calculate_prior_weights(mcmc)
    mean = mcmc.parameter_means(weight)
    cov = mcmc.parameter_cov(weight)
    return mean[1], cov[1,1]

mean1, var = calculate_slope_with_prior(powerlaw_mcmc)

print('With the prior on slope, the inferred slope is {:.2f} +- {:.2f}'.
      format(mean1, np.sqrt(var)))


def plot_corner_powerlaw(mcmc):
    """Make a corner plot for the parameters of the ice flow model. Includes
    contours at 68% and 95% confidence levels."""

    samples = mcmc.chain_samples()
    matplotlib.rc('font', size=16)
    levels=(0.68, 0.95)
    fig = corner.corner(samples, labels=mcmc.names, levels=levels, bins=50)
    fig.set_size_inches(10,10)
    plt.show()

def plot_corner_powerlaw_prior(mcmc):
    """Make a corner plot for the parameters of the ice flow model, with the
    prior weighting."""

    weight = calculate_prior_weights(mcmc)
    samples = mcmc.chain_samples()
    matplotlib.rc('font', size=16)
    levels=(0.68, 0.95)
    fig = corner.corner(samples, labels=mcmc.names, levels=levels, bins=50,
                        weights=weight)
    fig.set_size_inches(10,10)
    plt.show()

plot_corner_powerlaw(powerlaw_mcmc)
plot_corner_powerlaw_prior(powerlaw_mcmc)
