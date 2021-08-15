
[![Build Status](https://travis-ci.com/daniel-furman/lwMCMC.svg?branch=main)](https://travis-ci.com/daniel-furman/lwMCMC)

## lwMCMC: lightweight Markov Chain Monte Carlo

---

`from lwMCMC import MCMC`

Parameter space sampling with MCMC. See Bayesian inference with the MCMC class below, for an Experimental Geophysics regression. 

Posterior distributions | MCMC model fit
:---------------------------------:|:----------------------------------------:
![](examples/data/grid_ice.png) | ![](examples/data/ice_scatter.png)


* The grid entries reveal the 1-dimensional posterior distributions of our parameters after setting our prior beliefs, as well as the pairwise projections with one and two sigma error contours. 

* With the slope parameters's 1.8 +- 0.225 prior, the Bayesian inferred slope is 1.70 +- 0.17.

---

### Package Layout

* [LICENSE](https://github.com/daniel-furman/lwMCMC/blob/main/LICENSE) - the MIT license, which applies to this package
* README.md - the README file, which you are now reading
* [requirements.txt](https://github.com/daniel-furman/lwMCMC/blob/main/requirements.txt) - prerequisites to install this package, used by pip
* [setup.py](https://github.com/daniel-furman/lwMCMC/blob/main/setup.py) - installer script
* [docs](https://github.com/daniel-furman/lwMCMC/tree/main/docs)/ - contains documentation on package installation and usage
* [examples](https://github.com/daniel-furman/lwMCMC/tree/main/examples)/ - use cases for Bayesian Modeling
* [lwMCMC](https://github.com/daniel-furman/lwMCMC/tree/main/lwMCMC)/ - the library code itself
* [tests](https://github.com/daniel-furman/lwMCMC/tree/main/test)/ - unit tests


