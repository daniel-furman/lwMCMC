
[![Build Status](https://travis-ci.com/daniel-furman/lwMCMC.svg?branch=main)](https://travis-ci.com/daniel-furman/lwMCMC)

## `lwMCMC` lightweight Markov Chain Monte Carlo

---

Parameter space sampling with lightweight MCMC powered by NumPy and Metropolis Hastings.

### Example 1: Experimental Geophysics Modeling with Laboratory Data

---

Posterior distributions with <img src="https://render.githubusercontent.com/render/math?math=\sigma"> contours | MCMC fit with a slope prior
:---------------------------------:|:----------------------------------------:
![](examples/data/grid_ice.png) | ![](examples/data/ice_scatter.png)

Recovered parameter constraints for a power law flow model for firn creep (ice compaction in nature). The grid entries reveal the 1-dimensional posterior distributions of our parameters, as well as the pairwise projections with one and two sigma modeling error contours. 

* With the slope parameters's 1.8 +- 0.225 prior, the Bayesian inferred slope is 1.70 +- 0.17.

## Example 2: Particle Decay Modeling with a Simulated Dataset

---

Posterior distributions with <img src="https://render.githubusercontent.com/render/math?math=\sigma"> contours | MCMC fit with a <img src="https://render.githubusercontent.com/render/math?math=\lambda"> prior
:---------------------------------:|:----------------------------------------:
![](examples/data/gridsims.png) | ![](examples/data/sims.png)


Recovered parameter constraints for a particle decay simulation: <img src="https://render.githubusercontent.com/render/math?math=\R(t) = A + B e^{-\lambda t}">. 

## `lwMCMC` Class Functions 

---

A lwMCMC object has class functions to perform Bayesian inference. 

* define the log(likelihood) function first (not in the class).

To run the chain:
        
* mcmc.stepchain_forward() takes a single step of the chain.

* mcmc.burnout(nburn) runs the chain for nburn steps.

* mcmc.run_forward(nsteps) runs the chain for nsteps steps, saving the results.

* mcmc.ratio_accepted() returns what fraction of the candidate steps
            were taken.
         
            
Manage the results:

* mcmc.clear_chain() clears the MCMC chain
* mcmc.chain_samples() returns the sampled param values in 2d numpy array.

* mcmc.hist_plotter() plots a histogram of the sample values for each
            parameter.
   
* mcmc.sample_plotter() plots the sample values over the course of the 
            chain.
            
* mcmc.calcmean() returns mean of all samples for each parameter.

* mcmc.calccov() returns the covariance matrix of the parameters.
