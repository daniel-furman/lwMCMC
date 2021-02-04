## `lwMCMC` Markov Chain Monte Carlo in Python

---

Parameter space sampling with lightweight Markov Chain Monte Carlo (MCMC) with the trivial metropolis hastings algorithmm. The operations are powered by NumPy. 


## Example 1: Simulation of Particle Decay

---

Decay model with <img src="https://render.githubusercontent.com/render/math?math=\lambda"> prior on the decay rate param | Posteriors for modeling parameters with <img src="https://render.githubusercontent.com/render/math?math=\sigma"> error contours 
:-------------------------------------------:|:------------------------------:
![](examples/data/scatter.png) | ![](examples/data/corners.png)

* Recovered parameter constraints for a particle decay simulation: <img src="https://render.githubusercontent.com/render/math?math=\R(t) = A + B e^{-\lambda t}">. We include a prior on the decay constant <img src="https://render.githubusercontent.com/render/math?math=\lambda">. In the corner plot (left), the diagonal entries reveal the 1-dimensional posterior distributions and the lower-left half shows the pairwise projections alongside the one and two sigma error contours. 

## Class Functions 

---

A lwMCMC object has the following functions:
        
* mcmc.step() takes a single step of the chain.

* mcmc.burn(nburn) runs the chain for nburn steps, but it doesn't save
            the values.

* mcmc.run(nsteps) runs the chain for nsteps steps, saving the results.

* mcmc.accept_fraction() returns what fraction of the candidate steps
            were taken.

* mcmc.get_samples() returns the sampled theta values in 2d numpy array.
* mcmc.plot_hist() plots a histogram of the sample values for each
            paramter.  As the chain runs for more steps, this should get
            smoother.
        
* mcmc.plot_samples() plots the sample values over the course of the 
            chain.  If the burn in is too short, it should be evident as a
            feature at the start of these plots.
    
* mcmc.calculate_mean() returns mean of all samples for each parameter.
* mcmc.calculate_cov() returns the covariance matrix of the paramters.
