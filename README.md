## Lightweight Markov Chain Monte Carlo

---

Multi-dimensional model parameter space sampling with Markov Chain Monte Carlo (MCMC).
We implement am object-oriented implementation of MCMC based on the
trivial metropolis hastings algorithm.


## Example: Particle Decay Modeling with Priors

---

Recovered Parameter Constraints | Final Model with Prior
:-------------------------------------------:|:------------------------------:
![](data/corners.png) | ![alt-text-2](data/scatter.png "Final")

We visualize the recovered parameter constraints on a grid, for lwMCMC on a simulated particle decay model:
<img src="https://render.githubusercontent.com/render/math?math=\R(t) = A + B e^{-\lambda t}">, with 
decay constant <img src="https://render.githubusercontent.com/render/math?math=\lambda">. The diagonal shows the
1-dimensional posterior distribution results and the lower-left half shows the pairwise projections alongside the
one and two sigma error contours. 



## Available Class Functions 

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
