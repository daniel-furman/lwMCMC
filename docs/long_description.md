
## `lwMCMC` lightweight Markov Chain Monte Carlo

---

Parameter space sampling: lightweight Markov Chain Monte Carlo (MCMC) with the trivial metropolis hastings algorithm. Powered by NumPy. See examples at GitHub (https://github.com/daniel-furman/lwMCMC).

## `lwMCMC` Class Functions 

---

A lwMCMC object has the following class functions:

To run the chain:
        
* mcmc.step() takes a single step of the chain.

* mcmc.burn(nburn) runs the chain for nburn steps.

* mcmc.run(nsteps) runs the chain for nsteps steps, saving the results.

* mcmc.accept_fraction() returns what fraction of the candidate steps
            were taken.
            
Manage the results:

* mcmc.get_samples() returns the sampled param values in 2d numpy array.

* mcmc.plot_hist() plots a histogram of the sample values for each
            parameter.
   
* mcmc.plot_samples() plots the sample values over the course of the 
            chain.
            
* mcmc.calculate_mean() returns mean of all samples for each parameter.

* mcmc.calculate_cov() returns the covariance matrix of the parameters.
