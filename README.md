A python class for parameter search with MCMC chains using Metropolis Hastings.

With the MCMC object, run the following functions:
        
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
