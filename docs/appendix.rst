
    The motivation of MCMC is a need to understand a complex probability 
    distribution, p(x). In applications to Bayesian statistics, this distr.
    is usually a posterior from a Bayesian analysis. A good way to understand 
    intractable distributions is to simulate realisations from it and study those.
    However, this often isn’t easy, either. The idea behind MCMC is to simulate a
    Markov chain whose equilibrium distribution is p(x). Metropolis-Hastings (M-H)
    provides a way to correct a fairly arbitrary transition kernel q(x’|x) (which
    typically won’t have p(x) as its equilibrium) to give a chain which does have
    the desired target. In M-H, the transition kernel is used to generate a
    proposed new value, x’ for the chain, which is then accepted as the new state
    at random with a particular probability a(x’|x)=min(1,A), where 
    A = p(x’)q(x|x’)/[p(x)q(x’|x)].

    For more see:
        https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/

    Step Function:
        
    We use log(likelihood) rather than likelihood here.
        

    
