The motivation of MCMC is a need to understand a complex probability 
distribution, p(x). In applications to Bayesian statistics, this distr.
is usually a posterior from a Bayesian analysis. A good way to understand 
intractable distributions is to simulate realisations from it and study those.
However, this often isn’t easy, either. The idea behind MCMC is to simulate a
Markov chain whose equilibrium distribution is p(x). Metropolis-Hastings (M-H)
provides a way to correct a fairly arbitrary transition kernel <img src="https://render.githubusercontent.com/render/math?math=\q(x’|x)"> (which
typically won’t have p(x) as its equilibrium) to give a chain which does have
the desired target. In M-H, the transition kernel is used to generate a
proposed new value, x’ for the chain, which is then accepted as the new state
at random with a particular probability <img src="https://render.githubusercontent.com/render/math?math=\a(x’|x)=min(1,A)">, where <img src="https://render.githubusercontent.com/render/math?math=\A = p(x’)q(x|x’)/[p(x)q(x’|x)]">.

For more see:
https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/

    
