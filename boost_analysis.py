import numpy as np
from likelihoods import *
from helper_functions import *
import cluster_toolkit as ct
import os, sys

#Guesses for the model starts
def starts(name):
    if   name == 'nfw': return [0.5, 1.0]
    elif name == 'pl':  return [0.1, 1.0, -1.0]
    return None #Error

#Test call to the likelihood
def test_call(args):
    guess = starts(args['model_name'])
    print "Test call: lnprob(start) = %.e2\n"%lnprob(guess, args)
    return

def do_best_fit(args, bfpath):
    guess = starts(args['model_name'])
    import scipy.optimize as op
    nll = lambda *args: -lnprob(*args)
    print "Running best fit"
    result = op.minimize(nll, guess, args=(args,), method='Powell')
    print result
    print "Best fit saved at :\n\t%s"%bfpath
    print "\tSuccess = %s\n\t%s"%(result['success'],result['x'])
    #print result
    np.savetxt(bfpath, result['x'])
    return

def plot_bf(args, bfpath, show=False):
    import matplotlib.pyplot as plt
    import model as mod
    guess = np.loadtxt(bfpath)
    i, j= args['zi'], args['lj']
    Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(i, j, alldata=False)
    args['Rb'] = Rb
    boost = mod.get_boost_model(guess, args)
    plt.plot(Rb, boost-1, label="%s model"%args['model_name'])
    plt.errorbar(Rb, Bp1-1, np.sqrt(Bcov.diagonal()))
    plt.legend()
    plt.xscale('log')
    #plt.yscale('log')
    plt.title("z%d l%d"%(i,j))
    plt.gcf().savefig("figures/boost_%s_%s_z%d_l%d.png"%(args['name'], args['model_name'],i,j))
    if show:
        plt.show()
    plt.clf()

def do_mcmc(args, bfpath, chainpath, likespath):
    nwalkers, nsteps = 10, 1000
    import emcee
    bf = np.loadtxt(bfpath)
    ndim = len(bf)
    pos = [bf + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args,), threads=4)
    print "Starting MCMC, saving to \n\t%s"%chainpath
    sampler.run_mcmc(pos, nsteps)
    print "MCMC complete"
    np.save(chainpath, sampler.flatchain)
    np.save(likespath, sampler.flatlnprobability)
    return

def view_chain(zi,lj,chainpath,show=False):
    import matplotlib.pyplot as plt
    import corner
    chain = np.load(chainpath+".npy")
    fig = corner.corner(chain)
    fig.savefig("figures/corner_z%d_l%d.png"%(zi,lj))
    if show:
        plt.show()
    plt.clf()

if __name__ == "__main__":
    name = 'y1'
    model_name = "nfw"
    #Model name can be nfw or pl

    #Base pathnames for saving
    bfbase = "bestfits/bf_boost_%s_%s"%(name, model_name)
    chainbase = "chains/chain_boost_%s_%s"%(name, model_name)
    likesbase = "chains/likes_boost_%s_%s"%(name, model_name)
    
    Nz, Nl = 3, 7
    for i in xrange(1, 0, -1):
        for j in xrange(6, 5, -1):
            bfpath = bfbase+"_z%d_l%d.txt"%(i, j)
            chainpath = chainbase+"_z%d_l%d"%(i, j)
            likespath = likesbase+"_z%d_l%d"%(i, j)
            
            print "Working at z%d l%d"%(i, j)
            Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(i, j, alldata=False, diag_only=False)
            args = {'Rb':Rb, 'Bp1':Bp1, 'iBcov':iBcov, 'Bcov':Bcov, 'zi':i, 'lj':j, 'model_name':model_name, 'name':name}
            
            test_call(args)
            do_best_fit(args, bfpath)
            plot_bf(args, bfpath, show=False)

            #do_mcmc(args, bfpath, chainpath, likespath)
            #view_chain(i,j,chainpath, show=False)
