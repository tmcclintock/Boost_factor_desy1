import numpy as np
from model import *

#Prior
def lnprior(params, args):
    B0, Rs = params[:2]
    if args['name'] == 'pl':
        alpha = params[3]

    #Scale radius can't be negative
    #Boost amplitude can't be negative
    #Scale radius can't be insanely large
    #Boost amplitude can't be insanely large
    if Rs <=0.0 or Rs > 100: return -np.inf
    if B0 < 0.00 or B0 > 100:return -np.inf
    #No issue
    #Replace this return statement if we get meaninful priors
    return 0

#Likelihood
def lnlike(params, args):
    Bp1 = args['Bp1'] #1 + boost
    iBcov = args['iBcov'] #C_{boost}^{-1}
    boost_model = get_boost_model(params, args)

    #Gaussian likelihood
    Xb = Bp1 - boost_model
    LLboost = -0.5*np.dot(Xb, np.dot(iBcov, Xb))
    return LLboost

#Posterior probability
def lnprob(params, args):
    lpr = lnprior(params, args)
    if not np.isfinite(lpr):
        return -1e99 #a big negative number
    return lpr + lnlike(params, args)


