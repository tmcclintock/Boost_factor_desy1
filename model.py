import numpy as np
import cluster_toolkit as ct

#Get the boost factor model.
#Allow for seemless swapping between a power-law fit and an NFW fit.
def get_boost_model(params, args):
    name = args['model_name']
    Rb = args['Rb'] #Mpc physical
    if name == 'nfw':
        B0, Rs = params #Rs in Mpc physical
        return ct.boostfactors.boost_nfw_at_R(Rb, B0, Rs)
    elif name == 'pl': #power law
        B0, Rs, alpha = params #Rs in Mpc physical
        return ct.boostfactors.boost_powerlaw_at_R(Rb, B0, Rs, alpha)
