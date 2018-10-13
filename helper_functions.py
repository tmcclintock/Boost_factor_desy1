import numpy as np

y1boostbase    = "data/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost.dat"
y1boostcovbase = "data/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost_cov.dat"

def get_boost_data_and_cov(zi, lj, lowcut=0.2, highcut=999, alldata=False, diag_only=False):
    """
    Args:
        zi: z-index
        lj: lambda-index
        lowcut: lower radii cut in Mpc physical 
        highcut: upper radii cut in Mpc physical
        alldata: return everything without scale cuts
        diag_only: return a diagonal covariance matrix
    """
    boostpath = y1boostbase%(lj, zi)
    bcovpath  = y1boostcovbase%(lj, zi)
    Bcov = np.loadtxt(bcovpath) #Covariance 
    Rb, Bp1, Be = np.genfromtxt(boostpath, unpack=True) #Radii, boost+1, error

    #Cut out bad data, always make this cut
    indices = np.fabs(Be) > 1e-6 #errors go to 0 if the data is bad
    Bp1 = Bp1[indices]
    Rb  = Rb[indices]
    Be  = Be[indices]
    Bcov = Bcov[indices]
    Bcov = Bcov[:,indices]

    #If we want all of the data, return now
    #This is usually used for plotting
    if alldata:
        return Rb, Bp1, np.linalg.inv(Bcov), Bcov

    #Make scale cuts
    indices = (Rb > lowcut)*(Rb < highcut)
    Bp1 = Bp1[indices]
    Rb  = Rb[indices]
    Be  = Be[indices]
    Bcov = Bcov[indices]
    Bcov = Bcov[:,indices]
    Njk = 100.
    D = len(Rb)
    
    #Make the Hartlap correction
    Bcov = Bcov*((Njk-1.)/(Njk-D-2)) #Hartlap correction

    #If we only want the diagonals of the covariance.
    if diag_only:
        print("USING DIAGONALS ONLY!!")
        Bcov = np.diag(Bcov.diagonal())

    #Return radii, boost+1, Bcov_inv, and Bcov
    return Rb, Bp1, np.linalg.inv(Bcov), Bcov
