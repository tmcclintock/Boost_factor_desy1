import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
import matplotlib

extent = np.log10([0.0323, 30, 0.0323, 30])

for i in xrange(2,3):
    for j in xrange(6,7):
        Rb, Bp1, icov, cov = hf.get_boost_data_and_cov(i, j, alldata=False, diag_only=False)
        w,v = np.linalg.eig(cov)
        print w
        print np.max(w)/np.min(w)
        exit()

        
        D = np.diag(np.sqrt(cov.diagonal()))
        Di = np.linalg.inv(D)
        corr = np.dot(Di, np.dot(cov, Di))
        vmin, vmax = -1, 1
        fig, ax = plt.subplots()
        cmap = "seismic"
        im =ax.imshow(corr, aspect='equal', extent=extent, interpolation=None, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        cbax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
        cb =fig.colorbar(im, cax=cbax, cmap=cmap, orientation='vertical',norm=matplotlib.colors.LogNorm())

        ax.set_xlabel(r"$\log_{10} R$")
        ax.set_ylabel(r"$\log_{10} R$")
        ax.set_title("z%dl%d"%(i,j))
        
        fig.savefig("figures/corr_z%d_l%d.png"%(i,j), dpi=300)
        
        #plt.show()
        plt.clf()
