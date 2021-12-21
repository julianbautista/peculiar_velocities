import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table 
from astropy.cosmology import FlatLambdaCDM
import pv_max_likelihood

cosmo = FlatLambdaCDM(H0=67.0, Om0=0.32, Tcmb0=2.725)

catalog = Table.read('/Users/julian/Work/DEMNUnii/surveys/ztf/LCDM_062_ztf_0000.dat.fits')
#w = catalog['dec'] > 0
#w &= (catalog['ra'] < 90)
#catalog = catalog[w]
#np.random.seed(10)
#w = np.random.rand(len(catalog)) < 10000/124820
#catalog = catalog[w]


#-- Angles in radians
cat = pv_max_likelihood.Catalog(
        ra = np.radians(catalog['ra'].data),
        dec = np.radians(catalog['dec'].data),
        #-- Comoving distance in Mpc/h
        comoving_distance = cosmo.comoving_distance(catalog['redshift'].data).value * cosmo.h ,
        #-- Velocities
        velocity = catalog['v_radial'].data,
        velocity_error = None)


#-- Put galaxies in a mesh
mesh_cell_size = 50. 
mesh = cat.to_mesh(mesh_cell_size=mesh_cell_size)
print(f'{cat.size} were put in a mesh with {mesh.size} voxels')

#-- Initialise the model
#-- Note that grid_size is also an argument here
pk_file = 'pk_regpt_demnunii_LCDM_062.txt'
k, _, _, pk = np.loadtxt(pk_file, unpack=1)
#pk_file = 'pk_camb_linear_demnunii_LCDM_062.txt'
#k, pk = np.loadtxt(pk_file, unpack=1)
k = k*1 #-- need this for numba !
sigma_8 = 0.846
pk /= sigma_8**2

model = pv_max_likelihood.Model(k, pk)
model.set_mesh(mesh_cell_size=mesh_cell_size)

#-- Initialise fitter 
maxlik = pv_max_likelihood.MaxLikelihood(catalog=mesh, model=model)
print('Computing covariance matrix...')
maxlik.get_cosmological_covariance()
print('Done computing covariance matrix')

#-- Run minimizer and find best-fit parameters
maxlik.fit_iminuit()

#-- Compute error bars
maxlik.minos('fsigma_8')
maxlik.print_minos('fsigma_8', decimals=3)
#maxlik.minos('sigma_v')

maxlik.get_contours('fsigma_8', 'sigma_v')
maxlik.plot_contours('fsigma_8', 'sigma_v')

#-- Save results into file 
#-- To read it back, simply do 
#-- maxlik = pv_max_likelihood.MaxLikelihood.load('pv_results.pkl')
maxlik.save('pv_results.pkl')

