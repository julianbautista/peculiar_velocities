import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table 
from astropy.cosmology import FlatLambdaCDM
import pv_max_likelihood

cosmo = FlatLambdaCDM(H0=67.0, Om0=0.27, Tcmb0=2.725)

catalog = Table.read('/Users/julian/Work/DEMNUnii/surveys/ztf/LCDM_062_ztf_0000.dat.fits')
#w = catalog['dec'] > 0
#w &= (catalog['ra'] < 90)
#catalog = catalog[w]
np.random.seed(10)
w = np.random.rand(len(catalog)) < 10000/124820
catalog = catalog[w]

#-- Create a dictionary containing the relevant catalog information
#-- for the MaxLikelihood class
data = {}

#-- Angles in radians
data['ra'] = np.radians(catalog['ra'].data)
data['dec'] = np.radians(catalog['dec'].data)
#-- Comoving distance in Mpc/h
data['r_comov'] = cosmo.comoving_distance(catalog['redshift'].data).value * cosmo.h 
#-- Velocities
data['velocity'] = catalog['v_radial'].data 
data['velocity_error'] = np.zeros(len(catalog))
#-- Some required fields
data['size'] = len(catalog)
data['n_gals'] = np.ones(len(catalog))

#-- Put galaxies in a mesh
grid_size = 50. 
grid = pv_max_likelihood.grid_velocities(data, grid_size=grid_size)
print(f'{len(catalog)} were put in a mesh with {grid["size"]} cells')

#-- Initialise the model
#-- Note that grid_size is also an argument here
pk_file = 'pk_regpt_demnunii_tt.txt'
#pk_file = 'pk_lin_camb_demnunii.txt'
sigma_8 = 0.846
model = pv_max_likelihood.Model(pk_file, sigma_8=sigma_8, grid_size=grid_size)

#-- Initialise fitter 
print('Computing covariance matrix...')
fitter = pv_max_likelihood.MaxLikelihood(catalog=grid, model=model)
print('Done computing covariance matrix')

#-- Run minimizer and find best-fit parameters
fitter.fit_iminuit()

#-- Compute error bars
fitter.minos('fsigma_8')
fitter.print_minos('fsigma_8', decimals=3)
#fitter.minos('sigma_v')

fitter.get_contours('fsigma_8', 'sigma_v')
fitter.plot_contours('fsigma_8', 'sigma_v')

#-- Save results into file 
#-- To read it back, simply do 
#-- fitter = pv_max_likelihood.MaxLikelihood.load('pv_results.pkl')
fitter.save('pv_results.pkl')

