import numpy as np
import time, pickle
import matplotlib.pyplot as plt
plt.ion()

from numba import jit, prange

import iminuit

def grid_velocities(catalog, grid_size=20.):
    ''' Transform a galaxy catalog into a voxel catalog,
        where voxels have grid_size in Mpc/h 
    '''
    if grid_size==0:
        return catalog

    x = catalog['r_comov']*np.cos(catalog['ra'])*np.cos(catalog['dec'])
    y = catalog['r_comov']*np.sin(catalog['ra'])*np.cos(catalog['dec'])
    z = catalog['r_comov']*np.sin(catalog['dec'])

    n_gals = np.ones(x.size) if 'n_gals' not in catalog else catalog['n_gals']

    position = np.array([x, y, z])
    pos_min = np.min(position, axis=1)
    pos_max = np.max(position, axis=1)
    #- Number of grid voxels per axis
    n_grid = np.floor((pos_max-pos_min)/grid_size).astype(int)+1
    #-- Total number of voxels
    n_pix = n_grid.prod()
    
    #-- Voxel index per axis
    index = np.floor( (position.T - pos_min)/grid_size ).astype(int)
    #-- Voxel index over total number of voxels
    i = (index[:, 0]*n_grid[1] + index[:, 1])*n_grid[2] + index[:, 2]

    #-- Perform averages per voxel
    sum_vel  = np.bincount(i, weights=catalog['velocity']*n_gals, minlength=n_pix)
    sum_vel_error2 = np.bincount(i, weights=catalog['velocity_error']**2*n_gals, minlength=n_pix)
    sum_n    = np.bincount(i, weights=n_gals, minlength=n_pix)

    #-- Consider only voxels with at least one galaxy
    w = sum_n > 0
    center_vel = sum_vel[w]/sum_n[w]
    center_vel_error = np.sqrt(sum_vel_error2[w])/sum_n[w]
    center_ngals = sum_n[w]

    #-- Determine the coordinates of the voxel centers
    i_pix = np.arange(n_pix)[w]
    i_pix_z = i_pix % n_grid[2]
    i_pix_y = ((i_pix - i_pix_z)/n_grid[2]) % n_grid[1]
    i_pix_x = i_pix // (n_grid[1]*n_grid[2])
    i_pix = [i_pix_x, i_pix_y, i_pix_z]
    center_position = np.array([(i_pix[i]+0.5)*grid_size + pos_min[i] for i in range(3)])
    
    #-- Convert to ra, dec, r_comov
    center_r_comov = np.sqrt(np.sum(center_position**2, axis=0))
    center_ra = np.arctan2(center_position[1], center_position[0])
    center_dec = np.pi/2 - np.arccos(center_position[2]/center_r_comov)

    return {'ra': center_ra, 
            'dec': center_dec, 
            'r_comov': center_r_comov, 
            'velocity': center_vel, 
            'velocity_error': center_vel_error,
            'n_gals': center_ngals,
            'size': center_ra.size}

@jit(nopython=True)
def angle_between(ra_0, dec_0, ra_1, dec_1):
    cos_alpha = (np.cos(ra_1-ra_0)*np.cos(dec_0)*np.cos(dec_1) 
                 + np.sin(dec_0)*np.sin(dec_1))
    return cos_alpha

@jit(nopython=True)
def j0(x):
   return np.sin(x)/x

@jit(nopython=True)
def j2(x):
    return (3/x**2-1)*np.sin(x)/x  - 3*np.cos(x)/x**2
    
@jit(nopython=True)
def window(k, r_0, r_1, cos_alpha):
    ''' Window function corresponding to a pair of galaxies 
        at distances r_0 and r_1 and separated by an angle alpha
        Eq. 5 from Johnson et al. 2014 
    '''
    r = np.sqrt(r_0**2 + r_1**2 - 2*r_0*r_1*cos_alpha)
    sin_alpha_squared = 1-cos_alpha**2
    win = 1/3*np.ones_like(k)
    if r > 0:
        j0kr = j0(k*r) 
        j2kr = j2(k*r)
        win = 1/3*(j0kr - 2*j2kr)*cos_alpha
        win = win+(r_0*r_1/r**2*sin_alpha_squared * j2kr)
    return win

@jit(nopython=True)
def get_pair_covariance(ra_0, dec_0, r_comov_0, 
                        ra_1, dec_1, r_comov_1, 
                        k, pk):
    ''' Get cosmological covariance for one given pair of galaxies 
        and a given power spectrum (k, pk) in units of h/Mpc and (Mpc/h)^3
    '''
    cos_alpha = angle_between(ra_0, dec_0, ra_1, dec_1)
    win = window(k, r_comov_0, r_comov_1, cos_alpha)
    cova = np.trapz(pk * win, x=k)
    return cova

@jit(nopython=True, parallel=True)
def build_covariance_matrix(ra, dec, r_comov, 
                            k, pk):
    ''' Builds a 2d array with the theoretical covariance matrix 
        based on the positions of galaxies (ra, dec, r_comov) 
        and a given power spectrum (k, pk)
    '''
    nh = ra.size
    cov_matrix = np.zeros((nh, nh))

    for i in prange(nh):
        ra_0 = ra[i]
        dec_0 = dec[i]
        r_comov_0 = r_comov[i]
        for j in range(i+1, nh):
            ra_1 = ra[j]
            dec_1 = dec[j]
            r_comov_1 = r_comov[j]
            cov = get_pair_covariance(ra_0, dec_0, r_comov_0, 
                                      ra_1, dec_1, r_comov_1, 
                                      k, pk)
            cov_matrix[i, j] = cov
            cov_matrix[j, i] = cov

    return cov_matrix

class Model:

    def __init__(self, pk_file, sigma_8=0.846, grid_size=0.):
        k, pk = np.loadtxt(pk_file, unpack=1)
        pk /= sigma_8**2
        self.k = k*1
        self.pk = pk
        self.pk_variance = np.trapz(pk, x=k)

        grid_window = None 
        pk_grid = np.copy(self.pk)
        pk_variance_grid = self.pk_variance*1
        if grid_size > 0:
            grid_window = self.get_grid_window(grid_size)
            pk_grid = pk*grid_window**2
            pk_variance_grid = np.trapz(pk_grid, x=k)        
        self.grid_window = grid_window 
        self.pk_grid = pk_grid
        self.pk_variance_grid = pk_variance_grid

    def add_rsd_bel(self, sigma8=0.84648):
        ''' Isotropic non-linear correction to velocity power spectrum
            in real space. Empirical formulas fitted to n-body simulations 
            by Bel et al. 2019
        '''
        a1 = -0.817+3.198*sigma8
        a2 = 0.877 - 4.191*sigma8
        a3 = -1.199 + 4.629*sigma8
        k = self.k 
        self.pk *= np.exp(-k*(a1+a2*k+a3*k**2))

    def add_rsd_koda(self, sigma_u=13.):
        ''' Add redshift-space term (only if using redshift_obs)
            based on fits to n-body simulations by 
            Koda et al. 2014
        '''
        k = self.k 
        D_u = np.sin(k*sigma_u)/(k*sigma_u)
        self.pk *= D_u**2

    def get_grid_window(self, grid_size, n=100):
        ''' Computes the grid window function from a given cell size
            Eq. 29 of Johnson et al. 2014 
        '''
        k = self.k
        window = np.zeros_like(k)
        theta = np.linspace(0, np.pi, n)
        phi = np.linspace(0, 2*np.pi, n)
        kx = np.outer(np.sin(theta), np.cos(phi))
        ky = np.outer(np.sin(theta), np.sin(phi))
        kz = np.outer(np.cos(theta), np.ones(n))
        dthetaphi = np.outer(np.sin(theta), np.ones(phi.size))
        #-- Can we do this without for loop ? 
        for i in range(k.size):
            ki = k[i]
            #-- the factor here has an extra np.pi 
            #-- because of the definition of np.sinc
            fact = (ki*grid_size)/2/np.pi
            func = np.sinc(fact*kx)*np.sinc(fact*ky)*np.sinc(fact*kz)*dthetaphi
            win_theta = np.trapz(func, x=phi, axis=1)
            win = np.trapz(win_theta, x=theta)
            #win *= 1/(4*np.pi)
            window[i] = win
        window *= 1/4/np.pi 
        return window 

    def plot(self, scale_k=1):

        k = self.k 
        pk = self.pk 
        grid_window = self.grid_window
        plt.figure()
        plt.plot(k, pk * k**scale_k)
        if not grid_window is None:
            plt.plot(k, pk*grid_window**2*k**scale_k)
        plt.xlabel('k')
        ylabel = f'$k^{{scale_k}}P(k)'
        plt.ylabel(ylabel)

    def check_integrals(self, n=10):

        k = self.k 
        pk = self.pk
        grid_window = self.grid_window
        kmaxes = np.logspace(-1, 1, n)
        variances = np.zeros(n)
        variances_grid = np.zeros(n)
        for i in range(n):
            kmax = kmaxes[i]
            w = k < kmax 
            variances[i] = np.trapz(pk[w], x=k[w])
            variances_grid[i] = np.trapz(pk[w]*grid_window[w]**2, x=k[w])
        
        plt.plot(kmaxes, 1-variances/self.pk_variance)
        plt.plot(kmaxes, 1-variances_grid/self.pk_variance_grid)
        plt.xlabel(r'$k_{\rm max}$', fontsize=14)
        plt.ylabel(r'$1-\sigma^2(k_{\rm max})/\sigma^2(\infty)$', fontsize=14)

        #return kmaxes, variances, variances_grid

    def get_cosmological_covariance(self, catalog):

        k = self.k 
        pk_grid = self.pk_grid 

        ra = catalog['ra']
        dec = catalog['dec']
        r_comov = catalog['r_comov']
        n_gals = catalog['n_gals']

        cosmo_cov_matrix = build_covariance_matrix(ra, dec, r_comov, k, pk_grid)
        
        #-- Account for grid in variance  
        #-- Eq. 22 of Howlett et al. 2017
        #-- Factor of 1/3 due to one component of velocity
        #-- or simply the window function at zero separation
        pk_variance = self.pk_variance/3 
        pk_variance_grid = self.pk_variance_grid/3
        np.fill_diagonal(cosmo_cov_matrix, 
            pk_variance_grid + (pk_variance-pk_variance_grid)/n_gals)

        #-- Pre-factor H0^2/(2pi^2)
        cosmo_cov_matrix *= (100)**2/(2*np.pi**2) 
        return cosmo_cov_matrix

def log_likelihood(x, cova):
    ''' Computes log of the likelihood from 
        a vector x and a covariance cova
    '''
    nx = x.size
    eigvals = np.linalg.eigvalsh(cova)
    #inv_matrix = np.linalg.inv(cova)
    #chi2 = x.T @ inv_matrix @ x
    chi2 = x.T @ np.linalg.solve(cova, x)
    log_like = -0.5*(nx*np.log(2*np.pi) 
                    + np.sum(np.log(eigvals))
                    + chi2)
    return log_like

class MaxLikelihood:

    def __init__(self, catalog=None, model=None): 
        self.catalog = catalog
        self.model = model

        if not catalog is None and not model is None:
            self.cosmo_cova = model.get_cosmological_covariance(catalog)
        
        #-- Names of fields to be saved 
        self.mig_fields = ['best_pars', 
                           'ndata', 'max_loglike', 'npar', 'ndof',
                           'contours']
        self.param_fields = ['number', 'value', 'error', 'merror', 
                             'lower_limit', 'upper_limit', 'is_fixed']
        self.output = None

    def fit_iminuit(self):
        ''' Runs the iMinuit minimiser 
        '''
        catalog = self.catalog 
        cosmo_cova = self.cosmo_cova 

        velocity = catalog['velocity']
        velocity_error = catalog['velocity_error']
        n_gals = catalog['n_gals']

        def get_log_like(fsigma_8, sigma_v):
            diag_cosmo_cova = np.diag(cosmo_cova)
            cov_matrix = cosmo_cova*fsigma_8**2 
            diag_total = diag_cosmo_cova*fsigma_8**2 + sigma_v**2/n_gals**2 + velocity_error**2
            np.fill_diagonal(cov_matrix, diag_total)
            log_like = log_likelihood(velocity, cov_matrix)
            return -log_like
        
        t0 = time.time()
        mig = iminuit.Minuit(get_log_like, fsigma_8=0.5, sigma_v=200.)
        mig.errordef = iminuit.Minuit.LIKELIHOOD
        mig.limits['fsigma_8'] = (0., 2.)
        mig.limits['sigma_v'] = (0., 3000)
        mig.migrad()
        t1 = time.time()
        print(f'iMinuit fit lasted: {(t1-t0)/60:.2f} minutes')

        #-- Recover best-fit parameters and model 
        best_pars = {k: mig.params[k].value for k in mig.parameters}
        self.best_pars = best_pars
        self.mig = mig 
        self.max_loglike = -mig.fval
        self.ndata = velocity.size
        self.npar = mig.nfit
        self.ndof = self.ndata - self.npar
        self.mig = mig

    def minos(self, parameter_name):
        self.mig.minos(parameter_name)

    def print_minos(self, parameter_name, symmetrise=False, decimals=None):
        ''' Prints the output of minos for a given parameter
            
            If symmetrise=True, it will show a single error value
                that corresponds to the average of the lower and upper
                errors
            Decimals (int) is the number of decimals to be shown for the 
                result.
        '''
        if self.output is None:
            self.get_output_from_minuit()

        par_details = self.output['best_pars_details'][parameter_name]
        value = par_details['value']
        error_low, error_upp = par_details['merror']
        error = (error_upp - error_low)/2

        if not decimals is None:
            value = f'{value:.{decimals}f}'
            error_low = f'{-error_low:.{decimals}f}'
            error_upp = f'{error_upp:.{decimals}f}'
            error = f'{error:.{decimals}f}'

        if symmetrise:     
            print(f'{parameter_name}: {value} +/- {error}')
        else:
            print(f'{parameter_name}: {value} + {error_upp} - {error_low}')

    def get_contours(self, parameter_name_1, parameter_name_2, 
                     confidence_level=0.685, n_points=30):
        ''' Computes confidence contours for two parameters
            given a confidence level and a number of points 
            defining the contour. 
            It is basically a wrapper of iMinuit.mncontour()
        '''
        if self.output is None:
            self.get_output_from_minuit()
        output = self.output 

        contour_xy = self.mig.mncontour(parameter_name_1, parameter_name_2, 
                                    cl=confidence_level, size=n_points)
        
        if not 'contours' in output:
            output['contours'] = {}
        key = (parameter_name_1, parameter_name_2)
        if not key in output['contours']:
            output['contours'][key] = {}
        output['contours'][key][confidence_level] = contour_xy

    def plot_contours(self, parameter_name_1, parameter_name_2):
        ''' Plot 2D contours for two given parameters
        '''
        if self.output is None or not 'contours' in self.output:
            print('Error: Need to compute contours first.')
            return

        contours = self.output['contours'][parameter_name_1, parameter_name_2]
        confidence_levels = np.sort(list(contours.keys()))

        plt.figure()      
        for confidence_level in confidence_levels[::-1]:
            contour = contours[confidence_level]
            plt.fill(contour[:, 0], contour[:, 1], alpha=0.3, color='C0', 
                     label=f'{confidence_level}')
        plt.xlabel(parameter_name_1)
        plt.ylabel(parameter_name_2)
        #plt.legend()

    def get_output_from_minuit(self):
        ''' Converts outputs from iMinuit to a simple dictionary,
            which is used to save the results. 
        '''
        output = {}
        
        for field in self.mig_fields:
            try:
                output[field] = self.__getattribute__(field)
            except:
                pass

        details = {}
        for parameter in self.mig.params:
            details[parameter.name] = {}
            for field in self.param_fields:
                details[parameter.name][field] = parameter.__getattribute__(field)

        output['best_pars_details'] = details
        self.output = output
    
    def save(self, filename):
        ''' Saves the output into a file
        '''
        if self.output is None:
            self.get_output_from_minuit()
        pickle.dump(self.output, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        ''' Reads a pickle file and stores the content 
            in the output dictionary
        '''
        output = pickle.load(open(filename, 'rb'))
        maxlik = MaxLikelihood()
        #-- fill chi with output 
        #-- todo
        for field in maxlik.mig_fields:
            if field in output:
                maxlik.__setattr__(field, output[field])
        
        maxlik.output = output
        return maxlik
