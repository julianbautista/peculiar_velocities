import numpy as np
import time, pickle
import matplotlib.pyplot as plt
plt.ion()

from numba import jit, prange

import iminuit

class Catalog:
    
    def __init__(self, ra=None, dec=None, comoving_distance=None, 
                 velocity=None, velocity_error=None, 
                 n_galaxies=None, mesh_cell_size=0):

        if not ra is None and (ra.min() < -2*np.pi or ra.max()>2*np.pi):
            print('RA has to be in radians and between -2pi and 2pi')
        if not dec is None and (dec.min() < -np.pi/2 or dec.max()>np.pi/2):
            print('DEC has to be in radians and between -pi/2 and pi/2')

        if not ra is None:
            size = ra.size
            x = comoving_distance * np.cos(ra) * np.cos(dec)
            y = comoving_distance * np.sin(ra) * np.cos(dec)
            z = comoving_distance * np.sin(dec)
        else:
            size = None
            x = None 
            y = None 
            z = None

        if n_galaxies is None and ra is not None:
            n_galaxies = np.ones(size, dtype=int)

        if velocity_error is None and not velocity is None:
            velocity_error = np.zeros(size)
        
        self.ra = ra
        self.dec = dec
        self.comoving_distance = comoving_distance
        self.velocity = velocity
        self.velocity_error = velocity_error
        self.n_galaxies = n_galaxies
        self.size = size 
        self.x = x
        self.y = y 
        self.z = z 
        self.mesh_cell_size = mesh_cell_size

    def to_mesh(self, mesh_cell_size=20.):
        ''' Assigns a galaxy catalog or a voxel to a voxel catalog,
            where voxels have mesh_cell_size in same units
            as the comoving distances.
        '''
        if mesh_cell_size<self.mesh_cell_size:
            print(f'Error: mesh_cell_size={mesh_cell_size} has'
                  f'to be larger than {self.mesh_cell_size}')
            return

        n_galaxies = self.n_galaxies 
        vel_value = self.velocity 
        vel_error = self.velocity_error

        position = np.array([self.x, self.y, self.z])
        pos_min = np.min(position, axis=1)
        pos_max = np.max(position, axis=1)
        #- Number of grid voxels per axis
        n_grid = np.floor((pos_max-pos_min)/mesh_cell_size).astype(int)+1
        #-- Total number of voxels
        n_pix = n_grid.prod()
        
        #-- Voxel index per axis
        index = np.floor( (position.T - pos_min)/mesh_cell_size ).astype(int)
        #-- Voxel index over total number of voxels
        i = (index[:, 0]*n_grid[1] + index[:, 1])*n_grid[2] + index[:, 2]

        #-- Perform averages per voxel
        sum_vel_value = np.bincount(i, weights=vel_value*n_galaxies,    minlength=n_pix)
        sum_vel_error = np.bincount(i, weights=vel_error**2*n_galaxies, minlength=n_pix)
        sum_n_galaxies = np.bincount(i, weights=n_galaxies, minlength=n_pix)

        #-- Consider only voxels with at least one galaxy
        w = sum_n_galaxies > 0
        center_vel_value = sum_vel_value[w]/sum_n_galaxies[w]
        center_vel_error = np.sqrt(sum_vel_error[w])/sum_n_galaxies[w]
        center_n_galaxies = sum_n_galaxies[w]

        #-- Determine the coordinates of the voxel centers
        i_pix = np.arange(n_pix)[w]
        i_pix_z = i_pix % n_grid[2]
        i_pix_y = ((i_pix - i_pix_z)/n_grid[2]) % n_grid[1]
        i_pix_x = i_pix // (n_grid[1]*n_grid[2])
        i_pix = [i_pix_x, i_pix_y, i_pix_z]
        center_position = np.array([(i_pix[i]+0.5)*mesh_cell_size + pos_min[i] for i in range(3)])
        
        #-- Convert to ra, dec, r_comov
        center_comoving_distance = np.sqrt(np.sum(center_position**2, axis=0))
        center_ra = np.arctan2(center_position[1], center_position[0])
        center_dec = np.pi/2 - np.arccos(center_position[2]/center_comoving_distance)

        return Catalog( ra = center_ra, 
                        dec = center_dec, 
                        comoving_distance = center_comoving_distance, 
                        velocity = center_vel_value, 
                        velocity_error = center_vel_error,
                        n_galaxies = center_n_galaxies,
                        mesh_cell_size = mesh_cell_size)

    def cut(self, w):

        return Catalog( ra = self.ra[w],
                        dec = self.dec[w],
                        comoving_distance = self.comoving_distance[w],
                        velocity = self.velocity[w],
                        velocity_error = self.velocity_error[w],
                        n_galaxies = self.n_galaxies[w],
                        mesh_cell_size = self.mesh_cell_size)


@jit(nopython=True )
def angle_between(ra_0, dec_0, ra_1, dec_1):
    cos_alpha = (np.cos(ra_1-ra_0)*np.cos(dec_0)*np.cos(dec_1) 
                 + np.sin(dec_0)*np.sin(dec_1))
    return cos_alpha

@jit(nopython=True )
def j0(x):
   return np.sin(x)/x

@jit(nopython=True )
def j2(x):
    return (3/x**2-1)*np.sin(x)/x  - 3*np.cos(x)/x**2
    
@jit(nopython=True )
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

@jit(nopython=True )
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

@jit(nopython=True, parallel=True )
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

    def __init__(self, k, pk):
        
        self.k = k
        self.pk = pk
        self.pk_variance = np.trapz(pk, x=k)

        self.mesh_window_function = None 
        self.pk_mesh = None 
        self.pk_variance_mesh = None 

    def set_mesh(self, mesh_cell_size=20):
        
        k = self.k
        pk = self.pk
        pk_mesh = np.copy(pk)
        pk_variance_mesh = self.pk_variance*1

        mesh_window_function = self.get_mesh_window_function(mesh_cell_size)

        pk_mesh = pk*mesh_window_function**2
        pk_variance_mesh = np.trapz(pk_mesh, x=k)        

        self.mesh_window_function = mesh_window_function 
        self.pk_mesh = pk_mesh
        self.pk_variance_mesh = pk_variance_mesh

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

    def get_mesh_window_function(self, mesh_cell_size, n=100):
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
            fact = (ki*mesh_cell_size)/2/np.pi
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
        mesh_window_function = self.mesh_window_function
        plt.figure()
        plt.plot(k, pk * k**scale_k, label=r'$P(k)$')
        if not mesh_window_function is None:
            plt.plot(k, pk*mesh_window_function**2*k**scale_k, label=r'$P(k)W^2(k)$')
        plt.xlabel('k')
        ylabel = f'$k^{{{scale_k}}}P(k)$'
        plt.ylabel(ylabel)
        plt.legend()

    def check_integrals(self, n=1000):

        k = self.k 
        pk = self.pk
        mesh_window_function = self.mesh_window_function
        has_mesh = not mesh_window_function is None

        kmaxes = np.logspace(-1, 1, n)
        variances = np.zeros(n)
        variances_mesh = np.zeros(n)
        for i in range(n):
            kmax = kmaxes[i]
            w = k < kmax 
            variances[i] = np.trapz(pk[w], x=k[w])
            if has_mesh:
                variances_mesh[i] = np.trapz(pk[w]*mesh_window_function[w]**2, x=k[w])
        
        plt.plot(kmaxes, 1-variances/self.pk_variance, label=r'$P(k)$')
        if has_mesh:
            plt.plot(kmaxes, 1-variances_mesh/self.pk_variance_mesh, label=r'$P(k)W^2(k)$')
        plt.xlabel(r'$k_{\rm max}$', fontsize=14)
        plt.ylabel(r'$1-\sigma^2(k_{\rm max})/\sigma^2(\infty)$', fontsize=14)
        plt.legend()

class MaxLikelihood:

    def __init__(self, catalog=None, model=None): 
        self.catalog = catalog
        self.model = model
        self.cosmo_cova = None 

        #-- Names of fields to be saved 
        self.mig_fields = ['best_pars', 
                           'ndata', 'max_loglike', 'npar', 'ndof',
                           'contours']
        self.minos_fields = ['number', 'value', 'error', 'merror', 
                             'lower_limit', 'upper_limit', 'is_fixed']
        self.output = None

    def get_cosmological_covariance(self):

        cat = self.catalog 
        model = self.model 

        cosmo_cova_matrix = build_covariance_matrix(
                                cat.ra, cat.dec, cat.comoving_distance, 
                                model.k, model.pk_mesh
                            )
        
        #-- Account for grid in variance  
        #-- Eq. 22 of Howlett et al. 2017
        #-- Factor of 1/3 due to one component of velocity
        #-- or simply the window function at zero separation
        pk_variance = model.pk_variance/3 
        pk_variance_mesh = model.pk_variance_mesh/3
        np.fill_diagonal(cosmo_cova_matrix, 
            pk_variance_mesh + (pk_variance-pk_variance_mesh)/cat.n_galaxies)

        #-- Pre-factor (100 h)^2/(2pi^2)
        cosmo_cova_matrix *= (100)**2/(2*np.pi**2) 
        self.cosmo_cova = cosmo_cova_matrix

    def fit_iminuit(self):
        ''' Runs the iMinuit minimiser 
        '''
        cosmo_cova = self.cosmo_cova

        cat = self.catalog  
        velocity = cat.velocity
        velocity_error = cat.velocity_error
        n_galaxies = cat.n_galaxies

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

        def get_log_like(fsigma_8, sigma_v):
            diag_cosmo_cova = np.diag(cosmo_cova)
            cov_matrix = cosmo_cova*fsigma_8**2 
            diag_total = diag_cosmo_cova*fsigma_8**2 + sigma_v**2/n_galaxies**2 + velocity_error**2
            np.fill_diagonal(cov_matrix, diag_total)
            log_like = log_likelihood(velocity, cov_matrix)
            return -log_like
        
        t0 = time.time()
        mig = iminuit.Minuit(get_log_like, fsigma_8=0.5, sigma_v=200.)
        mig.errordef = iminuit.Minuit.LIKELIHOOD
        mig.limits['fsigma_8'] = (0., 2.)
        mig.limits['sigma_v'] = (0., 100000)
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
        if contour_xy.size == 0:
            print('Error in computing contours.')
            return 

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
            for field in self.minos_fields:
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
