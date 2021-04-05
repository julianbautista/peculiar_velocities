from operator import sub
import numpy as np
import pylab as plt
import os
import time

from astropy.table import Table

import scipy.integrate
import scipy.special
import scipy.stats

from numba import jit, prange

import iminuit

from cosmo import CosmoSimple

plt.ion()

def read_power_spectrum(non_linear=False, redshift_space=False, 
                        kmin=None, kmax=None, nk=None):

    #-- Read power spectrum from camb 
    #-- units are in h/Mpc and Mpc^3/h^3
    pk_table = Table.read('pk_lin_camb_demnunii.txt', format='ascii',names=('k', 'power'))
    k = pk_table['k'] 
    pk = pk_table['power'] 

    #-- apply non-linearities from Bel et al. 2019
    if non_linear: 
        sig8 = 0.8
        a1 = -0.817+3.198*sig8
        a2 = 0.877 - 4.191*sig8
        a3 = -1.199 + 4.629*sig8
        pk = pk*np.exp(-k*(a1+a2*k+a3*k**2))

    #-- Go to redshift-space (only if using redshift_obs)
    #-- based on Koda et al. 2014
    if redshift_space:
        sigma_u = 13. #- Mpc/h
        D_u = np.sin(k*sigma_u)/(k*sigma_u)
        pk *= D_u**2

    #-- reduce resolution of pk for speed
    k, pk = reduce_resolution(k, pk, kmin=kmin, kmax=kmax, nk=nk, linear=False)

    return k, pk

def reduce_resolution(k, pk, kmin=None, kmax=None, nk=None, linear=False):

    if kmin is None:
        kmin = k.min()
    if kmax is None:
        kmax = k.max()
    if nk is None:
        nk = k.size
    if linear:
        knew = np.linspace(kmin, kmax, nk)
    else:
        knew = 10**np.linspace(np.log10(kmin), np.log10(kmax), nk)
    pknew = np.interp(knew, k, pk)
    return knew, pknew

def read_halos(input_catalog, 
               cosmo=None, redshift_space=False, nhalos=None, zmax=None, 
               density_subsample=False, subsample_fraction=1.):

    #-- Read halo catalog
    halos = Table.read(input_catalog)
    
    #-- cut to small sky region for testing
    mask = np.ones(len(halos), dtype=bool)
    #mask = (halos['ra']<180) & (halos['ra']>0.  ) & (halos['dec']>0) & (halos['dec']<70.)
    #mask = (halos['ra']<360) & (halos['ra']>180.) & (halos['dec']>0) & (halos['dec']<70.)
    #mask = (halos['ra']<180) & (halos['ra']>0.  ) & (halos['dec']<0) & (halos['dec']>-70.)
    #mask = (halos['ra']<360) & (halos['ra']>180.) & (halos['dec']<0) & (halos['dec']>-70.)
    f_sky = np.sum(mask)/len(halos)

    if not zmax is None:
        mask &= (halos['redshift'] < zmax)
    halos = halos[mask]
    
    if density_subsample:
        #-- Downsampling to match 2MTF catalogs from Howlett et al. 2017
        #nz = density_z(halos['redshift'], f_sky, cosmo, nbins=30)
        #nz_gal = np.interp(halos['redshift'], nz['z_centers'], nz['density'])
        #prob_to_keep = 10**( (-4+2)/(0.03-0.002)*(halos['redshift']-0.002))#/nz_gal
        np.random.seed(10)
        r = np.random.rand(len(halos))
        mask = r <= subsample_fraction
        halos = halos[mask]

    if not nhalos is None:
        halos = halos[:nhalos]
    halos['ra'] = np.radians(halos['ra'])
    halos['dec'] = np.radians(halos['dec'])
    
    if redshift_space:
        z = halos['redshift_obs']
    else:
        z = halos['redshift']
    z = z.data.astype(float)

    #-- Compute comoving distances in Mpc/h units to match power-spectrum units
    r_comov = cosmo.get_comoving_distance(z)*cosmo.pars['h']

    ra = halos['ra'].data.astype(float)
    dec = halos['dec'].data.astype(float)
    vel = halos['v_radial'].data.astype(float)

    catalog = {'ra': ra,
               'dec': dec,
               'r_comov': r_comov,
               'vel': vel, 
               'redshift': z,
               'weight': np.ones(ra.size),
               'f_sky': f_sky}
    return catalog

def add_intrinsic_scatter(catalog, sigma_m=0.1, cosmo=None, seed=0):
    ''' Convert error in distance modulus into error in velocity 
        Draw Gaussian random errors for velocities
    '''
    z = catalog['redshift']
    sigma_v = cosmo.pars['c']*np.log(10)/5
    sigma_v /= (1 - cosmo.pars['c']*(1+z)**2/cosmo.get_hubble(z)/cosmo.get_DL(z)) 
    sigma_v *= -1*sigma_m
    np.random.seed(seed)
    vel_error = np.random.randn(z.size)*sigma_v
    catalog['vel'] += vel_error
    catalog['sigma_v'] = sigma_v

def grid_velocities(catalog, grid_size=20.):
    ''' Transform a galaxy catalog into a voxel catalog,
        where voxels have grid_size in Mpc/h 
    '''
    x = catalog['r_comov']*np.cos(catalog['ra'])*np.cos(catalog['dec'])
    y = catalog['r_comov']*np.sin(catalog['ra'])*np.cos(catalog['dec'])
    z = catalog['r_comov']*np.sin(catalog['dec'])

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
    sum_vel  = np.bincount(i, weights=catalog['vel']   *catalog['weight'], minlength=n_pix)
    sum_vel2 = np.bincount(i, weights=catalog['vel']**2*catalog['weight'], minlength=n_pix)
    sum_we   = np.bincount(i, weights=catalog['weight'], minlength=n_pix)
    sum_n    = np.bincount(i, minlength=n_pix)
    #-- Consider only voxels with at least one galaxy
    w = sum_we > 0
    center_vel = sum_vel[w]/sum_we[w]
    center_vel_error = np.sqrt(sum_vel2[w]/sum_we[w] - center_vel**2)/np.sqrt(sum_n[w])
    center_weight = sum_we[w]
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
            'vel': center_vel, 
            'vel_err': center_vel_error,
            'weight': center_weight,
            'ngals': center_ngals}

def density_z(z, f_sky, cosmo, zmin=None, zmax=None, nbins=50):

    if zmin is None:
        zmin = z.min()
    if zmax is None:
        zmax = z.max()
    bin_edges = np.linspace(zmin, zmax, nbins)
    counts, bin_edges = np.histogram(z, bins=bin_edges)

    r = cosmo.get_comoving_distance(bin_edges)*cosmo.pars['h']
    r3_diff = r[1:]**3 - r[:-1]**3
    vol_shell = f_sky * 4*np.pi/3 * r3_diff
    bin_centers = (bin_edges[1:]+bin_edges[:-1])*0.5
    density = counts / vol_shell
    density_err = np.sqrt(counts) / vol_shell
    volume = np.sum(vol_shell)

    return {'z_centers': bin_centers,
            'z_edges': bin_edges,
            'density': density,
            'density_err': density_err,
            'volume': volume}

@jit(nopython=True)
def angle_between(ra_0, dec_0, ra_1, dec_1):
    cos_alpha = np.cos(ra_1-ra_0)*np.cos(dec_0)*np.cos(dec_1) + np.sin(dec_0)*np.sin(dec_1)
    return cos_alpha

@jit(nopython=True)
def separation(r_0, r_1, cos_alpha):
    return np.sqrt(r_0**2 + r_1**2 - 2*r_0*r_1*cos_alpha)

def j0(x):
    ''' This doesn't work with numba '''
    return scipy.special.spherical_jn(0, x)

@jit(nopython=True)
def j0_alt(x):
   return np.sin(x)/x
    
def j2(x):
    ''' This doesn't work with numba '''
    return scipy.special.spherical_jn(2, x)

@jit(nopython=True)
def j2_alt(x):
    return (3/x**2-1)*np.sin(x)/x  - 3*np.cos(x)/x**2
    
@jit(nopython=True)
def window(k, r_0, r_1, cos_alpha):
    r = separation(r_0, r_1, cos_alpha)
    sin_alpha_squared = 1-cos_alpha**2
    win = 1/3*np.ones_like(k)
    if r > 0:
        j0kr = j0_alt(k*r) 
        j2kr = j2_alt(k*r)
        win = 1/3*(j0kr - 2*j2kr)*cos_alpha
        win = win+(r_0*r_1/r**2*sin_alpha_squared * j2kr)
    return win

def test_window():
    ''' Reproduce Fig 4 of Johnson et al. 2014 
        even though his labels are switched 
        between (a and e) and (b and d)
    '''
    k = 10**np.linspace(-4, 0, 1000)
    plt.figure()
    plt.plot(k, window(k, 86.6,  133.7, np.cos(0.393)), label='Wa', color='C1', ls='--')
    plt.plot(k, window(k, 76.8,  127.6, np.cos(1.313)), label='Wb', color='k', ls='--')
    plt.plot(k, window(k, 59.16, 142.5, np.cos(0.356)), label='Wc', color='b')
    plt.plot(k, window(k, 51.9,   91.1, np.cos(0.315)), label='Wd', color='C3', ls='--')
    plt.plot(k, window(k, 99.49, 158.4, np.cos(0.463)), label='We', color='C2')
    plt.legend()
    plt.xlim(5e-4, 2e-1)
    plt.xscale('log')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel(r'$W_{i,j}(k)$')

#@jit(nopython=True, parallel=True)
def grid_window(k, L, n=100):

    window = np.zeros_like(k)
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2*np.pi, n)
    kx = np.outer(np.sin(theta), np.cos(phi))
    ky = np.outer(np.sin(theta), np.sin(phi))
    kz = np.outer(np.cos(theta), np.ones(n))
    dthetaphi = np.outer(np.sin(theta), np.ones(phi.size))
    for i in prange(k.size):
        ki = k[i]
        #-- the factor here has an extra np.pi because of the definition of np.sinc
        fact = (ki*L)/2/np.pi
        func = np.sinc(fact*kx)*np.sinc(fact*ky)*np.sinc(fact*kz)*dthetaphi
        win_theta = np.trapz(func, x=phi, axis=1)
        win = np.trapz(win_theta, x=theta)
        win *= 1/(4*np.pi)
        window[i] = win
    return window

@jit(nopython=True)
def get_covariance(ra_0, dec_0, r_comov_0, ra_1, dec_1, r_comov_1, k, pk):
    ''' Get cosmological covariance for a given pair of galaxies 
        and a given power spectrum (k, pk) in units of h/Mpc and (Mpc/h)^3
    '''
    cos_alpha = angle_between(ra_0, dec_0, ra_1, dec_1)
    win = window(k, r_comov_0, r_comov_1, cos_alpha)
    cova = np.trapz(pk * win, x=k)
    return cova

@jit(nopython=True, parallel=True)
def build_covariance_matrix(ra, dec, r_comov, k, pk_nogrid, grid_win=None, ngals=None):
    ''' Builds a 2d array with the theoretical covariance matrix 
        based on the positions of galaxies (ra, dec, r_comov) 
        and a given power spectrum (k, pk)
    '''
    nh = ra.size
    cov_matrix = np.zeros((nh, nh))
    if not grid_win is None:
        pk = pk_nogrid*grid_win**2
    else:
        pk = pk_nogrid*1

    for i in prange(nh):
        ra_0 = ra[i]
        dec_0 = dec[i]
        r_comov_0 = r_comov[i]
        for j in range(i+1, nh):
            ra_1 = ra[j]
            dec_1 = dec[j]
            r_comov_1 = r_comov[j]
            cov = get_covariance(ra_0, dec_0, r_comov_0, ra_1, dec_1, r_comov_1, k, pk)
            cov_matrix[i, j] = cov
            cov_matrix[j, i] = cov

    #-- For diagonal, window = 1/3
    var = np.trapz(pk/3, x=k)
    np.fill_diagonal(cov_matrix, var)

    if not grid_win is None:
        var_nogrid = np.trapz(pk_nogrid/3, x=k)
        #-- Eq. 22 of Howlett et al. 2017
        np.fill_diagonal(cov_matrix, var + (var_nogrid-var)/ngals)

    #-- Pre-factor H0^2/(2pi^2)
    cov_matrix *= (100)**2/(2*np.pi**2) 

    return cov_matrix

def corr_from_cov(cov):
    diag = np.diag(cov)
    return cov/np.sqrt(np.outer(diag, diag))

def check_integrals():

    k, pk = read_power_spectrum(nk=100000)
    scales = [1., 3., 10., 30., 100.]

    plt.figure()
    for i_scale, scale in enumerate(scales):
        win_radial = window(k, 200., 200+scale, 1.)
        win_transv = window(k, 200., 200., np.cos(scale/200.))
        full_int_radial = np.trapz(win_radial*pk, x=k)
        full_int_transv = np.trapz(win_transv*pk, x=k)
        nmax = 1000
        i_k = np.arange(k.size//2, k.size, k.size//2//nmax)
        int_radial = np.zeros(i_k.size)
        int_transv = np.zeros(i_k.size)
        kmax = k[i_k]
        for j, i in enumerate(i_k):
            int_radial[j] = np.trapz(win_radial[:i]*pk[:i], x=k[:i])
            int_transv[j] = np.trapz(win_transv[:i]*pk[:i], x=k[:i])
        plt.plot(kmax, int_radial/full_int_radial-1, color=f'C{i_scale}', ls='--', label=f'dr = {scale}')
        plt.plot(kmax, int_transv/full_int_transv-1, color=f'C{i_scale}', ls=':')

    plt.legend()
    plt.xlabel(r'$k_{\rm max}$ [h/Mpc]')
    plt.ylabel(r'Relative error on $C_{ij}$')
    plt.ylim(-1, 1)
    plt.xscale('log')

@jit(nopython=True)
def log_likelihood(x, cova):
    ''' Computes log of the likelihood from 
        a vector x and a covariance cova
    '''
    nx = x.size
    eigvals = np.linalg.eigvalsh(cova)
    inv_matrix = np.linalg.inv(cova)
    chi2 = x.T @ inv_matrix @ x
    log_like = -0.5*(nx*np.log(2*np.pi) 
                      + np.sum(np.log(eigvals))
                      + chi2)
    return log_like



@jit(nopython=True, parallel=False)
def get_log_likes(vel, cov_cosmo, fs8_values, sig_values):
    ''' Computes 2d array containing the log likelihoood
        as a function f and sig
    '''
    diag_cosmo = np.diag(cov_cosmo)
    log_likes = np.zeros((fs8_values.size, sig_values.size))
    for i in prange(fs8_values.size):
        fs8 = fs8_values[i]
        cov_matrix = cov_cosmo*fs8**2 
        for j in range(sig_values.size):
            sig_v = sig_values[j]
            #-- Total matrix
            #-- Add intrinsic dispersion to diagonal
            diag_total = diag_cosmo*fs8**2 + sig_v**2
            np.fill_diagonal(cov_matrix, diag_total)
            #-- Compute likelihood
            log_likes[i, j] = log_likelihood(vel, cov_matrix)
    return log_likes

def read_likelihood(fin):
    fs8_values, sig_values, like = np.loadtxt(fin, unpack=1)
    fs8_values = np.unique(fs8_values)
    sig_values = np.unique(sig_values)
    like = np.reshape(like, (fs8_values.size, sig_values.size))
    return fs8_values, sig_values, like

def plot_likelihood(fin, fs8_expected=None):

    fs8_values, sig_values, likelihood = read_likelihood(fin)

    #-- Cumulative distribution at 1, 2, 3 sigma for one degree of freedom
    cdf = scipy.stats.chi2.cdf([9, 4, 1], 1)
    #-- corresponding chi2 values for 2 degrees of freedom
    chi2_values = scipy.stats.chi2.ppf(cdf, 2)
    #-- corresponding likelihood values
    like_contours = np.exp(-0.5*chi2_values)

    plt.figure(figsize=(5,4))
    plt.contour(fs8_values, sig_values, likelihood.T, levels=like_contours,
                colors='k', linestyles='--')
    plt.pcolormesh(fs8_values, sig_values, likelihood.T, shading='nearest', cmap='gray_r')
    if not fs8_expected is None:
        plt.axvline(fs8_expected, ls='--')
    plt.colorbar()
    plt.xlabel(r'$f \sigma_8$', fontsize=12)
    plt.ylabel(r'$\sigma_v$ [km/s]', fontsize=12)
    plt.tight_layout()

def fit_iminuit(vel, cov_cosmo):

    @jit(nopython=True, parallel=False)
    def get_log_like(fs8, sig_v):
        diag_cosmo = np.diag(cov_cosmo)
        cov_matrix = cov_cosmo*fs8**2 
        diag_total = diag_cosmo*fs8**2 + sig_v**2
        np.fill_diagonal(cov_matrix, diag_total)
        log_like = log_likelihood(vel, cov_matrix)
        return -log_like
    
    t0 = time.time()
    m = iminuit.Minuit(get_log_like, fs8=0.5, sig_v=200.)
    m.errordef = iminuit.Minuit.LIKELIHOOD
    m.limits['fs8'] = (0.1, 2.)
    m.limits['sig_v'] = (0., 1000)
    mig = m.migrad()
    minos = m.minos()
    t1 = time.time()
    print(f'iMinuit fit lasted: {(t1-t0)/60:.2f} minutes')
    return mig, m


def header_line(mig):
    npars = len(mig.parameters)
    line = '# name fval nfcn nfit valid '
    for pars in mig.parameters:
        line += f'{pars}_value {pars}_error {pars}_lower {pars}_lower_valid {pars}_upper {pars}_upper_valid '
    for i in range(npars):  
        pars1 = mig.parameters[i]
        for j in range(i+1, npars):
            pars2 = mig.parameters[j]
            line += f'cova_{pars1}_{pars2} '
    return line

def fit_to_line(mig, name):
    npars = len(mig.parameters)
    #-- Values
    line = name 
    line += f'  {mig.fval}  {mig.nfcn}  {mig.nfit}  {mig.valid*1}  '
    for pars in mig.parameters:
        line += f'{mig.values[pars]}  {mig.errors[pars]}  '
        minos = mig.merrors[pars]
        line += f'{minos.lower}  {minos.lower_valid*1}  {minos.upper}  {minos.upper_valid*1}  '
    for i in range(npars):
        pars1 = mig.parameters[i]
        for j in range(i+1, npars):
            pars2 = mig.parameters[j]
            line += f'{mig.covariance[pars1, pars2]}  '
    return line

def export_fit(output_fit, mig, name):

    fout = open(output_fit, 'w')
    #-- Header
    line = header_line(mig)
    print(line, file=fout)
    #-- Values
    line = fit_to_line(mig, name)
    print(line, file=fout)
    fout.close()

def main(name='test',
    input_catalog=None,
    nhalos = None,
    zmax = None,
    kmax = None,
    nk = 512,
    non_linear = False,
    redshift_space = False,
    grid_size = 0, 
    density_subsample=False,
    subsample_fraction=1.,
    fit=True, export_fit=False,
    scan=False, 
    n_values_scan = 20):

    t00 = time.time()

    cosmo = CosmoSimple(omega_m=0.32, h=0.67, mass_neutrinos=0.)
    sigma_8 = 0.84648 #-- DEMNUni value for m_nu = 0 (Castorina et al. 2015)
    f_expected = cosmo.get_growth_rate(0)
    fs8_expected = f_expected*sigma_8

    print('Name of run:', name)
    print('Input catalog:', input_catalog)
    print('Number of selected halos:', nhalos)
    print('kmax:', kmax)
    print('nk:', nk)
    print('Non-linear:', non_linear)
    print('Redshift-space:', redshift_space)
    print('Expected value of f*sigma_8:', fs8_expected)
    
    if fit and export_fit:
        output_fit = f'howlett/{name}_fit_{"non"*non_linear}linear_'
        output_fit += f'{"redshift"*redshift_space + "real"*~redshift_space}space_'
        output_fit += f'kmax{kmax:.2f}_{nhalos}halos_grid{grid_size}_'
        output_fit += f'zmax{zmax:.2f}.txt'
        print('Fit output file:', output_fit)
    if scan:
        output_likelihood = output_fit.replace('fit_', 'likegrid_')
        print('Likelihood output file:', output_likelihood)
        print('Number of values for scanning likelihood in a grid:', n_values_scan)
    
    #-- Read power spectrum model
    k, pk = read_power_spectrum(non_linear=non_linear, 
                                redshift_space=redshift_space, 
                                kmin=None, kmax=kmax, nk=nk)
    pk /= sigma_8**2

    #-- Read halo catalog and compute comoving distances
    catalog = read_halos(input_catalog,
        cosmo=cosmo, 
        redshift_space=redshift_space, 
        nhalos=nhalos, zmax=zmax,
        density_subsample=density_subsample,
        subsample_fraction=subsample_fraction)
    print('Final number of galaxies in catalog:', len(catalog['ra']))
    print(f'Radial velocity dispersion: {np.std(catalog["vel"]):.1f} km/s')

    #-- Grid halos and velocities
    if grid_size>0:
        print('Grid size :', grid_size)
        grid = grid_velocities(catalog, grid_size=grid_size)
        final_catalog = grid
        ngals = final_catalog['ngals']
        #-- Quick checks
        print('Number of grid cells with data: ', grid['ra'].size)
        print('Histogram of galaxies per cell:')
        unique_ngals, counts_ngals = np.unique(ngals, return_counts=True)
        print(unique_ngals)
        print(counts_ngals)
        print(f'Radial velocity dispersion (grid): {np.std(catalog["vel"]):.1f} km/s')

        #-- Compute grid window function
        grid_win = grid_window(k, grid_size)
    else:
        final_catalog = catalog
        grid_win = None
        ngals = None

    ra = final_catalog['ra']
    dec = final_catalog['dec']
    r_comov = final_catalog['r_comov']
    vel = final_catalog['vel']

    #-- Compute cosmological covariance matrix
    t0 = time.time()
    cov_cosmo = build_covariance_matrix(ra, dec, r_comov, k, pk, grid_win=grid_win, ngals=ngals)    
    t1 = time.time()
    print(f'Time elapsed calculating cov matrix {(t1-t0)/60:.2f} minutes')
    
    print('First five elements of cov_cosmo:')
    print(cov_cosmo[:5, :5])

    if fit:
        mig, m = fit_iminuit(vel, cov_cosmo)
        print(mig)
        if export_fit:
            export_fit(output_fit, mig, name)
        else:
            header = header_line(mig)
            line = fit_to_line(mig, name)

    #-- Scan over fs8 and sig_v values
    if scan:
        fs8_values = np.linspace(0., 1., n_values_scan)
        sig_values = np.linspace(50., 250., n_values_scan)
        print('Scanning values of fsigma8 and sigma_vel')
        log_likes = get_log_likes(vel, cov_cosmo, fs8_values, sig_values)
        t2 = time.time()
        print(f'Time elapsing with likelihood grid: {(t2-t1)/60:.2f} minutes'+
              f' or {(t2-t1)/n_values_scan**2:.2f} sec per entry')

        likelihood = np.exp(log_likes-log_likes.max())

        fout = open(output_likelihood, 'w')
        for i in range(n_values_scan):
            for j in range(n_values_scan):
                print(fs8_values[i], sig_values[j], likelihood[i, j], file=fout)
        fout.close()

    tt = time.time()
    print(f'Total time elapsed: {(tt-t00)/60:.2f} minutes')

    return line, header

#if __name__ == '__main__':
#    main()

    
