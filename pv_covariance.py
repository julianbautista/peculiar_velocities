import numpy as np
import astropy
from astropy.table import Table
import pylab as plt

import scipy.integrate
import scipy.special
import scipy.stats

from cosmo import CosmoSimple
from numba import jit, prange

import time
import multiprocessing 

plt.ion()

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
def build_covariance_matrix(ra, dec, r_comov, k, pk):
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
            cov = get_covariance(ra_0, dec_0, r_comov_0, ra_1, dec_1, r_comov_1, k, pk)
            cov_matrix[i, j] = cov
            cov_matrix[j, i] = cov

    #-- For diagonal, window = 1/3
    var = np.trapz(pk/3, x=k)
    np.fill_diagonal(cov_matrix, var)

    #-- Pre-factor H0^2/(2pi^2)
    cov_matrix *= (100)**2/(2*np.pi**2) 

    return cov_matrix

def corr_from_cov(cov):
    diag = np.diag(cov)
    return cov/np.sqrt(np.outer(diag, diag))

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

def read_halos(cosmo=None, redshift_space=False, nhalos=None):

    #-- Read halo catalog
    #halos = Table.read('/datadec/cppm/bautista/DEMNUnii/surveys/survey_LCDM_062.fits')
    halos = Table.read('/Users/julian/Work/supernovae/peculiar/survey_LCDM_062.dat.fits')
    #print('Number of halos', len(halos))

    #-- cut to small sky region for testing
    #mask = (halos['ra']>0) & (halos['ra']<10.) & (halos['dec']>0) & (halos['dec']<10.)
    #halos = halos[mask]
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

    return ra, dec, r_comov, vel

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
def get_log_likes(vel, cov_cosmo, f_values, sig_values):
    ''' Computes 2d array containing the log likelihoood
        as a function f and sig
    '''
    diag_cosmo = np.diag(cov_cosmo)
    log_likes = np.zeros((f_values.size, sig_values.size))
    for i in prange(f_values.size):
        f_value = f_values[i]
        cov_matrix = cov_cosmo*f_value**2 
        for j in range(sig_values.size):
            sig_value = sig_values[j]
            #-- Total matrix
            #-- Add intrinsic dispersion to diagonal
            diag_total = diag_cosmo*f_value**2 + sig_value**2
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

    plt.figure()
    plt.contour(fs8_values, sig_values, likelihood.T, levels=like_contours,
                colors='k', linestyles='--')
    plt.pcolormesh(fs8_values, sig_values, likelihood.T, shading='nearest', cmap='gray_r')
    if not fs8_expected is None:
        plt.axvline(fs8_expected, ls='--')
    plt.colorbar()
    plt.xlabel(r'$f \sigma_8$', fontsize=12)
    plt.ylabel(r'$\sigma_v$ [km/s]', fontsize=12)


def main():

    cosmo = CosmoSimple(omega_m=0.32, h=0.67, mass_neutrinos=0.)
    sigma_8 = 0.84648 #-- DEMNUni value for m_nu = 0 (Castorina et al. 2015)

    nhalos = 1000
    kmax = 0.1
    nk = 512
    non_linear = True
    redshift_space = False
    n_values = 20
    f_expected = cosmo.get_growth_rate(0)
    fs8_expected = f_expected*sigma_8
    output_likelihood = f'like_{"non"*non_linear}linear_realspace_kmax{kmax:.1f}_{nhalos}halos.txt'
    
    print('Number of selected halos:', nhalos)
    print('kmax:', kmax)
    print('nk:', nk)
    print('Non-linear:', non_linear)
    print('Redshift-space:', redshift_space)
    print('Likelihood output file:', output_likelihood)
    print('Number of values in likelihood grid:', n_values)
    print('Expected value of f*sigma_8:', fs8_expected)

    #-- Read power spectrum model
    k, pk = read_power_spectrum(non_linear=non_linear, 
                                redshift_space=redshift_space, 
                                kmin=None, kmax=kmax, nk=nk)
    pk /= sigma_8**2

    #-- Read halo catalog and compute comoving distances
    ra, dec, r_comov, vel = read_halos(cosmo=cosmo, 
        redshift_space=redshift_space, 
        nhalos=nhalos)

    #-- Compute cosmological covariance matrix
    t0 = time.time()
    cov_cosmo = build_covariance_matrix(ra, dec, r_comov, k, pk)    
    t1 = time.time()
    print(f'Time elapsed calculating cov matrix {(t1-t0)/60:.2f} minutes')

    #-- Scan over fs8 values
    fs8_values = np.linspace(0.2, 0.5, n_values)
    sig_values = np.linspace(100, 300., n_values)
    log_likes = get_log_likes(vel, cov_cosmo, fs8_values, sig_values)
    t2 = time.time()
    print(f'Time elapsing with likelihood grid: {(t2-t1)/60:.2f} minutes or {(t2-t1)/n_values**2:.2f} sec per entry')

    likelihood = np.exp(log_likes-log_likes.max())

    fout = open(output_likelihood, 'w')
    for i in range(n_values):
        for j in range(n_values):
            print(fs8_values[i], sig_values[j], likelihood[i, j], file=fout)
    fout.close()


#if __name__ == '__main__':
#    main()

    