import numpy as np
import astropy
from astropy.table import Table
import pylab as plt
import scipy.integrate
import scipy.special
from cosmo import CosmoSimple
from numba import jit
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
    return scipy.special.spherical_jn(0, x)

@jit(nopython=True)
def j0_alt(x):
   return np.sin(x)/x
    
def j2(x):
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
def covariance(k, pk, win):
    return np.trapz(pk * win, x=k)



@jit(nopython=True)
def get_covariance_matrix(ra, dec, r_comov, k, pk, h):
    
    variance = covariance(k, pk, 1/3)

    nh = len(ra)
    cov_matrix = np.zeros((nh, nh))
    for i in range(nh):
        cov_matrix[i, i] = variance
        for j in range(i+1, nh):
            cos_alpha = angle_between(ra[i], dec[i], ra[j], dec[j])
            win = window(k, r_comov[i], r_comov[j], cos_alpha)
            cov = covariance(k, pk, win)
            cov_matrix[i, j] = cov
            cov_matrix[j, i] = cov

    cov_matrix *= (h*100)**2/(2*np.pi**2) 
    return cov_matrix  

@jit(nopython=True)
def get_covariance_parallel(index):
    i, j = index 
    cos_alpha = angle_between(ra[i], dec[i], ra[j], dec[j])
    win = window(k, r_comov[i], r_comov[j], cos_alpha)
    cova = covariance(k, pk, win)
    return cova

def correlation(cov):
    diag = np.diag(cov)
    return cov/np.sqrt(np.outer(diag, diag))

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

def check_integrals(k, pk, scales=[1., 3., 10., 30., 100.]):
        
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
    halos = Table.read('/Users/julian/Work/supernovae/peculiar/survey_LCDM_062.fits')
    #print('Number of halos', len(halos))

    #-- cut to small sky region for testing
    #mask = (halos['ra']>0) & (halos['ra']<10.) & (halos['dec']>0) & (halos['dec']<10.)
    #halos = halos[mask]
    if not nhalos is None:
        halos = halos[:nhalos]
    halos['ra'] = np.radians(halos['ra'].astype(float))
    halos['dec'] = np.radians(halos['dec'].astype(float))
    
    #-- Compute comoving distances in Mpc/h units to match power-spectrum units
    redshift_space = False
    if redshift_space:
        #-- Redshift-space
        r_comov = cosmo.get_comoving_distance(halos['redshift_obs'].data.astype(float))*cosmo.pars['h']
    else:
        #-- Real-space
        r_comov = cosmo.get_comoving_distance(halos['redshift'].data.astype(float))*cosmo.pars['h']
   
    ra = halos['ra'].data.astype(float)
    dec = halos['dec'].data.astype(float)
    vel = halos['v_radial'].data.astype(float)
    
    return ra, dec, r_comov, vel

def build_covariance_matrix(ra, dec, r_comov, vel, k, pk, pool=None):

    nh = ra.size
    indices = [(i, j) for i in range(nh) for j in range(i+1, nh)]

    if not pool is None:
        results = pool.map(get_covariance_parallel, indices)

    cov_matrix = np.zeros((nh, nh))
    for ind, cov in zip(indices, results):
        i, j = ind
        cov_matrix[i, j] = cov
        cov_matrix[j, i] = cov
    np.fill_diagonal(cov_matrix, covariance(k, pk, 1/3))

    #-- Pre-factor H0^2/(2pi^2)
    cov_matrix *= (100)**2/(2*np.pi**2) 

    return cov_matrix

@jit(nopython=True)
def log_like(x, cova):
    nx = x.size
    eigvals = np.linalg.eigvalsh(cova)
    inv_matrix = np.linalg.inv(cova)
    chi2 = x.T @ inv_matrix @ x
    log_likes = -0.5*(nx*np.log(2*np.pi) 
                      + np.sum(np.log(eigvals))
                      + chi2)
    return log_likes

def get_log_likes(f_value):

    this_log_likes = np.zeros(sig_values.size)
    for j in range(sig_values.size):
        #-- Total matrix
        cov_matrix = cov_cosmo*f_value**2 
        #-- Add intrinsic dispersion to diagonal 
        d = np.einsum('ii->i', cov_matrix) 
        d += sig_values[j]**2
        #-- Compute likelihood
        this_log_likes[j] = log_like(vel, cov_matrix)
    return this_log_likes

def read_likelihood(fin):
    f_values, sig_values, like = np.loadtxt(fin, unpack=1)
    f_values = np.unique(f_values)
    sig_values = np.unique(sig_values)
    like = np.reshape(like, (f_values.size, sig_values.size))
    return f_values, sig_values, like

#ef main():

cosmo = CosmoSimple(omega_de=0.32, h=0.7, mass_neutrinos=0.)

nhalos = 1000
kmax = 1.
nk = 512
non_linear = True
redshift_space = False
parallel = True
one_dim = False
sig_v = 100. # km/s

print('Number of selected halos:', nhalos)
print('kmax:', kmax)
print('nk:', nk)
print('Non-linear:', non_linear)
print('Redshift-space:', redshift_space)
print('Parallel:', parallel)
print('sig_v:', sig_v)

k, pk = read_power_spectrum(non_linear=non_linear, 
                            redshift_space=redshift_space, 
                            kmin=None, kmax=kmax, nk=nk)

ra, dec, r_comov, vel = read_halos(cosmo=cosmo, 
    redshift_space=redshift_space, 
    nhalos=nhalos)

#-- Compute covariance matrix
t0 = time.time()
if parallel:
    context = multiprocessing.get_context('fork')
    print("Number of cpus available: ", multiprocessing.cpu_count())
    with context.Pool(processes=multiprocessing.cpu_count()) as pool:
        cov_cosmo = build_covariance_matrix(ra, dec, r_comov, vel, k, pk, pool=pool)    
else:
    cov_cosmo = build_covariance_matrix(ra, dec, r_comov, vel, k, pk)
t1 = time.time()
print(f'Time elapsed calculating cov matrix {(t1-t0)/60:.2f} minutes')


print('Cov:')
print(cov_cosmo[:5, :5])

#-- Scan over f values
f_expected = 0.523

n_values = 40
f_values = np.linspace(0.2, 0.5, n_values)
sig_values = np.linspace(100, 300., n_values)
if one_dim:
    log_likes = np.zeros( n_values )
    for i in range(n_values):
        print(i, n_values)
        #-- Total matrix
        cov_matrix = cov_cosmo*f_values[i]**2
        #-- Add intrinsic dispersion to diagonal 
        d = np.einsum('ii->i', cov_matrix) 
        d += sig_v**2
        #-- Compute likelihood
        log_likes[i] = log_like(vel, cov_matrix)
else:    
    log_likes = np.zeros( (n_values, n_values) )
    
    if 1==1:
        with context.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(get_log_likes, f_values)
        for i in range(n_values):
            for j in range(n_values):
                log_likes[i, j] = results[i][j]
    else:
        for i in range(n_values):
            print(i, n_values)
            log_likes[i] = get_log_likes(f_values[i])


t2 = time.time()
print(f'Time elapsing with likelihood grid: {(t2-t1)/60:.2f} minutes or {(t2-t1)/n_values**2:.2f} sec per entry')

likelihood = np.exp(log_likes-log_likes.max())

fout = open('like.txt', 'w')
for i in range(n_values):
    for j in range(n_values):
        print(f_values[i], sig_values[j], likelihood[i, j], file=fout)
fout.close()

plotit=False
if plotit:
    if one_dim: 
        plt.figure()
        plt.plot(f_values, likelihood)
        plt.axvline(f_expected, color='k', ls='--')
        plt.xlabel('Growth-rate f')
        plt.ylabel('Likelihood')
    else:
        from scipy.stats import chi2
        #-- Cumulative distribution at 1, 2, 3 sigma for one degree of freedom
        cdf = chi2.cdf([1, 4, 9], 1)
        #-- corresponding chi2 values for 2 degrees of freedom
        chi2_values = chi2.ppf(cdf, 2)
        #-- corresponding likelihood values
        like_contours = np.exp(-0.5*chi2_values)

        plt.figure()
        plt.pcolormesh(f_values, sig_values, likelihood.T, shading='nearest', cmap='gray_r')
        plt.contour(f_values, sig_values, likelihood.T, levels=like_contours,
                    colors='k', linestyles='--')
        plt.colorbar()
        plt.xlabel(r'Growth-rate $f$')
        plt.ylabel(r'$\sigma_v$ [km/s]')



#if __name__ == '__main__':
#    main()

    