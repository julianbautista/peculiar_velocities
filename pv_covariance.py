from threadpoolctl import threadpool_limits
import numpy as np
import pylab as plt
import time

from astropy.table import Table

import scipy.integrate
import scipy.special
import scipy.stats

from numba import jit, prange

import iminuit

from cosmo import CosmoSimple

plt.ion()

##-- Create a mock from a halo catalog 

def read_halos(input_halos, 
               cosmo=None, redshift_space=False, nhalos=None, zmin=None, zmax=None, 
               subsample_fraction=1.):

    #-- Read halo catalog
    halos = Table.read(input_halos)
    
    #-- cut to small sky region for testing
    mask = np.ones(len(halos), dtype=bool)
    #mask = (halos['ra']<180) & (halos['ra']>0.  ) & (halos['dec']>0) & (halos['dec']<70.)
    #mask = (halos['ra']<360) & (halos['ra']>180.) & (halos['dec']>0) & (halos['dec']<70.)
    #mask = (halos['ra']<180) & (halos['ra']>0.  ) & (halos['dec']<0) & (halos['dec']>-70.)
    #mask = (halos['ra']<360) & (halos['ra']>180.) & (halos['dec']<0) & (halos['dec']>-70.)
    f_sky = np.sum(mask)/len(halos)

    if redshift_space:
        z = halos['redshift_obs']
    else:
        z = halos['redshift']
    z = z.data.astype(float)

    if not zmin is None:
        mask &= (z > zmin)    
    if not zmax is None:
        mask &= (z < zmax)
    halos = halos[mask]
    z = z[mask]
    
    if subsample_fraction < 1.:
        #-- Downsampling to match 2MTF catalogs from Howlett et al. 2017
        #nz = density_z(halos['redshift'], f_sky, cosmo, nbins=30)
        #nz_gal = np.interp(halos['redshift'], nz['z_centers'], nz['density'])
        #prob_to_keep = 10**( (-4+2)/(0.03-0.002)*(halos['redshift']-0.002))#/nz_gal
        np.random.seed(10)
        r = np.random.rand(len(halos))
        mask = r <= subsample_fraction
        halos = halos[mask]
        z = z[mask]

    if not nhalos is None:
        halos = halos[:nhalos]
    halos['ra'] = np.radians(halos['ra'])
    halos['dec'] = np.radians(halos['dec'])

    #-- Compute comoving distances in Mpc/h units to match power-spectrum units
    r_comov = cosmo.get_comoving_distance(z)*cosmo.pars['h']

    ra = halos['ra'].data.astype(float)
    dec = halos['dec'].data.astype(float)
    vel = halos['v_radial'].data.astype(float)
    vel_error = np.zeros(ra.size)

    catalog = {'ra': ra,
               'dec': dec,
               'r_comov': r_comov,
               'vel': vel,
               'vel_error': vel_error, 
               'redshift': z,
               'weight': np.ones(ra.size),
               'f_sky': f_sky,
               'size': ra.size,
               'n_gals': np.ones(ra.size)}
    return catalog

def add_intrinsic_scatter(catalog, cosmo, sigma_m=0.1, seed=0):
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
    catalog['vel_error'] = sigma_v

def create_mock_catalog(input_halos, cosmo, 
    redshift_space=False,
    zmin=None, zmax=None,
    subsample_fraction=1.,
    nhalos=None, 
    sigma_m = 0, seed_sigma_m=0):

    catalog = read_halos(input_halos,
        cosmo=cosmo, 
        redshift_space=redshift_space, 
        nhalos=nhalos, zmin=zmin, zmax=zmax,
        subsample_fraction=subsample_fraction)

    #-- Add errors on velocity measurements from a given error in distance modulus
    if sigma_m != 0:
        add_intrinsic_scatter(catalog, cosmo, sigma_m=sigma_m, seed=seed_sigma_m)

    return catalog

def read_catalog(input_catalog, cosmo, use_true_vel=False):

    ra, dec, z, vpec_est, vpec_true, vp_error = np.loadtxt(input_catalog, unpack=1)
    r_comov = cosmo.get_comoving_distance(z)*cosmo.pars['h']
    if use_true_vel:
        vel = vpec_true
        vel_error = np.zeros(vel.size)
    else:
        vel = vpec_est
        vel_error = vp_error
    f_sky = 0.5
    w = z < 0.1
    ra = ra[w]
    dec = dec[w]
    r_comov = r_comov[w]
    z = z[w]
    vel = vel[w]
    vel_error = vel_error[w]


    catalog = {'ra': ra,
               'dec': dec,
               'r_comov': r_comov,
               'vel': vel,
               'vel_error': vel_error, 
               'redshift': z,
               'weight': np.ones(ra.size),
               'f_sky': f_sky,
               'size': ra.size,
               'n_gals': np.ones(ra.size)}
    return catalog

def grid_velocities(catalog, grid_size=20.):
    ''' Transform a galaxy catalog into a voxel catalog,
        where voxels have grid_size in Mpc/h 
    '''
    if grid_size==0:
        return catalog

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
    sum_vel_error = np.bincount(i, weights=catalog['vel_error']**2*catalog['weight'], minlength=n_pix)
    sum_we   = np.bincount(i, weights=catalog['weight'], minlength=n_pix)
    sum_n    = np.bincount(i, minlength=n_pix)

    #-- Consider only voxels with at least one galaxy
    w = sum_we > 0
    center_vel = sum_vel[w]/sum_we[w]
    #center_vel_std = np.sqrt(sum_vel2[w]/sum_we[w] - center_vel**2)/np.sqrt(sum_n[w])
    center_vel_error = np.sqrt(sum_vel_error[w]/sum_we[w])/np.sqrt(sum_n[w])
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
            'vel_error': center_vel_error,
            'weight': center_weight,
            'n_gals': center_ngals,
            'size': center_ra.size}

def density_z(z, f_sky, cosmo, zmin=None, zmax=None, nbins=50):
    ''' Compute comoving number density in [h^3/Mpc^3] as a function
        of redshift
        
        Input
        -----
        z: array with redshifts of galaxies
        f_sky: float - fraction of total sky covered
        cosmo: CosmoSimple instance - fiducial cosmology
        zmin: minimum redshift, default is np.min(z)
        zmax: maximum redshift, defautl is np.max(z)
        nbins: number of bins, default is 50

        Returns
        -----
        dict:  Containing 'z_centers', 'z_edges', 'density', 'density_err', 'volume'
    '''

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

def read_power_spectrum(non_linear='',
                        redshift_space=False):

    #-- Read power spectrum from camb 
    #-- units are in h/Mpc and Mpc^3/h^3
    pk_table = Table.read('pk_lin_camb_demnunii_1024.txt', format='ascii',names=('k', 'power'))
    k = pk_table['k'] 
    pk = pk_table['power'] 

    #-- apply non-linearities from Bel et al. 2019
    if non_linear == 'bel': 
        sig8 = 0.84648
        a1 = -0.817+3.198*sig8
        a2 = 0.877 - 4.191*sig8
        a3 = -1.199 + 4.629*sig8
        pk = pk*np.exp(-k*(a1+a2*k+a3*k**2))

    #-- Read RegPT theory for theta-theta
    if non_linear == 'regpt':
        k, pdd, pdt, ptt = np.loadtxt('pk_regpt_demnunii_1024.txt', unpack=1)
        pk = ptt

    #-- Go to redshift-space (only if using redshift_obs)
    #-- based on Koda et al. 2014
    if redshift_space:
        sigma_u = 13. #- Mpc/h
        D_u = np.sin(k*sigma_u)/(k*sigma_u)
        pk *= D_u**2

    return k, pk

def reduce_resolution(k, pk, kmin=None, kmax=None, nk=None, linear=False):

    if kmin is None:
        kmin = k.min()
    if kmax is None:
        kmax = k.max()
    if nk is None:
        w = (k>=kmin)&(k<=kmax)
        nk = np.sum(w)
    if linear:
        knew = np.linspace(kmin, kmax, nk)
    else:
        knew = 10**np.linspace(np.log10(kmin), np.log10(kmax), nk)
    pknew = np.interp(knew, k, pk)
    return knew, pknew

def optimize_k_range(k, pk, precision=1e-5):

    kmin = k[:k.size//2]
    kmax = k[k.size//2:]
    var_true = np.trapz(pk, x=k)

    def get_var(kl, pkl, kmin=1e-3, kmax=0.1, nk=None):
        w = (kl>=kmin)&(kl<=kmax)
        return np.trapz(pkl[w], x=kl[w])
    
    var_kmin = np.array([get_var(k, pk, kmin=ki, kmax=k[-1]) for ki in kmin])
    var_kmax = np.array([get_var(k, pk, kmin=k[0], kmax=ki)  for ki in kmax])
    error_kmin = 1-var_kmin/var_true
    error_kmax = 1-var_kmax/var_true
    kmin_opt = np.interp(precision/2, error_kmin, kmin)
    kmax_opt = np.interp(-precision/2, -error_kmax, kmax)

    k_opt, pk_opt = reduce_resolution(k, pk, kmin=kmin_opt, kmax=kmax_opt)
    error = 1-np.trapz(pk_opt, x=k_opt)/var_true
    print('kmin:', kmin_opt)
    print('kmax:', kmax_opt)
    return k_opt, pk_opt

def check_integrals():

    k, pk = read_power_spectrum(non_linear='regpt')
    scales = [1., 3., 10., 30., 100.]

    plt.figure()
    for i_scale, scale in enumerate(scales):
        win_radial = window(k, 200., 200+scale, 1.)
        win_transv = window(k, 200., 200., np.cos(scale/200.))
        full_int_radial = np.trapz(win_radial*pk, x=k)
        full_int_transv = np.trapz(win_transv*pk, x=k)
        i_k = np.arange(k.size//2, k.size)
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
    ''' From Johnson et al. 2014 '''
    r = separation(r_0, r_1, cos_alpha)
    sin_alpha_squared = 1-cos_alpha**2
    win = 1/3*np.ones_like(k)
    if r > 0:
        j0kr = j0_alt(k*r) 
        j2kr = j2_alt(k*r)
        win = 1/3*(j0kr - 2*j2kr)*cos_alpha
        win = win+(r_0*r_1/r**2*sin_alpha_squared * j2kr)
    return win

    
@jit(nopython=True)
def window_2(k, r_0, r_1, cos_alpha):
    ''' From Adams and Blake 2020
        with RSD, but do not account for wide-angle effects
    '''
    r = separation(r_0, r_1, cos_alpha)
    win = np.ones_like(k)
    if r == 0:
        return win
    cos_gamma_squared = (1+cos_alpha)/2*(r_1-r_0)**2/r**2
    l2 = 0.5*(3*cos_gamma_squared-1)
    j0kr = j0_alt(k*r) 
    j2kr = j2_alt(k*r)
    win = j0kr/3 + j2kr*(-2/3*l2)
    return win

@jit(nopython=True)
def window_3(k, r_0, r_1, cos_alpha):
    ''' From Castorina and White 2020
        with RSD, account for wide-angle effects
        gives same results as window from Ma. et al. 2011
    '''
    r = separation(r_0, r_1, cos_alpha)
    win = np.ones_like(k)
    if r == 0:
        return win*1/3
    cos_gamma_squared = (1+cos_alpha)/2*(r_1-r_0)**2/r**2
    l2 = 0.5*(3*cos_gamma_squared-1)
    j0kr = j0_alt(k*r) 
    j2kr = j2_alt(k*r)
    win = j0kr/3*cos_alpha - 2/3*j2kr*(l2- 1/4*(1-cos_alpha))
    return win

def test_window():
    ''' Reproduce Fig 4 of Johnson et al. 2014 
        even though his labels are switched 
        between (a and e) and (b and d)
    '''
    k = 10**np.linspace(-4, 0, 1000)
    plt.figure()
    #-- Values from Johnson et al. 2014
    r0_r1_angle = [ [86.6, 133.7, 0.393],
                    [76.8, 127.6, 1.313],
                    [59.16, 142.5, 0.356],
                    [51.9, 91.1, 0.315],
                    [99.449, 158.4, 1.463]]
    r0_r1_angle = [ [50., 50., 0.],
                    [50., 50., np.pi/2],
                    [50., 50., np.pi]]
    for win, alpha, ls in zip([window, window_2, window_3], [1, 0.5, 1.0], ['-', '--', ':']):
    #for win, alpha, ls in zip([window_2, window_3], [ 0.5, 1.0], [ '--', ':']):
        for i, [r0, r1, angle] in enumerate(r0_r1_angle):
            plt.plot(k, win(k, r0,  r1, np.cos(angle)), color=f'C{i}', ls=ls, alpha=alpha)
    plt.legend()
    plt.xlim(5e-4, 2e-1)
    plt.xscale('log')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel(r'$W_{i,j}(k)$')

#@jit(nopython=True, parallel=True)
def grid_window(k, L, n=100):
    if L == 0:
        return None

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
def build_covariance_matrix(ra, dec, r_comov, k, pk_nogrid, grid_win=None, n_gals=None):
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
        np.fill_diagonal(cov_matrix, var + (var_nogrid-var)/n_gals)

    #-- Pre-factor H0^2/(2pi^2)
    cov_matrix *= (100)**2/(2*np.pi**2) 

    return cov_matrix

def corr_from_cov(cov):
    diag = np.diag(cov)
    return cov/np.sqrt(np.outer(diag, diag))

#@jit(nopython=True, fastmath=True)
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

def fit_iminuit(vel, vel_error, n_gals, cov_cosmo):

    #@jit(nopython=True, parallel=False)
    def get_log_like(fs8, sig_v):
        diag_cosmo = np.diag(cov_cosmo)
        cov_matrix = cov_cosmo*fs8**2 
        diag_total = diag_cosmo*fs8**2 + sig_v**2/n_gals + vel_error**2
        np.fill_diagonal(cov_matrix, diag_total)
        log_like = log_likelihood(vel, cov_matrix)
        return -log_like
    
    t0 = time.time()
    m = iminuit.Minuit(get_log_like, fs8=0.5, sig_v=200.)
    m.errordef = iminuit.Minuit.LIKELIHOOD
    m.limits['fs8'] = (0.1, 2.)
    m.limits['sig_v'] = (0., 1000)
    m.migrad()
    m.minos()
    t1 = time.time()
    print(m)
    print(f'iMinuit fit lasted: {(t1-t0)/60:.2f} minutes')
    return m


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
    zmin = 0.,
    zmax = None,
    kmax = None,
    nk = 512,
    non_linear = '',
    redshift_space = False,
    grid_size = 0, 
    subsample_fraction=1.,
    sigma_m=0.,
    export_contours='',
    thread_limit=None,
    create_mock=False,
    use_true_vel=False):

    print('===========================================')
    print('Name of run:', name)
    print('Input catalog:', input_catalog)
    print('Number of selected halos:', nhalos)
    print('Redshift range:', zmin, zmax)
    print('kmax:', kmax)
    print('nk:', nk)
    print('Non-linear:', non_linear)
    print('Redshift-space:', redshift_space)
    print('Error in magnitude:', sigma_m)

    t00 = time.time()

    #-- Set up fiducial cosmology
    cosmo = CosmoSimple(omega_m=0.32, h=0.67, mass_neutrinos=0.)
    sigma_8 = 0.84648 #-- DEMNUni value for m_nu = 0 (Castorina et al. 2015)
    
    #-- Read power spectrum model
    k, pk = read_power_spectrum(non_linear=non_linear, 
                                redshift_space=redshift_space)
    k, pk = optimize_k_range(k, pk, precision=1e-6)

    #-- Normalise by sigma_8 of this template power spectra
    pk /= sigma_8**2

    #-- Create mock from halo catalog
    if create_mock:
        catalog = create_mock_catalog(input_catalog, cosmo, 
                                    redshift_space=redshift_space,
                                    zmin=zmin, zmax=zmax, 
                                    subsample_fraction=subsample_fraction,
                                    nhalos=nhalos, sigma_m=sigma_m)
    else:
    #-- Read halo catalog and compute comoving distances
        catalog = read_catalog(input_catalog, cosmo, use_true_vel=use_true_vel)

    print('Number of galaxies in catalog:', catalog['size'])
    print(f'Radial velocity dispersion: {np.std(catalog["vel"]):.1f} km/s')

    #-- Put halos and velocities in a grid
    catalog = grid_velocities(catalog, grid_size=grid_size)

    print('Number of grid cells with data: ', catalog['size'])
    print(f'Radial velocity dispersion (grid): {np.std(catalog["vel"]):.1f} km/s')

    #-- Compute grid window function
    grid_win = grid_window(k, grid_size)

    #-- Compute cosmological covariance matrix
    t0 = time.time()
    print(f'Computing cosmological covariance matrix...')
    cov_cosmo = build_covariance_matrix(catalog['ra'], 
                                        catalog['dec'], 
                                        catalog['r_comov'],
                                        k, pk, 
                                        grid_win=grid_win, 
                                        n_gals=catalog['n_gals'])    
    t1 = time.time()
    print(f'Time elapsed calculating cov matrix {(t1-t0)/60:.2f} minutes')
    

    #-- Print some elements of covariance matrix
    n_el = 10
    print(f'First {n_el} elements of cov_cosmo [10^5 km^2/s^2]:')
    for i in range(n_el):
        line = '   '
        for j in range(n_el):
            line+=f'{cov_cosmo[i, j]/1e5:.2f} '
        print(line)

    #print('Eigvalues:')
    #print(np.linalg.eigvalsh(cov_cosmo))

    #-- Perform fit of fsigma8
    print('Running iMinuit fit of fsigma8...')
    with threadpool_limits(limits=thread_limit, user_api='blas'):
        mig = fit_iminuit(catalog['vel'], 
                            catalog['vel_error'], 
                            catalog['n_gals'], 
                            cov_cosmo)
    header = header_line(mig)
    line = fit_to_line(mig, name)

    if export_contours != '':
        print('Computing one and two sigma contours...')
        one_sigma = mig.mncontour('fs8', 'sig_v', cl=0.685, size=30)
        two_sigma = mig.mncontour('fs8', 'sig_v', cl=0.95, size=30)
        np.savetxt(export_contours+'_one_sigma', one_sigma)
        np.savetxt(export_contours+'_two_sigma', two_sigma)



    tt = time.time()
    print(f'Total time elapsed for {name}: {(tt-t00)/60:.2f} minutes')
    print('')
    print('')

    return line, header

#if __name__ == '__main__':
#    main()

    
