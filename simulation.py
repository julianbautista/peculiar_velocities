from matplotlib.pyplot import sca
import cosmo
import numpy as np
import pylab as plt
from scipy.optimize import minimize_scalar
from astropy.table import Table

plt.ion()

plot_hd = False
plot_methods = False

c_light = 299792.458 #-- km/s

def sig_v(sigma_m, z, cosmology):
    ''' Convert an error in magnitude to an error in velocity
    '''
    hz = cosmology.get_hubble(z)
    dl = cosmology.get_DL(z)
    return -c_light*np.log(10)/5 / (1 - c_light*(1+z)**2/hz/dl) * sigma_m

def get_peculiar_redshift(v_p):
    ''' Computes redshift from velocities 
        relativistic Doppler effect
        which could be approximated by z_p = 1+v_p/c_light 
    '''
    return np.sqrt( (1+v_p/c_light)/(1-v_p/c_light)) - 1

def get_peculiar_velocity(z_p):
    ''' Computes velocity from redshift
        relativistic Doppler effect 
    '''
    a = (1+z_p)**2
    return c_light* (a-1)/(a+1)

def fit_peculiar_redshift(z_obs_in, mu_obs_in, z_th, mu_th):
    ''' Unbiased estimator of peculiar redshift 
        that accounts for effects both in redshift and mu
    '''
    def difference(z_pec):
        z_cos = (1+z_obs_in)/(1+z_pec)-1
        mu_cos = np.interp(z_cos, z_th, mu_th)
        mu_obs = mu_cos + 10*np.log10(1+z_pec)
        diff = (mu_obs-mu_obs_in)**2/1e-4**2
        return diff

    res = minimize_scalar(difference, bounds=(-0.05, 0.05), method='bounded')
    if res.success == False:
        print(res)

    return res.x

def create_sim(cosmo_truth=None, cosmo_measurement=None, 
               n_sn=10000, sigma_m=0.10, zmin=0.01, zmax=0.12, rms_v_p=300.,
               seed=0):

    options = {'n_sn': n_sn, 
                'sigma_m': sigma_m,
                'zmin': zmin, 
                'zmax': zmax,
                'rms_v_p': rms_v_p,
                'seed': seed}

    catalog = {}

    np.random.seed(seed)

    #-- Draw cosmological redshifts and compute distance moduli
    z_cosmo = zmin+np.random.rand(n_sn)*(zmax-zmin)
    mu_cosmo = cosmo_truth.get_distance_modulus(z_cosmo)

    catalog['z_cosmo'] = z_cosmo
    catalog['mu_cosmo'] = mu_cosmo

    #-- Draw peculiar velocities
    v_p = np.random.randn(n_sn)*rms_v_p
    z_p = get_peculiar_redshift(v_p)

    catalog['v_p_true'] = v_p
    catalog['z_p_true'] = z_p

    #-- Observed redshift
    z_obs = (1+z_cosmo)*(1+z_p) - 1

    #-- Observed distance modulus
    #-- From Eq. 18 of Davis et al. 2011 :
    #-- D_L(z_obs) = D_L_cosmo(z_cosmo) * (1 + z_pec)**2
    #-- "The two factors of (1 + z_pec) enter the luminosity distance SN
    #-- correction. One is due to the Doppler shifting of the photons, 
    #-- the other is due to relativistic beaming."
    mu_obs = mu_cosmo + 10*np.log10(1+z_p)

    #-- Add intrinsic scatter of magnitudes before (4*sigma) and after standardisation    
    mu_error = np.random.randn(n_sn)*sigma_m
    mu_obs_before = mu_obs + mu_error*4
    mu_obs = mu_obs + mu_error
    
    catalog['z_obs'] = z_obs
    catalog['mu_obs_before'] = mu_obs_before
    catalog['mu_obs'] = mu_obs
    catalog['mu_error'] = mu_error

    #=========== Measurements =============#

    #-- Use interpolation to estimate the cosmological redshift from 
    #-- the observed distance modulus
    z_th = np.linspace(1e-5, 0.5, 10000)
    mu_th = cosmo_measurement.get_distance_modulus(z_th)

    #-- Estimate 1 of the peculiar velocity
    #-- Simple assume that the change in luminosity is small and 
    #-- calculate the corresponding z_cosmo
    z_cosmo_est = np.interp(mu_obs, mu_th, z_th)
    z_p_est1 = (1+z_obs)/(1+z_cosmo_est) - 1
    v_p_est1 = get_peculiar_velocity(z_p_est1)
    catalog['v_p_est1'] = v_p_est1

    #-- Estimate 2 of the peculiar velocity
    #-- Alternatively, one can use the difference in magnitude
    #-- Eq. 1 in Johnson et al. 2014 or Eq. 15 in Hue and Greene 2006
    mu_obs_est = cosmo_measurement.get_distance_modulus(z_obs)
    v_p_est2 = np.log(10)/5 * (mu_obs-mu_obs_est) * c_light
    v_p_est2 /= (1 - c_light*(1+z_obs)**2/ 
                     cosmo_measurement.get_hubble(z_obs)/
                     cosmo_measurement.get_DL(z_obs)) 
    catalog['v_p_est2'] = v_p_est2

    #-- Estimate 3 of the peculiar velocity 
    #-- Fit for z_p 
    z_p_est3 = np.array([fit_peculiar_redshift(zo, muo, z_th, mu_th) for zo, muo in zip(z_obs, mu_obs)])
    v_p_est3 = get_peculiar_velocity(z_p_est3)
    catalog['v_p_est3'] = v_p_est3


    catalog = Table(catalog)
    catalog.meta = options

    return catalog

def get_profiles(x, y, bins=10, percentiles=[2.5, 16, 50, 84, 97.5]):
    n_per = len(percentiles)
    x_bins = np.linspace(x.min(), x.max(), bins+1)
    y_profiles = np.zeros((n_per, bins))
    n_entries = np.zeros(bins)
    for i in range(bins):
        w = (x>=x_bins[i]) & (x<x_bins[i+1])
        y_profiles[:, i] = np.percentile(y[w], percentiles)
        n_entries[i] = np.sum(w)
    x_centers = 0.5*(x_bins[:-1]+x_bins[1:])
    return x_centers, y_profiles, n_entries

def plot_profiles(x_centers, y_profiles, color=None, ls=None):
    for y in y_profiles:
        plt.plot(x_centers, y, color=color, ls=ls)


#-- Illustration of the effect of peculiar velocities on the Hubble Diagram
def plot_hubble_diagram(catalog):

    z_cosmo = catalog['z_cosmo']
    z_obs = catalog['z_obs']
    mu_cosmo = catalog['mu_cosmo']
    mu_obs = catalog['mu_obs']
    v_p = catalog['v_p_true']
    n_sn = v_p.size
    rms_v_p = catalog.meta['rms_v_p']

    plt.figure(figsize=(5,4))
    for i in range(n_sn):
        plt.plot([z_cosmo[i], z_obs[i]], [mu_cosmo[i], mu_obs[i]], 'k-', alpha=0.1)
    plt.scatter(z_obs, mu_obs, c=v_p, vmin=-5*rms_v_p, vmax=5*rms_v_p, 
                cmap='seismic', s=4, label=r'$\mu_{obs}(z_{obs})$')
    plt.colorbar(label=r'$v_p$ [km/s]')
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\mu$')
    plt.xscale('log')
    plt.title('Effect of velocities on Hubble Diagram')
    plt.tight_layout()

def plot_methods(catalog, cosmology, ylim=None): 

    z_cosmo = catalog['z_cosmo']
    v_p = catalog['v_p_true']
    rms_v_p = catalog['options']['rms_v_p']
    sigma_m = catalog['options']['sigma_m']

    #-- Compare the estimated versus true peculiar velocities
    for method in [1, 2, 3]:
        v_p_est = catalog['v_p_est'+str(method)]

        plt.figure()
        plt.scatter(z_cosmo, v_p_est-v_p, c=v_p, vmin=-5*rms_v_p, vmax=5*rms_v_p, 
                    cmap='seismic', s=4)
        x, y, ns = get_profiles(z_cosmo, v_p_est-v_p)
        plot_profiles(x, y, color='C2')
        for a in [1, -1, 2, -2]:
            plt.plot(x, a*sig_v(sigma_m, x, cosmology), 'k--')
        plt.xlabel(r'$z_{\rm cosmo}$')
        plt.ylabel(r'$(\hat{v}_p - v_p)$')
        plt.axhline(-rms_v_p, color='k', ls=':')
        plt.axhline(+rms_v_p, color='k', ls=':')
        plt.axhline(0, color='k', alpha=0.5, ls='--')
        plt.colorbar(label=r'$v_p$ [km/s]')
        plt.ylim(ylim)
        plt.title(f'Method {method}')
        plt.tight_layout()

def plot_malmquist_bias(catalog, 
    mu_max=38., nbins=30, 
    z_obs_min=0.015, z_obs_max=0.110,
    x_quantity='z_obs'):

    catalog['delta_v'] = catalog['v_p_est3'] - catalog['v_p_true']

    mask_0 = np.ones(len(catalog), dtype=bool)
    mask_mu = catalog['mu_obs_before'] < mu_max
    mask_z = (catalog['z_obs'] < z_obs_max) & (catalog['z_obs'] > z_obs_min)

    ylabel = {'mu_obs': r'Distance modulus $\mu$',
              'v_p_true': 'Mean true peculiar velocity [km/s]',
              'v_p_est3': 'Mean estimated peculiar velocity [km/s]',
              'delta_v': 'Mean (est-true) peculiar velocity [km/s]'}

    #-- Compute average mu and dispersion without cut
    for quantity in ['mu_obs', 'v_p_true', 'v_p_est3', 'delta_v']:
        plt.figure()
        plt.plot(catalog[x_quantity], catalog[quantity], 'k.', ms=2, alpha=0.1, zorder=0)
        plt.plot(catalog[x_quantity][mask_mu], catalog[quantity][mask_mu], 
                'C3.', ms=2, alpha=0.1, zorder=0.5)

        for mask in [mask_0, mask_mu, (mask_mu & mask_z)]:
            x, y, n = get_profiles(catalog[x_quantity][mask], catalog[quantity][mask], 
                            percentiles=[16., 50, 84], bins=nbins)
            yerr = np.array([y[1]-y[0], y[2]-y[1]])/np.sqrt(n)
            plt.errorbar(x, y[1], yerr, fmt='o', zorder=1)

        if quantity == 'mu_obs':
            plt.axhline(mu_max, color='C1', ls=':')
        else:
            plt.axhline(0, color='k', ls=':')
        plt.axvline(z_obs_min, color='C2', ls=':')
        plt.axvline(z_obs_max, color='C2', ls=':')
        plt.xlabel(f'Redshift {x_quantity}')           
        plt.ylabel(ylabel[quantity])
        plt.tight_layout()

        

#-- Initialize true cosmology
#cosmo_truth = cosmo.CosmoSimple(omega_m=0.31)
#-- Initialize assumed fiducial cosmology for measurements
#cosmo_measurement = cosmo.CosmoSimple(omega_m=0.29)
#cosmo_measurement = cosmo_truth

#cat = create_sim(cosmo_truth=cosmo_truth, cosmo_measurement=cosmo_measurement, 
#        n_sn=100000, sigma_m=0.1, zmin=0.01, zmax=0.12, rms_v_p=300., seed=1)

#plot_malmquist_bias(cat, mu_max=38, nbins=30, z_obs_min=0.015, z_obs_max=0.11)





