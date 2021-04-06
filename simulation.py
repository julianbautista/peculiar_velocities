import cosmo
import numpy as np
import pylab as plt
from scipy.optimize import minimize, minimize_scalar

plt.ion()


#-- Fixing the random seed
np.random.seed(1)

#-- Initialize cosmology
c_truth = cosmo.CosmoSimple()
c_light = c_truth.pars['c']

plot_hd = False
plot_methods = True

#-- Number of supernovae
n_sn = 10000

#-- Intrinsic scatter of magnitudes
sigma_m = 0.10
def sig_v(z, c):
    return -c_light*np.log(10)/5 / (1 - c_light*(1+z)**2/c.get_hubble(z)/c.get_DL(z)) * sigma_m

#-- Cosmological redshifts 
zmin = 0.01
zmax = 0.1

z_cosmo = zmin+np.random.rand(n_sn)*(zmax-zmin)
mu_cosmo = c_truth.get_distance_modulus(z_cosmo)

#-- Draw peculiar velocities
rms_v_p = 300. #- km/s
v_p = np.random.randn(n_sn)*rms_v_p
#-- Relativistic Doppler effect
#-- which could be approximated by z_p = 1+v_p/c_light
def get_peculiar_redshift(v_p):
    return np.sqrt( (1+v_p/c_light)/(1-v_p/c_light)) - 1

def get_peculiar_velocity(z_p):
    a = (1+z_p)**2
    return c_light* (a-1)/(a+1)

z_p = get_peculiar_redshift(v_p)

#-- Redshift total
z_obs = (1+z_cosmo)*(1+z_p) - 1
#-- From Eq. 18 of Davis et al. 2011 :
#-- D_L(z_obs) = D_L_cosmo(z_cosmo) * (1 + z_pec)**2
#-- "The two factors of (1 + z_pec) enter the luminosity distance SN
#-- correction. One is due to the Doppler shifting of the photons, 
#-- the other is due to relativistic beaming."
mu_obs = mu_cosmo + 10*np.log10(1+z_p)

#-- Add intrinsic scatter of magnitudes
mu_obs += np.random.randn(n_sn)*sigma_m

#=========== Measurements =============#

#-- Now we will try to estimate these velocities using a 
#-- fiducial cosmology 
c_fid = cosmo.CosmoSimple(omega_m=0.29)
c_fid = c_truth

#-- Use interpolation to estimate the cosmological redshift from 
#- the observed distance modulus
z_th = np.linspace(1e-5, 0.5, 10000)
mu_th = c_fid.get_distance_modulus(z_th)



#-- Estimate 1 of the peculiar velocity
#-- Simple assume that the change in luminosity is small and 
#-- calculate the corresponding z_cosmo
z_cosmo_est = np.interp(mu_obs, mu_th, z_th)
z_p_est = (1+z_obs)/(1+z_cosmo_est) - 1
v_p_est = get_peculiar_velocity(z_p_est)

#-- Alternatively, one can use the difference in magnitude
#-- Eq. 1 in Johnson et al. 2014 or Eq. 15 in Hue and Greene 2006
mu_obs_est = c_fid.get_distance_modulus(z_obs)
v_p_est2 = np.log(10)/5 * (mu_obs-mu_obs_est) * c_light
v_p_est2 /= (1 - c_light*(1+z_obs)**2/c_fid.get_hubble(z_obs)/c_fid.get_DL(z_obs)) 

#-- Method 3: Fit for z_cosmo and z_p simultaneously
def get_redshifts(z_obs_in, mu_obs_in):

    def difference(z_pec):
        z_cos = (1+z_obs_in)/(1+z_pec)-1
        mu_cos = np.interp(z_cos, z_th, mu_th)
        mu_obs = mu_cos + 10*np.log10(1+z_pec)
        diff = (mu_obs-mu_obs_in)**2/1e-4**2
        #print(z_pec, diff)
        return diff

    res = minimize_scalar(difference, bounds=(-0.01, 0.01), method='bounded')
    if res.success == False:
        print(res)

    return res.x

v_p_est3 = np.array( [ get_peculiar_velocity(get_redshifts(zo, muo)) for zo, muo in zip(z_obs, mu_obs)])

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
if plot_hd:
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

if plot_methods:
    #-- Compare the estimated versus true peculiar velocities
    plt.figure()
    plt.scatter(z_cosmo, v_p_est-v_p, c=v_p, vmin=-5*rms_v_p, vmax=5*rms_v_p, 
                cmap='seismic', s=4)
    x, y, ns = get_profiles(z_cosmo, v_p_est-v_p)
    plot_profiles(x, y, color='C2')
    for a in [1, -1, 2, -2]:
        plt.plot(x, a*sig_v(x, c_truth), 'k--')
    plt.xlabel(r'$z_{\rm cosmo}$')
    plt.ylabel(r'$(\hat{v}_p - v_p)$')
    plt.axhline(-rms_v_p, color='k', ls=':')
    plt.axhline(+rms_v_p, color='k', ls=':')
    plt.axhline(0, color='k', alpha=0.5, ls='--')
    plt.colorbar(label=r'$v_p$ [km/s]')
    plt.title(r'Using $z_{\rm cosmo}(\mu_{\rm obs})$')
    plt.tight_layout()

    #-- Compare the estimated versus true peculiar velocities
    plt.figure()
    plt.scatter(z_cosmo, v_p_est2-v_p, c=v_p, vmin=-5*rms_v_p, vmax=5*rms_v_p, 
                cmap='seismic', s=4)
    x, y, ns = get_profiles(z_cosmo, v_p_est2-v_p)
    plot_profiles(x, y, color='C2')
    for a in [1, -1, 2, -2]:
        plt.plot(x, a*sig_v(x, c_truth), 'k--')
    plt.xlabel(r'$z_{\rm cosmo}$')
    plt.ylabel(r'$(\hat{v}_p - v_p)$')
    plt.axhline(-rms_v_p, color='k', ls=':')
    plt.axhline(+rms_v_p, color='k', ls=':')
    plt.axhline(0, color='k', alpha=0.5, ls='--')
    plt.colorbar(label=r'$v_p$ [km/s]')
    plt.title(r'Using $\mu_{\rm obs} - \mu_{\rm cosmo}(z_{\rm obs})$')
    plt.tight_layout()

    #-- Compare the estimated versus true peculiar velocities
    plt.figure()
    plt.scatter(z_cosmo, v_p_est3-v_p, c=v_p, vmin=-5*rms_v_p, vmax=5*rms_v_p, 
                cmap='seismic', s=4)
    x, y, ns = get_profiles(z_cosmo, v_p_est3-v_p)
    plot_profiles(x, y, color='C2')
    for a in [1, -1, 2, -2]:
        plt.plot(x, a*sig_v(x, c_truth), 'k--')
    plt.xlabel(r'$z_{\rm cosmo}$')
    plt.ylabel(r'$(\hat{v}_p - v_p)$')
    plt.axhline(-rms_v_p, color='k', ls=':')
    plt.axhline(+rms_v_p, color='k', ls=':')
    plt.axhline(0, color='k', alpha=0.5, ls='--')
    plt.colorbar(label=r'$v_p$ [km/s]')
    plt.title(r'Using simultaneous fit')
    plt.tight_layout()





