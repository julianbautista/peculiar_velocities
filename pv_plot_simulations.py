import numpy as np
import pylab as plt
from cosmo import CosmoSimple
import pv_covariance
plt.ion()


input_catalog = ('/Users/julian/Work/supernovae/peculiar/surveys/ztf/'
                +'LCDM_062_ztf_0000.dat.fits')
#input_catalog = ('/Users/julian/Work/supernovae/peculiar/surveys/2mtf/'
#                +'LCDM_062_2mtf_0000.dat.fits')


cosmo = CosmoSimple(omega_m=0.32, h=0.67)

redshift_space = False
add_grid = False
add_grid_arrows=True
plot_nz = False

zmax = 0.1
subsample_fraction = 1.
grid_size = 30.
sigma_m = 0.

catalog = pv_covariance.read_halos(input_catalog,
                    cosmo=cosmo, 
                    redshift_space=redshift_space, 
                    zmax=zmax,
                    subsample_fraction=1)
print('Number of galaxies:', len(catalog['ra']))

nz = pv_covariance.density_z(catalog['redshift'], 1, cosmo)

pv_covariance.add_intrinsic_scatter(catalog, sigma_m=sigma_m, cosmo=cosmo)

grid = pv_covariance.grid_velocities(catalog, grid_size=grid_size)

def get_xyz(cat):
    x = cat['r_comov']*np.cos(cat['ra'])*np.cos(cat['dec'])
    y = cat['r_comov']*np.sin(cat['ra'])*np.cos(cat['dec'])
    z = cat['r_comov']*np.sin(cat['dec'])
    return x, y, z

x, y, z = get_xyz(catalog)
xg, yg, zg = get_xyz(grid)

v = catalog['vel']
vg = grid['vel']

position = np.array([x, y, z])
pos_min = np.min(position, axis=1)
pos_max = np.max(position, axis=1)
#- Number of grid voxels per axis
n_grid = np.floor((pos_max-pos_min)/grid_size).astype(int)+1

w = (x>0) & (y > 0) & (z>-grid_size/2) & (z < grid_size/2)
wg = (xg>0) & (yg > 0) & (zg>-grid_size/2) & (zg < grid_size/2)
print('Number of galaxies:', np.sum(w))
print('Number of grid centers:', np.sum(wg))

grid_x = np.arange(n_grid[0]+1)*grid_size+pos_min[0]
grid_y = np.arange(n_grid[1]+1)*grid_size+pos_min[1]


#f = plt.figure(figsize=(12, 5))
f = plt.figure(figsize=(6, 5))
plt.scatter(x[w], y[w], c=catalog['vel'][w], cmap='seismic', s=2, 

            vmin=-1000, vmax=1000)
f.axes[0].set_aspect('equal')
plt.xlabel(r'x [$h^{-1}$ Mpc]')
plt.ylabel(r'y [$h^{-1}$ Mpc]')
cbar = plt.colorbar()
cbar.set_label('Velocity [km/s]', rotation=270)


if add_grid:
    for g in grid_x: 
        plt.axvline(g, color='k', ls='--', lw=1, alpha=0.5)
    for g in grid_y: 
        if g>=-grid_size:
            plt.axhline(g, color='k', ls='--', lw=1, alpha=0.5)

    wg = (yg > -grid_size/2) & (zg>-grid_size/2) & (zg < grid_size/2)
    plt.autoscale(False)
    plt.scatter(xg[wg], yg[wg], c=vg[wg], s=1400, 
            cmap='seismic', vmin=-1000, vmax=1000,
            marker='s', alpha=0.5)

if add_grid_arrows:
    rg = np.sqrt(xg[wg]**2+yg[wg]**2)
    vx = vg[wg]*xg[wg]/rg
    vy = vg[wg]*yg[wg]/rg
    plt.quiver(xg[wg], yg[wg], vx, vy, vg[wg], cmap='seismic', edgecolors='k')


if plot_nz:
    nz = pv_covariance.density_z(catalog['redshift'], 1, cosmo, nbins=20)
    plt.figure(figsize=(4,3))
    plt.errorbar(nz['z_centers'], nz['density']*subsample_fraction, 
        nz['density_err']*subsample_fraction, fmt='o')
    plt.xlabel('Redshift')
    plt.ylabel(r'$\bar{n}(z)$ [$h^3$ Mpc$^{-3}$]')
    plt.ylim(0, None)
    plt.tight_layout()
    