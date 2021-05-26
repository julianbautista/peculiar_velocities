import numpy as np
import pylab as plt
plt.ion()


root = 'bastien_v0'

plt.figure()
x, y = np.loadtxt(root+'_two_sigma', unpack=1)
plt.fill(x, y, color='C0', alpha=0.3)
x, y = np.loadtxt(root+'_one_sigma', unpack=1)
plt.fill(x, y, color='C0', alpha=1)
plt.xlabel(r'$f\sigma_8$', fontsize=16)
plt.ylabel(r'$\sigma_v$ [km/s]', fontsize=16)
plt.xlim(0.3, 0.6)
plt.ylim(150, 300)
plt.axvline(0.4505, color='k', ls='--')
plt.tight_layout()
