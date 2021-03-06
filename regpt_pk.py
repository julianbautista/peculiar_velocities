import pyregpt
import numpy as np
import pylab as plt

nthreads=30
k, pklin = np.loadtxt('pk_lin_camb_demnunii_1024.txt', unpack=1)
mask = (k<=1.0)
k = k[mask]
pklin = pklin[mask]
regpt0 = pyregpt.Spectrum2Loop()
regpt0.set_pk_lin(k, pklin)
regpt0.set_terms(k)
regpt0.run_terms('delta', 'delta', nthreads=30)
p_dd = regpt0.pk()

regpt1 = pyregpt.Spectrum2Loop()
regpt1.set_pk_lin(k, pklin)
regpt1.set_terms(k)
regpt1.run_terms('delta', 'theta', nthreads=30)
p_dt = regpt1.pk()


regpt2 = pyregpt.Spectrum2Loop()
regpt2.set_pk_lin(k, pklin)
regpt2.set_terms(k)
regpt2.run_terms('theta', 'theta', nthreads=30)
p_tt = regpt2.pk()

fout = open('pk_regpt_demnunii_1024.txt', 'w')
print( '# k[h/Mpc]   P_delta,delta   P_delta,theta   P_theta,theta ', file=fout)
for i in range(k.size):
    print(f'{k[i]}   {p_dd[i]}    {p_dt[i]}   {p_tt[i]}', file=fout)
fout.close()


