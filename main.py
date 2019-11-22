"""
Script to solve linear perturbation analysis for a hydrogel bilayer
under homogeneous swelling.
Written for Python 3.6.1

Written by: Arne Ilseng
Department of Structural Engineering,
Norwegian University of Science and Technology

The implementation is based on the work
Wu et al., International Journal of Solids and Structures, 2013
http://dx.doi.org/10.1016/j.ijsolstr.2012.10.022
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import bilayer_lpa as bl
import matplotlib

#  Material properties
Nv1 = 1e-2  # Normalized stiffness of lower layer
n = 2  # Stiffness ratio between upper and lower layer
chi = 0.5  # Chi parameter for lower layer
eta = 0.1  # Fraction of the gel being the upper layer

#  Testing values
num_data = int(1e2) # Number of data points in calculation
wH_list = np.linspace(0.1, 50, num=num_data)

# Initialize lists for results
mu_res = []  # mu with instability
l1_res = []  # Stretch in bottom layer
l2_res = []  # Stretch in top layer
wH_res = []  # Wave numbers with instability
# Find mu for instability for each wave number
for m, wH in enumerate(wH_list):
    a = bl.matrix_zero(-20, wH, Nv1, n, chi, eta)
    b = bl.matrix_zero(-1e-20, wH, Nv1, n, chi, eta)
    if np.sign(a) != np.sign(b):
        mu = brentq(bl.matrix_zero, -20, -1e-20, args=(wH, Nv1, n, chi, eta),
            maxiter = 500)
        mu_res.append(mu)
        l1_res.append(brentq(bl.s22_func, 1+1e-10, 1000, args=(Nv1, chi, mu)))
        l2_res.append(brentq(bl.s22_func, 1+1e-10, 1000, args=(n*Nv1, chi, mu)))
        wH_res.append(wH)

ltot_res = np.array(l1_res)*(1-eta)+np.array(l2_res)*eta

# Plot results
plt.figure()
plt.plot(wH_res, np.transpose(ltot_res))
plt.xlabel('Perturbation wave number $\omega$H')
plt.ylabel('Instability swelling ratio $\lambda_i$')
plt.show()
