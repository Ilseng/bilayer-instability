# Functions for bi-layer linear perturbation analysis
import numpy as np
from scipy.optimize import brentq

def s22_func(l2, Nv, chi, mu):
    s22 =  np.log(1-1/l2) + 1/l2 + chi/l2**2 + Nv*(l2-1/l2) - mu
    return s22

def matrix_zero(mu, wH, Nv1, n, chi, eta):
    Nv2 = n*Nv1  # Name bottom layer as 2
    l1 = brentq(s22_func, 1+1e-10, 1000, args=(Nv1, chi, mu))
    l2 = brentq(s22_func, 1+1e-10, 1000, args=(Nv2, chi, mu))
    xhi1 = 1/l1 + (1/Nv1)*(1/(l1-1)-(1/l1)-(2*chi)/(l1**2))
    xhi2 = 1/l2 + (1/Nv2)*(1/(l2-1)-(1/l2)-(2*chi)/(l2**2))
    b1 = np.sqrt((1+l1*xhi1)/(l1**2+l1*xhi1))
    b2 = np.sqrt((1+l2*xhi2)/(l2**2+l2*xhi2))
    wh1 = wH*(1-eta)*l1
    wh = wH*(l1*(1-eta)+l2*eta)
    D = np.zeros((8,8))
    D[0,0] = 1; D[0,1] = 1; D[0,2] = 1; D[0, 3] = 1
    D[1,0] = l1; D[1,1] = -l1; D[1,2] = b1; D[1,3] = -b1
    D[2,0] = np.exp(wh1/l1); D[2,1] = np.exp(-wh1/l1);
    D[2,2] = np.exp(b1*wh1); D[2,3] = np.exp(-b1*wh1);
    D[2,4] = -np.exp(wh1/l2); D[2,5] = -np.exp(-wh1/l2);
    D[2,6] = -np.exp(b2*wh1); D[2,7] = -np.exp(-b2*wh1);
    D[3,0] = l1*np.exp(wh1/l1); D[3,1] = -l1*np.exp(-wh1/l1);
    D[3,2] = b1*np.exp(b1*wh1); D[3,3] = -b1*np.exp(-b1*wh1);
    D[3,4] = -l2*np.exp(wh1/l2); D[3,5] = l2*np.exp(-wh1/l2);
    D[3,6] = -b2*np.exp(b2*wh1); D[3,7] = b2*np.exp(-b2*wh1);
    D[4,0] = 2*Nv1*l1*np.exp(wh1/l1); D[4,1] = 2*Nv1*l1*np.exp(-wh1/l1);
    D[4,2] = Nv1*(l1+1/l1)*np.exp(b1*wh1); D[4,3] = Nv1*(l1+1/l1)*np.exp(-b1*wh1);
    D[4,4] = -2*Nv2*l2*np.exp(wh1/l2); D[4,5] = -2*Nv2*l2*np.exp(-wh1/l2);
    D[4,6] = -Nv2*(l2+1/l2)*np.exp(b2*wh1); D[4,7] = -Nv2*(l2+1/l2)*np.exp(-b2*wh1);
    D[5,0] = Nv1*(l1**2+1)*np.exp(wh1/l1); D[5,1] = -Nv1*(l1**2+1)*np.exp(-wh1/l1);
    D[5,2] = 2*Nv1*l1*b1*np.exp(b1*wh1); D[5,3] = -2*Nv1*l1*b1*np.exp(-b1*wh1);
    D[5,4] = -Nv2*(l2**2+1)*np.exp(wh1/l2); D[5,5] = Nv2*(l2**2+1)*np.exp(-wh1/l2);
    D[5,6] = -2*Nv2*l2*b2*np.exp(b2*wh1); D[5,7] = 2*Nv2*l2*b2*np.exp(-b2*wh1);
    D[6,4] = 2*l2*np.exp(wh/l2); D[6,5] = 2*l2*np.exp(-wh/l2);
    D[6,6] = (l2+1/l2)*np.exp(b2*wh); D[6,7] = (l2+1/l2)*np.exp(-b2*wh);
    D[7,4] = (l2+1/l2)*np.exp(wh/l2); D[7,5] = -(l2+1/l2)*np.exp(-wh/l2);
    D[7,6] = 2*b2*np.exp(b2*wh); D[7,7] = -2*b2*np.exp(-b2*wh);

    logdet = np.linalg.slogdet(D)
    return logdet[0]*np.exp(logdet[1])
