import numpy as np
import matplotlib.pyplot as plt

def gaussian (x,sigma):
    # Global RBF
    psi = np.exp(-(x**2)/(2*sigma**2))
    return psi

def ctpsc2b (x,r):
    zetta = x/r
    if zetta > 0 and zetta <= 1:
        psi = 1 - 20*zetta**2 + 80*zetta**3 - 45*zetta**4 - 16*zetta**5 \
         + (60*zetta**4 * np.log(zetta))
    elif zetta == 0:
        psi = 1
    else:
        psi = 0

    return psi

def ctpsc1 (x,r):
    zetta = x/r
    if zetta > 0 and zetta <= 1:
        psi = 1 + (80/3)*zetta**2 - 40*zetta**3 + 15*zetta**4 - (8/3)*zetta**5 \
         + (20*zetta**2 * np.log(zetta))
    elif zetta == 0:
        psi = 1
    else:
        psi = 0

    return psi