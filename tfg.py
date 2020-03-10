# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:54:38 2020
@author: Carlos Sánchez Muñoz
"""

import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import integrate, init_printing
from sympy.abc import x

#######################
###   FUNCIONES   ###
#######################

""" Sucesión de funciones del sistema de Haar. """
def H_n(n, x):
	# Calculamos k
	l = math.log(n,2);
	k = int(l);
	if l - k == 0:
		k = k-1
	# Calculamos s
	s = n - 2**k

	# Evaluamos x
	den = 2**(k+1)
	if (2*s-2)/den <= x and x < (2*s-1)/den: return 1
	elif (2*s-1)/den <= x and x < (2*s)/den: return -1
	else: return 0


#######################
###       MAIN      ###
#######################

""" Programa principal. """
def main():
	print(H_n(2, 0.4))
	print(H_n(5, 0.23))
	f = x**2 - 3*x + 2
	print(f.subs(x,1))
	y = np.piecewise(x, [x<0, x>=0], [lambda x: -x, lambda x: x])
	print(y.subs(x,1))


if __name__ == "__main__":
    main()
