# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:54:38 2020
@author: Carlos Sánchez Muñoz
"""

import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import *
x = Symbol('x', real=True)

def matrix_example(x):
    row = np.array(haar_row(x))
    final = np.array(np.transpose(haar_row(np.transpose(row))))
    print(x)
    print()
    print(row)
    print()
    print(final)
    print()

def test_matrix():
    x = np.array([[12,12,8,8],
                  [12,12,8,8],
                  [10,10,8,8],
                  [10,10,8,8]])

    y = np.array([[12,12,12,12,8,8,10,10],
                  [12,12,12,12,8,8,10,10],
                  [10,10,10,10,8,8,10,10],
                  [10,10,10,10,8,8,10,10],
                  [22,22,22,22,8,8,16,16],
                  [22,22,22,22,8,8,16,16],
                  [22,20,20,20,14,14,4,4],
                  [22,20,20,20,14,14,4,4]])

    matrix_example(x)

#######################
###   FUNCIONES   ###
#######################

""" Sucesión de funciones del sistema de Haar. Sólo para evaluar un x. """
def Hn_eval(n, x):
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

""" Sucesión de funciones del sistema de Haar. """
def Hn(n):
	# Calculamos k
	l = math.log(n,2);
	k = int(l);
	if l - k == 0:
		k = k-1
	# Calculamos s
	s = n - 2**k

	# Calculamos la función
	den = 2**(k+1)
	v1 = (2*s-2)/den; v2 = (2*s-1)/den; v3 = 2*s/den
	hn = Piecewise( (1, And(x>=v1, x<v2)), (-1, And(x>=v2, x<v3)), (0, True) )
	return hn

""" Construcción de un diccionario con las n primeras funciones de Haar. """
def Haar_dictionary(n):
	h1 = Piecewise( (1, True) )
	d = [h1]
	for i in range(2,n+1):
		d.append(Hn(n))
	return d

""" Integra correctamente las funciones de Haar """
def int_midpoint(fun):
	last = 6
	while(str(fun._args[1][1])[last]!=")"):
		last = last + 1
	mid = float(str(fun._args[1][1])[6:last])
	return integrate(fun, (x,0,mid), conds='piecewise') - integrate(fun, (x,mid,1), conds='piecewise')

def Ff(fun):
	0;

def trial1():
	haar = Hn(3) + Hn(4)
	print(haar.as_leading_term(x))
	#int_midpoint(haar)
	#print(haar.subs(x, 0.6))
	integ = integrate(haar, x, conds='piecewise')
	#print(integ)
	#f = x**2 - 3*x + 2
	#print(f.subs(x,1))

	#d = Haar_dictionary(2)
	#print(d)

def trial2():
	a=1

#######################
###       MAIN      ###
#######################

""" Programa principal. """
def main():
	trial1()

if __name__ == "__main__":
    main()
