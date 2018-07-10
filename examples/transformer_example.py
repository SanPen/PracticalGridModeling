#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:23:27 2017

@author: santi
"""
from numpy import sqrt, array, hstack, vstack, cos, sin, pi
from numpy.linalg import inv

def f(c):
    return '{0:.4f}'.format(c)
def formatc(c):
    val = str(f(c.real)) 
    if c.imag >= 0:
        val+= ' + j' + str(f(c.imag))
    else:
        val+= ' - j' + str(f(abs(c.imag)))
    return val

def formatcm(mat):
    n, m = mat.shape
    
    db = '\\'*2
    
    v = '\\left[ \\begin{array}{' + str('c' * m) + '}\n'
    for i in range(n):
        for j in range(m):
            
            if j < (m-1):
                v += formatc(mat[i, j]) + ' & '
            else:
                v += formatc(mat[i, j])
        
        if i < (n-1):
            v += db + '\n'
            
    v += '\n\\end{array} \\right]'
    return v

# Data
# 0.5 MVA 20/0.4 kV Dyn7 ASEA

Sn = 0.5 # MVA
Uhv = 20  #kV
Ulv = 0.4 # kV

Usc = 6  # %
pcu = 6 # kW
pfe = 1.4  # kW
I0 = 0.28  # %

GRhv = 0.5
GXhv = 0.5

alpha = 1

beta = 1

################################

Zn_hv = Uhv**2 / Sn
Zn_lv = Ulv**2 / Sn
zsc = Usc / 100
rsc = (pcu/1000) / Sn
xsc = 1 / sqrt(zsc**2 - rsc**2)

rcu_hv = rsc * GRhv
rcu_lv = rsc * (1 - GRhv)
xs_hv = xsc * GXhv
xs_lv = xsc * (1 - GXhv)

rfe = Sn / (pfe/1000)

zm = 1 / (I0/100)

xm = 1 / sqrt((1/(zm**2)) - (1/(rfe**2)))

z_series = rsc + 1j * xsc

y_series = 1 / z_series

zl = rfe + 1j * xm

y_shunt = 1 / zl

yt = 3 / z_series

print('$$Zn_{hv} = ', f(Zn_hv), '\quad \Omega$$')
print('$$Zn_{lv} = ', f(Zn_lv), '\quad \Omega$$')
print('$$z_{sc} = ', f(zsc), '\quad p.u.$$')
print('$$r_{sc} = ', f(rsc), '\quad p.u.$$')
print('$$x_{sc} = ', f(xsc), '\quad p.u.$$')

print('$$r_{cu,hv} = ', f(rcu_hv), '\quad p.u.$$')
print('$$r_{cu,lv} = ', f(rcu_lv), '\quad p.u.$$')
print('$$x_{s,hv} = ', f(xs_hv), '\quad p.u.$$')
print('$$x_{s,lv} = ', f(xs_lv), '\quad p.u.$$')

print('$$r_{fe} = ', f(rfe), '\quad p.u.$$')
print('$$z_m = ', f(zm), '\quad p.u.$$')
print('$$x_m = ', f(xm), '\quad p.u.$$')

print('$$z_{series} = ', formatc(z_series), '\quad p.u.$$')
print('$$y_{series} = ', formatc(y_series), '\quad p.u.$$')
print('$$y_{shunt} = ', formatc(y_shunt), '\quad p.u.$$')
print('$$y_t = ', formatc(yt), '\quad p.u.$$')


################################
# Build the branch admittance matrices
################################

YI = [[yt, 0, 0], 
      [0, yt, 0], 
      [0, 0, yt]]
YI = array(YI)

YII = [[2*yt, -yt, -yt], 
       [-yt, 2*yt, -yt], 
       [-yt, -yt, 2*yt]]
YII = array(YII)

YIII = [[-yt, yt, 0], 
        [0, -yt, yt], 
        [yt, 0, -yt]]
YIII = array(YIII)

print('YI, YII, YIII')
print(formatcm(YI))
print(formatcm(YII))
print(formatcm(YIII))

Ypp = (1 / alpha**2) * YII
Yss = (1 / beta**2) * YI
Yps = (1 / alpha / beta) * YIII
Ysp = Yps

print('Ypp, Yss, Yps')
print(formatcm(Ypp))
print(formatcm(Yss))
print(formatcm(Yps))

Ytransformer = hstack((vstack((Ypp, Yps)), 
                       vstack((Ysp, Yss))))

print('Ytransformer')
print(formatcm(Ytransformer))


################################
# Pass to sequence components
################################
ang = 2/3*pi
a = cos(ang) + 1j * sin(ang)
A = [[1, 1, 1], 
      [1, a*a, a], 
      [1, a, a*a]]
A = array(A)
Ai = inv(A)

ypp = Ai.dot(Ypp).dot(A)[1, 1]
yss = Ai.dot(Yss).dot(A)[1, 1]
yps = Ai.dot(Yps).dot(A)[1, 1]
ysp = yps

Ypos_seq = array([[ypp, yps], [ysp, yss]])
print('Y pos seq ')
print(formatcm(Ypos_seq))