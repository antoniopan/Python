# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:32:16 2018

@author: liangliang.pan
"""
import numpy as np
import matplotlib.pyplot as plt

def ResolveEllipseEquation(A, B, C, D, E, F):
    
    '''
    For a given equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    
    rotate the curve clockwise by theta, which equals to rotate the coordinate
    system anti-clockwise by theta
    
    x' = xcos(theta) + ysin(theta)
    y' = ycos(theta) - xsin(theta)
    Ax'^2 + Bx'y' + Cy'^2 + Dx' + Ey' + F = 0
    get
    A'x^2 + B'xy + C'y^2 + D'x + E'y + F = 0
    A' = Acos^2(theta) - Bcos(theta)sin(theta) + Csin^2(theta)
    B' = 2(A-C)cos(theta)sin(theta) + B(cos^2(theta) - sin^2(theta))
       = (A-C)sin(2theta) + Bcos(2theta)
    C' = Asin^2(theta) + Bcos(theta)sin(theta) + Ccos^2(theta)
    D' = Dcos(theta) - Esin(theta)
    E' = Dsin(theta) + Ecos(theta)
    
    B' needs to be zero to put the rotated curve along axes
    theta = arctan(B/(C-A)) / 2
    '''
    theta = (np.arctan(B / (C - A))) * 0.5
    print 'The angle is %f pi.'%(theta/np.pi)
    
    cosT = np.cos(theta)
    sinT = np.sin(theta)
    A1 = A * cosT**2 - B * cosT * sinT + C * sinT**2
    C1 = A * sinT**2 + B * cosT * sinT + C * cosT**2
    D1 = D * cosT - E * sinT
    E1 = D * sinT + E * cosT
    
    X, Y = np.mgrid[-10:10:500j, -10:10:500j]
    plt.contour(X, Y, A*X**2+B*X*Y+C*Y**2+D*X+E*Y+F, [0], colors='black')
    plt.contour(X, Y, A1*X**2+C1*Y**2+D1*X+E1*Y+F, [0], colors='red')
    plt.grid(True)
    
    '''
    For the equation: A'x^2 + D'x + C'y^2 + E'y + F = 0
    A'x^2 + D'x + D'^2/4A'^2 + C'y^2 + E'y + E'^2/4C'^2 + F - D'^2/4A'^2 -E'^2/4C'^2 = 0
    A'(x + D'/2A')^2 + C'(y + E'/2C')^2 = D'^2/4A'^2 + E'^2/4C'^2 - F
    make G = D'^2/4A'^2 + E'^2/4C'^2 - F
    A'(x + D'/2A')^2 + C'(y + E'/2C')^2 = G
    A'/G(x + D'/2A')^2 + C'/G(y + E'/2C')^2 = 1
    (x + D'/2A')^2/(G/A') + (y + E'/2C')^2/(G/C') = 1
    '''
    x0 = -D1/(2*A1)
    y0 = -E1/(2*C1)
    G = D1**2/(4*A1) + E1**2/(4*C1) - F
    AX = np.sqrt(G / A1)
    AY = np.sqrt(G / C1)
    
    #rotate (x0, y0) anti-clockwise by -theta to get the center of the ellipse
    x1 = x0*cosT+y0*sinT
    y1 = y0*cosT-x0*sinT
    print 'The center of the ellipse is (%f, %f).'%(x1, y1)
        
    matR = np.mat([[cosT, sinT],[-sinT, cosT]])
    matAxis = np.mat([[AX,.0,-AX,.0],[.0,AY,.0,-AY]])
    
    Points = np.asarray(matR*matAxis)
    
    plt.plot([x1], [y1], 'ro')
    plt.plot(Points[0]+x1, Points[1]+y1, 'r*')
    plt.axis('scaled')
    plt.show()
    
    return theta, np.array([[cosT, sinT],[-sinT, cosT]]), x1, y1, AX, AY

'''  
theta = np.pi * (1.0 / 6.0)

x0 = -2
y0 = 3
a = 5
b = 3

A11 = (np.cos(theta)/a)**2 + (np.sin(theta)/b)**2
A12 = 2 * np.cos(theta)*np.sin(theta) * (1.0/a**2 - 1.0/b**2)
A22 = (np.sin(theta)/a)**2 + (np.cos(theta)/b)**2
B1 = -2 * ((x0*np.cos(theta)**2+y0*np.cos(theta)*np.sin(theta))/a**2 + (x0*np.sin(theta)**2-y0*np.cos(theta)*np.sin(theta))/b**2)
B2 = -2 * ((y0*np.sin(theta)**2+x0*np.cos(theta)*np.sin(theta))/a**2 + (y0*np.cos(theta)**2-x0*np.cos(theta)*np.sin(theta))/b**2)
C = ((x0*np.cos(theta))**2+2*x0*y0*np.cos(theta)*np.sin(theta)+(y0*np.sin(theta))**2)/a**2 + ((y0*np.cos(theta))**2-2*x0*y0*np.cos(theta)*np.sin(theta)+(x0*np.sin(theta))**2)/b**2-1
'''
'''
X, Y = np.mgrid[-10:10:500j, -10:10:500j]
Theta = np.pi * np.linspace(0, 1, 10)
for i in range(len(Theta)):
    theta = Theta[i]
    A11 = (np.cos(theta)/a)**2 + (np.sin(theta)/b)**2
    A12 = 2 * np.cos(theta)*np.sin(theta) * (1.0/a**2 - 1.0/b**2)
    A22 = (np.sin(theta)/a)**2 + (np.cos(theta)/b)**2
    B1 = -2 * ((x0*np.cos(theta)**2+y0*np.cos(theta)*np.sin(theta))/a**2 + (x0*np.sin(theta)**2-y0*np.cos(theta)*np.sin(theta))/b**2)
    B2 = -2 * ((y0*np.sin(theta)**2+x0*np.cos(theta)*np.sin(theta))/a**2 + (y0*np.cos(theta)**2-x0*np.cos(theta)*np.sin(theta))/b**2)
    C = ((x0*np.cos(theta))**2+2*x0*y0*np.cos(theta)*np.sin(theta)+(y0*np.sin(theta))**2)/a**2 + ((y0*np.cos(theta))**2-2*x0*y0*np.cos(theta)*np.sin(theta)+(x0*np.sin(theta))**2)/b**2-1

    plt.contour(X, Y, A11*X**2+A12*X*Y+A22*Y**2+B1*X+B2*Y+C, [0])
'''
#ResolveEllipseEquation(A11, A12, A22, B1, B2, C)