# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 10:40:52 2018

@author: liangliang.pan
"""

import numpy as np
import mayavi.mlab as mlab
import resolve_ellipse as re

def CheckNormalized(x):
    return np.abs(np.sum(np.square(x)) - 1) < 1e-7
    
def CheckPerpendicular(x, y):
    return np.abs(np.sum(x * y)) < 1e-7
    
def CalcNormalizedVector(x):
    return x / np.linalg.norm(x)

def DrawArbitraryEllipsoid(a, b, c, Ex, Ey, Ez, E0, Px, Py, P0):
    """
    Args:
        a: length of the half axis along x direction
        b: length of the half axis along y direction
        c: length of the half axis along z direction
        Ex: normalized vector representing the x direction of the ellipsoid
        Ey: normalized vector representing the y direction of the ellipsoid
        Ez: normalized vector representing the z direction of the ellipsoid
        E0: center of the ellipsoid
        Px: normalized vector representing the x direction of the plane
        Py: normalized vector representing the y direction of the plane
        P0: origin of the plane
    """
    
    # check the validity of the arguments
    if a <= 0 or b <= 0 or c <= 0:
        return
        
    if (not CheckNormalized(Ex)) or (not CheckNormalized(Ey)) or (not CheckNormalized(Ez)):
        return
        
    if not CheckPerpendicular(Ex, Ey) or not CheckPerpendicular(Ey, Ez) or not CheckPerpendicular(Ez, Ex):
        return
        
    if (not CheckNormalized(Px)) or (not CheckNormalized(Py)):
        return
        
    if (not CheckPerpendicular(Px, Py)):
        return
        
    print 'x axis is %s'%Ex
    print 'y axis is %s'%Ey
    print 'z axis is %s'%Ez
    
    l = np.max([a, b, c]) + 1
    
    # plot the ellipsoid
    Dim = 101
    
    X, Y, Z = np.mgrid[-l:l:np.complex(0,Dim), -l:l:np.complex(0,Dim), -l:l:np.complex(0,Dim)]
    
    #plot the frame
    F = np.maximum(np.maximum(np.abs(X), np.abs(Y)), np.abs(Z))
    src = mlab.pipeline.scalar_field(F)
    mlab.pipeline.iso_surface(src, contours=[np.max(F)], opacity=0.1)
    
    '''
    E = (X+1)**2/a**2 + (Y+1)**2/b**2 + (Z+1)**2/c**2
    src = mlab.pipeline.scalar_field(E)
    mlab.pipeline.iso_surface(src, contours=[1], opacity=0.3)
    '''
    
    matR = np.mat([Ex, Ey, Ez]).transpose()
    M = np.mat([X.reshape(X.size), Y.reshape(Y.size), Z.reshape(Z.size)])
    
    MAfterTransform = np.asarray((matR*M).transpose()) - E0
    M = MAfterTransform.transpose()
    X = M[0].reshape(Dim, Dim, Dim)
    Y = M[1].reshape(Dim, Dim, Dim)
    Z = M[2].reshape(Dim, Dim, Dim)
    
    E = X**2/a**2 + Y**2/b**2 + Z**2/c**2
    
    src = mlab.pipeline.scalar_field(E)
    mlab.pipeline.iso_surface(src, contours=[1], opacity=0.3)
    #mlab.pipeline.iso_surface(src, contours=[s.max()-0.1*s.ptp(), ],)
    
    #plot the plane    
    vecN = CalcNormalizedVector(np.cross(Px, Py))
    
    X, Y, Z = np.mgrid[-l:l:np.complex(0,Dim), -l:l:np.complex(0,Dim), -l:l:np.complex(0,Dim)]
    P = vecN[0] * X + vecN[1] * Y + vecN[2] * Z - np.dot(vecN, P0)
    
    src = mlab.pipeline.scalar_field(P)
    mlab.pipeline.iso_surface(src, contours=[0])
    
    mlab.show()
    
    return

def PlotEllipsoidIntersectPlane(a, b, c, Ex, Ey, Ez, E0, Px, Py, P0):
    Pz = CalcNormalizedVector(np.cross(Px, Py))
    
    matB = np.mat([np.append(P0,1),np.append(P0+Px,1),np.append(P0+Py,1),np.append(P0+Pz,1)]).transpose()
    matF = np.mat([np.append(E0,1),np.append(E0+Ex,1),np.append(E0+Ey,1),np.append(E0+Ez,1)]).transpose()
    matR = np.asarray(matB * np.linalg.inv(matF))
    '''
    x' = R00*x + R01*y + R02*z + T0
    y' = R10*x + R11*y + R12*z + T1
    z' = R20*x + R21*y + R22*z + T2
    
    |x'^2/a^2 + y'^2/b^2 + z'^2/c^2 = 1
    |z = 0
    
    A11*x^2 + 2A12*x*y + A22*y^2 + 2B1*x + 2B2*y + C = 0
    '''
    # use the contour function to draw the standard ellipse according to the equation
    A11 = (matR[0][0]/a)**2 + (matR[1][0]/b)**2 + (matR[2][0]/c)**2
    A12 = matR[0][0]*matR[0][1]/(a*a) + matR[1][0]*matR[1][1]/(b*b) + matR[2][0]*matR[2][1]/(c*c)
    A22 = (matR[0][1]/a)**2 + (matR[1][1]/b)**2 + (matR[2][1]/c)**2
    B1 = matR[0][0]*matR[0][3]/(a*a) + matR[1][0]*matR[1][3]/(b*b) + matR[2][0]*matR[2][3]/(c*c)
    B2 = matR[0][1]*matR[0][3]/(a*a) + matR[1][1]*matR[1][3]/(b*b) + matR[2][1]*matR[2][3]/(c*c)
    C = (matR[0][3]/a)**2 + (matR[1][3]/b)**2 + (matR[2][3]/c)**2 - 1
    
    re.ResolveEllipseEquation(A11, 2*A12, A22, 2*B1, 2*B2, C)
    
    return
    
u = np.pi * (1.0/6)
v = np.pi * (.0/60)
R0 = np.mat([[np.cos(u), -np.sin(u), 0], [np.sin(u), np.cos(u), 0], [0, 0, 1]])
R1 = np.mat([[np.cos(v), 0, np.sin(v)], [0, 1, 0], [-np.sin(v), 0, np.cos(v)]])
R = np.asarray(np.linalg.inv(R0*R1)).transpose()

DrawArbitraryEllipsoid(4, 5, 6, R[0], R[1], R[2], np.array([2, 1, -1]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 1]))
PlotEllipsoidIntersectPlane(4, 5, 6, R[0], R[1], R[2], np.array([2, 1, -1]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 1]))