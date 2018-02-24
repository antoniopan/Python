# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 10:40:52 2018

@author: liangliang.pan
"""

import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
    
    
    fig = plt.figure()    
    ax = fig.add_subplot(111, projection='3d')
    
    
    # plot the ellipsoid
    thetaDim = 401
    phiDim = 201
    theta = np.linspace(0, 2*np.pi, thetaDim)
    phi = np.linspace(0,np.pi,phiDim)
    
    X = a * np.outer(np.cos(theta), np.sin(phi))
    Y = b * np.outer(np.sin(theta), np.sin(phi))
    Z = c * np.outer(np.ones(thetaDim), np.cos(phi))
    
    X, Y, Z = np.mgrid[]
    
    matR = np.array([Ex, Ey, Ez]).transpose()
    M = np.array([X.reshape(X.size), Y.reshape(Y.size), Z.reshape(Z.size)])
    
    MAfterTransform = np.dot(matR, M).transpose() + E0
    M = MAfterTransform.transpose()
    X = M[0].reshape(thetaDim, phiDim)
    Y = M[1].reshape(thetaDim, phiDim)
    Z = M[2].reshape(thetaDim, phiDim)
    
    ax.plot_surface(X, Y, Z, color='b', cmap=cm.gnuplot)
    
    
    #plot the plane    
    l = np.max([a, b, c])
    
    axisDim = 201
    Y, X = np.mgrid[-l:l:np.complex(axisDim), -l:l:np.complex(axisDim)]
    Z = np.zeros((axisDim, axisDim))

    angle = 0
    normal = np.cross(Px, Py)
    axis = np.cross(np.array([.0, .0, 1.0]), normal)
    if 0 != np.linalg.norm(axis):
        angle = np.arccos(np.dot(np.array([.0, .0, 1.0]), normal))
        axis = axis / np.linalg.norm(axis)
    cosA = np.cos(angle)
    sinA = np.sin(angle)
    matR = np.array([[cosA+(1-cosA)*np.square(axis[0]),(1-cosA)*axis[0]*axis[1]-sinA*axis[2],(1-cosA)*axis[0]*axis[2]+sinA*axis[1]],
                      [(1-cosA)*axis[1]*axis[0]+sinA*axis[2],cosA+(1-cosA)*np.square(axis[1]),(1-cosA)*axis[1]*axis[2]-sinA*axis[0]],
                      [(1-cosA)*axis[2]*axis[0]-sinA*axis[1],(1-cosA)*axis[2]*axis[1]+sinA*axis[0],cosA+(1-cosA)*np.square(axis[2])]])
    
    
    M = np.array([X.reshape(X.size), Y.reshape(Y.size), Z.reshape(Z.size)])
    MAfterTransform = np.dot(matR, M).transpose() + P0
    M = MAfterTransform.transpose()
    X = M[0].reshape(axisDim, axisDim)
    Y = M[1].reshape(axisDim, axisDim)
    Z = M[2].reshape(axisDim, axisDim)
    
    ax.plot_surface(X, Y, Z, color = 'b')
    
    ax.set_xlabel('X')
    ax.set_xlim(-2*l, 2*l)
    ax.set_ylabel('Y')
    ax.set_ylim(-2*l, 2*l)
    ax.set_zlabel('Z')
    ax.set_zlim(-2*l, 2*l)
    
    plt.show()
    

def PlotEllipsoidIntersectPlane(a, b, c, Ex, Ey, Ez, E0, Px, Py, P0):
    Pz = CalcNormalizedVector(np.cross(Px, Py))
    
    matB = np.array([Px, Py, Pz]).transpose()
    matF = np.array([Ex, Ey, Ez]).transpose()
    matR = matF * np.linalg.inv(matB)
    vecT = E0 - P0
    '''
    x' = R00*x + R01*y + R02*z + T0
    y' = R10*x + R11*y + R12*z + T1
    z' = R20*x + R21*y + R22*z + T2
    
    |x'^2/a^2 + y'^2/b^2 + z'^2/c^2 = 1
    |z = 0
    
    A11*x^2 + 2A12*x*y + A22*y^2 + 2B1*x + 2B2*y + C = 0
    '''
    
    A11 = np.square(matR[0][0]/a) + np.square(matR[1][0]/b) + np.square(matR[2][0]/c)
    A12 = matR[0][0]*matR[0][1]/(a*a) + matR[1][0]*matR[1][1]/(b*b) + matR[2][0]*matR[2][1]/(c*c)
    A22 = np.square(matR[0][1]/a) + np.square(matR[1][1]/b) + np.square(matR[2][1]/c)
    B1 = matR[0][0]*vecT[0]/(a*a) + matR[1][0]*vecT[1]/(b*b) + matR[2][0]*vecT[2]/(c*c)
    B2 = matR[0][1]*vecT[0]/(a*a) + matR[1][1]*vecT[1]/(b*b) + matR[2][1]*vecT[2]/(c*c)
    C = np.square(vecT[0]/a) + np.square(vecT[1]/b) + np.square(vecT[2]/c)
    
    '''
    x' = cos(theta)*x - sin(theta)*y
    y' = sin(theta)*x + cos(theta)*y
    
    D11*x^2 + 2E1*x + D22*y^2 + 2E2*y + C = 0
    '''
    l = np.max([a, b, c]) * 2
    x = y = np.linspace(-l, l, 500)
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, A11*X**2+2*A12*X*Y+A22*Y**2+2*B1*X+2*B2*Y, [-C])
    
    theta = np.arctan(2*A12/(A22-A11)) / 2
    D11 = A11*np.square(np.cos(theta)) + 2*A12*np.cos(theta)*np.sin(theta) + A22*np.square(np.sin(theta))
    D22 = A11*np.square(np.sin(theta)) - 2*A12*np.cos(theta)*np.sin(theta) + A22*np.square(np.cos(theta))
    E1 = B1 * np.cos(theta) + B2 * np.sin(theta)
    E2 = B2 * np.cos(theta) - B1 * np.sin(theta)
    
    plt.axis('scaled')
    plt.show()
    
    return
    
u = np.pi / 3
v = np.pi / 3
R0 = np.array([[np.cos(u), -np.sin(u), 0], [np.sin(u), np.cos(u), 0], [0, 0, 1]])
R1 = np.array([[np.cos(v), 0, np.sin(v)], [0, 1, 0], [-np.sin(v), 0, np.cos(v)]])
R = np.dot(R0, R1)

PlotEllipsoidIntersectPlane(4, 2.2, 1.8, R[0], R[1], R[2], np.array([-1, -1, -1]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0]))