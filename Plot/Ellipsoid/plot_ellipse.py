# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:39:44 2018

@author: liangliang.pan
"""

import numpy as np
import mayavi.mlab as mlab

x, y, z = np.mgrid[-10:10:100j, -10:10:100j, -10:10:100j]
#s = np.sin(x*y*z)/(x*y*z)
s = x**2/16 + y**2/9 + z**2/6.25

src = mlab.pipeline.scalar_field(s)
mlab.pipeline.iso_surface(src, contours=[1], opacity=0.3)
#mlab.pipeline.iso_surface(src, contours=[s.max()-0.1*s.ptp(), ],)

mlab.show()