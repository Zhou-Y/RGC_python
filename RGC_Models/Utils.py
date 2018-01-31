#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:33:00 2017

@author: zhouyu
"""
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def plotdiag(x,y,xer,yer,**kwargs):
    plt.errorbar(x, y, xerr = xer, yerr = yer, fmt = 'o',**kwargs)
    xmin,xmax = plt.xlim()
    ymin,ymax = plt.ylim()
    axmin, axmax = min(xmin,ymin),max(xmax,ymax)
    plt.plot([axmin, axmax],[axmin, axmax],linestyle = ':')
    
def contrastNormCDFFunc(x, a, b, gamma, e):
    resp = norm.cdf(x*b+gamma,loc = 0,scale = 1)
    return a*resp+e

def contrastNORMCDFFunc2(x, a, b, gamma, e):
    if isinstance(x,list):
        resp = norm.cdf(x[0]*x[1]*b+gamma,loc = 0,scale = 1)
    else:
        resp = norm.cdf(x[:,0]*x[:,1]*b+gamma,loc = 0,scale = 1)
    return a*resp+e

def inhContrast(x, func, para1, para2):
    if (len(x.shape)==0):
        if x<=0:
            return func(x,*para1)
        else:
            return func(x,*para2)
    elif (len(x.shape)==1):
        # fixed background
        xn = x[x<=0]
        yn = func(xn,*para1)
        xp = x[x>0]
        yp = func(xp,*para2)
    else:
        xn = x[x[:,0]<=0]
        yn = func(xn,*para2)
        xp = x[x[:,0]>0]
        yp = func(xp,*para2)
    return np.concatenate((yn,yp),axis = 0)
            

def contrastTanh2(x0, a, b):
    x = x0[:,0]*x0[:,1]
    res = np.exp(x)-np.exp(-x)
    res/=np.exp(x)+np.exp(-x)
    return b*np.exp(a*x)*res

def adjustContrast(cond):#  adjust contrast to avoid saturation
    if (cond[1]+cond[0]*cond[1]>1):
        cond[0] = (1-cond[1])/cond[1]
    return cond

def findNearest(bg,b):
    temp = bg.values
    idx = (np.abs(temp - b)).argmin()
    return temp[idx]

def findNearestBg(bg,b):
    temp = np.array(bg)
    idx = (np.abs(temp - b)).argmin()
    return temp[idx]

def matlab_style_gauss2D(shape=(50,50),sigma=2):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    Got from http://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def grate2EquiMean(Img,Sigma):
    RF = matlab_style_gauss2D(Img.shape,Sigma)
    #weightFxn = np.ones(Img.shape)*RF
    #weightFxn = weightFxn/np.sum(weightFxn)
    intensity = np.sum(RF*Img.astype('float'))
    return intensity

def convertRFArea(resp,Sigma,r0=300,r=200):
    c = 2*Sigma**2
    factor = (1-np.exp(-r0**2/c))/(1-np.exp(-r**2/c))
    return resp/factor