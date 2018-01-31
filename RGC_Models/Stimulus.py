#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:59:32 2017

@author: zhouyu
"""
import numpy as np
from RGC_Models.Utils import *
# all output images are scaled so that it is betwen [0,1]
# fixed parameters:
# um->pixel conversion factor
SCALING_FACTOR = 3.3
# size of images is 61pixel x 61 pixel
IMGSIZE = 61
# aperture radius
APERTURE_RAD = 30
# Resource dir
RESOURCEDIR = '../resource/VHsubsample_20160105/'
RAWIMGSIZE = (1024,1536)

def addCircularAperture(img,b):
    r = APERTURE_RAD
    x = np.arange(1,2*r+2)
    y = np.arange(1,2*r+2)
    rr,cc = np.meshgrid(x,y)
    apertureMatrix = np.sqrt((rr-r)**2+(cc-r)**2)<r
    img[apertureMatrix==0] = b
    return img

def getGratingStimulus(abscontra,mu,b,barwidth):
    bd = barwidth/SCALING_FACTOR
    center = IMGSIZE/2+1
    gratings = np.ones((IMGSIZE,1))
    wave = np.arange(1,IMGSIZE+1)
    wave = np.sin((wave-center)*np.pi/bd)
    wave[wave>=0] = 1
    wave[wave<0] = -1
    wave = wave.reshape((1,IMGSIZE))
    gratings = gratings.dot(wave)*abscontra+mu
    return gratings

def getUniformSpot(i):
    return np.ones((IMGSIZE,IMGSIZE))*i

def getRawImage(name):
    # pay special attention to the image orientation
    # matlab and python may differ
    imgname = 'imk'+name+'.iml'
    imgfile = open(RESOURCEDIR+imgname,'rb')
    img = np.fromfile(imgfile,dtype = '>u2')
    img = img.reshape(RAWIMGSIZE)
    return img.astype(float)

def getImagePatch(rawImg, locx, locy):
    # explictively transform x,y
    x = np.round(locy)-1
    y = np.round(locx)-1
    img = rawImg/np.max(rawImg)
    rad = APERTURE_RAD
    return img[x-rad:x+rad+1,y-rad:y+rad+1]


def getSkewedGrating(mu, sz, unitWidth, pos_num, neg_num, pos_center, f1_cancel=False, SigmaC=None):
    '''
    To be compatiable with the simulating purpose (fixed sz)
    '''
    graing_img, _ , _ = getSkewedGratingStimulus(mu, IMGSIZE, unitWidth, pos_num, neg_num, pos_center)
    return grating_img

def getSkewedGratingStimulus(mu, sz, unitWidth, pos_num, neg_num, pos_center, f1_cancel, SigmaC):
    # create skewed distribtuion grating images
    # note everything has been scaled with the SCALING_FACTOR
    sz = int(sz/SCALING_FACTOR)
    SigmaC = SigmaC/SCALING_FACTOR
    center = sz/2 + 1
    wave = np.ones((1, sz))
    pt = center
    unitWidth = int(unitWidth/SCALING_FACTOR)
    while (pt <= sz):
        if (pt == center):
            if (pos_center):
                rad = pos_num/2 * unitWidth
                wave[:,pt-rad:pt+rad+1] = 1
                pt = pt + rad + 1
                pos_tag = 0
            else:
                rad = neg_num/2 * unitWidth
                wave[:,pt-rad:pt+rad+1] = -1
                pt = pt + rad + 1
                pos_tag = 1
        elif(pos_tag):
            dia = pos_num * unitWidth
            wave[:,pt:min(sz, pt+dia)] = 1
            wave[:,max(0, 2*center-pt-dia):2*center-pt]=1
            pt = pt + dia
            pos_tag = 0
        elif(~pos_tag):
            dia = neg_num * unitWidth
            wave[:,pt:min(sz, pt+dia)] = -1
            wave[:,max(0, 2*center-pt-dia):2*center-pt] = -1
            pt = pt + dia
            pos_tag = 1
    gratingMatrix = np.ones((sz,1)).dot(wave)
    if not f1_cancel:
        if (np.sum(gratingMatrix>0) > np.sum(gratingMatrix<0)):
            neg_contra = -0.9
            gratingMatrix[gratingMatrix<0] = neg_contra
            pos_contra = -np.sum(gratingMatrix[gratingMatrix<0])/np.sum(gratingMatrix>0)
            gratingMatrix[gratingMatrix>0] = pos_contra
        else:
            pos_contra = 0.9
            gratingMatrix[gratingMatrix>0] = pos_contra
            neg_contra = -np.sum(gratingMatrix[gratingMatrix>0])/np.sum(gratingMatrix<0)
            gratingMatrix[gratingMatrix<0] = neg_contra
    else:
        r = sz/2
        x = np.arange(1, sz+1)
        y = np.arange(1, sz+1)
        rr, cc = np.meshgrid(x, y)
        apertureMatrix = np.sqrt((rr-r)**2+(cc-r)**2)<r
        RF = matlab_style_gauss2D(shape=(sz, sz), sigma=SigmaC)
        weightingFxn = apertureMatrix*RF
        weightingFxn = weightingFxn/np.sum(weightingFxn)
        weightedGratings = gratingMatrix*weightingFxn
        if np.sum(gratingMatrix>0) > np.sum(gratingMatrix<0):
            neg_contra = -0.9
            gratingMatrix[gratingMatrix<0] = neg_contra
            pos_contra = neg_contra * np.sum(weightedGratings[gratingMatrix<0]) / np.sum(weightedGratings[gratingMatrix>0])
            gratingMatrix[gratingMatrix>0] = pos_contra
        else:
            pos_contra = 0.9
            gratingMatrix[gratingMatrix>0] = pos_contra
            neg_contra = pos_contra*np.sum(weightedGratings[gratingMatrix>0]) / np.sum(weightedGratings[gratingMatrix<0])
            gratingMatrix[gratingMatrix<0] = neg_contra
    return (gratingMatrix*mu + mu, pos_contra, neg_contra )

def getSkewedSplitField(background, pos_contrast, neg_contrast, SigmaC, sz=IMGSIZE):
    '''
    create stimulus of skewed stimulus
    the skewness is determined by the balance of pos_contrast and neg_contrast
    to cancel F1
    '''
    r = sz/2
    x = np.arange(1, sz+1)
    y = np.arange(1, sz+1)
    rr, cc = np.meshgrid(x, y)
    apertureMatrix = np.sqrt((rr-r)**2+(cc-r)**2)<r
    RF = matlab_style_gauss2D(shape=(sz, sz), sigma=SigmaC)
    weightingFxn = apertureMatrix*RF
    weightingFxn = weightingFxn/np.sum(weightingFxn)
    # search for the minimum contrast difference point - delta_pos
    delta_pos = 0
    abs_diff = np.absolute((pos_contrast - neg_contrast)*sz*sz)
    for i in xrange(sz/2, sz):
        mat = np.zeros((sz, sz))
        mat[:, :i] = min(abs(pos_contrast), abs(neg_contrast))
        mat[:, i:] = max(abs(pos_contrast), abs(neg_contrast))
        mat = mat*weightingFxn
        diff = abs(np.sum(mat[:,:i]) - np.sum(mat[:, i:]))
        if diff < abs_diff:
            abs_diff = diff
            delta_pos = i
    # create image
    split_img = np.zeros((sz, sz))
    if abs(pos_contrast) < abs(neg_contrast):
        sign = 1;
    else:
        sign = -1;

    split_img[:, :delta_pos] = sign*min(abs(pos_contrast), abs(neg_contrast))
    split_img[:, delta_pos:] = -sign*max(abs(pos_contrast), abs(neg_contrast))
    #print np.min(split_img), np.max(split_img)
    split_img = background*split_img + background
    return addCircularAperture(split_img,background)
