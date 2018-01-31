#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:08:50 2017

@author: zhouyu
"""
from RGC_Models.Utils import *
from RGC_Models.Stimulus import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
IMGSIZE = 61

class RGC(object):
    def __init__(self,cellid):
        self.cellID = cellid

    def getGratings(self,grating):
        cols = ['RecType','currentAbsContrast','currentMeanLevel','currentBarWidth',
                'backgroundIntensity','rfSigmaCenter','MeanRespOn','SEM','stimulusTag']
        temp = grating[cols]
        f_cols = cols[:-3]
        #f_cols = cols.drop(['MeanRespOn','SEM','stimulusTag'])
        temp_grate = temp[temp.stimulusTag == 'grating']
        temp_grate.rename(columns = {'MeanRespOn':'grate_Mean','SEM':'grate_SEM'},inplace = True)
        temp_disc = temp[temp.stimulusTag == 'intensity']
        temp_disc.rename(columns = {'MeanRespOn':'disc_Mean','SEM':'disc_SEM'},inplace = True)
        Gratings = pd.merge(temp_grate,temp_disc,on = f_cols)
        Gratings.drop(['stimulusTag_x','stimulusTag_y'],axis = 1, inplace = True)
        Gratings['abs_BarWidth'] = np.absolute(Gratings['currentBarWidth'])
        Gratings.rename(columns = {'currentMeanLevel':'meanLevel'}, inplace = True)
        self.Gratings = Gratings
        
    def getContrastCurve(self, contrast):
        cols = ['RecType','currentSpotContrast','backgroundIntensity',
                'MeanRespOn','SEM']
        temp = contrast[cols]
        temp.rename(columns = {'backgroundIntensity':'background',
                               'currentSpotContrast':'Contrast'}, inplace = True)
        self.Contrast = temp
        
    def getImgPatch(self, patchResp):
        cols = ['RecType','imageName','currentPatchLocation_1','currentPatchLocation_2','equivalentIntensity',
                'backgroundIntensity','rfSigmaCenter','MeanRespOn','SEM','stimulusTag']
        temp = patchResp[cols]
        f_cols = cols[:-3]
        #f_cols = cols.drop(['MeanRespOn','SEM','stimulusTag'])
        temp_img = temp[temp.stimulusTag == 'image']
        temp_img.rename(columns = {'MeanRespOn':'img_Mean','SEM':'img_SEM'},inplace = True)
        temp_disc = temp[temp.stimulusTag == 'intensity']
        temp_disc.rename(columns = {'MeanRespOn':'disc_Mean','SEM':'disc_SEM'},inplace = True)
        Patches = pd.merge(temp_img,temp_disc,on = f_cols)
        Patches.drop(['stimulusTag_x','stimulusTag_y'],axis = 1, inplace = True)
        self.ImgPatch = Patches
        
    def getTexture(self, textures):
        cols = ['RecType','seed','contrast','background','centerSigma','rotation',
                'MeanRespOn','SEM']
        self.Textures = textures[cols]
    
    def plotGratings(self):
        print "Grating responses measured at the background of :"
        print self.Gratings.backgroundIntensity.unique()
        g = sns.FacetGrid(self.Gratings, row="RecType",col="meanLevel", hue = "abs_BarWidth",sharex=False, sharey=False)
        g = (g.map(plotdiag, "grate_Mean","disc_Mean","grate_SEM","disc_SEM").add_legend())
    
    def plotTextures(self):
        g = sns.FacetGrid(self.Textures, row="centerSigma",col="seed", hue = "RecType")
        g = (g.map(plt.errorbar,"rotation","MeanRespOn","SEM").add_legend())
    
    def plotContrast(self):
        g = sns.FacetGrid(self.Contrast,col = "RecType", col_wrap = 3,hue = "background",
                          sharex = False, sharey = False)
        g = (g.map(plt.errorbar,"Contrast","MeanRespOn","SEM").add_legend())
        
    def plotImgPatches(self):
        print "Img responses measured for images:"
        print self.ImgPatch.imageName.unique()
        g = sns.FacetGrid(self.ImgPatch, row="RecType",col="imageName",sharex=False, sharey=False)
        g = g.map(plotdiag, "img_Mean", "disc_Mean","img_SEM","disc_SEM")
        
    def assembleGratingX(self,rectype = "exc"):
        data = self.Gratings[self.Gratings.RecType ==rectype]
        x = []
        y = []
        bg = []
        for i in xrange(data.shape[0]):
            im_grate = getGratingStimulus(data.iloc[i].currentAbsContrast,data.iloc[i].meanLevel,
                                    data.iloc[i].backgroundIntensity,data.iloc[i].currentBarWidth)
            im_grate = addCircularAperture(im_grate, b= data.iloc[i].backgroundIntensity)
            x.append(im_grate)
            y.append(data.iloc[i].grate_Mean)
            bg.append(data.iloc[i].backgroundIntensity)
        x = np.array(x)
        y = np.array(y)
        y = y.reshape((y.shape[0],1))
        bg = np.array(bg)
        return x,y,bg

    def assembleImgPatchX(self, rectype = "exc"):
        data = self.ImgPatch[self.ImgPatch.RecType == rectype]
        x = []
        y = []
        bg = []
        imgLib = {}
        for name in data.imageName.unique():
            imgname = str(name)
            imgname = '0'*(5-len(imgname))+imgname
            imgLib[name] = getRawImage(imgname)
        for i in xrange(data.shape[0]):
            Img = imgLib[data.iloc[i].imageName]
            im_patch = getImagePatch(Img,data.iloc[i].currentPatchLocation_1,
                                      data.iloc[i].currentPatchLocation_2)
            im_patch = addCircularAperture(im_patch, b= data.iloc[i].backgroundIntensity)
            x.append(im_patch)
            y.append(data.iloc[i].img_Mean)
            bg.append(data.iloc[i].backgroundIntensity)
        x = np.array(x)
        y = np.array(y)
        bg = np.array(bg)
        y = y.reshape((y.shape[0],1))
        return x,y,bg
    
    def getNNTrainingSet(self, rectype = "exc",data = ['grate','img','spot']):
        x_grate,y_grate,bg_grate = [],[],[]
        if 'grate' in data:
            try:
                x_grate,y_grate,bg_grate = self.assembleGratingX(rectype = rectype)
            except AttributeError:
                print "no Grating response data"
        x_img,y_img,bg_img = [],[],[]
        if 'img' in data:    
            try:
                x_img,y_img,bg_img = self.assembleImgPatchX(rectype = rectype)
            except AttributeError:
                print "no Grating response data"
        x_spot, y_spot,bg_spot = [],[],[]
        if 'spot' in data:
            try:
                x_spot,y_spot,bg_spot = self.getScaledSpotX(rectype = rectype)
            except AttributeError:
                print "no Spot responses included"
        X = np.concatenate((x_grate,x_img,x_spot),axis = 0)
        y = np.concatenate((y_grate,y_img,y_spot),axis = 0)
        bg = np.concatenate((bg_grate,bg_img,bg_spot),axis = 0)
        X = X.reshape((X.shape[0],-1))
        return X,y,bg
    
    def getScaledSpotX(self, rectype = "exc"):
        m_img,bg_img,y_img = [],[],[]
        Sigma = 55
        if hasattr(self,'ImgPatch'):
            m_img = self.ImgPatch[self.ImgPatch['RecType'] == rectype].equivalentIntensity.values
            bg_img = self.ImgPatch[self.ImgPatch['RecType'] == rectype].backgroundIntensity.values
            y_img = self.ImgPatch[self.ImgPatch['RecType']==rectype].disc_Mean.values
            Sigma = self.ImgPatch[self.ImgPatch['RecType']==rectype].rfSigmaCenter.iloc[0]
            #print "get Img spots"
        m_grate,bg_grate,y_grate = [],[],[]
        if hasattr(self,'Gratings'):
            data_Grate = self.Gratings[self.Gratings['RecType']==rectype]
            bg_grate = data_Grate.backgroundIntensity.values
            y_grate= data_Grate.disc_Mean.values
            for i in xrange(data_Grate.shape[0]):
                grateImg = getGratingStimulus(data_Grate.iloc[i].currentAbsContrast,data_Grate.iloc[i].meanLevel,
                                    data_Grate.iloc[i].backgroundIntensity,data_Grate.iloc[i].currentBarWidth)
                # need to compute the actrual equivalent mean intensity
                m_grate.append(grate2EquiMean(grateImg,data_Grate.iloc[i].rfSigmaCenter))
            m_grate = np.array(m_grate)
            Sigma = data_Grate.rfSigmaCenter.iloc[0]
            #print "get Grate spots"
        m_contra, bg_contra, c_contra = [],[],[]
        if hasattr(self,'Contrast'):
            c_contra = self.Contrast[self.Contrast['RecType']==rectype].Contrast.values
            bg_contra = self.Contrast[self.Contrast['RecType']==rectype].background.values
            y_contra = self.Contrast[self.Contrast['RecType']==rectype].MeanRespOn.values
            m_contra = c_contra*bg_contra+bg_contra
            m_contra[m_contra>=1] = 1
            # need to linear scale the contrast spot responses, because they are measured 
            # with 300um aperture
            # while others are measured with 200um aperture
            y_contra =convertRFArea(y_contra,Sigma) 
            #print "get Contrast spots"
        m = np.concatenate((m_img,m_grate,m_contra),axis = 0)
        bg = np.concatenate((bg_img,bg_grate,bg_contra),axis = 0)
        y = np.concatenate((y_img,y_grate,y_contra),axis = 0)
        y = y.reshape((y.shape[0],1))
        X = np.zeros((m.shape[0],IMGSIZE,IMGSIZE))
        for i in xrange(m.shape[0]):
            X[i] = getUniformSpot(m[i])
            X[i] = addCircularAperture(X[i],bg[i])
        #print X.shape
        return X, y, bg
    
    def getEquivalentSpot(self, rectype = "exc",stitype = "Gratings"):
        m,bg,y = [],[],[]
        if stitype == "Gratings":
            if hasattr(self,'Gratings'):
                data_Grate = self.Gratings[self.Gratings['RecType']==rectype]
                bg = data_Grate.backgroundIntensity.values
                y= data_Grate.disc_Mean.values
                for i in xrange(data_Grate.shape[0]):
                    grateImg = getGratingStimulus(data_Grate.iloc[i].currentAbsContrast,data_Grate.iloc[i].meanLevel,
                                    data_Grate.iloc[i].backgroundIntensity,data_Grate.iloc[i].currentBarWidth)
                # need to compute the actrual equivalent mean intensity
                    m.append(grate2EquiMean(grateImg,data_Grate.iloc[i].rfSigmaCenter))
                m = np.array(m)
                Sigma = data_Grate.rfSigmaCenter.iloc[0]
            else:
                print " no Grating or equivalent discs found"
                return
        if stitype == "Images":
            if hasattr(self,'ImgPatch'):
                m = self.ImgPatch[self.ImgPatch['RecType'] == rectype].equivalentIntensity.values
                bg = self.ImgPatch[self.ImgPatch['RecType'] == rectype].backgroundIntensity.values
                y = self.ImgPatch[self.ImgPatch['RecType']==rectype].disc_Mean.values
                Sigma = self.ImgPatch[self.ImgPatch['RecType']==rectype].rfSigmaCenter.iloc[0] 
            else:
                print "no Iamge or equivalent discs found"
                return
        X = np.zeros((m.shape[0],IMGSIZE,IMGSIZE))
        for i in xrange(m.shape[0]):
            X[i] = getUniformSpot(m[i])
            X[i] = addCircularAperture(X[i],bg[i])
        return X, y, bg
            
            
            
                
        
        
        
            
        