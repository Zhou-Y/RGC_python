#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:53:50 2017

@author: zhouyu
"""

from RGC_Models.Utils import *
from RGC import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
SCALING_FACTOR = 3.3

class subunitBiophys(object):

    def __init__(self,cellid):
        self.cellID = cellid


    def fitExcContrast(self,**kwargs):
        contrast = kwargs['data'][kwargs['data'].RecType =='exc']
        fit_all = False
        if not kwargs.has_key('background'):
            print 'no background specified, fit with all background'
            fit_all = True
        else:
            b = kwargs['background']
            if (np.abs(findNearest(contrast.background,b)-b)<=0.05):
                b = findNearest(contrast.background,b)
            else:
                print 'no background matching, fit with all background'
                fit_all = True
        if (fit_all):
            x = contrast[['Contrast','background']].values
            y = contrast.MeanRespOn.values
            x = np.apply_along_axis(adjustContrast,axis =1,arr = x)
            self.excFitall,_ = curve_fit(contrastNORMCDFFunc2,x,y,p0 = [200,3,0,-0.4])
        else:
            data = contrast[contrast.background == b ]
            x = data.Contrast.values
            y = data.MeanRespOn.values
            if not hasattr(self,'excFit'):
                self.excFit = {}
            self.excFit[b],_ = curve_fit(contrastNormCDFFunc, x,y, p0 =[200,3,0,-0.4])
        r = None
        if (kwargs.has_key('plot') and kwargs['plot'] == 'on'):
            if(fit_all):
                plt.plot(x[:,0],y,'bo',label = 'data')
                plt.plot(x[:,0],contrastNORMCDFFunc2(x,*self.excFitall),'ro',label = 'fit')
                r = r2_score(y, contrastNORMCDFFunc2(x,*self.excFitall))
            else:
                plt.plot(x,y,'bo',label = 'data')
                plt.plot(x,contrastNormCDFFunc(x,*self.excFit[b]),'ro',label = 'fit');
                r = r2_score(y, contrastNormCDFFunc(x,*self.excFit[b]))
            plt.legend()
            plt.xlabel('Contrast Level')
            plt.ylabel('Response/pA')
        if (kwargs.has_key('reportr') and kwargs['reportr']=='on'):
            return r

    def predExcContrast(self,b,c):
        # if the background is close to available background(<=0.05)
        # fit with the single background model
        # otherwise fit with overall model
        if hasattr(self,'excFit'):
            nb = findNearestBg(self.excFit.keys(),b)
            if (np.abs(nb-b)<=0.05):
                return contrastNormCDFFunc(c,*self.excFit[nb])
        if hasattr(self,'excFitall'):
            return contrastNORMCDFFunc2([c,b],*self.excFitall)
        else:
            print "no availalbe trained model"

    def fitInhContrast(self,**kwargs):
        # similiar to fitting the excitatory currents
        # need additional processing to treat the function piece-wise
        contrast = kwargs['data'][kwargs['data'].RecType =='inh']
        fit_all = False
        if not kwargs.has_key('background'):
            print 'no background specified, fit with all background'
            fit_all = True
        else:
            b = kwargs['background']
            if (np.abs(findNearest(contrast.background,b)-b)<=0.05):
                b = findNearest(contrast.background,b)
            else:
                print 'no background matching, fit with all background'
                fit_all = True
        if (fit_all):
            x = contrast[['Contrast','background']].values
            y = contrast.MeanRespOn.values
            x = np.apply_along_axis(adjustContrast,axis =1,arr = x)
            x_n = x[x[:,0]<=0]
            y_n = y[x[:,0]<=0]
            x_p = x[x[:,0]>0]
            y_p = y[x[:,0]>0]
            self.inhFitall_n,_ = curve_fit(contrastNORMCDFFunc2,x_n,y_n,p0 = [1000,-5,0,-0.4]) # was 200,-5,0,-0.4
            self.inhFitall_p,_ = curve_fit(contrastNORMCDFFunc2,x_p,y_p,p0 = [200,3,0,-0.4]) # was 200,3,0,-0.4
        else:
            data = contrast[contrast.background == b ]
            x = data.Contrast.values
            y = data.MeanRespOn.values
            x_n = x[x<=0]
            y_n = y[x<=0]
            x_p = x[x>0]
            y_p = y[x>0]
            if not hasattr(self,'inhFit_n'):
                self.inhFit_n = {}
            if not hasattr(self,'inhFit_p'):
                self.inhFit_p = {}
            self.inhFit_n[b],_ = curve_fit(contrastNormCDFFunc, x_n,y_n, p0 =[6000,-200,5,-10]) # p0[1] can be set as either -2 or -5
            self.inhFit_p[b],_ = curve_fit(contrastNormCDFFunc, x_p,y_p, p0 =[200,3,0,-0.4])
        if (kwargs.has_key('plot') and kwargs['plot'] == 'on'):
            if(fit_all):
                plt.plot(x[:,0],y,'bo',label = 'data')
                y_pred = inhContrast(x,contrastNORMCDFFunc2,self.inhFitall_n,self.inhFitall_p)
                plt.plot(x[:,0],y_pred,'ro',label = 'fit')
            else:
                plt.plot(x,y,'bo',label = 'data')
                y_pred = inhContrast(x,contrastNormCDFFunc,self.inhFit_n[b],self.inhFit_p[b])
                plt.plot(x,y_pred,'ro',label = 'fit')
            plt.legend()
            plt.xlabel('Contrast Level')
            plt.ylabel('Response/pA')

        if (kwargs.has_key('reportr') and kwargs['reportr']=='on'):
            return r2_score(y, y_pred)

    def predInhContrast(self, b,c):
        # if the background is close to available background(<=0.05)
        # fit with the single background model
        # otherwise fit with overall model
        if (hasattr(self,'inhFit_n') and hasattr(self,'inhFit_p')):
            k = list(set(self.inhFit_n.keys()+self.inhFit_p.keys()))
            nb = findNearestBg(k,b)
            if (np.abs(nb-b)<=0.05):
                return inhContrast(c,contrastNormCDFFunc,self.inhFit_n[nb],self.inhFit_p[nb])
        if (hasattr(self,'inhFitall_n') and hasattr(self,'inhFitall_p')):
            return inhContrast(c,contrastNORMCDFFunc2,self.inhFitall_n,self.inhFitall_p)
        else:
            print "no availalbe trained model"

    def getSubResp(self,patch,bg,numSub1D,subRad,centerRad,rectype,**kwargs):
        contrastPatch = (patch-bg)/bg
        D = patch.shape[0]
        rad = D/2
        stride = np.round(D/numSub1D)
        xshift = 0
        yshift = 0
        if kwargs.has_key('xshift'):
            xshift = np.round(kwargs['xshift']/SCALING_FACTOR)
            xshift = xshift%stride
        if kwargs.has_key('yshift'):
            yshift = np.round(kwargs['yshift']/SCALING_FACTOR)
            yshift = yshift%stride
        subunitResp = np.ones((patch.shape))
        subrad = np.round(subRad/SCALING_FACTOR)
        subunitFilter = matlab_style_gauss2D(shape = patch.shape,sigma = subrad)
        for i in xrange(numSub1D**2):
            subLocx = i/numSub1D
            subLocy = i%numSub1D
            subLocx = np.round(stride/2+(subLocx*stride)+xshift).astype(int)%D
            subLocy = np.round(stride/2+(subLocy*stride)+yshift).astype(int)%D
            #print (subLocx, subLocy)
            resp = np.ones(patch.shape)*bg
            xoff = -subLocx+rad
            yoff = -subLocy+rad
            for x in xrange(max(0,subLocx-rad),min(D,subLocx+rad)):
                for y in xrange(max(0,subLocy-rad),min(D,subLocy+rad)):
                    resp[x+xoff][y+yoff] = contrastPatch[x][y]
            subresp = np.sum(resp*subunitFilter)
            if (rectype=='exc'):
                subresp = self.predExcContrast(bg,subresp)
            elif (rectype=='inh'):
                subresp = self.predInhContrast(bg, subresp)
            subunitResp[subLocx][subLocy] = subresp
        centerrad = np.round(centerRad/SCALING_FACTOR)
        centerFilter = matlab_style_gauss2D(shape = patch.shape,sigma = centerrad)
        Resp = np.sum(centerFilter*subunitResp)
        """
        if (rectype=='exc'):
            Resp = self.predExcContrast(bg,Resp)
        elif(rectype == 'inh'):
            Resp = self.predInhContrast(bg,Resp)
        """
        return Resp

    def getSubRespModel(self, X, y, bg, numSub1D, subRad, centerRad, rectype, **kwargs):
        resp = np.ones((X.shape[0],1))
        for i in xrange(X.shape[0]):
            resp[i] = self.getSubResp(X[i],bg[i],numSub1D,subRad,centerRad,rectype,**kwargs)
        lregr = linear_model.LinearRegression()
        lregr.fit(resp,y)
        r2 = lregr.score(resp,y)
        print "r2 is: ",r2
        return lregr, r2

    def setSubUnitModel(self, numSub1D, subRad, centerRad, rectype,lmodel):
        m = {}
        m['numSubunits'] = numSub1D
        m['subRadius'] = subRad
        m['centerRadius'] = centerRad
        m['linearAdjust'] = lmodel
        if rectype =='exc':
            self.excModel = m
        elif rectype =='inh':
            self.inhModel = m

    def predResp(self,X,bg,rectype):
        if (rectype == "exc"):
            resp = self.getSubResp(X,bg,self.excModel['numSubunits'],
                                   self.excModel['subRadius'],self.excModel['centerRadius'],"exc")
            return self.excModel['linearAdjust'].predict(resp)
        elif (rectype == "inh"):
            resp = self.getSubResp(X,bg,self.inhModel['numSubunits'],self.inhModel['subRadius'],
                                   self.inhModel['centerRadius'],"inh")
            return self.inhModel['linearAdjust'].predict(resp)


    def fitExcContrastTanh(self, b,plot = 'on'):
        # todo
        pass

    def fitInhContrastTanh(self, b, plot = 'on'):
        # todo
        pass
