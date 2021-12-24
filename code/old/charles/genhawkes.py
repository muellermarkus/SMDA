#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genhawkes.py

Purpose:
    Generate Hawkes Process data

Version:
    1       First start

Date:
    2021/11/23

Author:
    Charles Bos
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as si
import scipy.optimize as opt

from lib.transpar import *
from lib.grad import *
from lib.tracktime import *

###########################################################
### (dL, dA, dD, dE)= GetPar(vP)
def GetPar(vP):
    """
    Purpose:
        Extract parameters (stupid function, just fix the order...)

    Inputs:
        vP      iP vector of parameters

    Return value:
        dL, dA, dD, dE      doubles, lambda, alpha, delta, eta
    """
    (dL, dAf, dD)= vP[:3]
    dA= dAf*dD # SHOULD THIS NOT TAKE INTO ACCOUNT ** ETA?
    if (len(vP) == 3):
        return (dL, dA, dD)
    dE= vP[3]
    return (dL, dAf, dD, dE)

###########################################################
### vPhi= DecayH(vX, vP)
def DecayH(vX, vP):
    vX = np.array(vX) # other vX[vI] tries to subset a float when vX is only one element
    vPhi= np.zeros_like(vX)
    vI= vX > 0
    if (len(vP) == 3):
        (dL, dA, dD)= GetPar(vP)
        vPhi[vI]= dA * np.exp(-dD * vX[vI])
    else:
        (dL, dA, dD, dE)= GetPar(vP)
        vPhi[vI]= dA / (vX[vI] + dD)**(dE+1)
    return vPhi

###########################################################
### vLambda= IntensityHawkes(vt, vP, vT)
def IntensityHawkes(vt, vP, vT):
    """
    Purpose:
        Calculate the Hawkes intensity, given time t and (earlier) events at times vT

    Inputs:
        vt      vector, current times THIS IS NOT A VECTOR IN GENHAWKES
        vP      iP vector of parameters
        vT      iN vector of earlier events

    Return value:
        vLambda vector, intensities
    """
    (dL, dA, dD, dE)= GetPar(vP)
    vLambda= np.zeros_like(vt) + dL
    for dT in vT:
        vD= DecayH(vt-dT, vP)
        vLambda= vLambda + DecayH(vt-dT, vP)
        # print (vLambda, vD)

    return vLambda

###########################################################
### GenHawkes(vP, iN, iT, dTau= .01)
def GenHawkes(vP, iN, iT, dTau= .01):
    """
    Purpose:
        Generate observations from Hawkes process using exponential or power law kernel

    Inputs:
        vP      3 or 4-element vector of parameters
        iN      integer, max number of observations to sample
        iT      integer, max time span to sample for
        # dTau    (optional, default= 0.01) double, time step in which to sample

    Return value:
        dfH     dataframe, with hawkes process data, with columns 'root', 'offspring', 'generation', 'time', 'rate'
    """
    vT= []
    dt= 0
    dLambda= IntensityHawkes(dt, vP, vT)
    while (len(vT) < iN) and (dt < iT):
        dLambda0= dLambda
        dWait= st.expon.rvs(scale= 1/dLambda0)
        dt= dt + dWait
        dLambda= IntensityHawkes(dt, vP, vT)
        dS= np.random.rand()
        if (dS < dLambda/dLambda0):
            vT.append(dt)

    return vT

###########################################################
### LnLHawkes(vP, vT)
def LnLHawkes(vP, vT, iD= 0):
    """
    Purpose:
        Compute the loglikelihood of the Hawkes contributions

    Inputs:
        vP      iP vector of parameters
        vT      iN vector of timings

    Return value:
        vLL     iN vector of loglikelihoods
    """
    vLL= np.log(IntensityHawkes(vT, vP, vT))

    iN= len(vT)
    dL= 0
    # i= 0
    for i in range(iN):
        if (iD == 0):
            vQ= si.quadrature(IntensityHawkes, dL, vT[i], args=(vP, vT))
            vLL[i]-= vQ[0]
        else:
            dD= (vT[i] - dL)/iD
            vt= np.arange(dL, vT[i]+dD, dD)
            vInt= IntensityHawkes(vt, vP, vT)
            dQ= dD*vInt.sum()
            vLL[i]-= dQ
        dL= vT[i]

    print ('.', end='')

    return vLL

###########################################################
### (mPSS, dLL, sMess)= EstimateRegr(vY, mX)
def EstimateHawkes(vT, vP0, iD= 0):
    """
    Purpose:
      Estimate the Hawkes model

    Inputs:
      vT        iN vector of data
      vP0       iP vector of initial parameters
      iD        integer, number of steps to take for integrating intensity function. If 0, use full quadrature (slow but more precise)

    Return value:
    """
    iN= len(vT)
    adtPar= [{'name': 'l0', 'p': 1, 'trans': [0, np.inf]},
             {'name': 'alphaf', 'p': .2, 'trans': [0, 1]},
             {'name': 'delta', 'p': .5, 'trans': [0, 2]},
             {'name': 'eta', 'p': .5, 'trans': [0, np.inf]}]

    # Create lambda function returning NEGATIVE AVERAGE LL, as function of vP only
    AvgNLnLHawkes= lambda vP: -np.mean(LnLHawkes(vP, vT, iD), axis=0)
    AvgNLnLHawkesTr= lambda vPTr: -np.mean(LnLHawkes(TransBackPar(vPTr, adtPar), vT, iD), axis=0)

    vP0Tr= TransPar(vP0, adtPar)
    # TrackTime('D0')
    # iD= 0
    # dLL= -iN*AvgNLnLHawkesTr(vP0Tr)
    # print ("Initial LL= ", dLL, "\nvP0=", vP0)
    # TrackTime('D10')
    # iD= 10
    # dLL= -iN*AvgNLnLHawkesTr(vP0Tr)
    # print ("Initial LL= ", dLL, "\nvP0=", vP0)
    # TrackTime('D100')
    # iD= 100
    # dLL= -iN*AvgNLnLHawkesTr(vP0Tr)
    # print ("Initial LL= ", dLL, "\nvP0=", vP0)
    # TrackTime('D1000')
    # iD= 1000
    # dLL= -iN*AvgNLnLHawkesTr(vP0Tr)
    # print ("Initial LL= ", dLL, "\nvP0=", vP0)
    # TrackReport()
    dLL= -iN*AvgNLnLHawkesTr(vP0Tr)
    print ("Initial LL= %g using D= %i \nvP0=" % (dLL, iD), vP0)

    iD= 10
    res= opt.minimize(AvgNLnLHawkesTr, vP0Tr, method="BFGS")
    vP0Tr= np.copy(res.x)
    print ('First trial:', res.message)

    iD= 0
    res= opt.minimize(AvgNLnLHawkesTr, vP0Tr, method="BFGS")
    # res= opt.minimize(AvgNLnLHawkes, vP0, method="BFGS")
    vPTr= np.copy(res.x)
    vP= TransBackPar(vPTr, adtPar)   # Remember to transform back!

    # Get standard errors, using delta method
    mHnTr= hessian_2sided(AvgNLnLHawkesTr, vPTr)
    mHTr= -iN*mHnTr
    mS2Tr= -np.linalg.inv(mHTr)
    mG= jacobian_2sided(TransBackPar, vPTr, adtPar)  # Evaluate jacobian at vPTr
    mS2hd= mG @ mS2Tr @ mG.T                 # Cov(vP)

    # Notation of EQRM course...
    # mA= -hessian_2sided(AvgNLnLHawkesTr, vPTr)
    # mAi= np.linalg.inv(mA)
    # mS2Tr= -mAi/iN
    # mG= jacobian_2sided(TransBackPar, vPTr, adtPar)  # Evaluate jacobian at vPTr
    # mS2hd= mG @ mS2Tr @ mG.T                 # Cov(vP)
    vShd= np.sqrt(np.diag(mS2hd))            # s(vP)

    sMess= res.message
    dLL= -iN*res.fun
    print ("\nBFGS results in ", sMess, "\nPars: ", vP, "\nLL= ", dLL, ", f-eval= ", res.nfev)
    print ("\nJacobian of transformation, mG=\n", mG, "\nmS2hd, delta method:\n", mS2hd)

    dfRes= pd.DataFrame(vP0, index=[dtPar['name'] for dtPar in adtPar], columns=['p0'])


    return (np.vstack((vP, vSh, vShd)).T, dLL, sMess)

###########################################################
### PlotHawkes(vP, vT, sOut)
def PlotHawkes(vP, vT, out= None):
    iN= len(vT)
    iT= np.ceil(max(vT))

    vt= np.arange(0, iT, iT/1000)
    vIt= IntensityHawkes(vt, vP, vT)
    vIT= IntensityHawkes(vT, vP, vT)

    plt.figure()
    plt.plot(vt, vIt)
    plt.plot(vT, vIT, 'o')
    if (out is not None):
        plt.savefig(out)
    plt.show()


###########################################################
### main
def main():
    # Magic numbers
    vP= [1, .4, .5, .5]     # Parameters lambda, alpha-frac, delta, eta, where alpha= alpha-frac*delta
    iN= 200
    iT= 100
    iSeed= 1234

    # Initialisation
    np.random.seed(iSeed)

    # Estimation
    vT= GenHawkes(vP, iN, iT)

    # Output
    PlotHawkes(vP, vT) # YOUR DOTS SHOW AT THE BOTTOMS JUST BEFORE PEAK!
    
    # values are similar to mine but you have double the observations in the same time window. my intensity tendes to fluctuate around some mean whereas your is increasing

    vLL= LnLHawkes(vP, vT)

###########################################################
### start main
if __name__ == "__main__":
    main()
