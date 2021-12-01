#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
esthawkes.py

Purpose:
    Generate and estimate Hawkes Process data

Version:
    1       First start
    2       Estimation implemented, with analytical integrated intensity

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
    dA= dAf*dD
    if (len(vP) == 3):
        return (dL, dA, dD)
    dE= vP[3]
    return (dL, dA, dD, dE)

###########################################################
### vPhi= DecayH(vX, vP)
def DecayH(vX, vP, right= False):
    """
    Purpose:
        Get the decay function at tau= vX

    Inputs:
        vX      iX vector, locations for evaluation
        vP      iP vector, parameters
        right   (optional, default= False) boolean, if True include x=0 (effectively using limit from right instead of left)

    Return vector:
        vPhi    iX vector of decay values
    """
    vX = np.array(vX)
    vPhi= np.zeros_like(vX)
    vI= vX >= 0 if (right) else vX > 0
    if (len(vP) == 3):
        print ('warning: exponentional kernel not thoroughly implemented...')
        (dL, dA, dD)= GetPar(vP)
        vPhi[vI]= dA * np.exp(-dD * vX[vI])
    else:
        (dL, dA, dD, dE)= GetPar(vP)
        vPhi[vI]= dA / (vX[vI] + dD)**(dE+1)
    return vPhi

###########################################################
### vIPhi= IntDecayH(vX, vP)
def IntDecayH(vA, vB, vP):
    (dL, dA, dD, dE)= GetPar(vP)
    vIPhi= np.zeros_like(vA)
    vI= vA >= 0
    vIPhi[vI]= dA/dE * (1.0/(vA[vI] + dD)**dE - 1.0/(vB[vI] + dD)**dE)

    vI= (vA < 0) * (vB > 0)
    vIPhi[vI]= dA/dE * (1.0/(0 + dD)**dE - 1.0/(vB[vI] + dD)**dE)

    return vIPhi

###########################################################
### vLambda= IntensityHawkes(vt, vP, vT)
def IntensityHawkes(vt, vP, vT, right= False):
    """
    Purpose:
        Calculate the Hawkes intensity, given time t and (earlier) events at times vT

    Inputs:
        vt      vector, current times
        vP      iP vector of parameters
        vT      iN vector of earlier events
        right   (optional, default= False) boolean, if True include x=0 (effectively using limit from right instead of left)

    Return value:
        vLambda vector, intensities
    """
    (dL, dA, dD, dE)= GetPar(vP)
    vLambda= np.zeros_like(vt) + dL
    for dT in vT:
        vLambda= vLambda + DecayH(vt-dT, vP, right)

    return vLambda

###########################################################
### vIL= IntIntensityHawkes(vAB, vP, vT)
def IntIntensityHawkes(vAB, vP, vT):
    """
    Purpose:
        Calculate the integrated Hawkes intensity, given bounds [a, b] and events at times vT

    Inputs:
        vAB     iB vector, successive bounds a and b (should NOT contain any events in the interior)
        vP      iP vector of parameters
        vT      iN vector of earlier events

    Return value:
        vIL     iB-1 vector, integrated intensities
    """
    iB= len(vAB)
    (dL, dA, dD, dE)= GetPar(vP)
    vIL= np.zeros(iB-1)
    for b in range(1, iB):
        vA= vAB[b-1]-vT
        vB= vAB[b]-vT
        vIL[b-1]= (vAB[b]-vAB[b-1])*dL + np.sum(IntDecayH(vA, vB, vP))

    return vIL

###########################################################
### vT= GenHawkes(vP, iN, iT)
def GenHawkes(vP, iN, iT):
    """
    Purpose:
        Generate observations from Hawkes process using exponential or power law kernel

    Inputs:
        vP      3 or 4-element vector of parameters
        iN      integer, max number of observations to sample
        iT      integer, max time span to sample for
        # dTau    (optional, default= 0.01) double, time step in which to sample

    Return value:
        # dfH     dataframe, with hawkes process data, with columns 'root', 'offspring', 'generation', 'time', 'rate'
        vT      iN* vector of times of events, where (iN* <= iN, vT[-1] <= iT)
    """
    vT= []
    dt= 0
    dLambda= IntensityHawkes(dt, vP, vT)
    while (len(vT) < iN) and (dt < iT):
        dLambda0= IntensityHawkes(dt, vP, vT, right= True) # why right limit here? paper shows sum with t>T_i
        dWait= st.expon.rvs(scale= 1/dLambda0)
        dt= dt + dWait
        dLambda= IntensityHawkes(dt, vP, vT)
        dS= np.random.rand()
        if (dS < dLambda/dLambda0):
            vT.append(dt)

    return np.array(vT)

###########################################################
### LnLHawkes(vP, vT)
def LnLHawkes(vP, vT, iD= -1):
    """
    Purpose:
        Compute the loglikelihood of the Hawkes contributions

    Inputs:
        vP      iP vector of parameters
        vT      iN vector of timings

    Return value:
        vLL     iN vector of loglikelihoods
    """
    vLL= np.log(IntensityHawkes(vT, vP, vT, right= False))

    iN= len(vT)
    dL= 0
    # i= 0
    for i in range(iN):
        # ### Testing
        # vAB= [0, vT[i]] if (i == 0) else [vT[i-1], vT[i]]
        # dILa= IntIntensityHawkes(vAB, vP, vT)
        #
        # vQ= si.quadrature(IntensityHawkes, dL, vT[i], args=(vP, vT))
        # dILq= vQ[0]
        #
        # iD0= 100
        # dD= (vT[i] - dL)/iD0
        # vt= np.arange(dL, vT[i]+dD, dD)
        # vInt= IntensityHawkes(vt, vP, vT)
        # dILi= dD*vInt.sum()
        # print ('%i: a=%g, q=%g, i=%g' % (i, dILa, dILq, dILi))
        if (iD < 0):
            vAB= [0, vT[i]] if (i == 0) else [vT[i-1], vT[i]]
            vLL[i]-= IntIntensityHawkes(vAB, vP, vT)
        elif (iD == 0):
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

    # Create lambda function returning NEGATIVE AVERAGE LL, as function of vPTr only
    # AvgNLnLHawkes= lambda vP: -np.mean(LnLHawkes(vP, vT, iD), axis=0)
    AvgNLnLHawkesTr= lambda vPTr: -np.mean(LnLHawkes(TransBackPar(vPTr, adtPar), vT, iD), axis=0)

    vP0Tr= TransPar(vP0, adtPar)
    dLL0= -iN*AvgNLnLHawkesTr(vP0Tr)
    print ("Initial LL= %g using D= %i \nvP0=" % (dLL0, iD), vP0)

    res= opt.minimize(AvgNLnLHawkesTr, vP0Tr, method="BFGS")
    # vP0Tr= np.copy(res.x)
    # print ('First trial:', res.message)
    #
    # iD= 0
    # res= opt.minimize(AvgNLnLHawkesTr, vP0Tr, method="BFGS")
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
    # dLL= -iN*AvgNLnLHawkesTr(vPTr)


    dfRes= pd.DataFrame(vP0, index=[dtPar['name'] for dtPar in adtPar], columns=['p0'])
    dfRes['p']= vP
    dfRes['s']= vShd
    # dfRes.loc['LL/fev', ['p', 's']]= [dLL, res.nfev]

    print ("\nBFGS results in %s\nLL= %g, n-eval= %i" % (sMess, dLL, res.nfev), "\nPars: ")
    print (dfRes)
    print ("\nJacobian of transformation, mG=\n", mG, "\nmS2hd, delta method:\n", mS2hd)

    return (dfRes, dLL, sMess)

###########################################################
### PlotHawkes(vP, vT, sOut)
def PlotHawkes(vP, vT, out= None):
    iN= len(vT)
    iT= np.ceil(max(vT))

    vt= np.arange(0, iT, iT/1000)               # Get a grid
    vt= np.unique(np.concatenate((vt, vT)))     # Add observations to the grid
    vIt= IntensityHawkes(vt, vP, vT, right= True)
    vITr= IntensityHawkes(vT, vP, vT, right= True)

    plt.figure()
    plt.plot(vt, vIt)
    plt.plot(vT, vITr, 'o')
    if (out is not None):
        plt.savefig(out)
    plt.show()

###########################################################
### main
def main():
    # Magic numbers
    vP= [.1, .4, .5, .5]     # Parameters lambda, alpha-frac, delta, eta, where alpha= alpha-frac*delta
    iN= 200
    iT= 10000
    iSeed= 1234

    # Initialisation
    np.random.seed(iSeed)

    # Estimation
    vT= GenHawkes(vP, iN, iT)

    # Output
    PlotHawkes(vP, vT)

    print ('Trying out different likelihood approaches...')
    TrackTime('D-analytical')
    vLL= LnLHawkes(vP, vT, iD= -1)
    print ('LL=', np.sum(vLL))
    TrackTime('D-quadrature')
    vLL= LnLHawkes(vP, vT, iD= 0)
    print ('LL=', np.sum(vLL))
    TrackTime('D-1000')
    vLL= LnLHawkes(vP, vT, iD= 1000)
    print ('LL=', np.sum(vLL))
    TrackReport()

    vP0= vP.copy()
    EstimateHawkes(vT, vP0, iD= -1)



###########################################################
### start main
if __name__ == "__main__":
    main()
