#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transpar.py

Purpose:
   Provide library for transforming parameters

Version:
   1       First start
   2       Added TransParSpher and TransBackParSpher, but did not yet link them in fully...

Date:
   2021/10/4

Author:
   Charles Bos
"""
###########################################################

###########################################################
### Imports
import numpy as np
import json

from lib.vech import *

###########################################################
### vPTr= TransPar(vP, adtPar)
def TransPar(vP, adtPar):
    """
    Purpose:
        Transform the parameters of the model

    Inputs:
        vP      iP vector of parameters
        adtPar  iP list of dictionaries, description of parameters containing 'trans', a list of size two with the limits.

    Return value:
        vPTr    iP* vector of transformed, non-fixed parameters
    """
    vPTr= []
    iP= len(vP)
    iF= 0
    aCov= []
    for (p, dtPar) in enumerate(adtPar):
        vTrans= dtPar['trans'] if 'trans' in dtPar.keys() else [-np.inf, np.inf]

        if (isinstance(vTrans[0], str) and (vTrans[0] == 'cov')):
            iCov= vTrans[1]
            while (len(aCov) < iCov):
                aCov.append([])
            aCov[iCov].append(p)
        elif (vTrans[:2] == [-np.inf, np.inf]):        # No transform
            vPTr.append(vP[p])
        elif (vTrans[0] == vTrans[1]):           # Fix parameter
            iF+= 1
        elif (vTrans[0] == -np.inf):
            vPTr.append(np.log(-(vP[p] - vTrans[1])))
        elif (vTrans[1] == np.inf):
            vPTr.append(np.log(vP[p] - vTrans[0]))
        else:
            dC= (vP[p] - vTrans[0])/(vTrans[1] - vTrans[0])
            vPTr.append(np.log(dC / (1-dC)))

    for aiCov in aCov:
        print ('Transforming pars ', aiCov, ' with spherical coordinates')
        vPTr.append(TransParSpher(vP[aiCov]))

    return np.array(vPTr)

###########################################################
### vPSph= TransBackParSpher(vPTrSph)
def TransBackParSpher(vPTrSph):
    """
    Purpose:
        Transform back the covariance parameters of the model, see Pinh96A

    Inputs:
        vPTrSph   iK*(iK+1)/2 vector of spherical coordinates of vech of covariance matrix

    Return value:
        vPSph    iK*(iK+1)/2 vector of coordinates of covariance matrix
    """
    mTh= unvech(vPTrSph)
    iK= mTh.shape[0]
    mL= np.eye(iK)

    vS= np.ones(iK)
    for i in range(1, iK):
        mL[i:,i-1]= np.cos(mTh[i:,i]) * vS[i:]
        vS[i:]= vS[i:]*np.sin(mTh[i:,i])
        mL[i, i]= vS[i]

    # Get standard deviations
    vS= mTh[:,0]

    mR= mL@mL.T
    mS2= np.diag(vS) @ mR @ np.diag(vS)
    # print ("S2, S, mR, mL, mTh: ", mS2, vS, mR, mL, mTh)

    vPSph= vech(mS2)
    return vPSph

###########################################################
### vPTr= TransParSpher(vPSph)
def TransParSpher(vPSph):
    """
    Purpose:
        Transform the covariance parameters of the model, see Pinh96A

    Inputs:
        vPSph   iK*(iK+1)/2 vector of vech of covariance matrix

    Return value:
        vPTr    iK*(iK+1)/2 vector of spherical coordicates
    """
    # mS2= np.array([[1, .8], [.8, 2]])
    # vPSph= vech(mS2)
    mS2= unvech(vPSph)
    vS= np.sqrt(np.diagonal(mS2))
    mR= (mS2 / vS).T / vS  # Correlation matrix
    mL= np.linalg.cholesky(mR)

    iK= mS2.shape[0]
    # Place standard deviations, first row
    mTh= np.zeros_like(mS2)
    mTh[:,0]= vS
    vS= np.ones(iK)
    i= 1
    for i in range(1, iK):
        print ("i= ", i, mTh)
        mTh[i:,i]= np.arccos(mL[i:,i-1] / vS[i:])
        vS[i:]= vS[i:]*np.sin(mTh[i:,i])

    vPTr= vech(mTh)

    # print ("S2, S, mR, mL, mTh: ", mS2, np.sqrt(np.diagonal(mS2)), mR, mL, mTh)
    return vPTr

###########################################################
### vP= TransBackPar(vPTr, adtPar)
def TransBackPar(vPTr, adtPar):
    """
    Purpose:
        Transform back the parameters of the model

    Inputs:
        vPTr      iP vector of parameters
        adtPar  iP list of dictionaries, description of parameters containing 'trans', a list of size two with the limits.

    Return value:
        vPTr    iP* vector of transformed, non-fixed parameters
    """
    vP= []
    p= 0
    for dtPar in adtPar:
        vTrans= dtPar['trans'] if 'trans' in dtPar.keys() else [-np.inf, np.inf]

        if (vTrans[:2] == [-np.inf, np.inf]):        # No transform
            vP.append(vPTr[p])
        elif (vTrans[0] == vTrans[1]):           # Fix parameter
            vP.append(vTrans[0])
            p= p - 1
        elif (vTrans[0] == -np.inf):             # [-inf, up]
            vP.append(-np.exp(vPTr[p]) + vTrans[1])
        elif (vTrans[1] == np.inf):              # [low, inf]
            vP.append(np.exp(vPTr[p]) + vTrans[0])
        else:                                    # [low, up]
            dC= np.exp(vPTr[p]) / (1+np.exp(vPTr[p]))
            vP.append((vTrans[1] - vTrans[0])*dC + vTrans[0])
        p+= 1

    return np.array(vP)


###########################################################
### GetNamesPar(adtPar)
def GetNamesPar(adtPar):
    """
    Purpose:
        Get names of parameters

    Inputs:
        adtPar  dictionary, parameters description

    Outputs:
        adtPar  dictionary, adapted to name the parameter, if necessary

    Return value:
        asP     list of size iP, names
    """
    iP= len(adtPar)
    if ('name' not in adtPar[0]):
        for j in range(iP):
            adtPar[j]['name']= 'p%i' % j

    asP= [ dtPar['name'] for dtPar in adtPar ]

    return asP

###########################################################
### FixPar(i, vP, adtPar)
def FixPar(i, adtPar, dP= None, trans= None):
    """
    Purpose:
        Quickly fix a parameter

    Inputs:
        i       integer, or string, with single parameter to fix
        adtPar  dictionary, parameters description
        dP      (optional) double, value at which to fix the parameter
        trans   (optional) list of size two, with bound between which to fix the parameter; if given, parameter IS FIXED BETWEEN BOUNDS only

    Outputs:
        adtPar  dictionary, adapted to fix the parameter

    Return value:
        br      boolean, True if all went well
    """
    asP= GetNamesPar(adtPar)
    if (isinstance(i, int)):
        sP= asP[i]
    else:
        if (i in asP):
            sP= i
            i= asP.index(sP)
        else:
            print ('Parameter \'%s\' not found' % str(i))
            return False

    iP= len(adtPar)
    if ((i < 0) or (i >= iP)):
        print ('Index \'%i\' out of bounds [0,%i]' % (i, iP-1))
        return False

    if ('free' not in adtPar[0]):
        for j in range(iP):
            adtPar[j]['free']= not(adtPar[i]['trans'][0] == adtPar[i]['trans'][1])
    if (trans is not None):
        adtPar[i]['trans']= trans
    else:
        adtPar[i]['free']= False
    if (dP is not None):
        adtPar[i]['p']= dP

    return True

###########################################################
### FreePar(i, adtPar)
def FreePar(i, adtPar):
    """
    Purpose:
        Quickly free a parameter

    Inputs:
        i       integer, or string, with single parameter to free
        adtPar  dictionary, parameters description

    Outputs:
        adtPar  dictionary, adapted to free the parameter

    Return value:
        br      boolean, True if all went well
    """
    asP= GetNamesPar(adtPar)
    if (isinstance(i, int)):
        sP= asP[i]
    else:
        if (i in asP):
            sP= i
            i= asP.index(sP)
        else:
            print ('Parameter \'%s\' not found' % str(i))
            return False

    iP= len(adtPar)
    if ((i < 0) or (i >= iP)):
        print ('Index \'%i\' out of bounds [0,%i]' % (i, iP-1))
        return False

    if ('free' not in adtPar[0]):
        for j in range(iP):
            adtPar[j]['free']= True
    adtPar[i]['free']= True

    return True

###########################################################
### main
def main():
    # Magic numbers
    adtPar= [{'p': 5, 'trans': [-np.inf, np.inf]},
             {'p': 5, 'trans': [-np.inf, 10]},
             {'p': 5, 'trans': [3, np.inf]},
             {'p': 5, 'trans': [3, 6]},
             {'p': 5, 'trans': [5, 5]},
             {'p': 5, 'trans': [6, 10]}]

    # Initialisation
    vP= [ dtPar['p'] for dtPar in adtPar ]

    # Estimation
    vPTr= TransPar(vP, adtPar)
    vP1= TransBackPar(vPTr, adtPar)

    print ('p', vP)
    print ('pTr', vPTr)
    print ('p1', vP1)

    FixPar(2, adtPar)
    FreePar(3, adtPar)
    FixPar(5, adtPar, trans=[3, 10])

    print ('After fixing 2, freeing 3:', json.dumps(adtPar, indent= 2))

###########################################################
### start main
if __name__ == "__main__":
    main()
