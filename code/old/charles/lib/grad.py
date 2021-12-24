#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grad.py

Purpose:
    contain a series of routines used in the
      Principles of Programming for Econometrics
    course at VU/TI Amsterdam

Version:
    1       Extract from ppectr.py, excluding GetCovML()

Date:
    2017/8/21

Author:
    Charles Bos, based partially on Kevin Sheppard's hessian_2sided, with
      ideas/constants from Jurgen Doornik
"""
###########################################################
### Imports
import numpy as np
import scipy.optimize as opt
import math

###########################################################
### vh= _gh_stepsize(vP)
def _gh_stepsize(vP, second= False):
    """
    Purpose:
        Calculate stepsize close (but not too close) to machine precision

    Inputs:
        vP      1D array of parameters

    Return value:
        vh      1D array of step sizes
    """
    vh = 1e-8*(np.fabs(vP)+1e-8)   # Find stepsize
    dRice= 1e-4 if second else 5e-6  # Let stepsize depend on first/second derivatives
    vh= np.maximum(vh, dRice)        # Don't go too small

    return vh

###########################################################
### vG= gradient_2sided(fun, vP, *args)
def gradient_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical gradient, using a 2-sided numerical difference

    Author:
      Charles Bos, following Kevin Sheppard's hessian_2sided, with
      ideas/constants from Jurgen Doornik's Num1Derivative

    Inputs:
      fun     function, as used for minimize()
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      vG      iP vector with gradient

    See also:
      scipy.optimize.approx_fprime, for forward difference
    """
    iP = np.size(vP)
    vP= np.array(vP).reshape(iP)      # Ensure vP is 1D-array

    # f = fun(vP, *args)    # central function value is not needed
    vh= _gh_stepsize(vP)
    mh = np.diag(vh)        # Build a diagonal matrix out of h

    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):     # Find f(x+h), f(x-h)
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    vhr = (vP + vh) - vP    # Check for effective stepsize right
    vhl = vP - (vP - vh)    # Check for effective stepsize left
    vG= (fp - fm) / (vhr + vhl)  # Get central gradient

    return vG

###########################################################
### mG= jacobian_2sided(fun, vP, *args)
def jacobian_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical jacobian, using a 2-sided numerical difference

    Author:
      Charles Bos, following Kevin Sheppard's hessian_2sided, with
      ideas/constants from Jurgen Doornik's Num1Derivative

    Inputs:
      fun     function, return 1D array of size iN
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      mG      iN x iP matrix with jacobian

    See also:
      numdifftools.Jacobian(), for similar output
    """
    iP = np.size(vP)
    vP= vP.reshape(iP)      # Ensure vP is 1D-array

    vF = fun(vP, *args)     # evaluate function, only to get size
    iN= vF.size

    vh= _gh_stepsize(vP)
    mh = np.diag(vh)        # Build a diagonal matrix out of h

    mGp = np.zeros((iN, iP))
    mGm = np.zeros((iN, iP))

    for i in range(iP):     # Find f(x+h), f(x-h)
        mGp[:,i] = fun(vP+mh[i], *args)
        mGm[:,i] = fun(vP-mh[i], *args)

    vhr = (vP + vh) - vP    # Check for effective stepsize right
    vhl = vP - (vP - vh)    # Check for effective stepsize left
    mG= (mGp - mGm) / (vhr + vhl)  # Get central jacobian

    return mG

###########################################################
### mH= hessian_2sided(fun, vP, *args)
def hessian_2sided_old(fun, vP, *args):
    """
    Purpose:
      Compute numerical hessian, using a 2-sided numerical difference

    Author:
      Kevin Sheppard, adapted by Charles Bos

    Source:
      https://www.kevinsheppard.com/Python_for_Econometrics

    Inputs:
      fun     function, as used for minimize()
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      mH      iP x iP matrix with symmetric hessian
    """
    iP = np.size(vP,0)
    vP= vP.reshape(iP)    # Ensure vP is 1D-array

    f = fun(vP, *args)
    vh= _gh_stepsize(vP, second= True)
    vPh = vP + vh
    vh = vPh - vP

    mh = np.diag(vh)            # Build a diagonal matrix out of vh

    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    fpp = np.zeros((iP,iP))
    fmm = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            # if (i == j):
            #     fpp[i,i] = fun(vP + 2*mh[i], *args)
            #     fmm[i,i] = fun(vP - 2*mh[i], *args)
            # else:
            fpp[i,j] = fun(vP + mh[i] + mh[j], *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(vP - mh[i] - mh[j], *args)
            fmm[j,i] = fmm[i,j]

    vh = vh.reshape((iP,1))
    mhh = vh @ vh.T             # mhh= h h', outer product of h-vector

    mH = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            mH[i,j] = (fpp[i,j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i,j])/mhh[i,j]/2
            mH[j,i] = mH[i,j]

    return mH

###########################################################
### mH= hessian_2sided(fun, vP, *args)
def hessian_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical hessian, using a 2-sided numerical difference
      This version goes further away from the source of Sheppard, closer to the theoretical derivation of the gradient-of-gradient, and to the Ox implementation.

    Author:
      Kevin Sheppard, adapted by Charles Bos

    Source:
      https://www.kevinsheppard.com/Python_for_Econometrics

    Inputs:
      fun     function, as used for minimize()
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      mH      iP x iP matrix with symmetric hessian
    """
    iP = np.size(vP,0)
    vP= vP.reshape(iP)    # Ensure vP is 1D-array

    f = fun(vP, *args)
    vh= _gh_stepsize(vP, second= True)
    vPh = vP + vh
    vh = vPh - vP

    mh = np.diag(vh)            # Build a diagonal matrix out of vh

    fpp = np.zeros((iP,iP))
    fmm = np.zeros((iP,iP))
    fpm = np.zeros((iP,iP))
    fmp = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i, iP):
            fpp[i,j] = fun(vP + mh[i] + mh[j], *args)
            # fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(vP - mh[i] - mh[j], *args)
            # fmm[j,i] = fmm[i,j]
            if (j > i):
                fpm[i,j] = fun(vP + mh[i] - mh[j], *args)
                fmp[i,j] = fun(vP - mh[i] + mh[j], *args)

    vh = vh.reshape((iP,1))
    mhh = vh @ vh.T             # mhh= h h', outer product of h-vector

    mH = np.zeros((iP,iP))
    for i in range(iP):
        mH[i,i]= ((fpp[i,i] - f) - (f - fmm[i,i]))/(4*mhh[i,i])
        for j in range(i+1,iP):
            mH[i,j] = ((fpp[i,j] - fmp[i,j]) - (fpm[i,j] - fmm[i,j]))/(4*mhh[i,j])
            mH[j,i] = mH[i,j]

    return mH
