#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vech.py

Purpose:
   Provide library for transforming a matrix, getting the vech

Version:
   1       First start, extracting from oxpy/transpar

Date:
   2021/11/11

Author:
   Charles Bos
"""
###########################################################

###########################################################
### Imports
import numpy as np
import pandas as pd

#########################################################
def vech(mP):
    """
    Purpose:
        Get the vech of a square and symmetric matrix. Note that no checking is
        performed, and effectively the lower diagonal of the matrix is used.

    Inputs:
        mP      iK x iK symmetric matrix

    Return value:
        vPh     iK*(iK+1)/2 vector with vech(mP)
    """
    iK= mP.shape[0]
    mI= np.tril(np.ones((iK, iK), dtype=bool))

    return mP[mI].flatten()

#########################################################
def unvech(vPh):
    """
    Purpose:
        Get the unvech to create a square and symmetric matrix.

    Inputs:
        vPh     iK*(iK+1)/2 vector with vech(mP)

    Return value:
        mP      iK x iK symmetric matrix
    """
    iP= vPh.shape[0]
    iK= np.round(0.5*(-1+np.sqrt(1+8*iP))).astype(int)
    mP= np.zeros((iK, iK))

    mI= np.tril(np.ones((iK, iK), dtype=bool))
    mP[mI]= mP.T[mI]= vPh

    return mP

#########################################################
### vX= vec(mX)
def vec(mX):
    """
    Purpose:
        vectorize mX by column

    Inputs:
        mX      iN x iK matrix

    Return value
        vX      iN*iK vector
    """
    return mX.flatten('F')

#########################################################
### mX= unvec(vX)
def unvec(vX, iN= None, iK= None):
    """
    Purpose:
        rebuild mX by column

    Inputs:
        vX      iN*iK vector
        iN      (optional, default= sqrt(iN*iK)) integer, number of rows
        iK      (optional, default= iX/iN) integer, number of columns

    Return value
        mX      iN x iK matrix
    """
    iN= iN or int(np.sqrt(len(vX)))
    iK= iK or int(len(vX)/iN)
    return vX.reshape(iN, iK, order= 'F')

###########################################################
### main
def main():
    # Magic numbers
    iK= 3

    # Initialisation
    vS= np.random.randn(int(iK*(iK+1)/2))
    mS= unvech(vS)
    vS1= vech(mS)

    df= pd.DataFrame(vS, columns= ['s'])
    df['s1']= vS1
    df['diff']= vS-vS1
    print (df)

    iN= 4
    mX= np.arange(iN*iK).reshape(iN, iK)
    print ('Org:', mX)
    vX= vec(mX)
    mX1= unvec(vX, iN)
    print ('New:', mX1)

###########################################################
### start main
if __name__ == "__main__":
    main()
