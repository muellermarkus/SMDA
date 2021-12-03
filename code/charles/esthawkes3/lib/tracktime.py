#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracktime.py

Purpose:
    Provide info on time routines take

Version:
    1       First start, based on tracktime.ox
    2       Including info and Timer()

Date:
    2017/9/14, 2020/7/15

@author: cbs310
"""
###########################################################
### Imports
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.optimize as opt
# import seaborn as sns
# import statsmodels.api as sm
import time

###########################################################
### Globals
# s_info_iRep= 1
# s_info_iTot= None
# s_info_tStart= None
# s_info_tLast= None

###########################################################
class Timer(object):
    """
    Purpose:
        Measure time of a block of code
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print ('[%s]' % self.name),
        print ('Elapsed: %s' % (time.time() - self.tstart))

###########################################################
### bR= info(iI, iTot= None, iRep= None)
def info(iI, iTot= None, iRep= None):
    """
    Purpose:
        Show time to finish

    Inputs:
        iI      integer, current iteration
        iTot    integer, total number of iterations
        iRep    integer, number of iterations between showing (if positive), or -number of seconds between showing (if negative)

    Return value:
        bR      boolean, TRUE if info was displayed
    """
    global s_info_iRep, s_info_iTot, s_info_tStart, s_info_tLast
    if (not 's_info_iRep' in globals()):
        s_info_iRep= 1
        s_info_iTot= None
        s_info_tStart= None
        s_info_tLast= None

    if (iRep is not None):
        s_info_iRep= iRep
    if (iTot is not None):
        s_info_iTot= iTot

    if ((s_info_tStart is None) or (iI == 0)):
        s_info_tStart= time.time()
        s_info_tLast= s_info_tStart - 3600 # Set last display an hour ago

    tim= time.time()
    bDisplay= (iI == s_info_iTot) or ((iI % s_info_iRep == 0) if (s_info_iRep > 0) else (tim - s_info_tLast > -s_info_iRep))
    if (bDisplay):
        print ("\n")
        print ("-------------------------------------------------------------")
        print ("   Number of elapsed iterations:    %9i" % int(iI))
        if (s_info_iTot is not None):
            print ("   Number of iterations to go:      %9i" % int(s_info_iTot - iI))
        dElapsed= tim - s_info_tStart
        print ("   Elapsed time:                 %12s" % time.strftime("%H:%M:%S", time.gmtime(dElapsed)))

        if (iI == 0):
            print("   Time per iteration:                   .")
            print("   Estimate of remaining time:           .")
        else:
            dMeantime = (tim-s_info_tStart) / iI;
            dEsttime = dMeantime * (s_info_iTot - iI);
            if (dMeantime > 50):
                print("   Time per iteration:           %12.2f" % dMeantime)
            else:
                print("   Time per 100 iterations:      %12.2f" % (dMeantime*100))
            if (dEsttime is not np.nan):
                print("   Estimate of remaining time:   %12s" % time.strftime("%H:%M:%S", time.gmtime(dEsttime)))
        s_info_tLast= tim

    return bDisplay

###########################################################
### TrackInit()
def TrackInit():
    """
    Purpose:
      Initialise settings for timing routines through TrackTime()
    """
    global g_TT_names, g_TT_duration, g_TT_t0, g_TT_iR

    # print ("In TrackInit")
    g_TT_names= []
    g_TT_duration= []
    g_TT_t0= time.time()
    g_TT_iR= -1

###########################################################
### _TrackIndex()
def _TrackIndex(sR):
    """
    Purpose:
        Find the index of the routine, and store the name of the routine

    Inputs:
        sR      string, routine to search/place in index (or -1/None, for no routine)

    Return value:
        iR      integer, index of string in index, or -1 if no routine is tracked
    """
    global g_TT_names, g_TT_duration

    # In case sR= -1, just stop tracking time
    if ((sR == -1) | (sR is None)):
        return -1

    if (not (sR in g_TT_names)):
        g_TT_names.append(sR)
        g_TT_duration.append(0.0)
    # else:
    #     print ("Found ", sR, " at index ", g_TT_names.index(sR))

    return g_TT_names.index(sR)

###########################################################
### TrackTime(sR)
def TrackTime(sR):
    """
    Purpose:
        Track the time routine sR takes

    Input:
        sR      string, new routine to track

    Return value:
        sRold   string, previous routine that was tracked, or None
    """
    global g_TT_names, g_TT_duration, g_TT_t0, g_TT_iR

    if (not 'g_TT_names' in globals()):
        TrackInit()

    sRold= None
    if (g_TT_iR >= 0):
        g_TT_duration[g_TT_iR]+= time.time() - g_TT_t0
        sRold= g_TT_names[g_TT_iR]

    g_TT_iR= _TrackIndex(sR)
    g_TT_t0= time.time()

    return sRold

###########################################################
### TrackReport()
def TrackReport(bInit= True):
    """
    Purpose:
        Report the time routines took

    Inputs:
        bInit    (optional, default= True) boolean, if True re-initialise after showing the report.
    """
    global g_TT_names, g_TT_duration, g_TT_t0, g_TT_iR

    TrackTime(-1)       # Stop current timing

    vDur= np.array(g_TT_duration)
    dTot= np.sum(vDur)

    iRR= len(vDur)
    print ("Time spent in routines:")
    print ("%15s  %10s %8s" % ("Routine", "Secs", "Perc."))
    for i in range(iRR):
        print ("%15s: %10.2f %8.2f" % (g_TT_names[i], vDur[i], 100*vDur[i]/dTot))
    print ("Total: %10.2fs" % dTot)

    if (bInit):
        TrackInit()

###########################################################
def tic():
    #Homemade version of matlab tic and toc functions
    # import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
###########################################################
def toc():
    # import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time: " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")

###########################################################
### mY= lengthycalculation(iN, iS)
def lengthycalculation(iN, iS):
    mY= 0.0
    for j in range(iN):
        mX= np.random.randn(iS, iS)
        mY+= np.exp(mX)

    return mY

###########################################################
### main
def main():
    # Magic numbers
    iN= 100
    iSa= 5
    iSb= 10
    iR= 5

    # estimation
    for i in range(iN):
        info(i, iN, iR)
        TrackTime("Routine A")
        mX= lengthycalculation(iN*iR, iSa)

        TrackTime("Routine B")
        mX= lengthycalculation(iN*iR, iSb)

        TrackTime(-1)
    info(i, iN)

    print ('\nNow show every .1 second...')
    for i in range(iN):
        info(i, iN, -.1)
        TrackTime("Routine A")
        mX= lengthycalculation(iN*iR, iSa)

        TrackTime("Routine B")
        mX= lengthycalculation(iN*iR, iSb)

        TrackTime(-1)
    info(iN, iN)

    # Output
    TrackReport()
    print ("This is a test\n")

###########################################################
### start main
if __name__ == "__main__":
    main()
