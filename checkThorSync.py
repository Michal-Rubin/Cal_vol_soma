
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc
from scipy.optimize import curve_fit
from scipy.ndimage import filters 
import tifffile as tiff
from scipy.signal import find_peaks, butter, filtfilt
from scipy import signal
import os
import csv
import pandas as pd
from roipoly import MultiRoi
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# Extracting ThorSync File
def extracTHorsync(dataPath):
    expirment = h5py.File(dataPath, 'r')
    channelNames = list(expirment.keys())
    fullName = []
    for i , cl in enumerate(channelNames):
        fullName.append(list(expirment[cl].keys()))
    #fullName = np.array(fullName)
    return expirment, channelNames , fullName



# extract data from thorSync
def   frameSync(ThorF, KayN , subKay):
    frameCount = []
    for i , cl in enumerate(KayN):
        currC = ThorF[cl]
        for k , chan in enumerate(subKay[i]):
            if chan == 'YGalvo':
                galvoVec = currC[chan]
                galvoVec = np.array(galvoVec)
            if chan =='FrameCounter':
                frameCount = currC[chan]
                frameCount = np.array(frameCount)
            if chan == 'FrameOut':
                frameOut = currC[chan]
                frameOut = np.array(frameOut)
                frameOut = frameOut.flatten()
                print(np.max(frameOut))
                plt.plot(frameOut)
                plt.show()
            if chan == 'DAQ_Trigger':
                StartSig = currC[chan]
                StartSig = np.array(StartSig)
    return galvoVec, frameCount, frameOut , StartSig    

def startframe(Cal , Vol, DAQ ,CalTrace, VolTrace):
    #CalFrame = np.where(Cal < 1.5)
    DAQframe = np.where(DAQ == 1)
    IndexCal, _ = find_peaks(-Cal,prominence=(1,None))

    IndexCal=IndexCal[:CalTrace.shape[1]]
    Indexvol = np.argwhere(filters.convolve1d(Vol > 0, np.array([1, -1])))
    frame_startVol= Indexvol[::2].ravel()[:VolTrace.shape[1]]
    #IndexCal = CalFrame[0]
    IndexDaq = DAQframe[0]
    DAqDF = pd.DataFrame(DAQframe)
    #Indexvol = VolFrame[0]
    return IndexCal, IndexDaq ,frame_startVol, DAqDF

def fps_from_indices(frame_idx, SR=30000.0):
    frame_idx = np.asarray(frame_idx, dtype=int)
    frame_idx = frame_idx[np.isfinite(frame_idx)]
    frame_idx = np.unique(frame_idx)
    if frame_idx.size < 2:
        return np.nan, np.array([])
    dt = np.diff(frame_idx) / SR  # seconds
    fps = 1.0 / np.median(dt)
    return float(fps), dt



input_folder = r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-07-2025\fov8'

SR = 30000  # ThorSync sampling rate in Hz-+ *-


thorSyncF = os.path.join(input_folder, 'TS_0227','Episode_0000.h5')
FullThor , chanName , subName = extracTHorsync(thorSyncF)
calFrame , VolFrame, volFrameO, DAQ = frameSync(FullThor , chanName , subName)
# plt.plot(volFrameO)
# plt.plot(VolFrame)
# plt.show()
# print(np.max(DAQ))
VolFrameStep=np.insert(np.diff(volFrameO.squeeze()),0,0) 

# IndexCal, IndexDaq , Indexvol, DAQFull  = startframe(calFrame.squeeze() , VolFrameStep, DAQ, CalTrace, VolTrace)
t = np.arange(1, (len(calFrame)) + 1, 1) 

L =np.convolve(np.diff(calFrame.flatten()), np.ones(5)/5, mode='valid')
# plt.plot(L)
# plt.plot(calFrame)
cal_Ind, _ = find_peaks(-L, height=0.08, distance=80)
#plt.plot(-L)
#plt.plot(L)
if cal_Ind.size <= 1:
    #calFrame = -1*calFrame
    Lo = np.convolve((-1*calFrame).flatten(), np.ones(5)/5, mode='valid')
    cal_Ind, _ = find_peaks(L, height=0.08, distance=80)
else:
    Lo = np.convolve(calFrame.flatten(), np.ones(5)/5, mode='valid')

cal_Ind = cal_Ind[0:-1]
calS = cal_Ind[0]
calE = cal_Ind[-1]

cal_IndD, _ = find_peaks(Lo,height=0.08, distance=80)
DiffPick = cal_IndD - calS
DiffPickE = np.abs(cal_IndD - calE)
PdIffPick = [i for i in DiffPick if i < 0]
RcalSt = np.argmax(PdIffPick)
RcalEn = np.argmin(DiffPickE)
Fcal_Ind = cal_IndD[RcalSt:RcalEn+1]
fps_cal, dt_cal = fps_from_indices(Fcal_Ind, SR=SR)
print("Calcium (galvo) fps (median):", fps_cal)
print("Calcium dt stats [s]:", np.median(dt_cal), np.mean(dt_cal), np.std(dt_cal))

#print(np.shape(volFrameO))
#plt.plot(np.diff(volFrameO.flatten()))
CamInd = np.where(np.diff(volFrameO.flatten()) == 1)[0] 
fps_vol, dt_vol = fps_from_indices(CamInd, SR=SR)
print("Voltage camera fps (median):", fps_vol)
print("Voltage camera dt stats [s]:", np.median(dt_vol), np.mean(dt_vol), np.std(dt_vol))# this find points of differance between camra trigger meaning start or end of frame, -28 is for correction to actual
# print(np.shape(CamInd))
# print(CamInd[-1])
# print(np.shape(cal_Ind))
# TwoPInd = np.where(np.diff(np.convolve(np.diff(calFrame.flatten()), np.ones(7)/7, mode='valid')) > 0)[0] + 1
# print(np.shape(TwoPInd))

#   Sainty check - plot frames ofcalcium tnd voltage and cutting points
# plt.figure(1)
# plt.clf()
# #plt.plot(L)
# #plt.plot(cal_Ind, L[cal_Ind], 'r.', markersize=10)
# plt.plot(t,calFrame, alpha=0.5)
# plt.plot(t,volFrameO,alpha=0.5)   
# plt.plot(CamInd, volFrameO.flatten()[CamInd], 'b.', markersize=10)


CamExp = np.diff(CamInd)/SR
variable_range = np.concatenate(([0], np.cumsum(CamExp)))  # average camera exposure in sec
TwoPExp = np.mean(np.diff(cal_Ind))/SR  # average 2P exposure in sec
tpCamExp = CamInd/SR
x = np.arange(0,len(tpCamExp),1)
# plt.figure()
# plt.scatter(x,tpCamExp)
# plt.show()
# print(CamExp)
# print(TwoPExp)
calStart = cal_Ind[0]

# print( CamInd[-1])
# print( cal_Ind[-1])
volend = CamInd[-1]
# print()
if cal_Ind[0] > CamInd[0] and cal_Ind[-1] > CamInd[-1]: #calcium starts after camra and ends after
    k = 1
    VolStart = np.where((CamInd > calStart))[0][0]         
    CalEnd = np.where(cal_Ind > volend)[0][0] 
    # print(VolStart)
    # print(CalEnd)
    Dur = (cal_Ind[CalEnd] - cal_Ind[0])/SR
    # print(Dur)
    # plt.plot(CamInd[VolStart], volFrameO[CamInd[VolStart]], 'k*', markersize=10) #ploting the cal frame when 
    # plt.plot(cal_Ind[CalEnd], calFrame[cal_Ind[CalEnd]], 'k*', markersize=10)
    # plt.show()
    
    

if cal_Ind[0] > CamInd[0] and cal_Ind[-1] < CamInd[-1]: #calcium start after camre and end before camra
    k = 2
    VolStart = np.where((CamInd > calStart))[0][0]
    CalEnd = np.where(CamInd > cal_Ind[-1])[0][0]
    # print(VolStart)
    # print(CalEnd)
    Dur = (cal_Ind[-1] - cal_Ind[0])/SR
    # print(Dur)
    # plt.plot(CamInd[VolStart], volFrameO[CamInd[VolStart]], 'k*', markersize=10) #ploting the cal frame when 
    # plt.plot(CamInd[CalEnd], volFrameO[CamInd[CalEnd]], 'k*', markersize=10)
    # plt.show()
    

if cal_Ind[0] < CamInd[0] and cal_Ind[-1] < CamInd[-1]: # calcium start before camra and end before camra
    k = 3
    VolStart = np.where((cal_Ind > CamInd[0]))[0][0]
    CVol = np.where(CamInd < cal_Ind[VolStart])[0][-1]
    CalEnd = np.where(CamInd > cal_Ind[-1])[0][0]
    # print(VolStart)
    # print(CalEnd)
    Dur = (cal_Ind[-1] - cal_Ind[0])/SR
    # print(Dur)
    # plt.plot(cal_Ind[VolStart], calFrame[cal_Ind[VolStart]], 'k*', markersize=10) #ploting the cal frame when 
    # plt.plot(CamInd[CalEnd], volFrameO[CamInd[CalEnd]], 'k*', markersize=10)
    # plt.show()
    

if cal_Ind[0] < CamInd[0] and cal_Ind[-1] > CamInd[-1]: # calcium start before camra and end after camra
    k = 4
    VolStart = np.where((cal_Ind > CamInd[0]))[0][0]
    CVol = np.where(CamInd < cal_Ind[VolStart])[0][-1]
    CalEnd  = np.where(cal_Ind > CamInd[-1])[0][0]
    Dur = (cal_Ind[-1] - cal_Ind[0])/SR
    # print(k)
    # print(Dur)
    # print(VolStart)
    # print(CalEnd)
    # plt.plot(cal_Ind[VolStart], calFrame[cal_Ind[VolStart]], 'k*', markersize=10) #ploting the cal frame when 
    # plt.plot(cal_Ind[CalEnd], calFrame[cal_Ind[CalEnd]], 'k*', markersize=10)
    # plt.plot(CamInd[CVol], volFrameO[CamInd[CVol]], 'k*', markersize=10)

    # plt.show()
    Dur = (cal_Ind[CalEnd] - cal_Ind[VolStart])/SR


