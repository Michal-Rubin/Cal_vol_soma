import plotly.io as pio
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc
from scipy.optimize import curve_fit
from scipy.ndimage import filters 
import tifffile as tiff
from scipy.signal import find_peaks
import tifffile
from scipy.signal import resample
import os
import csv
import pandas as pd
from roipoly import MultiRoi
import plotly.graph_objects as go
import plotly.io as pio
import glob
from plotly.subplots import make_subplots
from scipy.io import loadmat
import glob
import plotly.io as pio
import h5py
import numpy as np

import scipy.signal as sc
from scipy.optimize import curve_fit
from scipy.ndimage import filters 
import tifffile as tiff
from scipy.signal import find_peaks
import os
import csv
import pickle
from scipy import signal
import cv2  # Make sure OpenCV is installed: pip install opencv-python
import logging
import pandas as pd
from skimage.morphology import dilation
from skimage.morphology import disk
from roipoly import MultiRoi
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from scipy.io import loadmat
import xmltodict
import xml.etree.ElementTree as ET
from scipy.ndimage import binary_dilation
from matplotlib.path import Path
from scipy import signal
from scipy.sparse.linalg import svds
from sklearn.linear_model import Ridge
from scipy import stats
import concurrent.futures
from caiman.base.movies import movie
from caiman.motion_correction import MotionCorrect
from caiman.paths import caiman_datadir
import caiman as cm
import sys
import scipy.signal as sc
import matplotlib as mpl
mpl.use("Qt5Agg")   # or "TkAgg" if Qt causes issues
# Add the folder containing your BatchRunSpatialFootLocal.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BatchRunSpatialFootLocal import get_rois_mask,volS,trace_extraction,save_volpy_plots,get_calROImask

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
# extract data from thorSync
def frameSync(ThorF, KayN , subKay):
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
                # print(np.max(frameOut))
                # plt.plot(frameOut)
                # plt.show()
            if chan == 'DAQ_Trigger':
                StartSig = currC[chan]
                StartSig = np.array(StartSig)
            if chan=="Motor":
                Motor=np.array(currC[chan]).squeeze()

    return galvoVec, frameCount, frameOut , StartSig , Motor    

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




def syncParam(TS_path):
    SR = 30000
    targetSR = 500
    FullThor , chanName , subName = extracTHorsync(TS_path)
    calFrame , VolFrame, volFrameO, DAQ, motoridx = frameSync(FullThor , chanName , subName)
    Frameout=np.insert(np.diff(VolFrame),0,0)
    calFrame = calFrame*-1
    print(np.max(DAQ))
    VolFrameStep=np.insert(np.diff(volFrameO.squeeze()),0,0) 
    t = np.arange(1, (len(calFrame)) + 1, 1) 
    L =np.convolve(np.diff(calFrame.flatten()), np.ones(5)/5, mode='valid')
    
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
    #plt.plot(L)
    #plt.plot(Lo)
    #plt.plot(cal_Ind,L[cal_Ind],'b.', markersize=10)
    #plt.plot(Fcal_Ind,calFrame.flatten()[Fcal_Ind],'k.', markersize=8)
    #plt.plot((calFrame.flatten()))
    #plt.show()
    #CamInd = np.where(np.diff(volFrameO.flatten()) == 1)[0] # this find points of differance between camra trigger meaning start or end of frame, -28 is for correction to actual
    CamInd = np.where(np.diff(volFrameO.flatten()) == 1)[0]
     # this find points of differance between camra trigger meaning start or end of frame, -28 is for correction to actual
    
    CamExp = np.diff(CamInd)/SR   # camera exposure in sec
    TwoPExp = np.diff(Fcal_Ind)/SR # 2P exposure in sec
    print(CamExp)
    print(TwoPExp)
    calStart = Fcal_Ind[0]
    volend = CamInd[-1]
    tMotor = np.arange(len(motoridx)) / SR  
    # time axis for voltage imaging frames
    tVOL = np.concatenate(([0], np.cumsum(np.diff(CamInd)/SR)))
    tCAL = np.concatenate(([0], np.cumsum(TwoPExp)))
    #f_motor = interp1d(tMotorI, motoridx, kind='nearest', 
   #                     bounds_error=False, fill_value="extrapolate")
    #Motor_aligned = f_motor(tVOL)
    
    
    
    
    
    if Fcal_Ind[0] > CamInd[0] and Fcal_Ind[-1] > CamInd[-1]: #calcium starts after camra and ends after
        k = 1
        VolStart = np.argmin(np.abs(CamInd - calStart)) +12 
        MotorStart =CamInd[VolStart]
        VolStart = np.max((VolStart,0))       
        CalEnd = np.argmin(np.abs(Fcal_Ind - volend)) 
        print(VolStart)
        print(CalEnd)
        Dur = (Fcal_Ind[CalEnd] - Fcal_Ind[0])/SR
        print(Dur)
         # interpolate Motor onto voltage imaging time base
        MotorStart = CamInd[VolStart]
        MotorEnd = np.argmin(np.abs(tMotor - tCAL[CalEnd]))
        #MotS = motoridx[Fcal_Ind[0]:CamInd[-1]+1]
        # threshold Motor (same logic you had before)
       
        #plt.plot(CamInd[VolStart], volFrameO[CamInd[VolStart]], 'k*', markersize=10) #ploting the cal frame when 
        #plt.plot(Fcal_Ind[CalEnd], calFrame[Fcal_Ind[CalEnd]], 'k*', markersize=10)
        #plt.show()
    if Fcal_Ind[0] > CamInd[0] and Fcal_Ind[-1] < CamInd[-1]: #calcium start after camre and end before camra
        k = 2
        VolStart = np.argmin(np.abs(CamInd - calStart)) 
        CalEnd =np.argmin(np.abs(CamInd - Fcal_Ind[-1]))
        print(VolStart)
        print(CalEnd)
        Dur = (Fcal_Ind[-1] - Fcal_Ind[0])/SR
        print(Dur)
        MotorStart = CamInd[VolStart]
        MotorEnd = CamInd[CalEnd]
        #MotS = motoridx[Fcal_Ind[0]:Fcal_Ind[-1]+1]
        #plt.plot(CamInd[VolStart], volFrameO[CamInd[VolStart]], 'k*', markersize=10) #ploting the cal frame when 
        #plt.plot(CamInd[CalEnd], volFrameO[CamInd[CalEnd]], 'k*', markersize=10)
        #plt.show()
    if Fcal_Ind[0] < CamInd[0] and Fcal_Ind[-1] < CamInd[-1]: # calcium start before camra and end before camra
        k = 3
        VolStart = np.argmin(np.abs(Fcal_Ind - CamInd[0])) - 1
        CVol = np.argmin(np.abs(CamInd - Fcal_Ind[VolStart]))
        CalEnd = np.argmin(np.abs(CamInd - Fcal_Ind[-1]))
        print(VolStart)
        print(CalEnd)
        Dur = (Fcal_Ind[-1] - Fcal_Ind[0])/SR
        print(Dur)
        MotorStart = CamInd[CVol]
        MotorEnd = CamInd[CalEnd]
        #MotS = motoridx[CamInd[0]:Fcal_Ind[-1]+1]
        plt.plot(t,calFrame, alpha=0.5)
        plt.plot(t,volFrameO,alpha=0.5)
        plt.plot(t,motoridx,alpha=0.5)
        
        
        plt.plot(CamInd, volFrameO.flatten()[CamInd], 'b.', markersize=10)
        plt.plot(Fcal_Ind[VolStart], calFrame[Fcal_Ind[VolStart]], 'k.', markersize=10) #ploting the cal frame when
        plt.plot(CamInd[CVol], volFrameO[CamInd[CVol]], 'k*', markersize=10)
        plt.plot(CamInd[CalEnd], volFrameO[CamInd[CalEnd]], 'k*', markersize=10)
        plt.show()
    if Fcal_Ind[0] < CamInd[0] and Fcal_Ind[-1] > CamInd[-1]: # calcium start before camra and end after camra
        k = 4
        VolStart = np.argmin(np.abs(Fcal_Ind - CamInd[0]))
        if VolStart > CamInd[0]:
            VolStart  = VolStart -1
            CVol = np.argmin(np.abs(CamInd - Fcal_Ind[VolStart])) -1
        else:
            CVol = np.argmin(np.abs(CamInd - Fcal_Ind[VolStart])) 
        #CVol = np.where(CamInd < Fcal_Ind[VolStart])[0][-1]
        CalEnd  =  np.argmin(np.abs(Fcal_Ind - CamInd[-1]))
        
        #MotS = motoridx[CamInd[0]:CamInd[-1]+1]
        # plt.plot(t,calFrame, alpha=0.5)
        # plt.plot(t,volFrameO,alpha=0.5)
        # plt.plot(t,motoridx,alpha=0.5)
        
        
        # plt.plot(CamInd, volFrameO.flatten()[CamInd], 'b.', markersize=10)
        # plt.plot(Fcal_Ind[VolStart], calFrame[Fcal_Ind[VolStart]], 'k.', markersize=10) #ploting the cal frame when
        # plt.plot(CamInd[CVol], volFrameO[CamInd[CVol]], 'k*', markersize=10)
        # plt.plot(Fcal_Ind[CalEnd], Fcal_Ind[CamInd[CalEnd]], 'k*', markersize=10)
        # plt.show()
        
        MotorStart = CamInd[CVol]
        MotorEnd = Fcal_Ind[CalEnd]
        Dur = (Fcal_Ind[-1] - Fcal_Ind[0])/SR
        print(k)
        print(Dur)
        print(VolStart)
        print(CalEnd)
        print(len(CamInd[CVol:-1])*0.002)
        print(CamExp[-1])
        Dur = (Fcal_Ind[CalEnd] - Fcal_Ind[VolStart])/SR
    MotS = motoridx[MotorStart:MotorEnd+1]    
    # thresh = min(MotS)+0.25
    # tMotor = np.arange(len(MotS)) / SR
    # tMotor = list(tMotor)
    # print(tMotor[-1])
    # motor_bin = (MotS > thresh).astype(int)
    # plt.figure(1)
    # plt.clf()
    # plt.plot(t,calFrame, alpha=0.5)
    # plt.plot(t,volFrameO,alpha=0.5)
    # plt.plot(t,motoridx,alpha=0.5)
    
       
    # plt.plot(CamInd, volFrameO.flatten()[CamInd], 'b.', markersize=10)
    # #plt.plot(volFrameO.flatten()[CamInd], 'b.', markersize=10)
    # plt.plot([t[MotorStart], t[MotorEnd]],
    #      [motoridx[MotorStart], motoridx[MotorEnd]],
    #      'b.', markersize=10)
    # plt.plot([t[VolStart], t[CalEnd]],
    #      [calFrame[Fcal_Ind[VolStart]], calFrame[Fcal_Ind[CalEnd]]],
    #      'r.', markersize=10)
    # plt.plot([CamInd[CVol], CamInd[-1]],
    #      [volFrameO[CamInd[CVol]], volFrameO[CamInd[-1]]],
    #      'k.', markersize=10)
    
    return k,VolStart,CalEnd,CVol,CamExp,TwoPExp, MotS

def SyncCalVol(thP,mc_Path,cal_Path,spatialFootTr):
    k,VolStart,CalEnd,CVol,CamExp,TwoPExp, MotorT, = syncParam(thP)
    
    thresh = np.median(MotorT)
    MotorSync = MotorT
    

    n_target = int(len(MotorSync) * 500 / 30000)  # expected number of samples
    MotorSync_ds = resample(MotorSync, n_target)
    # plt.figure()
    # plt.clf()
    # plt.plot(MotorSync_ds,alpha=0.5)
    # plt.plot(CamExp,alpha=0.5)
    # plt.scatter(VolStart,CamExp[VolStart])
    # plt.scatter(VolStart,MotorSync_ds[VolStart])
    # parentP = os.path.dirname(thP)
    # plt.savefig(os.path.join(parentP,f'dd.png'), dpi=300, bbox_inches="tight") 
    #plt.show() 
    MotAx = np.arange(len(MotorSync) )/ 500 
    #print(np.max(MotAx))
    Vol = tiff.imread(mc_Path)
    #Cal = CalNormCor(cal_Path)
    Cal = tiff.imread(cal_Path)
    # spatialFootTr can be a DataFrame from CSV (2D) or ndarray (2D/3D).
    # Convert once and always slice on axis 0 only.
    if isinstance(spatialFootTr, pd.DataFrame):
        spatialFootTr_arr = spatialFootTr.to_numpy()
    else:
        spatialFootTr_arr = np.asarray(spatialFootTr)

    def _slice_spatial(arr, s0, e0):
        a = np.asarray(arr)
        if a.ndim == 0:
            return a
        if a.ndim == 1:
            return a[s0:e0]
        if a.ndim == 2:
            return a[s0:e0, :]
        return a[s0:e0, :, :]

    dir_path = os.path.dirname(cal_Path)
    #MCcal = os.path.join(dir_path,f'motion_corrected_cal.tif')
    #tiff.imsave()
    tVOL = np.concatenate(([0], np.cumsum(CamExp)))  # Voltage imaging time stamps
    tCAL = np.concatenate(([0], np.cumsum(TwoPExp)))   # Calcium imaging time stamps
    if k == 1:
        VOLSync = Vol[VolStart:-1,:,:]
        spatialFootTrSync = _slice_spatial(spatialFootTr_arr, VolStart, -1)
        #VOLSync = VOLSync.reshape(np.size(VOLSync,1),1)
        CALSync = Cal[:CalEnd,:,:]
        #MotorSync = motor_bin[:CalEnd]
        tVOL = tVOL[VolStart:-1] - tVOL[VolStart]# Voltage imaging time stamps
        tVOL = np.array(tVOL).reshape(-1, 1)
        tCAL = tCAL[:CalEnd].T  # Calcium imaging time stamps
    if k == 2:
        VOLSync = Vol[VolStart:CalEnd,:,:]
        spatialFootTrSync = _slice_spatial(spatialFootTr_arr, VolStart, CalEnd)
        #MotorSync = motor_bin[:]
        #VOLSync = VOLSync.reshape(np.size(VOLSync,1),1)
        CALSync = Cal[:, :,:]
        tVOL = tVOL[VolStart:CalEnd] -tVOL[VolStart] # Voltage imaging time stamps
        tVOL = np.array(tVOL).reshape(-1, 1)
        tCAL = tCAL[:].T  # Calcium imaging time stamps
    if k == 3:
        VOLSync = Vol[CVol:CalEnd,:,:]
        spatialFootTrSync = _slice_spatial(spatialFootTr_arr, CVol, CalEnd)
        #VOLSync = VOLSync.reshape(np.size(VOLSync,1),1)
        
        CALSync = Cal[VolStart:-1,:,:]
        #MotorSync = motor_bin[VolStart:-1]
        tVOL = tVOL[CVol:CalEnd]- tVOL[CVol] # Voltage imaging time stamps
        tVOL = np.array(tVOL).reshape(-1, 1)
        tCAL = tCAL[VolStart:-1].T-tCAL[VolStart]  # Calcium imaging time stamps
        
    if k == 4:
        VOLSync = Vol[CVol:-1,:,:]
        spatialFootTrSync = _slice_spatial(spatialFootTr_arr, CVol, -1)
        #VOLSync = VOLSync.reshape(np.size(VOLSync,1),1)
        CALSync = Cal[VolStart:CalEnd,:,:]
        #MotorSync = motor_bin[VolStart:CalEnd]
        #RawCalSync =CalTrace[:,  VolStart:-1].T
        #Conv = ConvTrace[:, VolStart:CalEnd].T
        tVOL = tVOL[CVol:-1] -tVOL[CVol] # Voltage imaging time stamps
        tVOL = np.array(tVOL).reshape(-1, 1)
        tCAL = tCAL[VolStart:CalEnd].T-tCAL[VolStart]
   
    if len(tVOL) > np.size(VOLSync,0):
        x = len(tVOL)
        y =np.size(VOLSync,0)
        endF = len(tVOL)-np.size(VOLSync,0)
        tVOL = tVOL[0:-endF]
        tFinalV = tVOL[-1]
        CALp = tCAL - tFinalV
        indC = np.argmin(np.abs(CALp))
        tCAL = tCAL[0:indC+1]
        CALSync = CALSync[0:indC+1,:,:]
    thresh = min(MotorSync_ds)+0.25
    
    motor_bin = (MotorSync_ds > thresh).astype(int)
    
    # plt.plot(motor_bin,alpha=0.5)
    # plt.plot(CamExp,alpha=0.5)
    # plt.scatter(VolStart,CamExp[VolStart])
    # plt.scatter(VolStart,motor_bin[VolStart])
    parentP = os.path.dirname(thP)
    # plt.savefig(os.path.join(parentP,f'dd.png'), dpi=300, bbox_inches="tight")
    print(len(motor_bin))
    #manual cut
    # volCut = 3000
    # VOLSync = VOLSync[volCut:]
    # endF = len(tVOL)-np.size(VOLSync,0)
    
    # tFinalV = tVOL[volCut]
    # CALp = tCAL - tFinalV
    # indC = np.argmin(np.abs(CALp))
    # tCAL = tCAL[indC:]
    # tVOL = tVOL[volCut:]
    # CALSync = CALSync[indC:,:,:]
    
    #plt.clf()
    #plt.plot(tCAL,CALSync, alpha=0.5)
    #plt.plot(tVOL,VOLSync,alpha=0.5)
    #plt.plot(t,motoridx,alpha=0.5)
    #plt.plot(MotAx,MotorSync_ds,alpha=0.5)
       
    #plt.plot(CamInd, volFrameO.flatten()[CamInd], 'b.', markersize=10)
    #plt.plot(volFrameO.flatten()[CamInd], 'b.', markersize=10)
    #parentP = os.path.dirname(thP)
    #plt.savefig(os.path.join(parentP,f'ff.png'), dpi=300, bbox_inches="tight") 
    return CALSync,VOLSync, motor_bin,tVOL,tCAL, spatialFootTrSync

def CalNormCor(calPath):
    
    

    # Define parameters
    fnames = [calPath]

    # Motion correction parameters
    mc_dict = {
        'pw_rigid': True,                 # Use piecewise-rigid (non-rigid)
        'max_shifts': (6, 6),            # Maximum allowed rigid shift
        'strides': (48, 48),             # Start with these patch sizes for non-rigid correction
        'overlaps': (24, 24),            # Amount of overlap between patches
        'max_deviation_rigid': 3,        # Max deviation for rigid correction
        'shifts_opencv': True,           # Use OpenCV to speed up
        'border_nan': 'copy',            # How to handle borders
    }

    # Initialize motion correction object
    mc = MotionCorrect(fnames, dview=None, **mc_dict)

    # Run motion correction
    mc.motion_correct(save_movie=False)

    # Get corrected movie (optional: memory map)
    mmap_file = mc.mmap_file
    corrected_movie = cm.load(mmap_file)  # returns a memory-mapped array
    corrected_array = corrected_movie.asarray()
    return (corrected_array)

if __name__  == "__main__":
    input_folders = [#r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\17-06-2025\fov1',
               
            #     r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-07-2025\fov7',
                #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-07-2025\fov9',
               # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\28-07-2025-motor\fov1',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\28-07-2025-motor\fov7',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\28-07-2025-motor\fov9\2',
               
               
               
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\28-07-2025-motor\fov10',
            #   r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\16-06-2025\fov5',
            #   r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-08-2025-anst\fov7',
            #   r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-08-2025-anst\fov7\2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-08-2025-anst\fov9',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\01-07-2025\fov3',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-08-2025-anst\fov8',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\01-07-2025\fov1',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\01-07-2025\fov1\2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-07-2025\fov2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-07-2025\fov7',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-07-2025\fov8',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\15-07-2025\fov5',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\15-07-2025\fov9',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\15-07-2025\fov10',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\30-07-2025-motor\fov5',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\30-07-2025-motor\fov5\2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\30-07-2025-motor\fov7\2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\30-07-2025-motor\fov10\2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\30-07-2025-motor\fov11',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-08-2025-anst\fov10',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-08-2025-anst\fov1\2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-08-2025-anst\fov10',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-08-2025-anst\fov11\2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-07-2025\fov1',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC18\L\22-07-2025\fov5',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC18\L\03-07-25\fov1',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC18\L\04-08-2025\fov5\2',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC18\R\18.07.2025\fov10',
           
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC21\R\29-09-2025-motor\fov7\3',


            #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC21\R\08-09-2025-ans\fov7',


            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC21\X\15-09-2025-awake\fov5',


            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC21\X\20-08-20225-ANS\fov2',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC21\X\20-08-20225-ANS\fov3',

            #  r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\R\18-08-2025-ans\fov5',
            #   r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\22-10-2025-motor\fov8',
            #   r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\22-10-2025-motor\fov9',
            #   r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\22-10-2025-motor\fov10',
            #   r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\17-09-2025-rugc42-wh-s1-ans\fov5',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\21-10-2025-MOTOR\fov6',
            #     r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\21-10-2025-MOTOR\fov7',

            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\R\18-08-2025-ans\fov1',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\R\18-08-2025-ans\fov2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\R\18-08-2025-ans\fov3',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\R\18-08-2025-ans\fov4',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\R\18-08-2025-ans\fov5',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\R\07-08-2025\fov1',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\L\07-08-2025\FOV1',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\L\07-08-2025\fov2',
              # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\L\18-08-2025-ans\fov1',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\L\18-08-2025-ans\fov4',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\L\18-08-2025-ans\fov5',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\L\18-08-2025-ans\fov6',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\l\20-08-25-ans',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\l\20-08-25-ans\fov2',
               
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\l\20-08-25-ans\fov5',

            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\17-09-2025-rugc42-wh-s1-ans\fov1\2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\17-09-2025-rugc42-wh-s1-ans\fov5',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RB\20-08-25-ans\fov3',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RB\20-08-25-ans\fov4',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RW\30-09-2025-MOTOR\FOV1',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RW\30-09-2025-MOTOR\FOV1\2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RW\30-09-2025-MOTOR\fov2\2',
               
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\L\15-09-2025-ans\fov2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\RL-REAL\FOV1',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\RL-REAL\fov2',
            #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\RL-REAL\fov3',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\21-10-2025-MOTOR\fov7',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\05-11-2025-motor\fov13',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\05-11-2025-motor\fov14',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\05-11-2025-motor\fov15',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\05-11-2025-motor\fov16',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\05-11-2025-motor\fov17',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RW\28-10-2025-motor\fov6',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RW\28-10-2025-motor\fov7',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RW\28-10-2025-motor\fov9',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RW\28-10-2025-motor\fov10',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\05-11-2025-motor\fov16',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\19-11-2025-awake\fov18',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\19-11-2025-awake\fov19',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\19-11-2025-awake\fov20',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\19-11-2025-awake\fov22',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc43\17-11-2025-anst\fov1',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc26\l\17-11-2025-anst\fov1',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc26\l\17-11-2025-anst\fov2',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc26\l\17-11-2025-anst\fov3',
            # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc26\l\17-11-2025-anst\fov4'  
              #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RW\30-09-2025-MOTOR\FOV1'  
              #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RW\30-09-2025-MOTOR\FOV1' 
              #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc27\L\13-01-2026-Anst\fov1\2',
            #   r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc27\L\13-01-2026-Anst\fov2',
            #   #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc27\L\13-01-2026-Anst\fov3',
            #         #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc27\RL\15-12-2025-ANS\fov2',
            #     r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc27\RL\13-01-2026-ANS\FOV2',
            #     r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc46\RL2\04-01-2026-anst\fov1',
            #     r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\L\08-01-2026-ANST\fov1' ,
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\L\08-01-2026-ANST\fov2',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\L\08-01-2026-ANST\fov3',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\L\08-01-2026-ANST\fov4'  
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\r\08-01-2025-anst\fov1',
                #     r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\r\08-01-2025-anst\fov2',
                #     r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\r\08-01-2025-anst\fov3',
                #    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\r\08-01-2025-anst\fov4',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\21-01-2026-ans\fov16',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\21-01-2026-ans\fov18',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc43\x\01-12-2025-awake\fov6',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc43\x\01-12-2025-awake\fov8',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc43\x\01-12-2025-awake\fov9',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc25\01-12-2025-motor\fov1',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc45\r\12-09-2025-ans\fov5',
                #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc45\r\12-09-2025-ans\fov5\2',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc45\x\07-12-2025-ans\fov1',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc45\x\07-12-2025-ans\fov3',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\L\03-02-2026-awake\fov6',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\L\03-02-2026-awake\fov7',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\L\03-02-2026-awake\fov8',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\r\08-01-2025-anst\fov1',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\r\22-01-2026-motor\fov1',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\r\22-01-2026-motor\fov3',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\rl\03-02-2026-awake\fov1',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\rl\03-02-2026-awake\fov2',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc28\rl\03-02-2026-awake\fov4',
                
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc46\R\07-01-2025-ans',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc46\RL2\07-01-2025-ans\fov2',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc46\R\04-02-2026-MOTOR\FOV1',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc46\X\07-01-2026-ANS\fov4',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc29\03-02-2026-ansa\fov2',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc29\03-02-2026-ansa\fov3',
                
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc29\03-02-2026-ansa\fov5',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc29\03-02-2026-ansa\fov6',
                #                    r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\12-01-2025-awake\fov16',
                #                 r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\12-01-2025-awake\fov18',
                                #  r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\12-01-2025-awake\fov19',
                                   
                                   
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\26-11-2025-ANST\FOV4',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\26-11-2025-ANST\fov5',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\26-11-2025-ANST\fov7',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\26-11-2025-ANST\fov8',
                #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\24-11-2025-ANS\FOV1',
                #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\24-11-2025-ANS\FOV2',
                #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\09-12-2025-motor\fov10',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\12-01-2025-awake\fov17\2',
                # r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\24-11-2025-ANS\FOV1',
                #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC44\L\24-11-2025-ANS\fov2',
                r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\R\18-08-2025-ans\fov1',
                r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\RL-REAL\FOV1',
                
                r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\R\18-08-2025-ans\fov2',
                r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\L\18-08-2025-ans\fov6',
                r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\17-09-2025-rugc42-wh-s1-ans\fov1\2',
                r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc41\RW\30-09-2025-MOTOR\FOV1',
                r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\19-11-2025-awake\fov19',
                r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\19-11-2025-awake\fov20'
                  ]
    # DB = pd.read_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\NewMetaDataSSTgood.csv')

    # values = DB['SNR'].tolist()
    # #r = DB.iloc[[idx for idx,i in enumerate(values) if i > 4.1]]
    # r = DB
    # input_folders = list(r['Link'])
    for l in range(len(input_folders)):
        #currP = os.path.dirname(input_folders[l])
        currP = input_folders[l]
        print(currP)
        calTiff = os.path.join(currP,'cal','Image_scan_1_region_0_0.tif')
        VolMC = os.path.join(currP,'pipeline_results','motion_corrected','motion_corrected.tiff')
       


        spatialFootprint = os.path.join(currP, 'pipeline_results', 'motion_corrected', 'traces')

        csv_path = glob.glob(os.path.join(spatialFootprint, '*.csv'))[-1]
        df = pd.read_csv(csv_path)
        ts_folder = glob.glob(os.path.join(currP, "TS_*"))
        file_path = os.path.join(ts_folder[-1], 'Episode_0000.h5')  # change name as needed
        SyncCal, SyncVol,SyncMotor,volAX,calAX, spatialFootTrSync = SyncCalVol(file_path,VolMC,calTiff,df)
        new_folder = os.path.join(currP, "Sync")  # subfolder name
        
        # Create the folder if it doesn't exist
        os.makedirs(new_folder, exist_ok=True)
        new_folder_Cal = os.path.join(new_folder, 'cal')  # subfolder name
        os.makedirs(new_folder_Cal, exist_ok=True)
        volSpath = os.path.join(new_folder,'SyncVol.tif')
        calSpath = os.path.join(new_folder_Cal,'SyncCal.tif')
        #tifffile.imwrite(volSpath,SyncVol.astype('float16'),bigtiff=True)
        tifffile.imwrite(calSpath,SyncCal.astype('float16'),bigtiff=True,compression=None)
        MotorPath = os.path.join(new_folder,'MotorId.csv')
        df = pd.DataFrame(SyncMotor)  # create df with column name
        df.to_csv(MotorPath, index=False)
        rr = get_rois_mask(currP)
        cal_rr = get_calROImask(currP)
        print("Number of ROIs found:", len(rr))
        print("Number of ROIs found:", len(cal_rr))
        #if not rr:
           # raise RuntimeError(f"No ROIs found for path: {currP}")
        meanImage = np.mean(SyncVol,0)
        # load parameters
        cellL = list(range(len(rr)))
        exT = []
        sp = []
        # Prepare arguments
        #args_list = [(cellL[i], rr[i], images) for i in range(len(cellL))]

        exT = []
        sp = []
        VpOutput = {}
        midpoint = SyncVol.shape[0] // 2
        part1 = SyncVol[:midpoint, :, :]
        part2 = SyncVol[midpoint:, :, :]
        allP = [part1,part2]
        
        
        for i in range(len(cellL)):
            cal_roi_mask = cal_rr[i]
            # Get coordinates of all True pixels
            ys, xs = np.where(cal_roi_mask)
            spF = []
            fullT = []
            SyncCal
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            SyncCALcur = SyncCal[:,max(y_min-50,0):min(y_max + 50,np.size(SyncCal,1)),max(x_min-50,0):min(x_max + 50,np.size(SyncCal,2))]
            mask = rr[i].astype(float)
            mask[mask == 0] = np.nan
            Output, x = volS(cellL[i], rr[i], SyncVol, currP, lowFr_filt=0.5, Hp_Filt=20, param_meth = 'adaptive_threshold')
            Output['traw'] = Output['traw'] - Output['traw'].min()

            save_volpy_plots(currP, cellL[i] , meanImage, Output)
            combined_output = None
            ref_baseline = None
            #qixin says no chunking for the spatial foot print
            # for t,p in enumerate(allP):
            #     #diffult params for sst- 
            #     #adaptive_threshold
            #     #p_notm = 0.5 (increse for low snr)
            #     #diffult params for [yramidal- ]-
            #     #simple
            #     Output, x = volS(cellL[i], rr[i], p, currP, lowFr_filt=0.5, Hp_Filt=20, param_meth = 'adaptive_threshold')
            #     save_volpy_plots(currP, cellL[i] , meanImage, Output)

            #                 # --- BASELINE MATCH ---
            #     # curr_baseline = np.percentile(Output['traw'], 5)

            #     # if ref_baseline is None:
            #     #     ref_baseline = curr_baseline

            #     # baseline_shift = ref_baseline - curr_baseline
            #     Output['traw'] = Output['traw'] + Output['traw'].min()
               
                
            #     # First half — initialize combined_output
            #     if combined_output is None:
            #         combined_output = Output
            #     else:
            #         # 4. Combine with previous part
            #         for key in ['traw', 't', 'dFF']:
            #             if key in Output and key in combined_output:
            #                 combined_output[key] = np.concatenate((combined_output[key], Output[key]))
                    
            #         # Offset spikes from the second part
            #         if 'spikes' in Output and 'spikes' in combined_output:
            #             offset = len(combined_output['traw']) - len(Output['traw'])
            #             combined_output['spikes'] = np.concatenate(
            #                 (combined_output['spikes'], Output['spikes'] + offset)
            #             )
                    
            #         # Combine numeric single values safely (like mean, snr, etc.)
            #         for key in Output.keys():
            #             if key not in ['traw', 't', 'spikes', 'dFF']:
            #                 if key not in combined_output:
            #                     combined_output[key] = Output[key]
            
            #     # 5. Save combined dictionary
            #     VpOutput[cellL[i]] = combined_output
            trace = Output['traw']
            spikes = Output['spikes']
            #spF.extend(spikes+(midpoint))
            #fullT.extend(trace)
            new_folder = os.path.join(currP, f'cell{i}')  # subfolder name
                
            # Create the folder if it doesn't exist
            os.makedirs(new_folder, exist_ok=True)
            save_pathPick = os.path.join(new_folder,'output_data.pkl')  # you can change the file name or add a full path

            with open(save_pathPick, 'wb') as f:
                pickle.dump(Output, f)
            calSpath = os.path.join(new_folder,'SyncCal.tif')
            tifffile.imwrite(calSpath,SyncCALcur.astype('float16'),bigtiff=True)
            TracPpath = os.path.join(new_folder,'volTrace.csv')
            SpikeTimePath = os.path.join(new_folder,'SpikeIdx.csv')
            VOLaxPATH =   os.path.join(new_folder,'volTime.csv')
            calaxPATH =  os.path.join(new_folder,'calTime.csv')
            df = pd.DataFrame(trace, columns=['trace'])  # create df with column name
            df.to_csv(TracPpath, index=False)            # save without row index
            dfS = pd.DataFrame(spikes, columns=['spike_idx'])  # create df with column name
            dfS.to_csv(SpikeTimePath, index=False)
            dfv = pd.DataFrame(volAX, columns=['time'])  # create df with column name
            dfv.to_csv(VOLaxPATH, index=False)  
            dfc = pd.DataFrame(calAX, columns=['time'])  # create df with column name
            dfc.to_csv(calaxPATH, index=False)          # save without row index
        tracesese = trace_extraction(SyncVol,rr)
        AllCEllPath = os.path.join(currP,'volTraceAllCells.csv')
        tracesese.to_csv(AllCEllPath,index=False)
        for i, roi in enumerate(rr):
            if np.sum(roi) == 0:
                print(f"ROI {i} is empty!")

