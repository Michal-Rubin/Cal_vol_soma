
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc
from scipy.optimize import curve_fit
from scipy.ndimage import filters 
import tifffile as tiff
from scipy.signal import find_peaks
import os
import csv
import pandas as pd
from roipoly import MultiRoi
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Create figure of voltage and calciu, for each FOV
def plotVolCal(path, VolAX, CALax,CalTrace, VolTrace):
    pathFig = os.path.join(path,'syncVolCAL.html')
    if VolTrace.ndim > 1:
        for i in range(0,(np.size(VolTrace,0))):
            CalTraceToPlot = CalTrace[int(i),:]
            VolTraceToPlot = VolTrace[int(i),:]
                #VolDetrand = detrend_func(VolTraceToPlot)

            fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
            fig.add_trace(go.Scatter(x=VolAX, y=VolTraceToPlot, name="Voltage"),secondary_y=False,)
            fig.add_trace(go.Scatter(x= CALax, y= CalTraceToPlot, name="Calcium"),secondary_y=True,)

        # Add figure title
            fig.update_layout(title_text="calcium and voltage togeter")

        # Set x-axis title
            fig.update_xaxes(title_text="Time(ms)")

        # Set y-axes titles
            fig.update_yaxes(title_text="<b>Voltage</b> yaxis title", secondary_y=False)
            fig.update_yaxes(title_text="<b>Calcium</b> yaxis title", secondary_y=True)

            fig.show()
            fig.write_html(pathFig)
    if VolTrace.ndim == 1 :
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=VolAX, y=VolTrace, name="Voltage"),secondary_y=False,)
        fig.add_trace(go.Scatter(x= CALax, y= CalTrace, name="Calcium"),secondary_y=True,)
        fig.update_layout(title_text="calcium and voltage togeter")
        fig.update_xaxes(title_text="Time(ms)")
        fig.update_yaxes(title_text="<b>Calcium</b> yaxis title", secondary_y=True)
        fig.update_yaxes(title_text="<b>Voltage</b> yaxis title", secondary_y=False)
        fig.show()
        fig.write_html(pathFig)

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
def frameSync(ThorF, KayN , subKay):
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


def clicy (Path):
   img = tiff.imread(Path)
   img_array = np.array(img)
   mean_img = np.mean(img,0)
   plt.imshow(mean_img)
   
   multiroi_named = MultiRoi()
   masks=[]
   corrected_polygons=[]
   for roi in multiroi_named.rois.values():
       mask=roi.get_mask(mean_img)
       masks.append(mask)
       corrected_polygons.append(np.transpose([roi.x,roi.y]))

#Extract the traces
   mov2d = np.reshape(img,(np.shape(img)[0],-1))
   raw_traces = [np.mean(mov2d[:,mask.flatten()], axis=1) for mask in masks]
    #tracedf = [NormalizeData(trace) for trace in traces]
   return raw_traces

def SyncTrace (VolTime,CalTime,firstFrameCal, FirstFrameVol , CalTrace , VolTrace):
    CalTrace =  np.array(CalTrace)
    VolTrace = np.array(VolTrace)
    diffFrame = firstFrameCal - FirstFrameVol
    if diffFrame > 0:
        correctFrameVol = np.round(diffFrame/60)
        #correctFrameCal = np.size(CalTrace[0])
        VolTime = VolTime[int(correctFrameVol):]
        VolTime = VolTime - VolTime[0] 
        CalTime = CalTime - CalTime[0]
        print(correctFrameVol)
        correctCAl = CalTrace
        correctVol = VolTrace[:, (int(correctFrameVol)-1):-1]
    
    if diffFrame < 0:
        correctFrame = np.round(diffFrame/1000)
        correctCAl = np.copy(CalTrace[:, int(-correctFrame):-1])
        correctVol = np.copy(VolTrace)
        print(correctFrame)
    
    if diffFrame == 0:
        correctCAl = np.copy(CalTrace)
        correctVol = np.copy(VolTrace)
        
    return VolTime , CalTime, correctCAl, correctVol 

def conDeltaF(Vol, rangMEAN):
    bacground = np.mean(Vol[-1,:])
    NoBackVol = Vol - bacground
    dfVOL = np.zeros((np.size(Vol,0)-1, np.size(Vol,1)))
    normDf = np.zeros((np.size(Vol,0)-1, np.size(Vol,1)))
    for i in range(0,(np.size(NoBackVol,0)-1)):
        currTrace = NoBackVol[int(i),:]
        F0 = np.mean(currTrace[rangMEAN])
        dfVOL[i,:] = (currTrace -np.abs(F0))/np.abs(F0)
        normDf[i,:] = ((dfVOL[i,:] - np.max(dfVOL[i,:]))/(np.max(dfVOL[i,:]) - np.min(dfVOL[i,:]))) +1

    return normDf


def detrend_func(data):
    t = np.arange(np.size(data,1))
    detVOL = np.zeros((np.size(data, 0), np.size(data, 1)))  # Creating a 2D array
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c
    for i in range(0,(np.size(data,0))):
        currTrace = data[int(i),:]
        print(np.size(currTrace))
        popt, pcov = curve_fit(exp_func, t, currTrace, p0=[920, 1/10000, 0], maxfev=10000)
        detrend_Trace = currTrace - exp_func(t, *popt)
        detVOL[int(i),:] = detrend_Trace
    return detVOL




if __name__  == "__main__":
    input_folder = r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\flipfun\RUGC16\07-08-2023-RUGC16-X-S2\FOV1\RUN3'
    calTiff = os.path.join(input_folder, 'cal\calcium_Signal', 'Image_scan_1_region_0_0.tif')
    print(calTiff)
    CalTrace = clicy(calTiff)
    shape = np.shape(CalTrace)
    print(shape)
    CalTrace = np.array(CalTrace)
    
    volTiff = os.path.join(input_folder, 'VOL(137).tif')
    VolTrace = clicy(volTiff)
    VolTrace = np.array(VolTrace)
    print(np.shape(VolTrace))

    Calfull = pd.DataFrame(CalTrace)
    Volfull = pd.DataFrame(VolTrace)
    PathCAL = os.path.join(input_folder,'StuffForYoav\CAL.csv')
    PathVol = os.path.join(input_folder,'StuffForYoav\VOL.csv')
    CalFullCsv = Calfull.to_csv(PathCAL)
    VolFullCsv = Volfull.to_csv(PathVol)
  
    
    
    thorSyncF = os.path.join(input_folder, 'TS_0041','Episode_0000.h5')
    FullThor , chanName , subName = extracTHorsync(thorSyncF)
    calFrame , VolFrame, volFrameO, DAQ = frameSync(FullThor , chanName , subName)
    VolFrameStep=np.insert(np.diff(VolFrame.squeeze()),0,0) 
    IndexCal, IndexDaq , Indexvol, DAQFull  = startframe(calFrame.squeeze() , VolFrameStep, DAQ, CalTrace, VolTrace)
    timeStempC = np.arange(0, np.shape(calFrame)[0], 1)/30
    timeStempV = np.arange(0, np.shape(VolFrame)[0], 1)/30
    CalTime = timeStempC[IndexCal]
    VolTime = timeStempV[Indexvol]
    CalFrame = pd.DataFrame(IndexCal)
    VolFrame = pd.DataFrame(Indexvol)
    PathCALstemp = os.path.join(input_folder,'StuffForYoav\CALfRAME.csv')
    PathVolstemp = os.path.join(input_folder,'StuffForYoav\VOLfRAME.csv')
    PathDAQstemp = os.path.join(input_folder,'StuffForYoav\DAQ.csv')
    CalTraceCsv = CalFrame.to_csv(PathCALstemp)
    volTraceCsv = VolFrame.to_csv(PathVolstemp)
    DAqTraceCsv = DAQFull.to_csv(PathDAQstemp)
    if np.size(VolTime) == 0:
        VolTime = np.arange(1,10000,2)
        print('No thorsync for VOltage')
    VolTime = VolTime- VolTime[0]
    CalTime = CalTime - CalTime[0]
    firstCal = IndexCal[0]
    if np.size(Indexvol) == 0:
        firstVol = firstCal
    else:
        firstVol = Indexvol[0]
    
    # plt.figure()
    # plt.plot(calFrame)
    # plt.plot(VolFrame)
    # plt.plot(DAQ)
    # plt.show()
    
    VolAX, CALax, CalTracePlot, VolTracePLOT = SyncTrace(VolTime,CalTime,firstCal, firstVol,CalTrace, VolTrace)
    
    CalTracePlot = np.array(CalTracePlot)
    print(np.shape(CalTracePlot))
    CALrANGE = np.arange(150,170,1)
    dfCalTrace = conDeltaF(CalTracePlot,CALrANGE)
    VolTracePLOT = np.array(VolTracePLOT)
    VolrANGE = np.arange(1100,1300,1)
    detrandVol = detrend_func(VolTracePLOT)
    dfVolTrace = conDeltaF(detrandVol,VolrANGE)
    print(np.shape(dfVolTrace))
    plotVolCal(input_folder, VolAX, CALax, dfCalTrace , dfVolTrace,)
    
    
    VoldDF = pd.DataFrame(detrandVol)
    VoldDF.loc[len(VoldDF)] = VolAX
    print(np.shape(VoldDF))
    PathVoldCsv = os.path.join(input_folder,'VOLd.csv')
    VoldCsv = VoldDF.to_csv(PathVoldCsv,)

    
    MiceName = input("what is mice name?")
    ImagingSeshion = input("saeshion number")
    ImagingDate = input("What is the imaging DATE?")
    NumberOfFov = input("how many ROI in the FOV?")
    pathForFov = input("Insert FOV path?")
    NotesAboutTrace = input("Notes?")
    
    VolDF = pd.DataFrame(dfVolTrace)
    VolDF.loc[len(VolDF)] = VolAX
    print(np.shape(VolDF))
    PathVolCsv =  os.path.join(input_folder,'VOL.csv')
    CalDF = pd.DataFrame(dfCalTrace)
    CalDF.loc[len(CalDF)] = CALax
    print(np.shape(CalDF))
    PathCalCsv = os.path.join(input_folder,'CAL.csv')
    CalCsv = CalDF.to_csv(PathCalCsv)
    
    
    DB = pd.read_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\MetaData.csv')
    NewRow = {'Mice': MiceName, 'Imaging_Seshion':ImagingSeshion, 'Imaging_Date':ImagingDate, 'Num_of_ROI':NumberOfFov,'Notes':NotesAboutTrace, 'Link':pathForFov}
    DB = DB.append(NewRow, ignore_index = True)







