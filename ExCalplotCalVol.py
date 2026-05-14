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
import os
import csv
import pandas as pd
from roipoly import MultiRoi
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.io import loadmat
import glob
import plotly.io as pio
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
#from caiman.base.movies import movie
from scipy.ndimage import center_of_mass
import sys
# Add the folder containing your BatchRunSpatialFootLocal.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\code')
from BatchRunSpatialFootLocal import get_rois_mask,volS,trace_extraction,save_volpy_plots, plot_roi_masks
from SpikeDetection2 import SNRcalculate
pio.renderers.default = "vscode"

def calDF(Vol):
    bacground = np.mean([0,0,0])
    NoBackVol = Vol - bacground
    
    dfVOL = np.zeros((np.size(Vol,0), np.size(Vol,1)))
    for i in range(0,(np.size(NoBackVol,0))):
        currTrace = NoBackVol[int(i),]
        NoBacksORT = np.sort(currTrace)
        F0 = np.percentile(currTrace,8) # 8th percentile
        #F0 = np.mean(NoBacksORT[0:round(len(currTrace)*0.1)])
        dfVOL[i,:] = (currTrace-F0)/F0
    return dfVOL

def calDFvol(Vol):
    
    NoBacksORT = np.sort(Vol)
    F0 = np.mean(NoBacksORT[0:round(len(Vol)*0.1)])    
    dfVOL = (Vol-F0)/F0
    return dfVOL

def plot_roi_masks(rois):
    """
    Plots all ROI masks overlaid on a blank image.
    """
    n_rois, h, w = rois.shape
    canvas = np.zeros((h, w))
    
    for i, mask in enumerate(rois):
        canvas += mask * (i+1)  # Add intensity for each ROI

    plt.figure(figsize=(6, 6))
    plt.imshow(canvas, cmap='nipy_spectral')
    plt.title(f'{n_rois} ROI Masks')
    plt.colorbar(label='ROI Index')
    plt.axis('off')
    # add numbers at centroid of each ROI
    for i, mask in enumerate(rois):
        if mask.sum() > 0:  # skip empty masks
            cy, cx = center_of_mass(mask)   # (row, col)
            plt.text(cx, cy, str(i+1),
                    color="white", fontsize=8, ha="center", va="center",
                    weight="bold")

    plt.show()

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# def plotVolCal(path, VolAX, CALax, CalTrace, VolTrace,
#                 Name="syncVolCAL.html"):
#     """
#     Plot calcium + voltage trace for one cell, plus mean images of both movies.

#     Parameters
#     ----------
#     path : str
#         Save path for figures.
#     VolAX : array
#         Time axis for voltage.
#     CALax : array
#         Time axis for calcium.
#     CalTrace : 2D array (frames × cells)
#         Calcium traces.
#     VolTrace : 2D array (frames × cells)
#         Voltage traces.
#     VolMovie : 3D array (frames × height × width)
#         Voltage movie.
#     CalMovie : 3D array (frames × height × width)
#         Calcium movie.
#     cell_index : int
#         Which cell to plot (0-based).
#     Name : str
#         File name for saved html/svg.
#     """

#     # pick the trace for the chosen cell
#     CalTraceToPlot = CalTrace.flatten()
#     VolTraceToPlot = VolTrace.flatten()

   

#     fig = make_subplots(specs=[[{"secondary_y": True}]])
#     fig.add_trace(go.Scatter(x=VolAX, y=VolTrace.squeeze(), name="Voltage",line=dict(color='red', width=3)),secondary_y=False,)
#     fig.add_trace(go.Scatter(x= CALax, y= CalTrace.squeeze(), name="Calcium",line=dict(color='blue', width=3)),secondary_y=True,)
#     fig.update_layout(title_text="calcium and voltage togeter")
#     fig.update_xaxes(title_text="Time(ms)")
#     fig.update_yaxes(title_text="<b>Calcium</b>", secondary_y=True)
#     fig.update_yaxes(title_text="<b>Voltage</b> ", secondary_y=False)
#     fig.show()
#     fig.update_layout(
#     title="cal-vol",
#     plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
#     paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
    
#     fig.update_layout(
#         width=1250,  # Set the figure width in pixels
#         height=375, # Set the figure height in pixels
#     )
    

#     # save
#     pathFig = os.path.join(path, Name)
#     #fig.write_html(pathFig)
#     # fig.write_image(os.path.join(path, "syncVolCAL.svg"), format="svg")
#     # fig.show()
from scipy.signal import butter, filtfilt
def highpass_filter(trace, fs, cutoff_hz=0.5, order=3):
    ny = 0.5 * fs
    b, a = butter(order, cutoff_hz / ny, btype='highpass')
    return filtfilt(b, a, trace)


def fps_from_indices(frame_idx, SR=30000.0):
    frame_idx = np.asarray(frame_idx, dtype=float).ravel()
    frame_idx = frame_idx[np.isfinite(frame_idx)]
    if frame_idx.size < 2:
        return np.nan, np.array([])
    frame_idx = np.unique(frame_idx.astype(int))
    if frame_idx.size < 2:
        return np.nan, np.array([])
    dt = np.diff(frame_idx) / float(SR)  # seconds
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan, np.array([])
    fps = 1.0 / np.median(dt)
    return float(fps), dt


def estimate_cal_sr_from_thorsync(thor_h5_path, SR=30000.0, default_sr=30.0):
    if (thor_h5_path is None) or (not os.path.isfile(thor_h5_path)):
        return float(default_sr)
    try:
        with h5py.File(thor_h5_path, "r") as f:
            ygalvo = None
            for gk in list(f.keys()):
                grp = f[gk]
                if not hasattr(grp, "keys"):
                    continue
                for dk in list(grp.keys()):
                    if str(dk).lower() == "ygalvo":
                        ygalvo = np.asarray(grp[dk]).squeeze()
                        break
                if ygalvo is not None:
                    break
        if ygalvo is None:
            return float(default_sr)
        ygalvo = np.asarray(ygalvo, dtype=float).ravel()
        if ygalvo.size < 500:
            return float(default_sr)
        L = np.convolve(np.diff(ygalvo), np.ones(5) / 5.0, mode="valid")
        cal_idx, _ = find_peaks(-L, height=0.08, distance=80)
        if cal_idx.size <= 1:
            cal_idx, _ = find_peaks(L, height=0.08, distance=80)
        if cal_idx.size < 2:
            return float(default_sr)
        fps_cal, _ = fps_from_indices(cal_idx, SR=SR)
        if not np.isfinite(fps_cal) or fps_cal <= 0:
            return float(default_sr)
        return float(fps_cal)
    except Exception as e:
        print(f"[WARN] failed CALsr from ThorSync ({thor_h5_path}): {e}")
        return float(default_sr)

def plotVolCal(path, VolAX, CALax, CalTrace, VolTrace, spikeIDX, Name="syncVolCAL.html"):
    """
    Plot calcium + voltage trace for one cell.
    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Voltage trace (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=VolAX,
            y=VolTrace.squeeze(),
            name="Voltage",
            line=dict(color="red", width=2),
        ),
        secondary_y=False,
    )


    # Voltage trace (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=VolAX[spikeIDX],
            y=VolTrace[spikeIDX].squeeze(),
            name="spikes",
            mode="markers",                 # <-- show points instead of lines
        marker=dict(color="black", size=5),  # <-- black dots, adjust size if you want
    ),
        secondary_y=False,)
    
    # Calcium trace (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=CALax,
            y=CalTrace.squeeze(),
            name="Calcium",
            line=dict(color="blue", width=2),
        ),
        secondary_y=True,
    )

    # Layout
    fig.update_layout(
        title="Calcium and Voltage Together",
        xaxis_title="Time (ms)",
        yaxis_title="Voltage",
        yaxis2_title="Calcium",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        width=1250,
        height=375,
    )

    # Save before showing
    fig.write_html(os.path.join(path, Name))
    fig.write_image(os.path.join(path, "syncVolCAL.svg"))

    # Show last
    fig.show()

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # Voltage trace (left y-axis)
    fig2.add_trace(
        go.Scatter(
            x=VolAX,
            y=VolTrace.squeeze(),
            name="Voltage",
            line=dict(color="red", width=2),
        ),
        secondary_y=False,
    )


    # Voltage trace (left y-axis)
    fig2.add_trace(
        go.Scatter(
            x=VolAX[spikeIDX],
            y=VolTrace[spikeIDX].squeeze(),
            name="spikes",
            mode="markers",                 # <-- show points instead of lines
        marker=dict(color="black", size=4),  # <-- black dots, adjust size if you want
    ),
        secondary_y=False,)
    # Layout
    fig2.update_layout(
        title="Calcium and Voltage Together",
        xaxis_title="Time (ms)",
        yaxis_title="Voltage",
        yaxis2_title="Calcium",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        width=1250,
        height=375,
    )

    # Save before showing
    fig2.write_html(os.path.join(path, "spikeDET.html"))
    fig2.write_image(os.path.join(path, "spikeDET.svg"))
## load  vidios
homePath = r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc46\RL2\07-01-2025-ans\fov2'
rr = get_rois_mask(homePath)
plot_roi_masks(rr)
suit2Path = os.path.join(homePath,'sync','cal','suite2p','plane0','F.npy')
MeanImagePath = os.path.join(homePath,'Mean.tif')
MeanImage = tiff.imread(MeanImagePath)
ts_folder = glob.glob(os.path.join(homePath, "TS_*"))
if len(ts_folder) > 0:
    file_path = os.path.join(sorted(ts_folder)[-1], 'Episode_0000.h5')
else:
    file_path = None
CAL_SR = estimate_cal_sr_from_thorsync(file_path, SR=30000.0, default_sr=30.0)
#print(f"Estimated CALsr from ThorSync: {CAL_SR:.6f} Hz")
if os.path.exists(suit2Path):
        CalTrace = np.load(suit2Path,allow_pickle=True)
        selctROI = input("Enter ROI numbers (check relevnt file in suite2p folder):")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        selctROI =  selctROI.split(',')
        selctROI = [int(x) for x in selctROI]
        CalTrace = np.array(CalTrace[selctROI])


MiceName = input("what is mice name?")
globalImagingSeshion = input("seshion number")
ImagingDate = input("What is the imaging DATE?")

BrainState = input("Brain State?")
for i in range(len(rr)):
    mask = rr[i].astype(float)
    currP = os.path.join(homePath, f'cell{i}')
    suit2Path = os.path.join(currP,'suite2p','plane0','F.npy')
    suit2PathNorophil = os.path.join(currP,'suite2p','plane0','Fneu.npy')
    if os.path.exists(suit2Path):
        F = np.load(suit2Path, allow_pickle=True)
        Fneu = np.load(suit2PathNorophil, allow_pickle=True)

        selctROI = input("Enter ROI numbers (check relevnt file in suite2p folder):")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        selctROI =  selctROI.split(',')
        selctROI = [int(x) for x in selctROI]
        CalTrace = np.array(CalTrace[selctROI])
        F = F[selctROI]
        Fneu = Fneu[selctROI]

        # neuropil subtraction
        neuropil_coeff = 0.7
        CalTrace = F - neuropil_coeff * Fneu
        CalTrace =CalTrace[:,:]
    CalDf = calDF(CalTrace)
    TracPpath = os.path.join(currP,'volTrace.csv')
    SpikeTimePath = os.path.join(currP,'SpikeIdx.csv')
    spikeTime = pd.read_csv(SpikeTimePath)
    spikeTime = np.array(spikeTime).flatten()
    Trace = pd.read_csv(TracPpath)
    
    Trace = Trace.to_numpy().flatten()
    deTrendVol = highpass_filter(Trace,500)
    TraceDF = calDFvol(deTrendVol)
    TraceDF = np.abs(TraceDF)
  
    Snr = SNRcalculate(Trace,spikeTime)
    RelCal = CalTrace[i,:]
    RelCalDF = CalDf[i,:]
    CalPath = os.path.join(currP,'calTraceNB.csv')
    CalDFPath = os.path.join(currP,'calTraceNBdf..csv')
    VolDFPath = os.path.join(currP,'volTrace.csv')
    
    tVOL = np.linspace(0, (len(TraceDF)/500), len(TraceDF))
   
    tCal = np.linspace(0, (len(RelCalDF)/CAL_SR), len(RelCalDF))

    df = pd.DataFrame(RelCal)  # create df with column name
    df.to_csv(CalPath, index=False)
    df = pd.DataFrame(RelCalDF)  # create df with column name
    df.to_csv(CalDFPath, index=False)
    df = pd.DataFrame(TraceDF)  # create df with column name
    df.to_csv(VolDFPath, index=False)
    plotVolCal(currP,tVOL,tCal,RelCalDF,TraceDF,spikeTime)
    save = input("Do you want to save to SST database? (Y/N)")
    save = save.upper()
    if save == 'Y':
        
        NumberOfFov = f'cell{i}'
        pathForFov = currP
        #NotesAboutTrace = input("Notes?")
        
        if not os.path.exists(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\SST.csv'):
            DB = pd.DataFrame(columns=['Mice', 'Imaging_Seshion', 'Imaging_Date', 'Num_of_ROI','Notes', 'Link','brainState','SNR', 'CALsr'])
            DB.to_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\SST.csv',index=False)

        DB = pd.read_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\SST.csv')
        NewRow = {'Mice': MiceName, 'Imaging_Seshion':globalImagingSeshion, 'Imaging_Date':ImagingDate, 'Num_of_ROI':NumberOfFov,
                  'Link':pathForFov, 'brainState':BrainState, 'SNR' : Snr, 'CALsr': CAL_SR}
        DB = pd.concat([DB, pd.DataFrame([NewRow])], ignore_index=True)
        DB.to_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\SST.csv',index=False)



    save = input("Do you want to save to pyramidal database? (Y/N)")
    save = save.upper()
    if save == 'Y':
        # MiceName = input("what is mice name?")
        # ImagingSeshion = input("seshion number")
        # ImagingDate = input("What is the imaging DATE?")
        # NumberOfFov = input("how many ROI in the FOV?")
        # pathForFov = input("Insert FOV path:")
        # NotesAboutTrace = input("Notes?")
        NumberOfFov = f'cell{i}'
        pathForFov = currP
        if not os.path.exists(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\Pyr.csv'):
            DB = pd.DataFrame(columns=['Mice', 'Imaging_Seshion', 'Imaging_Date', 'Num_of_ROI','Notes', 'Link','brainState','SNR', 'CALsr'])
            DB.to_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\Pyr.csv',index=False)

        DB = pd.read_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\Pyr.csv')
        NewRow = {'Mice': MiceName, 'Imaging_Seshion':globalImagingSeshion, 'Imaging_Date':ImagingDate, 'Num_of_ROI':NumberOfFov,
                  'Link':pathForFov, 'brainState':BrainState, 'SNR' : Snr, 'CALsr': CAL_SR}
        DB = pd.concat([DB, pd.DataFrame([NewRow])], ignore_index=True)
        DB.to_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\Pyr.csv',index=False)
