import h5py
import re
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
import ast
#from SomaAnalsys import detactCS
from scipy.stats import linregress
from SSTanlasys import  smoothTRACE
from Calciumdetection import str_to_list,baseTrace,CorrectedBasline,compute_dff,detect_events,Baseclicy
from calciumStat import choosenSpike, SpikeISI
from TRY import LongLIST




def choosenSpikeN (CalIdx,SpikeId,nspike):
    finalSpikeId = []
    finalNspike = []
    finalISI = []
    finalISIIDX = []
    NonChoseSpike = []
    NonChosenNspike = []
    finalCalId = []
    NonChoseCal = []
    startSpike = []
    z = 0
    for i,vi in enumerate(SpikeId):
        
        if i < len(SpikeId) -1 and i > 0:
            lSpike = max(SpikeId[i])
            lSpikeP = max(SpikeId[i-1])
            firstSpike = min(SpikeId[i])
            firstSpikeN = min(SpikeId[i+1])
            if lSpike < int(firstSpikeN - 166) and lSpikeP < int(firstSpike - 166):
                finalSpikeId.append(SpikeId[i])
                finalCalId.append(CalIdx[i])
                finalNspike.append(nspike[i])
                startSpike.append(SpikeId[i][0])
                x = SpikeId[i]
                if nspike[i] > 1:
                    finalISI.append(SpikeISI(SpikeId[i]))
                # finalISIIDX.append(range(z,z+nspike[i]-1))
                # z = z+nspike[i]-1
            elif lSpike >= int(firstSpikeN - 166) and lSpikeP >= int(firstSpike - 166):
                NonChoseSpike.append(SpikeId[i])
                NonChoseCal.append(CalIdx[i])
                NonChosenNspike.append(nspike[i])
                # z = z+nspike[i]-1
        elif i < len(SpikeId) -1 and i == 0:
            lSpike = max(SpikeId[i])
            firstSpike = min(SpikeId[i])
            firstSpikeN = min(SpikeId[i+1])
            if lSpike < int(firstSpikeN - 166):
                # finalISIIDX.append(range(z,z+nspike[i]-1))
                # z = z+nspike[i]-1
                if nspike[i] > 1:
                    finalISI.append(SpikeISI(SpikeId[i]))
                finalSpikeId.append(SpikeId[i])
                finalCalId.append(CalIdx[i])
                finalNspike.append(nspike[i])
                startSpike.append(SpikeId[i][0])
            elif lSpike >= int(firstSpikeN - 166):
                NonChoseSpike.append(SpikeId[i])
                NonChoseCal.append(CalIdx[i])
                NonChosenNspike.append(nspike[i])
                #z = z+nspike[i]-1
        elif not i < len(SpikeId) -1:
            lSpike = max(SpikeId[i])
            lSpikeP = max(SpikeId[i-1])
            firstSpike = min(SpikeId[i])
            if CalIdx[i][-1] - CalIdx[i][0] >= 500 and lSpikeP < int(firstSpike - 166):
                finalSpikeId.append(SpikeId[i])
                finalCalId.append(CalIdx[i])
                finalNspike.append(nspike[i])
                startSpike.append(SpikeId[i][0])
                if nspike[i] > 1:
                    finalISI.append(SpikeISI(SpikeId[i]))
                # x = IsI[z:z+nspike[i]-1]
                # finalISIIDX.append(range(z,z+nspike[i]-1))
                # z = z+nspike[i]-1
    return finalSpikeId,finalCalId,finalNspike,NonChoseSpike,NonChoseCal,NonChosenNspike,startSpike,finalISI


def intList(lst_str):
    intlist = []
    # If the input is a string, use ast.literal_eval to safely parse it
    for c in lst_str:
        if isinstance(c, str):
            numbers = re.findall(r'np\.(?:int64|float64)\(([\d\.eE+-]+)\)', c)
            lst = []
            for num in numbers:
                    # Check if the number contains a decimal point or scientific notation to determine if it's a float
                if '.' in num or 'e' in num or 'E' in num:
                    lst.append(float(num))
                else:
                    lst .append(int(num))  # Convert to int if it's an integer

            intlist.append(lst)      
    
        else:
            # If it's already a list, convert each element
            intlist.append([float(x) if isinstance(x, float) else int(x) for x in lst_str]) 
        
    # # Iterate over elements and convert to Python int
    # for i in range(len(lst)):
    #     intlist.append([int(x) for x in list[i]])
    
    return intlist

def FindIDXn(spike, Voltime,Caltime):
    volS = Voltime[spike[0]]
    volE = Voltime[spike[-1]]
    differenceS = np.abs(Caltime - volS)
    differenceE = np.abs(Caltime - volE)
    Startidx = np.argmin(differenceS) - 1
    EndIDX = Startidx + 500
    if EndIDX > len(Caltime) - 1:
        EndIDX = len(Caltime) - 1
    return [Startidx,EndIDX]


def devCaln (spikeL, Voltime,Caltime):
    CaLIdx = []
    for v in spikeL:
        CaLIdx.append(FindIDXn(v,Voltime,Caltime))
    return(CaLIdx)

def Fullcal (TR,time):
    xp = time
    x = np.linspace(0,time[-1],int(time[-1]*1000))
    fp = TR
    fCal = np.interp(x,xp,fp)
    return  x, fCal

def BaseCal (calT):
    STDc = np.std(calT)
    baseT = [i for i in calT if i < STDc]
    return baseT

def findCidx (val,ampIDx,cal,tranId):
    pId = np.where(cal == val)[0][0]
    HalfDif = val/2.0
    left_idx = np.where(cal[:ampIDx] <= HalfDif)[0]
    right_idx = np.where(cal[ampIDx:] <= HalfDif)[0] + ampIDx

    if not left_idx.size or not right_idx.size:
        return None, None  # Unable to find the half-maximum crossing points

    # Select the last point before the peak that crosses the half-maximum
    left_crossing = left_idx[-1]
    
    # Select the first point after the peak that crosses the half-maximum
    right_crossing = right_idx[0]
    # startL = HalfDif[tranId[0]:pId+5]
    # endL = HalfDif[pId:pId + 1000]
    # # plt.plot(HalfDif, linewidth = 1)
    # # plt.plot(cal, linewidth = 0.5)
    # #plt.show()
    # rHalf = tranId[0]+ np.argmin(startL)
    # dHalf = pId+ np.argmin(endL)
    return left_crossing,right_crossing

def calS (CALt,CALtime, calIDX):
    Amp = []
    halfR = []
    FullHw = []
    FullHwInd = []
    AmpId = []
    for i,c in enumerate(calIDX):
        tranz = CALt[c[0]:c[1]]
        #tranzAX = CALtime[c[0]:c[1]]
        #tForF = CALt[c[0]:]
        if i == len(calIDX) -2:
            v =22
        peak = max(tranz[0:300])
        ampID = c[0]+np.where(tranz == peak)[0]
        AmpId.append(ampID[0])
        hR,hD = findCidx(peak,ampID[0],CALt,c)
        FullHwInd.append([hR,hD])
        if not hR == None:
            if c[0] <= hR or c[-1] + 300 <= hD :
                halfR.append(0)
                halfWidth = 0
            if c[0] > hR and c[-1] + 300 > hD:
                halfR.append(CALtime[hR] - CALt[c[0]])
                halfWidth = CALtime[hD] - CALtime[hR]
        else:
           halfR.append(0)
           halfWidth = 0 
        FullHw.append(halfWidth)
        Amp.append(peak)
    return halfR,FullHw,Amp, AmpId,FullHwInd


if __name__ == '__main__':
    DB = pd.read_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\MetaData.csv')
    reader_obj = csv.reader(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\MetaData.csv')
    # SpikeIn = pd.read_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\SpikeInfo.csv')  
    # Spikes = SpikeIn['BurstSpikeNum']
    # Spikes = Spikes.apply(str_to_list)
    # vSpikeIdx = SpikeIn['BurstSpikeIdx']
    # vSpikeIdx = vSpikeIdx.apply(str_to_list)
    # print(vSpikeIdx)

    #want to add function to open the path base on inserting mice name and imaging seshion
    r = DB[DB['SNR']>5.5]
    p = list(r['Link'])
    z = 22
    path = p[z]
    # IsIpath = os.path.join(path,'VolIsI.csv')
    # IsI = pd.read_csv(IsIpath)
    # IsI = np.array(IsI)
    # IsIt = IsI[:,1:]
    burstpath = os.path.join(path,'SpikeBurstInfo30ms.csv')
    burstInfo  =pd.read_csv(burstpath)
    print(burstInfo.keys())
    BurstIdx =burstInfo['BurstIndex']
    try:
        BurstIdx = BurstIdx.astype(str).apply(ast.literal_eval)
    except:
        BurstIdx =  BurstIdx.tolist()
        BurstIdx= intList(BurstIdx)
    print(type(BurstIdx))
    print(BurstIdx[1])
    lBurstIdx = LongLIST(BurstIdx)
    ISIburst = burstInfo['ISIofBurst']
    try:
        ISIburst = ISIburst.astype(str).apply(ast.literal_eval)
    except:
        ISIburst =  ISIburst.tolist()
        ISIburst= intList(ISIburst)
    print(ISIburst)
    SpikeNum =burstInfo['NumberSpike']
    print(type(SpikeNum))
    print(SpikeNum)
    SpikeNum = SpikeNum.tolist()
    SpikeNum = np.array(SpikeNum)
    print(type(SpikeNum[0]))
    TracePathCal = os.path.join(path,'CAL.csv')
    TracePathVol = os.path.join(path,'VOL.csv')
    VolTrace = pd.read_csv(TracePathVol)
    VolTrace = np.array(VolTrace)
    VolTrace = VolTrace[:,1:]
    VolAX = VolTrace[-1,:]
    Trace = VolTrace[0,:]
    CalTrace = pd.read_csv(TracePathCal)
    CalTrace = np.array(CalTrace)
    CalTrace = CalTrace[:,1:]
    CalAX = CalTrace[-1,:]
    TraceC = CalTrace[0,:]

    smoCal = smoothTRACE(TraceC,3)
    print(np.shape(smoCal))
    fCalAx,ffCal = Fullcal(smoCal,CalAX)
    print(np.shape(ffCal))
    # plt.plot(CalAX, TraceC,'o')
    # plt.plot(CalAX, smoCal,'x')
    # plt.plot(fCalAx, ffCal,linewidth = 0.3)
    # plt.show()
    # print(fCalAx[0])
    # print(fCalAx[500])
    # print(fCalAx[1000])
    # print(fCalAx[2500])
    # print(fCalAx[8000])
    calIDn = devCaln(BurstIdx, VolAX, fCalAx)
    chSpikeID, chCalID, chSpikeN, NchSpikeID, NchCalID, NchSpikeN, startEvent, BurstISI = choosenSpikeN(calIDn,BurstIdx,SpikeNum)

    HalfRise, fwhm, amplitude,amplitudeId,fwhmId = calS(ffCal,fCalAx,chCalID)
    chBurst = [item for item in chSpikeID if len(item)>1]
    lburstIdx = LongLIST(chBurst)
    chSS = [item for item in chSpikeID if len(item) == 1]
    lsIDX = LongLIST(chSS)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=VolAX, y=Trace, name="Voltage"),secondary_y=False,)
    fig.add_trace(go.Scatter(x= [VolAX[time] for time in lburstIdx], y= [Trace[time] for time in lburstIdx], mode='markers',name="burst",
                    marker=dict(color='brown', size=10, symbol='x')), secondary_y=False,)   
    fig.add_trace(go.Scatter(x= [VolAX[time] for time in lsIDX], y= [Trace[time] for time in lsIDX], mode='markers',name="Simple Spike",
                marker=dict(color='green', size=10, symbol='x')), secondary_y=False,)
    fig.add_trace(go.Scatter(x= CalAX, y= TraceC, name="Calcium"),secondary_y=True,)
    
        # Add figure title
    fig.update_layout(title_text="calcium and voltage togeter")

        # Set x-axis title
    fig.update_xaxes(title_text="Time(ms)")

        # Set y-axes titles
    fig.update_yaxes(title_text="<b>Voltage</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Calcium</b>", secondary_y=True)

    fig.show()


    calStat = pd.DataFrame(data = {'FinalspikeIDX': chSpikeID,'FinalSpikeNum':chSpikeN, 'calAmp': amplitude,'calAmpIDX':amplitudeId,'RiseTime':HalfRise,'fwhm':fwhm,'fwhmIDX':fwhmId},columns=['FinalspikeIDX','FinalSpikeNum','calAmp','calAmpIDX','RiseTime','fwhm','fwhmIDX'])
    calStatPath = os.path.join(path,'FinalStat30ms.csv')
    calStat.to_csv(calStatPath,index=False)


# NoslowTrace = baseTrace(ffCal)
# trF = NoslowTrace
# ddf,Bs, BsT,BsIDX = compute_dff(trF)
# NoTranzTrace,th, CorBL,CorSt = CorrectedBasline(trF)

# cTR = go.Scatter(
#     x=fCalAx,
#     y=ddf,  # Assuming there's only one ROI in TraceVolL
#     mode='lines',
#     name='no slow transients', 
#     line=dict(color='blue', width=2)
    
# )
# bs = go.Scatter(
#     x=fCalAx,
#     y=BsT,  # Assuming there's only one ROI in TraceVolL
#     mode='lines',
#     name='no big transiant', 
#     line=dict(color='green', width=2)
    
# )
# fig = go.Figure(data=[cTR,bs])
# fig.add_shape(
#     # Line Vertical
#     dict(
#         type="line",
#         x0=0,
#         y0=Bs,
#         x1=CalAX[-1],
#         y1=Bs,
#         line=dict(
#             color="pink",
#             width=2,
#             dash="dashdot",
#         ),
#     )
# )

# # Add title and labels
# fig.update_layout(
#     title='ROI Trace with Spike Markers',
#     xaxis_title='Time (s)',
#     yaxis_title='Amplitude',
# )




# # Show the plot

# #fig.show()






# TR = go.Scatter(
#     x=fCalAx,
#     y=trF,  # Assuming there's only one ROI in TraceVolL
#     mode='lines',
#     name='no slow transients', 
#     line=dict(color='blue', width=2)
    
# )
# corTR = go.Scatter(
#     x=fCalAx,
#     y=ffCal,  # Assuming there's only one ROI in TraceVolL
#     mode='lines',
#     name='no big transiant', 
#     line=dict(color='green', width=2)
    
# )
# fig2 = go.Figure(data=[TR,corTR])
# fig2.add_shape(
#     # Line Vertical
#     dict(
#         type="line",
#         x0=0,
#         y0=th,
#         x1=CalAX[-1],
#         y1=th,
#         line=dict(
#             color="pink",
#             width=2,
#             dash="dashdot",
#         ),
#     )
# )

# # Add title and labels
# fig2.update_layout(
#     title='ROI Trace with Spike Markers',
#     xaxis_title='Time (s)',
#     yaxis_title='Amplitude',
# )




# # Show the plot

# #fig2.show()



# corRange = Baseclicy(fCalAx,trF)

# #corRange = round(0.1*len(trF))
# #plt.plot()
# Base = sorted(trF)
# CalR = trF[corRange]
# transients,sig,meanB = detect_events(trF,CalR,CalR,4,1)



# spike_trace = go.Scatter(
#     x=[fCalAx[time] for time in transients],
#     y=[trF[time] for time in transients],
#     mode='markers',
#     marker=dict(size=8, color='red'),
#     showlegend=False,
# )
# trc = go.Scatter(
#     x=fCalAx,
#     y=trF,
#     mode='markers',
#     marker=dict(size=3, color='green'),
#     showlegend=False,
# )

# fig1 = go.Figure(data=[TR,trc, spike_trace])
# fig1.add_shape(
#     # Line Vertical
#     dict(
#         type="line",
#         x0=0,
#         y0=sig*2,
#         x1=CalAX[-1],
#         y1=sig*2,
#         line=dict(
#             color="Black",
#             width=2,
#             dash="dashdot",
#         ),
#     )
# )

# # Add title and labels
# fig1.update_layout(
#     title='transient detecion',
#     xaxis_title='Time (s)',
#     yaxis_title='Amplitude',
# )


# # Show the plot
# fig1.show()

# Notes = input("any notes")
# calDet = pd.DataFrame(data = {'basiline': meanB,'std':sig,'transient':[transients],'Notes':Notes},columns=['basiline','std','transient','notes'])
    

# PathCalT = os.path.join(path,'calciumTranAutRec.csv')    
# SaveSTat =input("Do Want to save stats?")
# #calDet = pd.concat([calDet, pd.DataFrame([NewRow])], ignore_index=True)
# if SaveSTat.lower() == 'y':    
#     calDet.to_csv(PathCalT,index=False)


