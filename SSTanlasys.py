import h5py
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc
from scipy.optimize import curve_fit
from scipy.ndimage import filters 
import tifffile as tiff
from scipy.signal import find_peaks, correlate,correlation_lags
import os
import plotly.express as px
import csv
import pandas as pd
from roipoly import MultiRoi
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import ast
#from SomaAnalsys import detactCS
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
import scipy.stats as stat
import statsmodels.api as sm

def calculate_firing_rate(spike_indices, trace ,window_size, step_size,calAx, caltRACE,time):
    # Initialize an array to hold the firing rates
    firing_rates = []
    window_starts = range(0, len(caltRACE) - (window_size + 1), step_size)
    cBinn = []
    vBinn =[]

    # Calculate the firing rate for each window
    for start in window_starts:
        differenceS = np.abs(time - calAx[start])
        Startidx = np.argmin(differenceS)
        end = start + window_size
        differenceE = np.abs(time - calAx[end])
        Endidx = np.argmin(differenceE)
        spikes_in_window = [spike for spike in spike_indices if Startidx <= spike < Endidx]
        firing_rate = len(spikes_in_window) / (window_size/30)  # FR = number of spikes / window size
        firing_rates.append(firing_rate)
        cBinn.append(calAx[start])
        
        vBinn.append(Startidx)
    calAvg = meanCal(window_starts,caltRACE)

    return firing_rates,calAvg,cBinn, vBinn

def smoothTRACE (calcium_trace,ws):
    calcium_trace = pd.Series(calcium_trace)
    calcium_trace = calcium_trace.interpolate(method='linear')
    # smoothing Vm with a running average (10 point, ∼20 ms) boxcar procedure
    calcium_trace = calcium_trace.rolling(ws, center=True).mean()
    calcium_trace = calcium_trace.fillna(method='ffill').fillna(method='bfill')

    return calcium_trace


def binData(trace, binsize,time,calAx):
    BinnedTRace =[]
    cBinnedidx = []
    binindex = np.arange(0,len(trace),binsize)
    for i in range(0,len(binindex)):
        if i < len(binindex) - 1:
            Cbin = trace[binindex[i]:binindex[i+1]]
            differenceS = np.abs(calAx - time[binindex[i]])
            Startidx = np.argmin(differenceS)
        else:
            Cbin = trace[i:-1]
            differenceS = np.abs(calAx - time[binindex[i]])
            Startidx = np.argmin(differenceS)
        
        cBinnedidx.append(Startidx)  
        BinnedTRace.append(Cbin)
    return BinnedTRace,binindex,cBinnedidx

def calFiringRate (binindex,spikes,timeax):
    fr = []
    for i in range(len(binindex)):
        if  i == 0 and not binindex[0]  == 0:
            differencee = np.abs(spikes - binindex[i])
            finifh = np.argmin(differencee)
            time = timeax[0:binindex[i]]
            NofS = spikes[0:finifh]
            cFR = NofS/time


        if i < len(binindex) -1 and binindex[0]  == 0:
            differences = np.abs(spikes - binindex[i])
            start = np.argmin(differences)
            differencee = np.abs(spikes - binindex[i+1])
            finifh = np.argmin(differencee)
            time = timeax[binindex[i+1]] - timeax[binindex[i]]
            NofS = len(spikes[start:finifh])
            cFR = NofS/time
        

        elif i < len(binindex) and  not binindex[0]  == 0 and not i == 0: 
            differences = np.abs(spikes - binindex[i-1])
            start = np.argmin(differences)
            differencee = np.abs(spikes - binindex[i])
            finifh = np.argmin(differencee)
            time = timeax[binindex[i]] - timeax[binindex[i-1]]
            NofS = len(spikes[start:finifh])
            cFR = NofS/time
        fr.append(cFR)
    return fr


def meanCal (binn, tRACE):
    avgS = []
    for i in range(len(binn)):
        if i < len(binn) -1:
            currBin = tRACE[binn[i]:binn[i+1]]
            avgS.append(np.mean(currBin))
        

        elif i  == len(binn) -1 : 
            currBin = tRACE[binn[i]:-1]
            avgS.append(np.mean(currBin))
    return avgS

def calslope (binindex,cBinInd, smoothTrace,fr):
    slope = []
    frDiffrance = []
    frSlope = []
    #fr = fr[0:-1]
    for i in range(len(cBinInd)):
        if i < len(cBinInd) - 1:
             yfr = fr[i+1] - fr[i]
             xfr = (binindex[i+1] + binindex[i])*0.0020007934920317394
             y = smoothTrace[int(cBinInd[i+1])] - smoothTrace[int(cBinInd[i])]
             x = (cBinInd[i+1] - cBinInd[i])*0.03334305120167189
             frDiffrance.append(fr[i+1]-fr[i])
             frSlope.append(yfr/xfr)
        else:
            y = smoothTrace[len(smoothTrace)-1] - smoothTrace[int(cBinInd[i])]
            x = (len(smoothTrace) - cBinInd[i])*0.03334305120167189
        slope.append(y*100/x)
    return slope,frDiffrance

def LongLIST (LIST):
    lL = []
    for i in range(len(LIST)):
        lL = lL +LIST[i]
    return(lL)


def diffcal (trace):
    diffVec = []
    for i in range(len(trace)):
        if i < len(trace) -1:
            diffVec.append(trace[i+1] - trace[i]) 
    return(diffVec)

def corrCall (difFR,difCal):
    frToCal = np.array([difFR,difCal])

    slope, intercept, r_value, p_value, std_err = linregress(frToCal[0,:], frToCal[1,:])
    #print(slope)
    #print(p_value)
    #r = np.corrcoef(frToCal[0,:], frToCal[1,:])
    rPearson = stat.pearsonr(frToCal[0,:], frToCal[1,:])[0]    # Pearson's r
    rSpear =  stat.spearmanr(frToCal[0,:], frToCal[1,:])[0]   # Spearman's rho
    rKend  =stat.kendalltau(frToCal[0,:], frToCal[1,:])[0]  # Kendall's tau
    x_fit = np.linspace(min(difFR), max(difFR), 100)
    y_fit = slope * x_fit + intercept
    return frToCal,rPearson,rSpear,rKend,x_fit,y_fit


def calcCrossCoralation(FR,calDF):
    
    correlation = sm.tsa.stattools.ccf(FR,calDF, adjusted=False, nlags = 25)
    lags = np.arange(0,25,1)
    OpDirecorrelation = sm.tsa.stattools.ccf(calDF, FR, adjusted=False, nlags = 25)
    flipOCORR = np.flip(OpDirecorrelation)
    lagsO = np.arange(-25,0,1)
    flagsO = lagsO
    FullCORR = np.hstack([flipOCORR[0:-1], correlation])
    FullLag = np.hstack([flagsO[1:], lags])
    return FullCORR,FullLag

def crosCorDiffWS (spike_indices, trace ,window_size, calT,volT,calciumTrace,StepS):  
    corrarray = []
    fig3= go.Figure()
    for ws in window_size:
        step_size = StepS
        sWsc = ws/30
        frc2, calAvg2, Binnedidxws2,vBinnedidxws2 = calculate_firing_rate(spike_indices,trace,ws,step_size,calT,calciumTrace,volT)
        #ampCws2 = meanCal(Binnedidxws2,calciumTrace)
        corrl, corrlLag = calcCrossCoralation(frc2,calAvg2)
        if ws == window_size[-1]:
            minLen = len(corrl)
            maxT = volT[vBinnedidxws2[-2]]
            
        if ws == window_size[0]:
            maxLEN = len(corrl)
            x = corrlLag *0.1
        # Calculate padding lengths for both sides
        total_padding = maxLEN - len(corrl)
        start_padd = total_padding//2
        end_padd = total_padding - start_padd
    
    # Pad the normalized correlation to the maximum length with NaNs
        padded_correlation = np.pad(corrl, (start_padd,end_padd), mode='constant', constant_values=np.nan)
        corrarray.append(padded_correlation)
        corrws = go.Scatter(
        x=corrlLag,
        y=corrl,  # Assuming there's only one ROI in TraceVolL
        mode='markers',
        name='differance in cal and fr',
        line=dict(color='blue', width=2, backoff=0.6),showlegend=False 
        )

        fig3.add_traces (go.Scatter(x=x, y=padded_correlation, name=f"croos correlation for window size{sWsc}"))
    fig3.show()
    fcorrarray = np.vstack(corrarray)
    fifcorrarray = fcorrarray[:,0:minLen]

    
    return fcorrarray,corrlLag,x, fig3


def CrosscorrDifparts(FR,tAx, calDF,nSw,nSs,p):
    window_starts = np.arange(0, len(FR) - nSs , nSs)
    Pcorrarray = []

    # Calculate the firing rate for each window
    for start in window_starts:
        end = start + nSw
        if end > len(FR) - 1:
            end = len(FR) -1
        bFR = FR[start:end]
        bcalDF = calDF[start:end]
        corrP,lags = calcCrossCoralation(bFR,bcalDF)
        if start == window_starts[0]:
            maxl = len(corrP)
            endT = tAx[start]
            y = np.arange(0,maxl,1)
            y = (y - (maxl//2))*0.1
         # Calculate padding lengths for both sides
        total_padding = maxl - len(corrP)
        start_padd = total_padding//2
        end_padd = total_padding - start_padd
    
    # Pad the normalized correlation to the maximum length with NaNs
        padded_correlation = np.pad(corrP, (start_padd,end_padd), mode='constant', constant_values=np.nan)
        Pcorrarray.append(padded_correlation)
    


    # Calculate the firing rate for each window
    pathFig6 =os.path.join(p,f'CroscORDelay{nSs}{nSw}.html')
    pathFigSVG6 = os.path.join(p,f'CroscORDelay{nSs}{nSw}.svg')
    pathFigEPS6 = os.path.join(p,f'CroscORDelay{nSs}{nSw}.png')
    

    fcorrarray = np.vstack(Pcorrarray)
    # print(fcorrarray[49,:])
    # print(Pcorrarray[49])
  
    ffcorrarray = np.transpose(fcorrarray)
    # print(ffcorrarray[:,49])
    x = np.arange(1,len(window_starts)+1,1)
    # print(ffcorrarray[12,:])
    # print(ffcorrarray[12,49])
    df = pd.DataFrame(ffcorrarray)
    # print(df.iloc[12,:])
    # print(ffcorrarray[12,:])
    #print(ffcorrarray[:,90])
    import plotly.graph_objects as go

    # Create heatmap using graph_objects for better control
    heatmap = go.Heatmap(
        z=df.values,  # Pass data to heatmap
        x=x,          # x-axis values (e.g., window number)
        y=y,          # y-axis values (e.g., time lags)
        colorscale='RdBu_r'
    )

    # Create a figure
    fig6 = go.Figure(data=heatmap)

    # Customize layout for heatmap size and figure size
    fig6.update_layout(
        title=f'Cross Correlation for window size of {nSw} time points step size {nSs} time points',
        xaxis_title="Window number",
        yaxis_title=f"Time lags, {nSs} frames apart",
        autosize=False,
        width=1200,  # Total figure width
        height=800,  # Total figure height
        margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins to fit the heatmap
    )

    # Customize the axis tick labels to make sure they don't affect the plot size
    fig6.update_xaxes(
        tickangle=45,  # Rotate x-axis labels to avoid overlap
        automargin=True  # Automatically adjust margins if necessary
    )
    fig6.update_yaxes(
        automargin=True  # Automatically adjust margins for y-axis
    )

    # Show the figure
    fig6.show()

    # Save as HTML
    fig6.write_html(pathFig6)

    # Save as SVG with higher resolution
    fig6.write_image(pathFigSVG6, format="svg", width=1200, height=800, scale=1000)
    fig6.write_image(pathFigEPS6, format="png")


    # fig6 = px.imshow(df,
    #                 labels=dict(x="window number", y=f"time lags,{nSs} frames apart",),
    #                 title= f'Cross Corralation for windo size of {nSw} time points step Size {nSs} time points',
    #                 x = x,
    #                 y = y,
    #                 color_continuous_scale='RdBu_r',
    #     )
    # # fig6.update_layout(width=1200, height=2000)
    
  
   
 
    
    # fig6.show()
    # fig6.write_html(pathFig6)
    # fig6.write_image(pathFigSVG6, format="svg",  scale=4)
    # #print(fig6.trace)

    return df,x,y,fig6

if __name__  == "__main__":
    DB = pd.read_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\MetaData.csv')
    r = DB[DB['SNR']>5.5]
    AllPath = list(r['Link'])
    #AllPath = DB['Link']
    AllFR = []
    AllCal = []
    AlldifFR =[]
    AlldifCal = []
    AllFRws = []
    AllCalws = []
    AlldifFRws =[]
    AlldifCalws = []
    for i in range(len(AllPath)- 1, len(AllPath)):
        # Path = AllPath[i]
        Path = r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\flipfun\srugc8\09-06-2024-srugc8-r-s1\1'
        TracePathCal = os.path.join(Path,'CAL.csv')
        TracePathVol = os.path.join(Path,'VOL.csv')
        TracePathConv = os.path.join(Path,'ConvSignal.csv')
        TracePathSPIKE = os.path.join(Path,'spiketimeHC.csv')
        #print(TracePathSPIKE)
        spikeTime = pd.read_csv(TracePathSPIKE)
        spikeTime = np.array(spikeTime.iloc[:,-1])
        spike = list(spikeTime)
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
        TraceN = (TraceC+abs((min(TraceC))))
        TraceN = TraceN/max(TraceN)
        ConvTrace = pd.read_csv(TracePathConv)
        ConvTrace = np.array(ConvTrace)
        ConvTrace = ConvTrace[:,1:]
        ConvAX = ConvTrace[-1,:]
        TraceCv = ConvTrace[0,:]
        #plt.plot(CalAX,TraceC)
        # plt.show()
        fig100 = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig100.add_trace(go.Scatter(x=VolAX, y=Trace, name="Voltage"),secondary_y=False,)
        fig100.add_trace(go.Scatter(x= CalAX, y= TraceC, name="Calcium", yaxis='y2'),secondary_y=True,)

        # Add figure title
        fig100.update_layout(title_text="calcium and voltage togeter")

        # Set x-axis title
        fig100.update_xaxes(title_text="Time(ms)")

        # Set y-axes titles
        fig100.update_yaxes(title_text="<b>Voltage</b> df/f", secondary_y=False)
        fig100.update_yaxes(title_text="<b>Calcium</b> df/f", secondary_y=True)
        #fig100.show()

        ws = 15
        sWs = ws/30
        StepS = 1
        #BinnedTRace,binindex,cBinnedidx = binData(Trace, ws,VolAX,CalAX)
        #(binindex)
        smoothCal = smoothTRACE(TraceC,40)
        #frB = calFiringRate(binindex,spike,VolAX)firing_rates,calAvg,cBinn, window_starts
        frc, AvgC, Binnedidxws,vBinnedidxws = calculate_firing_rate(spike,Trace,ws,StepS,CalAX, TraceC, VolAX)
        #print(vBinnedidxws)
        #print(Binnedidxws)
        #AllFR.append(frB)
        AllFRws.append(frc)
        #ampC = meanCal(cBinnedidx,TraceC)
        #ampCws = meanCal(Binnedidxws,TraceC)
        corrar,xH, yH, fig102 = CrosscorrDifparts(frc,VolAX ,AvgC, nSw=500,nSs = 1, p = Path)
        print(np.shape(corrar))
        AllCalws.append(AvgC)
        #AllCal.append(ampC)
        #difFR = diffcal(frB)
        difFRws = diffcal(frc)
        #AlldifFR.append(difFR)
        AlldifFRws.append(difFRws)
        #difCal = diffcal(ampC)
        difCalws = diffcal(AvgC)
        #AlldifCal.append(difCal)
        AlldifCalws.append(difCalws)
        correlationSC,lagsSC = calcCrossCoralation(frc,AvgC)
        
        #plt.show()

        #frToCal,rPearson,rSpear,rKend,x_fit,y_fit = corrCall(difFR,difCal)
        frToCalws,rPearsonws,rSpearws,rKendws,x_fitws,y_fitws, = corrCall(difFRws,difCalws)

        wsRange = [8,15,30,45,60,75,90]
        StepS = 3
        corrarray,corrlLag, Xax, fig103 = crosCorDiffWS (spike,Trace,wsRange,CalAX,VolAX,TraceC,StepS)
        Fcorrarray = np.reshape(corrarray,(np.size(corrarray,1),np.size(corrarray,0)))
        maxLEN = (len(range(0, len(Trace) - wsRange[0] + 1, StepS)) -1)
        mInlen = -(len(range(0, len(Trace) - wsRange[0] + 1, StepS)) -1)
        y = np.arange(0,len(wsRange),1)
        x = Xax
        
        df = pd.DataFrame(corrarray)
        print(np.shape(df))
        pathFig =os.path.join(Path,f'CroscSCDifrrenSizeW.html')

        fig = px.imshow(df,
                        labels=dict(x="tIMElag", y="window size", color="Correlation"),
                        title=('croos corralation of whole cell for difrent window size and step zise of 0.1 ms'),
                        x = x,
                        y = y,
                        color_continuous_scale='RdBu_r', origin='lower'
                    )
        
        #fig.show()
        fig.write_html(pathFig)

        
        #print(np.shape(corrarray))
        
        # print(rPearson)
        # print(rSpear)
        # print(rKend)
        fig101 = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        pathFig101 =os.path.join(Path,f'FRtOcALws{sWs}.html')
        pathFigSvg101 = os.path.join(Path,f'FRtOcALws{sWs}.svg')
        fig101.add_trace(go.Scatter(x=Binnedidxws, y=AvgC, name="aVG CAL"),secondary_y=False,)
        fig101.add_trace(go.Scatter(x= Binnedidxws, y=AvgC, mode='markers',
                                name="AVG CAL", marker=dict(color='red', size=5, symbol='x')),secondary_y=False)
        fig101.add_trace(go.Scatter( x= [VolAX[time] for time in vBinnedidxws], y= frc, name="FR"),secondary_y=True,)
        fig101.add_trace(go.Scatter(x= [VolAX[time] for time in vBinnedidxws], y= frc, mode='markers',name="FR",
                    marker=dict(color='black', size=5, symbol='x')), secondary_y=True,)

        # Add figure title
        fig101.update_layout(title_text=f"calcium and voltage togeter {sWs}")

        # Set x-axis title
        fig101.update_xaxes(title_text="Time(ms)")

        # Set y-axes titles
        fig101.update_yaxes(title_text="<b>FR</b>", secondary_y=False)
        fig101.update_yaxes(title_text="<b>AVG DF/F</b>", secondary_y=True)

        #fig101.show()
        fig101.write_html(pathFig101)
        




        #print(np.shape(frToCal))
        # text =f"<b>Corralation coaficents:</b><br>Pearson = {rPearson}<br> SpearMAN = {rSpear}<br> Kendall = {rKend}"


        # fig = make_subplots(specs=[[{"secondary_y": True}]])

        # # Add traces
        # pathFig =os.path.join(Path,f'FRtOcAL{sWs}.html')
        # fig.add_trace(go.Scatter(x=[CalAX[time] for time in cBinnedidx], y=ampC, name="aVG CAL"),secondary_y=False,)
        # fig.add_trace(go.Scatter(x=[CalAX[time] for time in cBinnedidx], y=ampC, mode='markers',
        #                         name="AVG CAL", marker=dict(color='red', size=10, symbol='x')),secondary_y=False)
        # fig.add_trace(go.Scatter( x= [VolAX[time] for time in binindex], y= frB, name="FR"),secondary_y=True,)
        # fig.add_trace(go.Scatter(x= [VolAX[time] for time in binindex], y= frB, mode='markers',name="FR",
        #             marker=dict(color='black', size=10, symbol='x')), secondary_y=True,)

        # # Add figure title
        # fig.update_layout(title_text="calcium and voltage togeter")

        # # Set x-axis title
        # fig.update_xaxes(title_text="Time(ms)")

        # # Set y-axes titles
        # fig.update_yaxes(title_text="<b>FR</b>", secondary_y=False)
        # fig.update_yaxes(title_text="<b>AVG DF/F</b>", secondary_y=True)

        # #fig.show()
        # fig.write_html(pathFig)

        # slope,frDiffrance = calslope(binindex,cBinnedidx,smoothCal,frB)
        # frDiffrancel= frDiffrance
        
        # pathFigCorr = os.path.join(Path,f'corrSC{sWs}.html')
        # # Add the linear fit line
        # corr = go.Scatter(
        #     x=frToCal[0,:],
        #     y=frToCal[1,:],  # Assuming there's only one ROI in TraceVolL
        #     mode='markers',
        #     name='differance in cal and fr',
        #     line=dict(color='blue', width=2, backoff=0.6),showlegend=False 
        # )

        # fig2 = go.Figure(data=[corr])

        # fig2.add_trace(go.Scatter(
        #     x=x_fit,
        #     y=y_fit,
        #     mode='lines',
            
        #     showlegend=False,

        #     line=dict(color='blue', width=1,dash='dash'),

        # ))

    
        # fig2.add_annotation(
        #     x=10, y=max(frToCal[1,:]), # Text annotation position
        #     xref="x", yref="y", # Coordinate reference system
        #     text=text, # Text content
        #     showarrow=False # Hide arrow 
        # )
        # text =f"<b>Corralation coaficents:</b><br>Pearson = {rPearson}<br> SpearMAN = {rSpear}<br> Kendall = {rKend}"
        # # Update the layout to set the background to transparent and add black axis lines
        # fig2.update_layout(
        #     title_text=f'differance in cal and frfor a window size of {sWs} sec',
        #     plot_bgcolor='rgba(0,0,0,0)',  # Set the plot area background to transparent
        #     paper_bgcolor='rgba(0,0,0,0)',  # Set the paper (outside the plot area) background to transparent
        #     xaxis=dict(
        #         showline=True,  # Show the x-axis line
        #         linecolor='black',  # Set the color of the x-axis line to black
        #         title_text="<b>fr spikes per sec<b>"
        #     ),
        #     yaxis=dict(
        #         showline=True,  # Show the y-axis line
        #         linecolor='black',  # Set the color of the y-axis line to black
        #         title_text="<b>calcim mean df/f</b>"
        #     ),
        #     legend=dict(
        #         x=0.9,
        #         y=0.1,
        #         traceorder="reversed",
        #     )
        #  )

        # # fig2.show()
        
        # fig2.write_html(pathFigCorr)

        


        pathFigCorr = os.path.join(Path,f'corrSCws{sWs}.html')
        # Add the linear fit line
        corrws = go.Scatter(
            x=frToCalws[0,:],
            y=frToCalws[1,:],  # Assuming there's only one ROI in TraceVolL
            mode='markers',
            name='differance in cal and fr',
            line=dict(color='blue', width=2, backoff=0.6),showlegend=False 
        )

        fig3 = go.Figure(data=[corrws])

        fig3.add_trace(go.Scatter(
            x=x_fitws,
            y=y_fitws,
            mode='lines',
            
            showlegend=False,

            line=dict(color='blue', width=1,dash='dash'),

        ))
        text =f"<b>Corralation coaficents:</b><br>Pearson = {rPearsonws}<br> SpearMAN = {rSpearws}<br> Kendall = {rKendws}"

    
        fig3.add_annotation(
            x=2*max(frToCalws[0,:]), y=max(frToCalws[1,:]), # Text annotation position
            xref="x", yref="y", # Coordinate reference system
            text=text, # Text content
            showarrow=False # Hide arrow 
        )
        
        
        # Update the layout to set the background to transparent and add black axis lines
        fig3.update_layout(
            title_text=f'differance in cal and frfor a window size of {sWs} sec',
            plot_bgcolor='rgba(0,0,0,0)',  # Set the plot area background to transparent
            paper_bgcolor='rgba(0,0,0,0)',  # Set the paper (outside the plot area) background to transparent
            xaxis=dict(
                showline=True,  # Show the x-axis line
                linecolor='black',  # Set the color of the x-axis line to black
                title_text="<b>fr spikes per sec<b>"
            ),
            yaxis=dict(
                showline=True,  # Show the y-axis line
                linecolor='black',  # Set the color of the y-axis line to black
                title_text="<b>calcim mean df/f</b>"
            ),
            legend=dict(
                x=0.9,
                y=0.1,
                traceorder="reversed",
            )
        )

        fig3.show()
        
        fig3.write_html(pathFigCorr)
        fig102.show()



        pathFig1000 =os.path.join(Path,f'Summry.html')
        fig1000 = make_subplots(rows=3, cols = 2, specs=[[{"colspan": 1,"secondary_y": True},{"colspan": 1}],
            [{"colspan": 2, "secondary_y": True},None], 
            [{"colspan": 2},None]],
                subplot_titles=("Raw Trace", "Cross correlation of whole trace for diffrent size window", "FR to avg calcium", "Cross corralation of diffrent parts of trace"), 
        )
        
        for trace in fig100.data:
            if trace.name.lower() == 'voltage':
                fig1000.add_trace(trace, row=1, col=1,secondary_y=False)
            if not trace.name.lower() == 'voltage':
                fig1000.add_trace(trace, row=1, col=1,secondary_y=True)

        
            

            

        # Add all traces from fig2 to the second subplot
        for trace in fig101.data:
            if trace.name.lower() == 'avg cal':
                fig1000.add_trace(trace, row=2, col=1,secondary_y=False)
            if not trace.name.lower() == 'avg cal':
                fig1000.add_trace(trace, row=2, col=1,secondary_y=True)
            
            

        #Add all traces from fig3 to the third subplot
        for trace in fig102.data:
            fig1000.add_trace(trace, row=3, col=1)

        # Add all traces from fig4 to the fourth subplot
        for trace in fig103.data:
            fig1000.add_trace(trace, row=1, col=2)
        # fig1000.add_trace(fig100.data[0], row=1, col=1)
        # fig1000.add_trace(fig101.data[0], row=2, col=1)
        # fig1000.add_trace(fig102.data[0], row=3, col=1)
        # fig1000.add_trace(fig103.data[0], row=2, col=2)


        # Adjust layout for the figure
        fig1000.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),  # Adjust the margins
            title_text="Summarized Data",
            height=800,  # Adjust the height as necessary
        )

        # Update x and y axes for each subplot
        # For the first subplot (row=1, col=1)
        fig1000.update_xaxes(title_text="Time(sec)", row=1, col=1)
        fig1000.update_yaxes(title_text="<b>Voltage</b> df/f", secondary_y=False, row=1, col=1)
        fig1000.update_yaxes(title_text="<b>Calcium</b> df/f", secondary_y=True, row=1, col=1)

        # For the second subplot (row=1, col=2)
        fig1000.update_xaxes(title_text="Time Lag (sec)", row=1, col=2)
        fig1000.update_yaxes(title_text="Correlation", row=1, col=2)

        # For the third subplot (row=2, col=1)
        fig1000.update_xaxes(title_text="Time(sec)", row=2, col=1)
        fig1000.update_yaxes(title_text="<b>AVG DF/F</b>", secondary_y=False, row=2, col=1)
        fig1000.update_yaxes(title_text="<b>FR</b>", secondary_y=True, row=2, col=1)

        # For the fourth subplot (row=3, col=1)
        fig1000.update_xaxes(title_text="window number", row=3, col=1)
        fig1000.update_yaxes(title_text="Time lag (sec)", row=3, col=1,autorange='reversed')



    

        fig1000.update_layout(
        coloraxis1=dict(
            colorscale='RdBu_r',
            colorbar=dict(
                title="Correlation",
                
                len=0.25,  # Adjust length of the color bar relative to the subplot height
                y=0.12,  # Center the color bar vertically within its subplot
                #yanchor='middle',
                x=0.95,  # Position the color bar close to the subplot
                xanchor='left'
                ) ),                   
            )
        

        fig1000.show()
        fig1000.write_html(pathFig1000)
        fig1000.write_image(os.path.join(Path, "sst.svg"), format="svg",  scale=4)





    # # Add traces
    # FAlldifFR = LongLIST(AlldifFR)
    # FAlldifCal = LongLIST(AlldifCal)
    # frToCal,rPearson,rSpear,rKend,x_fit,y_fit = corrCall(FAlldifFR,FAlldifCal)
    # sWs = ws/30
    #     # Add the linear fit line
    # corr = go.Scatter(
    #     x=frToCal[0,:],
    #     y=frToCal[1,:],  # Assuming there's only one ROI in TraceVolL
    #     mode='markers',
    #     name='differance in cal and fr',
    #     line=dict(color='blue', width=2, backoff=0.6),showlegend=False 
    # )

    # fig4 = go.Figure(data=[corr])

    # fig4.add_trace(go.Scatter(
    #     x=x_fit,
    #     y=y_fit,
    #     mode='lines',
        
    #     showlegend=False,

    #     line=dict(color='blue', width=1,dash='dash'),

    # ))
    # for i in range(len(AlldifFR)):
    #     fig4.add_trace(go.Scatter(
    #     x=AlldifFR[i],
    #     y=AlldifCal[i],
    #     mode='markers',
    #     name = f'cell num{i}',
        
    #     showlegend=True

    #     ))


    # text =f"<b>Corralation coaficents:</b><br>Pearson = {rPearson}<br> SpearMAN = {rSpear}<br> Kendall = {rKend}"

    # fig4.add_annotation(
    #     x=10, y=max(frToCal[1,:]), # Text annotation position
    #     xref="x", yref="y", # Coordinate reference system
    #     text=text, # Text content
    #     showarrow=False # Hide arrow 
    # )
    # # Update the layout to set the background to transparent and add black axis lines
    # fig4.update_layout(
    #     title_text=f'differance in cal and frfor a window size of {sWs} sec',
    #     plot_bgcolor='rgba(0,0,0,0)',  # Set the plot area background to transparent
    #     paper_bgcolor='rgba(0,0,0,0)',  # Set the paper (outside the plot area) background to transparent
    #     xaxis=dict(
    #         showline=True,  # Show the x-axis line
    #         linecolor='black',  # Set the color of the x-axis line to black
    #         title_text="<b>diff in fr spikes per sec<b>"
    #     ),
    #     yaxis=dict(
    #         showline=True,  # Show the y-axis line
    #         linecolor='black',  # Set the color of the y-axis line to black
    #         title_text="<b> diff in calcim mean df/f</b>"
    #     ),
    #     legend=dict(
    #         x=0.9,
    #         y=0.1,
    #         traceorder="reversed",
    #     )
    # )

    # fig4.show()
    # pathFigCorr =  f'Z:\Adam-Lab-Shared\Data\Michal_Rubin\stat\sst\calVolCorr{sWs}.html'
    # fig4.write_html(pathFigCorr)




    FAlldifFRws = LongLIST(AlldifFRws)
    FAlldifCalws = LongLIST(AlldifCalws)
    frToCalws,rPearsonws,rSpearws,rKendws,x_fitws,y_fitws = corrCall(FAlldifFRws,FAlldifCalws)
    sWs = ws/500
        # Add the linear fit line
    corrws = go.Scatter(
        x=frToCalws[0,:],
        y=frToCalws[1,:],  # Assuming there's only one ROI in TraceVolL
        mode='markers',
        name='differance in cal and fr',
        line=dict(color='blue', width=2, backoff=0.6),showlegend=False 
    )

    fig5 = go.Figure(data=[corrws])

    fig5.add_trace(go.Scatter(
        x=x_fitws,
        y=y_fitws,
        mode='lines',
        
        showlegend=False,

        line=dict(color='blue', width=1,dash='dash'),

    ))
    for i in range(len(AlldifFRws)):
        fig5.add_trace(go.Scatter(
        x=AlldifFRws[i],
        y=AlldifCalws[i],
        mode='markers',
        name = f'cell num{i}',
        
        showlegend=True

        ))


    text =f"<b>Corralation coaficents:</b><br>Pearson = {rPearsonws}<br> SpearMAN = {rSpearws}<br> Kendall = {rKendws}"

    fig5.add_annotation(
        x=2*max(frToCalws[0,:]), y=max(frToCalws[1,:]), # Text annotation position
        xref="x", yref="y", # Coordinate reference system
        text=text, # Text content
        showarrow=False # Hide arrow 
    )
    # Update the layout to set the background to transparent and add black axis lines
    fig5.update_layout(
        title_text=f'differance in cal and frfor a window size of {sWs} sec',
        plot_bgcolor='rgba(0,0,0,0)',  # Set the plot area background to transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Set the paper (outside the plot area) background to transparent
        xaxis=dict(
            showline=True,  # Show the x-axis line
            linecolor='black',  # Set the color of the x-axis line to black
            title_text="<b>diff in fr spikes per sec<b>"
        ),
        yaxis=dict(
            showline=True,  # Show the y-axis line
            linecolor='black',  # Set the color of the y-axis line to black
            title_text="<b> diff in calcim mean df/f</b>"
        ),
        legend=dict(
            x=0.9,
            y=0.1,
            traceorder="reversed",
        )
    )

    fig5.show()
    pathFigCorrws =  f'Z:\Adam-Lab-Shared\Data\Michal_Rubin\stat\sst\calVolCorrws{sWs}.html'
    fig5.write_html(pathFigCorrws)
    c = 5
