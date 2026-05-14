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
from calVol_new import plotVolCal
from scipy.integrate import trapz
from scipy.signal import freqz, lfilter
from scipy import signal
import ast

# def hp_flter_median_method(trace):
#     window_size = 30 # 10 data points from each side for calculating median of a point
#     #trace = trace.to_numpy()
#     trace = trace[0]
#     filtered_signal = median_filter(trace, window_size)
#     median_filterdd = trace - filtered_signal
# #     median_filterdd = median_filterdd * [median_filterdd >0]
#     # median_filterdd = median_filterdd * [median_filterdd <0.1]
#     median_filterdd = median_filterdd.flatten()
#     return median_filterdd

# def design_fir_bandpass_filter(fs, f_p1, f_p2, f_s1, f_s2, N):

#     # Normalize frequencies
#     omega_p1 = 2 * np.pi * f_p1 / fs
#     omega_p2 = 2 * np.pi * f_p2 / fs

#     # Ideal impulse response of bandpass filter
#     alpha = (N - 1) / 2
#     n = np.arange(0, N)
#     h_d = np.where(n == alpha,
#                    (omega_p2 - omega_p1) / np.pi,
#                    (np.sin(omega_p2 * (n - alpha)) - np.sin(omega_p1 * (n - alpha))) / (np.pi * (n - alpha)))

#     # Window function (Hamming window)
#     w = np.hamming(N)

#     # Apply window to ideal impulse response
#     h = h_d * w

#     return h

# def apply_fir_filter(signal, fs, f_p1, f_p2, f_s1, f_s2, N):
   
#     # Design the filter
#     h = design_fir_bandpass_filter(fs, f_p1, f_p2, f_s1, f_s2, N)
    
#     # Apply the filter to the signal
#     filtered_signal = lfilter(h, 1.0, signal)
    
#     return filtered_signal


# def FIR_detection(trace, std_num, chosen_y=None):
#     th = trace.mean() + std_num * trace.std()
#     if chosen_y is not None:
#         th = chosen_y
#     spikes_time = signal.find_peaks(trace, height=th, distance=3)[0]
#     print(type(spikes_time))
#     spikes_time = list(spikes_time)
#     SNR = SNRcalculate(trace,spikes_time)
#     spikes_time = np.array(spikes_time)
#     return spikes_time, th, SNR

# def median_filter_detection(trace, fr, std_num, chosen_y=None):
#     hp_trace = hp_flter_median_method(trace)
#     th = hp_trace.mean() + std_num * hp_trace.std()
#     if chosen_y is not None:
#         th = chosen_y
#     spikes_time = signal.find_peaks(hp_trace, height=th, distance=3)[0]
#     print(type(spikes_time))
#     spikes_time = list(spikes_time)
#     SNR = SNRcalculate(hp_trace,spikes_time)
#     spikes_time = np.array(spikes_time)
#     return spikes_time, hp_trace, th, SNR


# def median_filter(signal, window_size):
#     filtered_signal = []
#     for i in range(len(signal)):
#         # Select a window of samples centered at the current sample
#         window_start = i - window_size // 2
#         window_end = i + window_size // 2 + 1
#         if (window_start >= 0) and (window_end <= len(signal)):
#             window = signal[window_start:window_end]
#             filtered_signal.append(np.median(window))
#         else:
#             if (window_start < 0):
#                 zeros = np.zeros(np.abs(window_start))
#                 filtered_signal.append(np.median(np.concatenate([zeros, signal[:window_end]])))
#             if (window_end > len(signal)):
#                 zeros = np.zeros(window_end - len(signal))
#                 filtered_signal.append(np.median(np.concatenate([signal[window_start:], zeros])))
#     return np.array(filtered_signal)

# def SNRcalculate(filter_trace,spikes_time):

#     spikes = filter_trace[spikes_time]
#     signal_amplitude = np.mean(spikes)

#     noise_mask = np.ones_like(filter_trace, dtype=bool)
#     noise_mask[spikes_time] = False
#     #noise_mask = list(noise_mask)
#     noise_region = filter_trace[noise_mask]
#     noise_level = np.std(noise_region)

#     SNR = signal_amplitude/noise_level
#     return(SNR)


# def signal_filter(sg, freq, fr, order=3, mode='high'):
#     normFreq = freq / (fr / 2)
#     b, a = signal.butter(order, normFreq, mode)
#     sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
#     return sg

# def spike_detector(traces, hp_freq, t_val, path):
#     spikes_count = []
#     spikes_timing = []
#     # high-pass filter the signal for spike detection
#     if np.ndim(traces) > 1 :
#        fr = np.shape(traces)[1]
#     if np.ndim(traces)== 1:
#        fr = len(traces)
#     print(fr)
#     flt_traces = signal_filter(traces, hp_freq, fr, order=5)
#     flt_traces = flt_traces - np.median(flt_traces)
#     bool_vec = np.zeros((len(flt_traces),fr), dtype=int)
#     for k in range(len(flt_traces)): 
#         t_avg = np.mean(flt_traces[k])
#         t_std = 2*np.std(flt_traces[k]) 
#         thresh = (t_val*t_std) + t_avg
#         spikes_indxs, _ = signal.find_peaks(flt_traces[k],thresh, distance=3)
#         bool_vec[k, spikes_indxs] = 1
#         spikes_timing.append(list(np.where(bool_vec[k] == 1)[0]))
#         # if all(bool_vec[k] == 0) == True:
#         #     spikes_timing.append(list(0))
#         # else:
#         #     spikes_timing.append(list(np.where(bool_vec[k][0]))
#     spikes_count.append(bool_vec.sum())
#     spikes_timing.append(np.where(bool_vec)[0])
      
#     roi_cols_name = ["roi_" + str(i+1) for i in range(len(traces))]
#     traces_df = pd.DataFrame(data = traces.T, columns = roi_cols_name)
#     spike_cols_name = ["spikes_" + str(i+1) for i in range(len(traces))]
#     spike_df = pd.DataFrame(data = bool_vec.T, columns = spike_cols_name)
#     df = pd.concat([traces_df,spike_df], axis = 1) 
#     return  thresh,flt_traces,spikes_count, spikes_timing, df


# def avgFR(vec,timeVec):
#     vec[vec.astype(bool)]
#     numSpike = np.sum(vec)
#     Time = timeVec[-1] - timeVec[0]
#     FR = numSpike / Time
#     return(FR)

def baseTrace(trace):
    corrected_series = np.zeros_like(trace)
    for i in range(len(trace)):
        start = max(0, i - 1500)
        end = min(len(trace), i + 1500)
        local_distribution = trace[start:end]
        percentile_value = np.percentile(local_distribution, 8)
        corrected_series[i] = trace[i] - percentile_value
    return corrected_series

def CorrectedBasline(trace):
    basline = np.median(trace)
    BASEstd = np.std(trace)
    corTrace = np.zeros_like(trace)
    for i in range(len(trace)):
        if trace[i]<(BASEstd):
            corTrace[i] = trace[i]
    thres = BASEstd
    CorBaline = np.mean(corTrace)
    CorStd = np.std(corTrace)
    return corTrace,thres, CorBaline,CorStd


def Baseclicy (Time, Trace):
   

   plt.plot(Time,Trace)
   plt.scatter(Time,Trace, color = 'r', s=5)
   plt.title('choose region whit now big transients')
   baserange = MultiRoi()
   base=[]
   endIdx = 0
   mark = 0
   t = baserange.rois.values()
   print(t)
   print(np.shape(t))
   

   for roi in baserange.rois.values():
       mark +=1
       difference = np.abs(Time - roi.x)
       
       add = np.argmin(difference)
       if not (mark % 2) == 0:
           startIdx = add
       if (mark % 2) == 0 and mark > 1:
           endIdx = add
           x = [item for item in np.arange(startIdx,endIdx,1)]
           base = base + x


   return base


def compute_dff(raw_fluorescence):
    # Initial baseline estimate
    threshold = 3 * np.std(raw_fluorescence)
    baseline_points = raw_fluorescence[raw_fluorescence < threshold]
    baseline_IDX = [i for i, val in enumerate(raw_fluorescence) if val in baseline_points]
    initial_baseline = np.mean(baseline_points)
    corrected_fluorescence = raw_fluorescence
    dff = corrected_fluorescence / initial_baseline
    return dff,initial_baseline,baseline_points,baseline_IDX

def detect_events(fluorescence_trace, baseline_region, sigma_region,sTH=3, eTH = 1):
    baseline = np.mean(baseline_region)
    sigma = np.std(sigma_region)
    events = []
    event_start = None
    
    for i, value in enumerate(fluorescence_trace):
        if event_start is None and value > baseline +  sTH*sigma:
            event_start = i
        elif event_start is not None and value < baseline + eTH*sigma:
            event_end = i
            event_peak = np.max(fluorescence_trace[event_start:event_end])
            event_amplitude = (event_peak - baseline) / sigma
            event_duration = event_end - event_start
            events = events + [event_start,event_end]
            
            event_start = None
    
    return events,sigma,baseline

def str_to_list(s):
    return ast.literal_eval(s)

def FindIDX(spike, Voltime,Caltime):
    volS = Voltime[spike]
    differenceS = np.abs(Caltime - volS)
    Startidx = np.argmin(differenceS) - 1
    EndIDX = Startidx + 13
    if EndIDX > len(Caltime) - 1:
        EndIDX = len(Caltime) - 1
    return [Startidx,EndIDX]


def devCal (spikeL, Voltime,Caltime):
    CaLIdx = []
    for v in spikeL:
        CaLIdx.append(FindIDX(v[0],Voltime,Caltime))
    return(CaLIdx)

def LongLIST (LIST):
    lL = []
    for i in range(len(LIST)):
        lL = lL +LIST[i]
    return(lL)
        



if __name__  == "__main__":

    
    DB = pd.read_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\MetaData.csv')
    r = DB[DB['SNR']>5.5]
    path = list(r['Link'])
    SpikeIn = pd.read_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\SpikeInfo.csv')  
    Spikes = SpikeIn['BurstSpikeNum']
    Spikes = Spikes.apply(str_to_list)
    vSpikeIdx = SpikeIn['BurstSpikeIdx']
    vSpikeIdx = vSpikeIdx.apply(str_to_list)

    for i in range (0,len(Spikes)):
        if i == 1:
            print(type(path))
            P = path[1]
            pp = 20
        TracePathVol = os.path.join(path[i],'VOL.csv')
        TracePathCal = os.path.join(path[i],'CAL.csv')
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
        sVspikeIDX = vSpikeIdx[i]
        lSpike = LongLIST(sVspikeIDX)
        calciumArea = devCal(sVspikeIDX, VolAX, CalAX)
        lCalIdx = LongLIST(calciumArea)
        PathCALa = os.path.join(path[i],'CalReg.csv')
        cacalciumAreaDf = pd.DataFrame(data = calciumArea)
        cacalciumAreaDf.to_csv(PathCALa,index=False)
        pathFig =os.path.join(path[i],'CalVOLselect.html')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=VolAX, y=Trace, name="Voltage"),secondary_y=False,)
        fig.add_trace(go.Scatter(x=[VolAX[time] for time in lSpike],y=[Trace[time] for time in lSpike], mode='markers',
                                name="SPikes - Voltage", marker=dict(color='red', size=10, symbol='x')),secondary_y=False)
        fig.add_trace(go.Scatter(x= CalAX, y= TraceC, name="Calcium"),secondary_y=True,)
        fig.add_trace(go.Scatter(x= [CalAX[time] for time in lCalIdx], y= [TraceC[time] for time in lCalIdx], mode='markers',name="AUC - Calcium",
                    marker=dict(color='black', size=10, symbol='x')), secondary_y=True,)

        # Add figure title
        fig.update_layout(title_text="calcium and voltage togeter")

        # Set x-axis title
        fig.update_xaxes(title_text="Time(ms)")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Voltage</b> yaxis title", secondary_y=False)
        fig.update_yaxes(title_text="<b>Calcium</b> yaxis title", secondary_y=True)

        fig.show()
        fig.write_html(pathFig)




    #want to add function to open the path base on inserting mice name and imaging seshion
    r =     list(DB['Link'])

    
    FR = 30
    std = 1
    path = r[5]
    print(path)
    PathCalT = os.path.join(path,'calciumTran.csv')
    print(path)
    TracePathCal = os.path.join(path,'CAL.csv')
    CalTrace = pd.read_csv(TracePathCal)
    CalTrace = CalTrace.to_numpy()
    TraceCalL = CalTrace[0:-1,:]
    CalAX = CalTrace[-1]
    CalAX = CalAX[1:]
    # print(np.shape(TraceVolL))
    TraceCalL = TraceCalL[:,1:]
    tr = np.reshape(TraceCalL,np.size(TraceCalL,1))
    NoslowTrace = baseTrace(TraceCalL)
    trF = np.reshape(NoslowTrace,np.size(NoslowTrace,1))
    ddf,Bs, BsT,BsIDX = compute_dff(trF)
    NoTranzTrace,th, CorBL,CorSt = CorrectedBasline(trF)

    cTR = go.Scatter(
        x=CalAX,
        y=ddf,  # Assuming there's only one ROI in TraceVolL
        mode='lines',
        name='no slow transients', 
        line=dict(color='blue', width=2)
        
    )
    bs = go.Scatter(
        x=CalAX,
        y=BsT,  # Assuming there's only one ROI in TraceVolL
        mode='lines',
        name='no big transiant', 
        line=dict(color='green', width=2)
        
    )
    fig = go.Figure(data=[cTR,bs])
    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=0,
            y0=Bs,
            x1=CalAX[-1],
            y1=Bs,
            line=dict(
                color="pink",
                width=2,
                dash="dashdot",
            ),
        )
    )

    # Add title and labels
    fig.update_layout(
        title='ROI Trace with Spike Markers',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
    )

 


    # Show the plot

    fig.show()






    TR = go.Scatter(
        x=CalAX,
        y=trF,  # Assuming there's only one ROI in TraceVolL
        mode='lines',
        name='no slow transients', 
        line=dict(color='blue', width=2)
        
    )
    corTR = go.Scatter(
        x=CalAX,
        y=tr,  # Assuming there's only one ROI in TraceVolL
        mode='lines',
        name='no big transiant', 
        line=dict(color='green', width=2)
        
    )
    fig2 = go.Figure(data=[TR,corTR])
    fig2.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=0,
            y0=th,
            x1=CalAX[-1],
            y1=th,
            line=dict(
                color="pink",
                width=2,
                dash="dashdot",
            ),
        )
    )

    # Add title and labels
    fig2.update_layout(
        title='ROI Trace with Spike Markers',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
    )

 


    # Show the plot

    fig2.show()



    corRange = Baseclicy(CalAX,trF)

    #corRange = round(0.1*len(trF))
    #plt.plot()
    Base = sorted(trF)
    CalR = trF[corRange]
    transients,sig,meanB = detect_events(trF,CalR,CalR)



    spike_trace = go.Scatter(
        x=[CalAX[time] for time in transients],
        y=[trF[time] for time in transients],
        mode='markers',
        marker=dict(size=8, color='red'),
        showlegend=False,
    )
    trc = go.Scatter(
        x=CalAX,
        y=trF,
        mode='markers',
        marker=dict(size=3, color='green'),
        showlegend=False,
    )

    fig1 = go.Figure(data=[TR,trc, spike_trace])
    fig1.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=0,
            y0=sig*2,
            x1=CalAX[-1],
            y1=sig*2,
            line=dict(
                color="Black",
                width=2,
                dash="dashdot",
            ),
        )
    )

    # Add title and labels
    fig1.update_layout(
        title='transient detecion',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
    )


    # Show the plot
    fig1.show()


    Notes = input("any notes")
    calDet = pd.DataFrame(data = {'basiline': meanB,'std':sig,'transient':[transients],'Notes':Notes},columns=['basiline','std','transient','notes'])
    

    
    
    # SaveSTat =input("Do Want to save stats?")
    # #calDet = pd.concat([calDet, pd.DataFrame([NewRow])], ignore_index=True)
    # if SaveSTat.lower() == 'y':    
    #     calDet.to_csv(PathCalT,index=False)






    


    #spikes_time, hp_trace, th, snr = median_filter_detection(TraceCalL, fr = FR, std_num = std, chosen_y=None)
    #print(np.shape(spikes_time))
  

    # DB.loc[DB[31],'SNR'] = snr
        
    # filtTR = go.Scatter(
    #     x=CalAX,
    #     y=hp_trace,  # Assuming there's only one ROI in TraceVolL
    #     mode='lines',
    #     name='Filter trace', 
    #     line=dict(color='green', width=2)
        
    # )

    # # threshold = go.Scatter(
    # #     x=VolAX,
    # #     y=thresh,  # Assuming there's only one ROI in TraceVolL
    # #     mode='lines',
    # #     name='ROI Trace', 
    # # )

    # # Create red dots at spike times
    # #spike_timesR = spikes_time[0]  # Assuming spikes_timing contains spike times for the single ROI
    # spike_traceR = go.Scatter(
    #     x=[CalAX[time] for time in spikes_time],
    #     y=[TraceCalL[0][time] for time in spikes_time],
    #     mode='markers',
    #     marker=dict(size=8, color='red'),
    #     showlegend=False,
    # )

    # trace = go.Scatter(
    #     x=CalAX,
    #     y=TraceCalL[0],  # Assuming there's only one ROI in TraceVolL
    #     mode='lines',
    #     name='ROI Trace',
    #     line=dict(color='blue', width=2, backoff=0.6)
    # )


    # fig2 = go.Figure(data=[filtTR,trace, spike_traceR])
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

    # fig2.show()
    # pathFig =  os.path.join(path,'Caldetection.html')
    # fig2.write_html(pathFig)


    # CaldDF = pd.DataFrame(spikes_time)
    # #VoldDF.loc[len(VoldDF)] = tVOL
    # print(path)
    # PeakTimeCsv = os.path.join(path,'CalPeack.csv')
    # spikes_timeCSV = CaldDF.to_csv(PeakTimeCsv,index=False)

    # fs = 30.0  # Sampling frequency in Hz
    # f_p1, f_p2 = 1.0, 5.0  # Passband frequencies in Hz (assuming calcium signals)
    # f_s1, f_s2 = 0.05, 6.0 
    # N = 101  # Filter order
    # filtered_signal = apply_fir_filter(TraceCalL, FR, f_p1, f_p2, f_s1, f_s2, N)
    # filtered_signal = filtered_signal.reshape((np.size(filtered_signal,1)))
    # spikes_timeF, thF, SNRF = FIR_detection(filtered_signal,std)
    



    # FIRflit = go.Scatter(
    #     x=CalAX,
    #     y=filtered_signal,  # Assuming there's only one ROI in TraceVolL
    #     mode='lines',
    #     name='Filter trace', 
    #     line=dict(color='green', width=2)
        
    # )

    # # threshold = go.Scatter(
    # #     x=VolAX,
    # #     y=thresh,  # Assuming there's only one ROI in TraceVolL
    # #     mode='lines',
    # #     name='ROI Trace', 
    # # )

    # # Create red dots at spike times
    # #spike_timesR = spikes_time[0]  # Assuming spikes_timing contains spike times for the single ROI
    # FIRspike_traceR = go.Scatter(
    #     x=[CalAX[time] for time in spikes_timeF],
    #     y=[TraceCalL[0][time] for time in spikes_timeF],
    #     mode='markers',
    #     marker=dict(size=8, color='red'),
    #     showlegend=False,
    # )

   

    # fig3 = go.Figure(data=[FIRflit,trace, FIRspike_traceR])
    # fig3.add_shape(
    #     # Line Vertical
    #     dict(
    #         type="line",
    #         x0=0,
    #         y0=thF,
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
    # fig3.update_layout(
    #     title='ROI Trace with Spike Markers',
    #     xaxis_title='Time (s)',
    #     yaxis_title='Amplitude',
    # )

 


    # # Show the plot

    # fig3.show()
    # pathFig3 =  os.path.join(path,'firCaldetection.html')
    # fig3.write_html(pathFig3)

