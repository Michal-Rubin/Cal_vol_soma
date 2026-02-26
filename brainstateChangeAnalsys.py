import h5py
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.signal as sc
from scipy.optimize import curve_fit
from scipy.ndimage import filters 
import tifffile as tiff
import plotly.tools as tls
from scipy.signal import find_peaks
import plotly.express as px
from scipy.optimize import least_squares
import os
import csv
import pandas as pd
from roipoly import MultiRoi
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import ast
from AnalasysFunction import plotFR,devideTr,plotVolCal,VolToCalIdx,CalSmooth,CorrWindow,ChooseSpk,CalInt,CalAmp,calculate_firing_rate,ChooseCom,LongLIST,SingleSpk,linear_model,quadratic_model,exponential_model,MeanRes,lagOptimaizre
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, linregress,ttest_ind
from NewinternueronsAnalsys import analyze_block
import math
from plotnine import ggsave
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



from plotnine import ggplot, aes, geom_line, geom_smooth, theme_bw, facet_wrap

def build_df(cal_list, fr_list, state, time_window=(-3, 3)):
    # Convert nested lists to arrays
    
    # pad calcium and firing rate traces
    cal_arr = pad_to_maxBig(cal_list, fill=0)
    fr_arr  = pad_to_maxBig(fr_list, fill=0)

    if cal_arr.ndim == 1:   # only one trial
        cal_arr = cal_arr[np.newaxis, :]  # make it 2D: (1, n_timepoints)
        fr_arr = fr_arr[np.newaxis, :]  # make it 2D: (1, n_timepoints)

    # Infer SRs from lengths
    cal_sr = cal_arr.shape[1] / (time_window[1] - time_window[0])
    fr_sr  = fr_arr.shape[1]  / (time_window[1] - time_window[0])

    # Resample calcium to FR length
    target_len = fr_arr.shape[1]
    cal_resampled = np.array([sc.resample(tr, target_len) for tr in cal_arr])

    # Time axis based on FR SR
    time = np.linspace(time_window[0], time_window[1], target_len)

    # Build tidy dataframe with all trials
    dfs = []
    for trial, (c, f) in enumerate(zip(cal_resampled, fr_arr)):
        df_trial = pd.DataFrame({
            "time": np.tile(time, 2),
            "value": np.concatenate([c, f]),
            "signal": ["Calcium"]*target_len + ["Firing rate"]*target_len,
            "trial": [trial]*2*target_len,
            "state": [state]*2*target_len
        })
        dfs.append(df_trial)

    return pd.concat(dfs, ignore_index=True)

def plot_state(df, state):
    p = (
        ggplot(df[df["state"] == state], aes("time", "value", color="signal", group="trial"))
        + geom_line(alpha=0.2)  # all trials faint
        + geom_smooth(aes(group="signal"), method="lowess", span=0.2, se=False, size=1.5)  # mean smoothed
        + facet_wrap("~signal", ncol=1, scales="free_y")  # separate Calcium & FR
        + theme_bw()
    )
    return p

def plot_traces_subplot(cal_traces, vol_traces, title="Motor ON"):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,   # 👈 no shared x-axis
        vertical_spacing=0.1,
        subplot_titles=("Calcium", "Voltage")
    )

    def add_traces(traces, row, name):
        # x-axis depends on trace length
        x_axes = [np.linspace(-3, 3, len(tr)) for tr in traces]

        # stack to compute mean/std on interpolated grid
        max_len = max(len(tr) for tr in traces)
        common_x = np.linspace(-3, 3, max_len)
        resampled = [np.interp(common_x, x, tr) for tr, x in zip(traces, x_axes)]
        resampled = np.vstack(resampled)
        mean_trace = resampled.mean(axis=0)
        std_trace  = resampled.std(axis=0)

        # all traces (native sampling)
        for tr, x in zip(traces, x_axes):
            fig.add_trace(
                go.Scatter(
                    x=x, y=tr,
                    mode="lines",
                    line=dict(color="rgba(150,150,150,0.4)", width=1),
                    showlegend=False
                ),
                row=row, col=1
            )

        # mean + shaded std (on common_x)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([common_x, common_x[::-1]]),
                y=np.concatenate([mean_trace-std_trace, (mean_trace+std_trace)[::-1]]),
                fill="toself",
                fillcolor="rgba(200,200,200,0.5)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False
            ),
            row=row, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=common_x, y=mean_trace,
                mode="lines",
                line=dict(color="black", width=3),
                name=f"{name} Mean"
            ),
            row=row, col=1
        )

    # Calcium subplot
    add_traces(cal_traces, row=1, name="Calcium")

    # Voltage subplot
    add_traces(vol_traces, row=2, name="Voltage")

    fig.update_layout(
        height=600,
        width=900,
        title_text=title,
        template="simple_white"
    )
    return fig
# def plot_traces_subplot(cal_traces, vol_traces,
#                        cal_axes=None, vol_axes=None, add_fill=True):
#     """
#     Plot multiple calcium and voltage traces with mean in bold and others in light colors.
#     cal_traces, vol_traces: lists of arrays (can be different lengths).
#     """

#     fig2 = make_subplots(specs=[[{"secondary_y": True}]])

#     # --- Voltage traces ---
#     for i, tr in enumerate(vol_traces):
#         x = np.linspace(-3, 3, len(tr)) if vol_axes is None else vol_axes[i]
#         fig2.add_trace(
#             go.Scatter(
#                 x=x,
#                 y=tr,
#                 mode="lines",
#                 line=dict(color="red", width=1),
#                 opacity=0.2,
#                 showlegend=(i == 0),
#                 name="Voltage (traces)",
#             ),
#             secondary_y=False,
#         )
#     # Voltage mean
#     min_len_vol = min(len(tr) for tr in vol_traces)
#     vol_stack = np.vstack([tr[:min_len_vol] for tr in vol_traces])
#     vol_mean = np.mean(vol_stack, axis=0)
#     vol_std = np.std(vol_stack, axis=0)
#     x_vol = np.linspace(-3, 3, min_len_vol)

#     fig2.add_trace(
#         go.Scatter(
#             x=x_vol,
#             y=vol_mean,
#             mode="lines",
#             line=dict(color="red", width=3),
#             name="Voltage (mean)",
#         ),
#         secondary_y=False,
#     )
#     if add_fill:
#         fig2.add_trace(
#             go.Scatter(
#                 x=np.concatenate([x_vol, x_vol[::-1]]),
#                 y=np.concatenate([vol_mean - vol_std, (vol_mean + vol_std)[::-1]]),
#                 fill="toself",
#                 fillcolor="rgba(255,0,0,0.2)",
#                 line=dict(color="rgba(255,0,0,0)"),
#                 name="Voltage ± STD",
#             ),
#             secondary_y=False,
#         )

#     # --- Calcium traces ---
#     for i, tr in enumerate(cal_traces):
#         x = np.linspace(-3, 3, len(tr)) if cal_axes is None else cal_axes[i]
#         fig2.add_trace(
#             go.Scatter(
#                 x=x,
#                 y=tr,
#                 mode="lines",
#                 line=dict(color="blue", width=1),
#                 opacity=0.2,
#                 showlegend=(i == 0),
#                 name="Calcium (traces)",
#             ),
#             secondary_y=True,
#         )
#     # Calcium mean
#     min_len_cal = min(len(tr) for tr in cal_traces)
#     cal_stack = np.vstack([tr[:min_len_cal] for tr in cal_traces])
#     cal_mean = np.mean(cal_stack, axis=0)
#     cal_std = np.std(cal_stack, axis=0)
#     x_cal = np.linspace(-3, 3, min_len_cal)

#     fig2.add_trace(
#         go.Scatter(
#             x=x_cal,
#             y=cal_mean,
#             mode="lines",
#             line=dict(color="blue", width=3),
#             name="Calcium (mean)",
#         ),
#         secondary_y=True,
#     )
#     if add_fill:
#         fig2.add_trace(
#             go.Scatter(
#                 x=np.concatenate([x_cal, x_cal[::-1]]),
#                 y=np.concatenate([cal_mean - cal_std, (cal_mean + cal_std)[::-1]]),
#                 fill="toself",
#                 fillcolor="rgba(0,0,255,0.2)",
#                 line=dict(color="rgba(0,0,0,0)"),
#                 name="Calcium ± STD",
#             ),
#             secondary_y=True,
#         )

#     # Layout
#     fig2.update_layout(
#         title="Calcium & Voltage (traces + mean ± STD)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)",
#         width=2500,
#         height=750,
#     )
#     fig2.update_xaxes(title_text="Time (s)")
#     fig2.update_yaxes(title_text="<b>Voltage</b>", secondary_y=False)
#     fig2.update_yaxes(title_text="<b>Calcium</b>", secondary_y=True)

    # Save
    #fig2.write_html(pathFig)
    #fig2.write_image(os.path.join(new_folder, f"volCal{k}MotorTS.svg"), format="svg")

    return fig2
# Sigmoid function
def sigmoid(x, L ,x0, k, b):
    return L / (1 + np.exp(-k*(x-x0))) + b
def flatten(nested):
    flat = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            flat.extend(flatten(item))  # go deeper
        else:
            flat
def pad_to_maxBig(traces, fill=0):
    """
    Pad a list of 1D arrays/lists to the same length with `fill` values.
    Padding is added to the BEGINNING of each trace.
    Returns a 2D NumPy array.
    """
    max_len = max(len(tr) for tr in traces)
    padded = np.full((len(traces), max_len), fill, dtype=float)
    for i, tr in enumerate(traces):
        padded[i, -len(tr):] = tr   # align to the end, pad at the start
    return padded
def pad_to_max(traces, fill=0):
    """
    Pad a list of 1D arrays/lists to the same length with `fill` value.
    Returns a 2D NumPy array (n_traces, max_len).
    """
    max_len = max(len(tr) for tr in traces)
    padded = np.full((len(traces), max_len), fill)
    for i, tr in enumerate(traces):
        padded[i, :len(tr)] = tr
    return padded
def BSchange(motor, cal, vol, spikeID, path):
    bsC = np.abs(np.diff(motor))
    bsCidx = np.argwhere(bsC == 1)
    volBS = []
    calBS = []
    volAxBs = []
    calAxBs = []
    spikeBs = []
    chFR = []
    calDiffA = []
    VolAX = np.linspace(0, (len(vol)/500), len(vol)) 
    CalAX = np.linspace(0, (len(cal)/30), len(cal))
    
    parentP = os.path.dirname(path)
    pathFig = os.path.join(parentP,r'changeT.html')
    #fig_Raw = make_subplots(rows=len(bsCidx), cols=1, shared_yaxes=True)
    figTrans = go.Figure()
    figTrans.add_trace(go.Scatter(x=VolAX, y = vol.squeeze(),line=dict(color='red', width=2),name="vol"))
    #figTrans.add_trace(go.Scatter(x=VolAX[bsCidx], y = vol[bsCidx].squeeze(),mode="markers",marker=dict(color="black", size=3),name="change"))
    print(VolAX[bsCidx])
    for xc in VolAX[bsCidx]:
        figTrans.add_vline(
            x=float(xc),
            line_width=3,       # thinner lines look nicer if many
            line_dash="dash",
            line_color="black"
        )
    figTrans.update_layout(
        title="cal-vol",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
    figTrans.write_html(pathFig)
    figTrans.update_layout(
        width=2500,  # Set the figure width in pixels
        height=750, # Set the figure height in pixels
    )
    figTrans.write_image(os.path.join(parentP, f'changeT.svg'), format="svg")
    MeanFR = []
    motS = []
    for i,r in enumerate(bsCidx):
        if bsCidx[0]< 200:
            if i < len(bsCidx)-1:
                sID = int(r)
                eID = bsCidx[i+1]
                spikeIn = [s for s in spikeID if sID <= s < eID]
                dur = VolAX[eID]-VolAX[sID]
                FR = len(spikeIn)/float(dur)
                MeanFR.append(FR)
            if i == len(bsCidx)-1:
                sID = r
                eID = -1
                spikeIn = [s for s in spikeID if sID <= s]
                dur = VolAX[eID]-VolAX[sID]
                FR = len(spikeIn)/float(dur)
                MeanFR.append(FR)
        if bsCidx[0]> 200:
            if i == 0:
                sID = 0
                eID = int(r)
                spikeIn = [s for s in spikeID if sID <= s < eID]
                dur = VolAX[eID]-VolAX[sID]
                FR = len(spikeIn)/float(dur)
                MeanFR.append(FR)
            if i < len(bsCidx)-1 and i > 0:
                sID = int(r)
                eID = bsCidx[i+1]
                spikeIn = [s for s in spikeID if sID <= s < eID]
                dur = VolAX[eID]-VolAX[sID]
                FR = len(spikeIn)/float(dur)
                MeanFR.append(FR)
            if i == len(bsCidx)-1:
                sID = r
                eID = -1
                spikeIn = [s for s in spikeID if sID <= s]
                dur = VolAX[eID]-VolAX[sID]
                FR = len(spikeIn)/float(dur)
                MeanFR.append(FR)
        if motor[bsCidx[i] -10] == 1:
            motS.append('ON')
        if motor[bsCidx[i] -10] == 0:
            motS.append('OFF')

        new_folder = os.path.join(path, f"transition{i}")  # subfolder name  
        # Create the folder if it doesn't exist
        os.makedirs(new_folder, exist_ok=True)
        pathFig = os.path.join(new_folder,f'bsSync{i}.html')
        volStart =  np.max([int(r) -1500,0])
        volEnd = np.min([int(r) + 1500,len(vol)-1])
        spikeIn = [s - volStart for s in spikeID if volStart <= s < volEnd]
        spikeB  = [s - volStart for s in spikeID if volStart <= s < r]
        spikeA  = [s - volStart for s in spikeID if r <= s < volEnd]
        frB = len(spikeB)/np.max([1,float(VolAX[r]-VolAX[volStart])])
        frA = len(spikeA)/np.max([float(VolAX[volEnd]-VolAX[r]),1])
        chFR.append([frB,frA])
        spikeBs.append(spikeIn)
        calStart = VolToCalIdx(volStart,VolAX,CalAX)
        calEnd = VolToCalIdx(volEnd,VolAX,CalAX)
        calChange = VolToCalIdx(int(r),VolAX,CalAX)
        volBS.append(vol[volStart:volEnd+1])
        calBS.append(cal[calStart:calEnd+1])
        VolAXB = np.linspace(0, (len(vol[volStart:volEnd+1])/500), len(vol[volStart:volEnd+1])) 
        CalAXB = np.linspace(0, (len(cal[calStart:calEnd+1])/30), len(cal[calStart:calEnd+1]))

        volAxBs.append(VolAXB)
        calAxBs.append(CalAXB)
        # Cut voltage and calcium signals
        volCut = vol[volStart:volEnd+1].squeeze()
        calCut = cal[calStart:calEnd+1].squeeze()
        BefCal = np.mean(cal[calStart:calChange+1])
        AftCal = np.mean(cal[calChange:calEnd+1])
        calDiff = AftCal-BefCal
        calDiffA.append(calDiff)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=VolAX[volStart:volEnd+1].squeeze(), y=volCut, name="Voltage",line=dict(color='red', width=2)),secondary_y=False,)
        #fig.add_trace(go.Scatter(x=VolAX[volStart:volEnd+1], y = volCut[spikeIn].squeeze(),mode="markers",marker=dict(color="black", size=3),name="Spike"),secondary_y=False)
        fig.add_trace(go.Scatter(x= CalAX[calStart:calEnd+1], y= cal[calStart:calEnd+1].squeeze(), name="Calcium",line=dict(color='blue', width=2)),secondary_y=True,)
        fig.update_layout(title_text="calcium and voltage togeter")
        fig.update_xaxes(title_text="Time(ms)")
        fig.update_yaxes(title_text="<b>Calcium</b>",secondary_y=True)
        fig.update_yaxes(title_text="<b>Voltage</b> ", secondary_y=False)
        #fig.show()
        fig.update_layout(
        title="cal-vol",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
        fig.write_html(pathFig)
        fig.update_layout(
            width=2500,  # Set the figure width in pixels
            height=750, # Set the figure height in pixels
        )
        fig.write_image(os.path.join(new_folder, f'bsSync{i}.svg'), format="svg")


    return volBS,calBS,volAxBs, calAxBs, spikeBs,MeanFR,bsCidx,chFR,motS,calDiffA

def combine_figs(fig_list):
    n = len(fig_list)
    # Compute rows/cols close to square
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    # Create the subplot canvas
    combined = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Fig {i+1}" for i in range(len(fig_list))])

    for i, fig in enumerate(fig_list):
        row = i // cols + 1
        col = i % cols + 1

        for trace in fig.data:
            combined.add_trace(trace, row=row, col=col)

        # Optionally carry over axis titles/layout from each fig
        if 'xaxis' in fig.layout and 'title' in fig.layout.xaxis:
            combined.update_xaxes(title_text=fig.layout.xaxis.title.text, row=row, col=col)
        if 'yaxis' in fig.layout and 'title' in fig.layout.yaxis:
            combined.update_yaxes(title_text=fig.layout.yaxis.title.text, row=row, col=col)

    combined.update_layout(height=800, width=1200, showlegend=False)
    return combined

def sigmoidFit(xdata,ydata):
    # Initial guess for parameters [L, x0, k, b]
    p0 = [max(ydata), np.median(xdata), 1, min(ydata)]
    try:
        popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
        # generate fitted curve
        sigX = np.linspace(np.min(xdata), np.max(xdata), 500)
        sigY = sigmoid(sigX, *popt)
        return sigX, sigY, popt

    except RuntimeError:
        # fitting failed → skip
        print("⚠️ Sigmoid fit failed, skipping this dataset")
         # residual function
        def residuals(p, x, y):
            return sigmoid(x, *p) - y
        
        res = least_squares(residuals, p0, args=(xdata, ydata), method='dogbox')
        
        # even if not converged, res.x is the "closest" params
        popt = res.x
        sigX = np.linspace(np.min(xdata), np.max(xdata), 500)
        sigY = sigmoid(sigX, *popt)
        
        return sigX, sigY, popt
    # Fit

def FRchange(volT,spikeID,volAx,calT,calAx,calW,StS,path):
    volCHidx = []
    calCHidx = []
    pathFig = os.path.join(path,f'FR.html')
    FR_vol,cal_avg,cal_Bin, vol_Bin,MvIDX,z = calculate_firing_rate(spikeID,volT,calW,StS,volAx,calT,calAx)
    plt.plot(FR_vol)
    plt.show()
    plt.title("FR_vol trace")
    plt.xlabel("Time")
    plt.ylabel("FR")
    plt.savefig(os.path.join(path,f'FR{calW}.png'), dpi=300, bbox_inches="tight")  # save as PNG
    fig = plt.gcf()  # get current matplotlib figure
    plotly_fig = tls.mpl_to_plotly(fig)  # convert to plotly
    import plotly.io as pio
    pio.write_html(plotly_fig, pathFig)
    diffFR = np.diff(FR_vol)
    spikes_time = find_peaks(FR_vol, height=2*np.std(FR_vol), distance=2)[0]
    spikes_time = list(spikes_time)
    FR_vol = np.array(FR_vol)
    plt.plot(FR_vol)
    plt.plot(diffFR)
    plt.scatter(spikes_time, FR_vol[spikes_time], color='black', s=20, label="Spikes")
    plt.title("FR_vol trace")
    plt.xlabel("Time")
    plt.ylabel("FR")
    plt.show()
    plt.savefig(os.path.join(path,f'FRandDiffFindP{calW}.png'), dpi=300, bbox_inches="tight")  # save as PNG
    thresh = np.mean(sorted(diffFR)[0:int(0.1*len(diffFR))])+4*np.std(diffFR)
    pathFig = os.path.join(path,f'FRandDiff.html')
    plt.plot(FR_vol)
    plt.plot(diffFR)
    plt.axhline(y=thresh, color='black', linestyle='--', linewidth=2, label=f"Threshold = {thresh:.2f}")
    plt.title("FR_vol trace")
    plt.xlabel("Time")
    plt.ylabel("FR")
    plt.show()
    plt.savefig(os.path.join(path,f'FRandDiff{calW}.png'), dpi=300, bbox_inches="tight")  # save as PNG
    
    
    CHidx = np.argwhere(diffFR>thresh)
    CHidx = [int(k) for k in CHidx]
    Bthresh = -1*thresh
    bCHidx = np.argwhere(diffFR<Bthresh)
    bCHidx = [int(k) for k in bCHidx]
    cbCHidx = []
    for i in range(1,len(bCHidx)-1):
        if bCHidx[i] - bCHidx[i-1] < 40 and bCHidx[i] - bCHidx[i+1] < -35:
            cbCHidx.append(bCHidx[i])
    FR_vol = np.array(FR_vol)
    
    FCHidx = [0] + CHidx
    diffCHidx = np.diff(FCHidx)
    FoolCHidx = np.argwhere(diffCHidx>35)
    
    FoolCHidx = [int(i) for i in FoolCHidx]
    CHidx=np.array(CHidx)
    idx = CHidx[FoolCHidx]
    FoolCHidx=np.array(FoolCHidx)
    diffCHidx=np.array(diffCHidx)
    startAndEnd = [int(i) for i in idx]+(cbCHidx)
    startAndEnd = sorted(startAndEnd)
    FR_vol=np.array(FR_vol)
    pathFig = os.path.join(path,f'frWinCh.html')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=list(range(len(FR_vol))), y=FR_vol, name="Voltage",line=dict(color='red', width=2)),secondary_y=False,)
    fig.add_trace(go.Scatter(x=startAndEnd, y = FR_vol[startAndEnd].squeeze(),mode="markers",marker=dict(color="black", size=3),name="Spike"),secondary_y=False)
    fig.update_layout(
        title="cal-vol",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
    fig.write_html(pathFig)
    fig.update_layout(
        width=2500,  # Set the figure width in pixels
        height=750, # Set the figure height in pixels
    )
    fig.write_image(os.path.join(path, f'frWinch{calW}.svg'), format="svg")
    spikeBs = []
    volBS =[]
    calBS =[]
    volAxBs = []
    fTimeIdx = []
    calAxBs = []
    timeIdx= np.array(vol_Bin)[startAndEnd]
    for i in range(len(timeIdx)):
        if i == 0 and timeIdx[i] > 13000:
            fTimeIdx.append(timeIdx[i])
        if i >0:
            if len(fTimeIdx)>0:
                if  timeIdx[i] - fTimeIdx[-1] > 11000:
                    fTimeIdx.append(timeIdx[i])
            else:
                if timeIdx[i] - timeIdx[i-1] > 11000:
                    fTimeIdx.append(timeIdx[i])
    for k,idx in enumerate(fTimeIdx):
        new_folder = os.path.join(path, f"transition{k}FR")  # subfolder name  
        # Create the folder if it doesn't exist
        os.makedirs(new_folder, exist_ok=True)
        pathFig = os.path.join(new_folder,f'bsSync{k}FR.html')
        volStart =  np.max([int(idx) - 2500,0])
        volEnd = np.min([int(idx) + 2500,len(volT)-1])
        spikeIn = [s - volStart for s in spikeID if volStart <= s < volEnd]
        spikeBs.append(spikeIn)
        calStart = VolToCalIdx(volStart,volAx,calAx)
        calChange = VolToCalIdx(int(idx),volAx,calAx)
        calEnd = VolToCalIdx(volEnd,volAx,calAx)
        volBS.append(volT[volStart:volEnd+1])
        calBS.append(calT[calStart:calEnd+1])
        VolAXB = np.linspace(0, (len(volT[volStart:volEnd+1])/500), len(volT[volStart:volEnd+1])) 
        CalAXB = np.linspace(0, (len(calT[calStart:calEnd+1])/30), len(calT[calStart:calEnd+1]))
        volAxBs.append(VolAXB)
        calAxBs.append(CalAXB)
        # Cut voltage and calcium signals
        volCut = volT[volStart:volEnd+1].squeeze()
        calCut = calT[calStart:calEnd+1].squeeze()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=VolAXB, y=volCut, name="Voltage",line=dict(color='red', width=2)),secondary_y=False,)
        fig.add_trace(go.Scatter(x=spikeIn, y = volCut[spikeIn].squeeze(),mode="markers",marker=dict(color="black", size=3),name="Spike"),secondary_y=False)
        fig.add_trace(go.Scatter(x= CalAXB, y= calT[calStart:calEnd+1].squeeze(), name="Calcium",line=dict(color='blue', width=2)),secondary_y=True,)
        fig.update_layout(title_text="calcium and voltage togeter")
        fig.update_xaxes(title_text="Time(ms)")
        fig.update_yaxes(title_text="<b>Calcium</b>",secondary_y=True)
        fig.update_yaxes(title_text="<b>Voltage</b> ", secondary_y=False)
        #fig.show()
        fig.update_layout(
        title="cal-vol",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
        fig.write_html(pathFig)
        fig.update_layout(
            width=2500,  # Set the figure width in pixels
            height=750, # Set the figure height in pixels
        )
        fig.write_image(os.path.join(new_folder, f'bsSync{k}FR.svg'), format="svg")
        
    return  volBS,calBS,volAxBs, calAxBs, spikeBs

LagList = np.arange(0,1.5,0.033)
DB = pd.read_csv(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\NewMetaDataSST.csv')

values = DB['SNR'].tolist()
#r = DB.iloc[[idx for idx,i in enumerate(values) if i > 4.1]]
r = DB
awake = r['Notes']
bs = list(r['brainState'])


slope = []
t0 = []
print(type(r))
path = [row['Link'] for _, row in r.iterrows() if row['brainState'].lower() == 'motor']
#path = [r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\28-07-2025-motor\fov9\2\cell0',r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\28-07-2025-motor\fov10\cell0']
for l in range(len(path)):
    
    TracePathCal = os.path.join(path[l],'calTraceDF.csv')
    TracePathVol = os.path.join(path[l],'volTraceDF.csv')
    TracePathSPIKE = os.path.join(path[l],'SpikeIdx.csv')
    VolTrace = pd.read_csv(TracePathVol)
    VolTrace = np.array(VolTrace)
    VolTrace = VolTrace.flatten()
    Trace = VolTrace
    CalTrace = pd.read_csv(TracePathCal)
    CalTrace = np.array(CalTrace)
    CalTrace = CalTrace.flatten()
    
    StepSize = 5 #330 ms
   
    spikeId = pd.read_csv(TracePathSPIKE)
    spikeId = np.array(spikeId)
    spikeId = spikeId.flatten()
    parentP = os.path.dirname(path[l])
    MotPath = os.path.join(parentP,'Sync','MotorId.csv')
    VolAX = np.linspace(0, (len(Trace)/500), len(Trace)) 
    CalAX = np.linspace(0, (len(CalTrace)/30), len(CalTrace))
    MotT = pd.read_csv(MotPath)
    MotT = np.array(MotT)
    MotT = MotT.flatten()
    MotT =MotT[1:]
    MotAX = np.linspace(0, (len(MotT)/500), len(MotT)) 
    sW = 1
    CalWindowSize = 5 #150 ms
    FR_vol,cal_avg,cal_Bin, vol_Bin,MvIDX,z = calculate_firing_rate(spikeId,Trace,CalWindowSize,sW,VolAX,CalTrace,CalAX)
    FRax = np.linspace(0, VolAX[-1], len(FR_vol))
    CalAax = np.linspace(0, CalAX[-1], len(cal_avg))
    pathFig =os.path.join(path[l],'FR.html')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=CalAax, y=cal_avg, name="bin cal",line=dict(color='blue', width=2)),secondary_y=False,)
    fig.add_trace(go.Scatter(x= FRax, y= FR_vol, name="FR",line=dict(color='red', width=2)),secondary_y=True,)
    fig.update_layout(title_text="calcium fit and FR")
    fig.update_xaxes(title_text="Time(ms)")
    fig.update_yaxes(title_text="<b>Calcium</b>",secondary_y=True)
    fig.update_yaxes(title_text="<b>Voltage</b> ", secondary_y=False)
    fig.write_html(pathFig)
    fig.update_layout(
        width=2500,  # Set the figure width in pixels
        height=750, # Set the figure height in pixels
    )
    fig.write_image(os.path.join(parentP, f'FR.svg'), format="svg")
    #volTCbs, calTbs,volAxBs,calAxBs,spikeIdxBs = BSchange(MotT,CalTrace,VolTrace,spikeId,path[l])
    sW = 5
    SmoothCal = CalSmooth(CalTrace,sW)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=CalAX, y=SmoothCal, name="smooth calcium",line=dict(color='blue', width=2)),secondary_y=False,)
    fig.add_trace(go.Scatter(x= VolAX, y= Trace, name="voltage",line=dict(color='red', width=2)),secondary_y=True,)
    fig.add_trace(go.Scatter(x= MotAX, y= MotT, name="voltage",line=dict(color='black', width=2)),secondary_y=True,)
    fig.update_layout(title_text="calcium fit and FR")
    fig.update_xaxes(title_text="Time(ms)")
    fig.update_yaxes(title_text="<b>Calcium</b>",secondary_y=True)
    fig.update_yaxes(title_text="<b>Voltage</b> ", secondary_y=False)
    #volTCbs, calTbs,volAxBs,calAxBs,spikeIdxBs = BSchange(MotT,CalTrace,VolTrace,spikeId,path[l])
    figList = []
    print(path[l])
    #sW = 3
    #CalWindowSize = 10 #500 ms
    #volTCbs, calTbs,volAxBs,calAxBs,spikeIdxBs = FRchange(VolTrace,spikeId,VolAX,CalTrace,CalAX,CalWindowSize,StepSize,path[l])
    
    # if len(volTCbs) > 0:
    #     figSig = make_subplots(rows=len(volTCbs),cols = 1)
    #     for k in range(len(volTCbs)):
    #         new_folder = os.path.join(path[l], f"transition{k}FR")
    #         currVol = volTCbs[k]
    #         currCal = calTbs[k]
    #         currVolAx = volAxBs[k]
    #         currCalAx = calAxBs[k]
            
    #         MaxSloeLag,MaxSlopeFig,MaxSlope,MaxCLId,MaxVLId,MaxCorLag,MaxCorrFig,MaxCor,c_Map,MfrW,tIMEmap,FR_vol,cal_avg,cal_Bin, vol_Bin,MvIDX,z,correlation, p_value,predXZ,y_pred_PlotZ,r2,Linear_r2Z,fig,fig18, fig_Lin, corDur, corWs, CorWval,CorrWFr,CorrEdFR = analyze_block(path[l],sW, calTbs[k],volTCbs[k], calAxBs[k],volAxBs[k], CalWindowSize,spikeIdxBs[k],StepSize, LagList,suffix ='')
            
    #         pathFig = os.path.join(new_folder, f"sigFR{k}FR.html")
    #         #calForFit,
            
    #         sigX,sigY,sigPar =sigmoidFit(currCalAx,currCal)
    #         if sigX is not None:
    #             slope.append(sigPar[2])
    #             t0.append(sigPar[1]) 
    #             currFRAx = np.linspace(min(sigX), max(sigX), len(FR_vol))
    #             fig = make_subplots(specs=[[{"secondary_y": True}]])
    #             fig.add_trace(go.Scatter(x=sigX, y=sigY, name="calcium",line=dict(color='red', width=2)),secondary_y=False,)
    #             fig.add_trace(go.Scatter(x= currFRAx, y= FR_vol, name="fr",line=dict(color='blue', width=2)),secondary_y=True,)
    #             fig.update_layout(title_text="calcium fit and FR")
    #             fig.update_xaxes(title_text="Time(ms)")
    #             fig.update_yaxes(title_text="<b>Calcium</b>",secondary_y=True)
    #             fig.update_yaxes(title_text="<b>Voltage</b> ", secondary_y=False)
    #             #fig.show()
    #             fig.update_layout(
    #             title="cal-vol",
    #             plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    #             paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
    #             fig.write_html(pathFig)
    #             fig.update_layout(
    #                 width=2500,  # Set the figure width in pixels
    #                 height=750, # Set the figure height in pixels
    #             )
    #             fig.write_image(os.path.join(new_folder, f'sigFR{k}FR.svg'), format="svg")
                
    #             figSig.add_trace(go.Scatter(x=sigX, y=sigY, name="calcium",line=dict(color='red', width=2)),row=k+1,col=1)
   
    #     pathFig = os.path.join(path[l], f'sigCellFR.html')    
    #     figSig.update_layout(
    #     title="cal-vol",
    #     plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    #     paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
    #     figSig.write_html(pathFig)
    #     figSig.update_layout(
    #         width=2500,  # Set the figure width in pixels
    #         height=750, # Set the figure height in pixels
    #     )
    #     figSig.write_image(os.path.join(path[l], f'sigCellFR.svg'), format="svg")
    #     CalWindowSize = 5 #500 ms
    #     StepSize = 5 #330 ms
    
# if len(volTCbs) > 0:
#     slopeAx = list(range(0, len(slope)))
#     t0Ax = list(range(0, len(t0)))
#     figS = go.Figure()
#     figS.add_trace(go.Scatter(x= slopeAx, y= slope, name="AvgCal",mode="lines+markers",  line=dict(dash="dash", width=1, color="rgba(0, 0, 255, 0.5)"), marker=dict(color='blue')))
#     pathFig = os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'slopesTranFR.html')    
#     figS.update_layout(
#     title="cal-vol",
#     plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
#     paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
#     figS.write_html(pathFig)
#     figS.update_layout(
#         width=2500,  # Set the figure width in pixels
#         height=750, # Set the figure height in pixels
#     )
#     figS.write_image(os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'slopesTranFR.svg'), format="svg")


#     figT = go.Figure()
#     figT.add_trace(go.Scatter(x= t0Ax, y= t0, name="AvgCal",mode="lines+markers",  line=dict(dash="dash", width=1, color="rgba(0, 0, 255, 0.5)"), marker=dict(color='blue')))
#     pathFig = os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'MidPointTranFR.html')    
#     figT.update_layout(
#     title="cal-vol",
#     plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
#     paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
#     figT.write_html(pathFig)
#     figT.update_layout(
#         width=2500,  # Set the figure width in pixels
#         height=750, # Set the figure height in pixels
#     )
#     figT.write_image(os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'MidPointTranFR.svg'), format="svg")
FullFR = []
FullMS = []

FullCAL = []
calDiffAll = []
FrDiffBA = []
for l in range(len(path)):
    FrDiffSC = []
    TracePathCal = os.path.join(path[l],'calTraceDF.csv')
    TracePathVol = os.path.join(path[l],'volTraceDF.csv')
    TracePathSPIKE = os.path.join(path[l],'SpikeIdx.csv')
    VolTrace = pd.read_csv(TracePathVol)
    VolTrace = np.array(VolTrace)
    VolTrace = VolTrace.flatten()
    Trace = VolTrace
    CalTrace = pd.read_csv(TracePathCal)
    CalTrace = np.array(CalTrace)
    CalTrace = CalTrace.flatten()
    sW = 3
    CalWindowSize = 25 #500 ms
    StepSize = 25 #330 ms
    SmoothCal = CalSmooth(CalTrace,sW)
    spikeId = pd.read_csv(TracePathSPIKE)
    spikeId = np.array(spikeId)
    spikeId = spikeId.flatten()
    parentP = os.path.dirname(path[l])
    MotPath = os.path.join(parentP,'Sync','MotorId.csv')
    VolAX = np.linspace(0, (len(Trace)/500), len(Trace)) 
    CalAX = np.linspace(0, (len(CalTrace)/30), len(CalTrace))
    MotT = pd.read_csv(MotPath)
    MotT = np.array(MotT)
    MotT = MotT.flatten()
    volTCbs, calTbs,volAxBs,calAxBs,spikeIdxBs,MeanFR,BSidx,bsFR,motState,calDIff = BSchange(MotT,CalTrace,VolTrace,spikeId,path[l])
    calDiffAll.append(calDIff)
    
    
    figList = []
    FRall = []
    #MeanFR = []
    #volTCbs, calTbs,volAxBs,calAxBs,spikeIdxBs = FRchange(VolTrace,spikeId,VolAX,CalTrace,CalAX,CalWindowSize,StepSize,path[l])
    if len(volTCbs) > 0:
        FullMS.append(motState)
        figSig = make_subplots(rows=len(volTCbs),cols = 1)
        for k in range(len(volTCbs)):
            calIDX = VolToCalIdx(BSidx[k],VolAX,CalAX)
            VerySmooth = CalSmooth(CalTrace[calIDX:np.min([calIDX+300,len(CalTrace)-1])],12)

            if k ==1:
                v=00
            new_folder = os.path.join(path[l], f"transition{k}MotorTS")
            os.makedirs(new_folder, exist_ok=True)
            currVol = volTCbs[k]
            currCal = calTbs[k]
            currVolAx = volAxBs[k]
            currCalAx = calAxBs[k]
            FR_vol,cal_avg,cal_Bin, vol_Bin,MvIDX,z = calculate_firing_rate(spikeIdxBs[k],volTCbs[k],CalWindowSize,StepSize,volAxBs[k], calTbs[k],calAxBs[k])
            FRall.append(FR_vol)
            mid = len(FR_vol) // 2 
            BeforeFR = np.mean(FR_vol[:mid])
            AfterFR = np.mean(FR_vol[mid:])
            FRdiff = AfterFR - BeforeFR
            FrDiffBA.append(FRdiff)
            FrDiffSC.append(FRdiff)
            #MaxSloeLag,MaxSlopeFig,MaxSlope,MaxCLId,MaxVLId,MaxCorLag,MaxCorrFig,MaxCor,c_Map,MfrW,tIMEmap,FR_vol,cal_avg,cal_Bin, vol_Bin,MvIDX,z,correlation, p_value,predXZ,y_pred_PlotZ,r2,Linear_r2Z,fig,fig18, fig_Lin, corDur, corWs, CorWval,CorrWFr,CorrEdFR = analyze_block(path[l],sW, calTbs[k],volTCbs[k], calAxBs[k],volAxBs[k], CalWindowSize,spikeIdxBs[k],StepSize, LagList,suffix ='')
            #MeanFR.append(np.mean(FR_vol))
            pathFig = os.path.join(new_folder, f"sigFR{k}MotorTS.html")
            #calForFit,
            
            sigX,sigY,sigPar =sigmoidFit(currCalAx,currCal)
            if sigX is not None:
                slope.append(sigPar[2])
                t0.append(sigPar[1]) 
                currFRAx = np.linspace(min(sigX), max(sigX), len(FR_vol))
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=sigX, y=sigY, name="calcium",line=dict(color='red', width=2)),secondary_y=False,)
                fig.add_trace(go.Scatter(x= currFRAx, y= FR_vol, name="fr",line=dict(color='blue', width=2)),secondary_y=True,)
                fig.update_layout(title_text="calcium fit and FR")
                fig.update_xaxes(title_text="Time(ms)")
                fig.update_yaxes(title_text="<b>Calcium</b>",secondary_y=True)
                fig.update_yaxes(title_text="<b>Voltage</b> ", secondary_y=False)
                #fig.show()
                fig.update_layout(
                title="cal-vol",
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
                fig.write_html(pathFig)
                fig.update_layout(
                    width=2500,  # Set the figure width in pixels
                    height=750, # Set the figure height in pixels
                )
                fig.write_image(os.path.join(new_folder, f'sigFR{k}MotorTS.svg'), format="svg")


                pathFig = os.path.join(new_folder, f"VolCal{k}MotorTS.html")
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=currVolAx, y=currVol, name="voltage",line=dict(color='red', width=2)),secondary_y=False,)
                fig2.add_trace(go.Scatter(x= currCalAx, y= currCal, name="calcium",line=dict(color='blue', width=2)),secondary_y=True,)
                fig2.update_layout(title_text="calcium fit and FR")
                fig2.update_xaxes(title_text="Time(ms)")
                fig2.update_yaxes(title_text="<b>Calcium</b>",secondary_y=True)
                fig2.update_yaxes(title_text="<b>Voltage</b> ", secondary_y=False)
                #fig.show()
                fig2.update_layout(
                title="cal-vol",
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
                fig2.write_html(pathFig)
                fig2.update_layout(
                    width=2500,  # Set the figure width in pixels
                    height=750, # Set the figure height in pixels
                )
                fig2.write_image(os.path.join(new_folder, f'volCal{k}MotorTS.svg'), format="svg")
                
                figSig.add_trace(go.Scatter(x=sigX, y=sigY, name="calcium",line=dict(color='red', width=2)),row=k+1,col=1)
        pathFig = os.path.join(path[l], f'sigCellFR.html')    
        figSig.update_layout(
        title="cal-vol",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
        figSig.write_html(pathFig)
        figSig.update_layout(
            width=2500,  # Set the figure width in pixels
            height=750, # Set the figure height in pixels
        )
        figSig.write_image(os.path.join(path[l], f'sigCellMotorTS.svg'), format="svg")
        FullFR.append(FRall)
        FullCAL.append([tr.tolist() for tr in calTbs])
        print(len(FullFR))
        print(len(FullCAL))

        MotorPath = os.path.join(path[l],'MeanFRbs.csv')
        df = pd.DataFrame(MeanFR) 
        df.to_csv(MotorPath, index=False) # create df with column name
        MotorPath = os.path.join(path[l],'brainchaneIDX.csv')
        df = pd.DataFrame(BSidx) 
        df.to_csv(MotorPath, index=False) # create df with column name
        MotorPath = os.path.join(path[l],'bsWinFR.csv')
        df = pd.DataFrame(bsFR) 
        df.to_csv(MotorPath, index=False) # create df with column name
        CalWindowSize = 3 #500 ms
        CalWindowSize = 5 #500 ms
        StepSize = 5 #330 ms
    # Boolean masks
    motOn_mask  = np.array(motState) == 'ON'
    motOff_mask = np.array(motState) == 'OFF'

    # Use masks to index traces
    calOn  = [calTbs[i] for i in range(len(calTbs)) if motOn_mask[i]]
    calOff = [calTbs[i] for i in range(len(calTbs)) if motOff_mask[i]]
    FrOn   = [FRall[i]  for i in range(len(FRall))  if motOn_mask[i]]
    FrOff  = [FRall[i]  for i in range(len(FRall))  if motOff_mask[i]]
    pathOn = os.path.join(path[l],"motOn_plot.svg")
    pathOff = os.path.join(path[l],"motOff_plot.svg")
    pathOnH = os.path.join(path[l],"motOn_plot.html")
    pathOffH = os.path.join(path[l],"motOff_plot.html")
    fig_on  = plot_traces_subplot(calOn, FrOn, "Motor ON")
    fig_off = plot_traces_subplot(calOff, FrOff, "Motor OFF")

    fig_on.show()
    fig_off.show()

    # Save
    fig_on.write_html(pathOnH)
    fig_on.write_image(pathOn)
    fig_off.write_html(pathOffH)
    fig_off.write_image(pathOff)
    
    df_on  = build_df(calOn, FrOn, state="motOn")
    df_off = build_df(calOff, FrOff, state="motOff")

    # Make plots
    p_on  = plot_state(df_on, "motOn")
    p_off = plot_state(df_off, "motOff")
   # p_on.save(, width=8, height=6, units="in")
   # p_off.save(os.path.join(path[l],"motOff_plot.svg"), width=8, height=6, units="in")
    
    #ggsave(plot=p_on, filename=pathOn, width=8, height=6, units="in")
    #ggsave(plot=p_off, filename=pathOff, width=8, height=6, units="in")
    #fig_on_plotly = tls.mpl_to_plotly(p_on.draw())  # converts plotnine (matplotlib) to plotly
    #fig_off_plotly = tls.mpl_to_plotly(p_off.draw())

    # Save as interactive HTML
    #pio.write_html(fig_on_plotly, os.path.join(path[l],"motOn_plot.html"), auto_open=False)
    #pio.write_html(fig_off_plotly, os.path.join(path[l],"motOn_plot.html"), auto_open=False)
    X = list(range(0, len(calDIff)))
    MotorPath = os.path.join(path[l],'calDiff.csv')
    df = pd.DataFrame(calDIff) 
    df.to_csv(MotorPath, index=False)
    MotorPath = os.path.join(path[l],'FrDiff.csv')
    df = pd.DataFrame(FrDiffSC) 
    df.to_csv(MotorPath, index=False)
    figD = go.Figure()
    figD.add_trace(go.Scatter(x=FrDiffSC, y = calDIff,mode="markers",marker=dict(color="black", size=3),name="cal fr diff"))
    pathFig = os.path.join(path[l], f'CalDiffToFRdiff.html')    
    figD.update_layout(
    title="cal-vol",
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
    figD.write_html(pathFig)
    figD.update_layout(
        width=2500,  # Set the figure width in pixels
        height=750, # Set the figure height in pixels
    )
    figD.write_image(os.path.join(path[l], f'CalDiffToFRdiff.svg'), format="svg")
    figA = go.Figure()
    figA.add_trace(go.Scatter(x=X, y = calDIff,mode="markers",marker=dict(color="black", size=3),name="cal diff"))
    figA.add_trace(go.Scatter(x=X, y = FrDiffSC,mode="markers",marker=dict(color="black", size=3),name="fr diff"))
    pathFig = os.path.join(path[l], f'CalDiffAndFRdiff.html')    
    figA.update_layout(
    title="cal-vol",
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
    figA.write_html(pathFig)
    figA.update_layout(
        width=2500,  # Set the figure width in pixels
        height=750, # Set the figure height in pixels
    )
    figA.write_image(os.path.join(path[l], f'CalDiffAndFRdiff.svg'), format="svg")
    

    
if len(volTCbs) > 0:
    slopeAx = list(range(0, len(slope)))
    t0Ax = list(range(0, len(t0)))
    figS = go.Figure()
    figS.add_trace(go.Scatter(x= slopeAx, y= slope, name="AvgCal",mode="lines+markers",  line=dict(dash="dash", width=1, color="rgba(0, 0, 255, 0.5)"), marker=dict(color='blue')))
    pathFig = os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'slopesTranMotorTS.html')    
    figS.update_layout(
    title="cal-vol",
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
    figS.write_html(pathFig)
    figS.update_layout(
        width=2500,  # Set the figure width in pixels
        height=750, # Set the figure height in pixels
    )
    figS.write_image(os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'slopesTranMotorTS.svg'), format="svg")


    figT = go.Figure()
    figT.add_trace(go.Scatter(x= t0Ax, y= t0, name="AvgCal",mode="lines+markers",  line=dict(dash="dash", width=1, color="rgba(0, 0, 255, 0.5)"), marker=dict(color='blue')))
    pathFig = os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'MidPointTranMotorTS.html')    
    figT.update_layout(
    title="cal-vol",
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
    figT.write_html(pathFig)
    figT.update_layout(
        width=2500,  # Set the figure width in pixels
        height=750, # Set the figure height in pixels
    )
    figT.write_image(os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'MidPointTranMotorTS.svg'), format="svg")


flatCalDiff = [i for i in calDiffAll]
flatFrDiff = [i for i in FrDiffBA]

X = list(range(0, len(flatCalDiff)))
X2 = list(range(0, len(flatFrDiff)))
figD = go.Figure()
figD.add_trace(go.Scatter(x=flatFrDiff, y = flatCalDiff,mode="markers",marker=dict(color="black", size=3),name="cal fr diff"))
pathFig = os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'CalDiffToFRdiff.html')    
figD.update_layout(
title="cal-vol",
plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
figD.write_html(pathFig)
figD.update_layout(
    width=2500,  # Set the figure width in pixels
    height=750, # Set the figure height in pixels
)
figD.write_image(os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'CalDiffToFRdiff.svg'), format="svg")
figA = go.Figure()
figA.add_trace(go.Scatter(x=X, y = calDIff,mode="markers",marker=dict(color="black", size=3),name="cal diff"))
figA.add_trace(go.Scatter(x=X2, y = FrDiffSC,mode="markers",marker=dict(color="black", size=3),name="fr diff"))
pathFig = os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'CalDiffAndFRdiff.html')    
figA.update_layout(
title="cal-vol",
plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
figA.write_html(pathFig)
figA.update_layout(
    width=2500,  # Set the figure width in pixels
    height=750, # Set the figure height in pixels
)
figA.write_image(os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST', f'CalDiffAndFRdiff.svg'), format="svg")


FullMSL = [item for sublist in FullMS for item in sublist]
FullCALL = [item for sublist in FullCAL if sublist is not None for item in sublist]
FullFRL = [item for sublist in FullFR if sublist is not None for item in sublist]
#FullCALL=np.array(FullCALL)
#FullFRL = np.array(FullFRL)
# Boolean masks
motOn_mask  = np.array(FullMSL) == 'ON'
motOff_mask = np.array(FullMSL) == 'OFF'

# Use masks to index traces
calOn  = [FullCALL[i] for i in range(len(FullCAL)) if motOn_mask[i]]
calOff = [FullCALL[i] for i in range(len(FullCAL)) if motOff_mask[i]]
FrOn   = [FullFRL[i]  for i in range(len(FullFR))  if motOn_mask[i]]
FrOff  = [FullFRL[i]  for i in range(len(FullFR))  if motOff_mask[i]]
# calOn  = FullCALL[FullMSL == 'ON']
# calOff = FullCALL[FullMSL == 'OFF']
# FrOn  = FullFRL[FullMSL == 'ON']
# FrOff = FullFRL[FullMSL == 'OFF']

#df_on  = build_df(calOn, FrOn, state="motOn")
#df_off = build_df(calOff, FrOff, state="motOff")

# Make plots
##p_on  = plot_state(df_on, "motOn")
#p_off = plot_state(df_off, "motOff")
pathOn = os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST',"motOn_plot.svg")
pathOff = os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST',"motOff_plot.svg")
pathOnH = os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST',"motOn_plot.html")
pathOffH = os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST',"motOff_plot.html")
fig_on  = plot_traces_subplot(calOn, FrOn, "Motor ON")
fig_off = plot_traces_subplot(calOff, FrOff, "Motor OFF")

fig_on.show()
fig_off.show()

# Save
fig_on.write_html(pathOnH)
fig_on.write_image(pathOn)
fig_off.write_html(pathOffH)
fig_off.write_image(pathOff)
#p_on.save(os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST',r"motOn_plotS.svg"), width=8, height=6, units="in")
#p_off.save(os.path.join(r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2025\SST',r"motOff_plotS.svg"), width=8, height=6, units="in")