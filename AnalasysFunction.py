import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc
from scipy.optimize import curve_fit
from scipy.ndimage import filters 
import tifffile as tiff
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, linregress
import os
import csv
import pandas as pd
from roipoly import MultiRoi
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import ast
from caliumPlay import smoothTRACE, Fullcal,intList
from TRY import LongLIST,str_to_list
from SomaAnalsys import detect_CS
from caliumPlay import Fullcal
from SSTanlasys import smoothTRACE
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
import pickle


def _edit_spikes_gui(raw, fs, highpass=100, verbose=True):
    """Interactive threshold-based spike editor for a 1D signal chunk.

    Returns: (np.ndarray or None, fraw)
      - np.ndarray: indices of detected spikes (relative to `raw`)
      - None: user cancelled (pressed 'q')
    """
    raw = np.asarray(raw).ravel()
    ny = fs / 2.0
    Wn = highpass / ny
    if not (0 < Wn < 1):
        raise ValueError(f"Invalid cutoff {highpass} Hz for fs={fs} Hz")
    b, a = butter(2, Wn, btype="highpass")
    fraw = filtfilt(b, a, raw)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fraw, color="gray", lw=0.6, label=f"HP {highpass} Hz")
    thr_line = ax.axhline(0, color="red", ls="--", label="threshold")
    ax.set_title("Click to set threshold; Enter/s to save; q to cancel (closing window = save empty)")
    ax.legend()
    plt.tight_layout()

    state = {"new_spikes": None, "done": False, "has_clicked": False}

    def on_click(evt):
        if evt.inaxes is not ax:
            return
        thr = evt.ydata
        thr_line.set_ydata(thr)
        idxs = np.flatnonzero(fraw > thr)
        # clear previous scatter
        for coll in list(ax.collections):
            coll.remove()
        if idxs.size:
            ax.scatter(idxs, fraw[idxs], c="red", s=8, label="new spikes")
        ax.legend()
        fig.canvas.draw_idle()
        state["new_spikes"] = np.asarray(idxs, dtype=int)
        state["has_clicked"] = True

    def on_key(evt):
        retirnSp = []
        k = evt.key
        if k in ("enter", "return", "s"):
            if not state["has_clicked"]:
                state["new_spikes"] = np.array([], dtype=int)
                if verbose:
                    print("[!] No threshold set, saving empty spike list.")
            if verbose:
                print(f"[✓] Saved {state['new_spikes'].size} spikes.")
            if k == 's':
                retirnSp = state["new_spikes"]
            state["done"] = True
            plt.close(fig)
        elif k == "q":
            if not state["has_clicked"]:
                state["new_spikes"] = np.array([], dtype=int)
                if verbose:
                    print("[×] Quit without editing — returning empty list.")
            else:
                if verbose:
                    print(f"[×] Quit — keeping last {state['new_spikes'].size} spikes.")
            state["done"] = True
            plt.close(fig)
        

    def on_close(evt):
        # If user closes window with the X: treat as "save empty" (not cancel)
        # This avoids hanging the GUI loop. Change behavior here if you prefer "cancel".
        if not state["has_clicked"]:
            state["new_spikes"] = np.array([], dtype=int)
            if verbose:
                print("[!] Window closed — saving empty spike list.")
        state["done"] = True

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    #fig.canvas.mpl_connect("close_event", on_close)

    # Non-blocking show, then wait for user action
    plt.show(block=False)
    while not state["done"]:
        plt.pause(0.05)

    

    return state["new_spikes"], fraw


def correct_spikes(trace, fs, old_spikes=None, highpass=100, verbose=True):
    """
    Interactive spike correction for a single trace.
    Keys:
      e : open editor and set threshold
      z : finish and return final spikes
      q : quit without saving

    Args:
      trace        : 1D array-like, raw trace
      fs           : sampling rate (Hz)
      old_spikes   : optional array of previously detected spikes
      highpass     : high-pass cutoff frequency (Hz)
      verbose      : print status messages

    Returns:
      np.ndarray of final spike indices (or None if cancelled)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    trace = np.asarray(trace).ravel()
    spikes, fraw = _edit_spikes_gui(trace, fs, highpass=highpass, verbose=verbose)
    if spikes is None:
        spikes = np.array([], dtype=int)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trace, color='gray', lw=0.6)

    # plot old spikes in black if available
    if old_spikes is not None and len(old_spikes) > 0:
        old_spikes = np.asarray(old_spikes, dtype=int)
        ax.scatter(old_spikes, trace[old_spikes], c='black', s=10, label='old spikes')

    # plot current spikes in red
    pts = ax.scatter(spikes, trace[spikes], c='red', s=8, label='current spikes')

    ax.legend()
    ax.set_title("Press 'e' to edit | 'z' to finish | 'q' to quit")
    plt.tight_layout()

    state = {'done': False, 'spikes': spikes}

    def redraw():
        # remove old red points and re-draw
        for coll in list(ax.collections):
            if coll.get_facecolor().shape[0] and np.allclose(coll.get_facecolor()[0][:3], [1, 0, 0]):
                coll.remove()
        ax.scatter(state['spikes'], trace[state['spikes']], c='red', s=8, label='current spikes')
        fig.canvas.draw_idle()

    def on_key(evt):
        k = evt.key
        if k == 'e':
            new, _ = _edit_spikes_gui(trace, fs, highpass=highpass, verbose=verbose)
            if new is not None:
                state['spikes'] = new
                if verbose:
                    print(f"[edit] Updated {len(new)} spikes.")
                redraw()
        elif k == 'z':
            if verbose:
                print(f"[✓] Finalized with {len(state['spikes'])} spikes.")
            state['done'] = True
            plt.close(fig)
        elif k == 'q':
            if verbose:
                print("[×] Quit without saving.")
            state['spikes'] = None
            state['done'] = True
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=False)
    while not state['done']:
        plt.pause(0.05)

    return state['spikes']


def splitTrace_from_arrays(trace, spikeIdx, n_chunks=2):
    """
    Split a voltage trace and corresponding spike indices into chunks.

    Parameters
    ----------
    trace : array-like (1D)
        Voltage trace
    spikeIdx : array-like (1D)
        Spike indices relative to the full trace
    n_chunks : int
        Number of chunks to split into

    Returns
    -------
    trace_chunks : list of np.ndarray
        Trace segments
    spikes_chunks : list of np.ndarray
        Spike indices relative to each chunk
    startlist : list of int
        Start index of each chunk in the original trace
    trace : np.ndarray
        Original trace (returned for compatibility)
    """

    trace = np.asarray(trace).astype(float)
    spikeIdx = np.asarray(spikeIdx).astype(int)

    trace_chunks = []
    spikes_chunks = []
    startlist = []

    N = len(trace)
    chunk_size = N // n_chunks

    for i in range(n_chunks):
        start = i * chunk_size
        end = N if i == n_chunks - 1 else (i + 1) * chunk_size

        # Trace segment
        segment = trace[start:end]

        # Spikes that fall inside this segment (re-indexed)
        spike_seg = spikeIdx[(spikeIdx >= start) & (spikeIdx < end)] - start

        trace_chunks.append(segment)
        spikes_chunks.append(spike_seg)
        startlist.append(start)

    return trace_chunks, spikes_chunks, startlist, trace

def splitTrace(home_path,n_chunks = 2):
    cell_num = int(os.path.basename(home_path).replace("cell", ""))
    pkl_path = os.path.join(home_path, "output_data.pkl")
    Spike_path = os.path.join(home_path, "SpikeIdx.csv")
    spikeId = pd.read_csv(Spike_path)
    spikeId = np.array(spikeId).flatten()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    #data = Fulldata[cell_num]
    
    #record = data[cell_num]
    record = data
    trace = np.asarray(record.get("traw")).astype(float)
    trace_chunks = []
    spikes_chunks = []
    N = len(trace)
    chunk_size = N // n_chunks
    startlist = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = N if i == n_chunks - 1 else (i + 1) * chunk_size
        segment = trace[start:end]
        spike_seg = np.array([i for i in spikeId if start < i < end]) - start
        spikes_chunks.append(spike_seg)
        trace_chunks.append(segment)
        startlist.append(start)
    return trace_chunks,spikes_chunks,startlist,trace
def Split_cal(chane_p,VolT,CalT,volax,calax,mot,spikeIdx):
    calMot = []
    volMot = []
    volRes = []
    calRes = []
    spikeMot = []
    spikeRes = []
    if chane_p[0]<5:
        chane_p = chane_p[1:]
    for i in range(len(chane_p)+1):
        if i == 0:
            sIDX = 0
            eIdx = chane_p[i]
        elif i > 0 and i < len(chane_p):
            sIDX = chane_p[i-1]
            eIdx = chane_p[i]
        elif i ==len(chane_p):
            sIDX = chane_p[-1]
            eIdx = len(VolT)-1
        #print(s)
        calS = VolToCalIdx(sIDX,volax,calax)
        calE = VolToCalIdx(eIdx,volax,calax)
        if mot[sIDX+7] == 1:
            #print(f'motor{sIDX}')
            spikeId = [i for i in spikeIdx if i >= sIDX and i < eIdx]
            spikeMot.append(np.array(spikeId) - sIDX)
            calMot.append(CalT[calS:calE+1])
            volMot.append(VolT[sIDX:eIdx+1])
        if mot[sIDX+7] == 0:
            #print(f'rest{sIDX}')
            
            spikeId = [i for i in spikeIdx if i >= sIDX and i < eIdx]
            spikeRes.append(np.array(spikeId) - sIDX)
            calRes.append(CalT[calS:calE+1])
            #print(len(calRes))
            volRes.append(VolT[sIDX:eIdx+1])
            #print(len(volRes))
    return calMot,calRes,spikeMot,spikeRes,volMot,volRes
def splitIDX(MotIdx,calMotID,RestIdx,calRestId):
    splitsVolOn = np.where(np.diff(MotIdx) > 1)[0] + 1
    motorVolOn = np.split(MotIdx, splitsVolOn)
    splitsVolOff = np.where(np.diff(RestIdx) > 1)[0] + 1
    motorVolOff = np.split(RestIdx, splitsVolOff)
    splitsCalOn = np.where(np.diff(calMotID) > 1)[0] + 1
    motorCalOn = np.split(calMotID, splitsCalOn)
    splitsCalOff = np.where(np.diff(calRestId) > 1)[0] + 1
    motorCalOff = np.split(calRestId, splitsCalOff)
    

    return motorVolOn,motorVolOff,motorCalOn,motorCalOff

def split_spikes_by_bout(bouts, spikeIdx, idx_map):
    """
    bouts    : list of arrays (original indices per bout)
    spikeIdx : spike indices relative to sliced trace (volMot or volRest)
    idx_map  : original → sliced index map
    """
    spike_by_bout = []

    # invert map: sliced → original
    inv_map = {v: k for k, v in idx_map.items()}

    # convert spike indices back to original indices
    spike_orig = np.array([inv_map[s] for s in spikeIdx])

    for bout in bouts:
        mask = np.isin(spike_orig, bout)
        spike_by_bout.append(
            list(np.where(mask)[0])  # indices relative to spike list
        )

    return spike_by_bout

def clean_and_parse(x):
    try:
        if isinstance(x, str):
            # Fix cases where items might be space-separated instead of comma-separated
            # (Based on row 673 in your CSV source which lacks commas)
            if ' ' in x and ',' not in x:
                x = x.replace(' ', ',')
            return ast.literal_eval(x)
        return x
    except:
        return [] # Return empty list if parsing fails
    
def motorSp(calT,volT,motorId,calAX,volAX,spId):
    changePoint =np.argwhere(np.abs(np.diff(motorId)) == 1)
    motorId = motorId[0:len(volT)]
    MotIdx = np.argwhere(motorId == 1).ravel()   # -> 1D array
    RestIdx = np.argwhere(motorId == 0).ravel()  # -> 1D array
    calMotID = []
    calRestId = []
    calMotID = np.unique([VolToCalIdx(idx, volAX, calAX) for idx in MotIdx])
    calRestId = np.unique([VolToCalIdx(idx, volAX, calAX) for idx in RestIdx])
    volMot = volT[MotIdx]
    volRest = volT[RestIdx]
    calMot = calT[list(set(calMotID))]
    calRest = calT[list(set(calRestId))]
    # split spikes into motor/rest
    spikeMot = []
    spikeRest = []

    # make a mapping from old index → new index
    mot_map = {orig: new for new, orig in enumerate(MotIdx)}
    rest_map = {orig: new for new, orig in enumerate(RestIdx)}
    motorVolOn,motorVolOff,motorCalOn,motorCalOff = splitIDX(MotIdx, calMotID, RestIdx, calRestId)

    for s in spId:
        if s in mot_map:
            spikeMot.append(mot_map[s])
        elif s in rest_map:
            spikeRest.append(rest_map[s])
    
    spikeMotByBout = split_spikes_by_bout(
    motorVolOn, spikeMot, mot_map
        )
    spikeRestByBout = split_spikes_by_bout(
    motorVolOff, spikeRest, rest_map
    )   
    return volMot, volRest, calMot, calRest, spikeMotByBout, spikeRestByBout,changePoint,motorVolOn,motorCalOn,motorVolOff,motorCalOff
    
def CalSmooth(CalTrace, window_size =3):
    return np.convolve(CalTrace, np.ones(window_size)/window_size, mode='same')

def BurstC(Spike, threshold):
    if not Spike:  # If the list is empty, return an empty list
        return []

    result = []
    current_sublist = [Spike[0]]
    NuSpikeBurst = []
    n = 1
    for i in range(0, len(Spike) -1):
        if abs(Spike[i+1]-Spike[i]) > threshold:
            result.append(current_sublist)
            NuSpikeBurst.append(n)
            current_sublist = [Spike[i+1]]
            n=1
        else:
            current_sublist.append(Spike[i+1])
            n+=1
    
    # Append the last sublist
    result.append(current_sublist)
    NuSpikeBurst.append(n)

    return result,NuSpikeBurst

def plot_spike_shapes_plotly_4states(
    out_svg, out_html,
    Anst, Awake, Motor, Rest,
    peak_idx_Anst=None, peak_idx_Awake=None, peak_idx_Motor=None, peak_idx_Rest=None,
    subplot_titles=None,
):

    # Helper: ensure 2D
    def as_2d(M):
        M = np.asarray(M)
        if M.ndim == 1:
            M = M[None, :]
        return M

    # Colors you requested
    state_specs = [
        ("Anst",  Anst,  "blue",    peak_idx_Anst),
        ("Awake", Awake, "magenta", peak_idx_Awake),
        ("Motor", Motor, "orange",  peak_idx_Motor),
        ("Rest",  Rest,  "green",   peak_idx_Rest),
    ]

    titles = subplot_titles if subplot_titles is not None else [s[0] for s in state_specs]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}]],
    )


    for idx, (state_name, pack, ca_color, peak_idx) in enumerate(state_specs):
        r = 1 if idx < 2 else 2
        c = 1 if idx % 2 == 0 else 2

        Vmat, Cmat, xV, xC = pack
        Vmat = as_2d(Vmat)
        Cmat = as_2d(Cmat)
        xV = np.asarray(xV)
        xC = np.asarray(xC)

        # ---- Voltage single traces (grey, transparent) ----
        for i in range(Vmat.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=xV, y=Vmat[i, :],
                    mode="lines",
                    line=dict(color="rgba(120,120,120,0.20)", width=1),
                    showlegend=False,
                    name="Vm"
                ),
                row=r, col=c, secondary_y=False
            )

        # ---- Voltage mean (black) ----
        v_mean = np.nanmean(Vmat, axis=0)
        fig.add_trace(
            go.Scatter(
                x=xV, y=v_mean,
                mode="lines",
                line=dict(color="black", width=2.5),
                showlegend=(idx == 0),
                name="Vm mean"
            ),
            row=r, col=c, secondary_y=False
        )
        a = {'blue':'135,206,235','magenta':'255,0,255','orange':'255,165,0','green':'0,128,0'}
        # ---- Calcium single traces (colored, transparent) ----
        for i in range(Cmat.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=xC, y=Cmat[i, :],
                    mode="lines",
                    line=dict(color=f"rgba({a[ca_color]},0.1)", width=1),showlegend=False,name="Ca"),row=r, col=c, secondary_y=True)

        # ---- Calcium mean (colored, strong) ----
        c_mean = np.nanmean(Cmat, axis=0)
        fig.add_trace(
            go.Scatter(
                x=xC, y=c_mean,
                mode="lines",
                line=dict(color=ca_color, width=2.5),
                showlegend=(idx == 0),
                name="Ca mean"
            ),
            row=r, col=c, secondary_y=True
        )

        # ---- Optional: mark peak points per trace (black markers) ----
        # peak_idx can be:
        #  - list of ints (one peak index per trace)
        #  - list of arrays/lists (multiple indices per trace)
        if peak_idx is not None:
            # make list length match n_traces if possible
            if len(peak_idx) == Cmat.shape[0]:
                for i in range(Cmat.shape[0]):
                    p = peak_idx[i]
                    if p is None:
                        continue
                    p = np.asarray(p).astype(int)
                    p = p[(p >= 0) & (p < Cmat.shape[1])]
                    if p.size == 0:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=xC[p],
                            y=Cmat[i, p],
                            mode="markers",
                            marker=dict(size=5, color="black", symbol="circle"),
                            showlegend=False,
                            name="Ca peak"
                        ),
                        row=r, col=c, secondary_y=True
                    )

        # ---- Axes formatting per subplot ----
        # (Plotly doesn't let you set per-subplot y ranges as simply without addressing axis IDs;
        # so we keep defaults unless you want explicit ranges.)
        fig.update_xaxes(showgrid=False, showline=True, linecolor="black", row=r, col=c)
        fig.update_yaxes(showgrid=False, showline=True, linecolor="black", row=r, col=c, secondary_y=False)
        fig.update_yaxes(showgrid=False, showline=True, linecolor="black", row=r, col=c, secondary_y=True)

    # Global layout (like yours: clean, no grid, transparent background)
    fig.update_layout(
        width=1400, height=900,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.85, y=0.98),
        margin=dict(l=60, r=60, t=60, b=60),
    )

    # Save
    fig.write_image(out_svg)
    fig.write_html(out_html)
    fig.show()

    return fig
def ChooseSpk(VOL,BurstIdxVol,SpikeNum,threshoAf,threshoBe):
    FinalEventList =[]
    FinalSpikeNum = []
    e = np.size(VOL)
    if  BurstIdxVol[1][0] - BurstIdxVol[0][-1] > threshoAf:
        FinalEventList.append(BurstIdxVol[0])
        FinalSpikeNum.append(SpikeNum[0])

    for l in range(1, len(BurstIdxVol) - 1):
        if l == 290:
            x = False
        if  BurstIdxVol[l+1][0] - BurstIdxVol[l][-1] > threshoAf and BurstIdxVol[l][0] - BurstIdxVol[l-1][-1] > threshoBe:
            FinalEventList.append(BurstIdxVol[l])
            FinalSpikeNum.append(SpikeNum[l])


    if BurstIdxVol[-1][0] - BurstIdxVol[-2][-1] > threshoBe and e - BurstIdxVol[-1][-1] > 126:
        FinalEventList.append(BurstIdxVol[-1])
        FinalSpikeNum.append(SpikeNum[-1])
    return FinalEventList,FinalSpikeNum

def CalInt(CalTrace,CalXaX,nSr=1000):
    # Define the new time points for 1000 Hz
    new_sampling_rate = nSr  # in Hz
    new_num_samples = int(CalXaX[-1] * new_sampling_rate)
    new_time = np.linspace(0, CalXaX[-1], new_num_samples)

    # Interpolate the trace
    interpolator = interp1d(CalXaX, CalTrace, kind='cubic')  # 'cubic' for smooth interpolation
    interpolated_trace = interpolator(new_time)
    return interpolated_trace , new_time

def VolToCalIdx(volIdx,VolXaX,CalXax):
    t = VolXaX[volIdx]
    CalIdx = np.argmin(np.abs(CalXax - t))
    return(CalIdx)

def CalAmp(FBurstIdx,CalTrace,VolXax,CalXax,AmpTh):
    amplitudes = []
    calIDX = []
    ampIdx = []
    Zscoring = []
    for l in FBurstIdx:
        calSidx = VolToCalIdx(l[0],VolXax,CalXax)
        calEidx = VolToCalIdx(l[-1],VolXax,CalXax) + 250
        Prestart_idx = max(0, calSidx - 50)
        end_idx = min(len(CalTrace), calEidx)
        calIDX.append([calSidx,calEidx])

        # Extract baseline and response windows
        baseline_window = CalTrace[Prestart_idx:calSidx]
        response_window = CalTrace[calSidx:end_idx]
        zBase = np.sort(CalTrace)
        zBaseReg = zBase[0:int(len(zBase) * 0.08)]
        baseline = np.mean(baseline_window)
        baselineZ = np.mean(zBaseReg)
        std_baseline = np.std(zBaseReg)
        zTrace = (CalTrace - baselineZ)/std_baseline
        response_window_Z = zTrace[calSidx:end_idx]
        

        # Subtract baseline from response window
        response_window_corrected = response_window - baseline

        # Find the peak amplitude
        if np.max(response_window_corrected) > AmpTh:
            amplitude = np.max(response_window_corrected)
            amplitudeZscore = np.max(response_window_Z)
            rMax = np.where(response_window_corrected == np.max(response_window_corrected))[0]
            Idx = calSidx + rMax
        else:
            amplitude = 0  # No response detected
            Idx = calSidx + 125
        ampIdx.append(Idx)

        amplitudes.append(amplitude)
        Zscoring.append(amplitudeZscore)

    return amplitudes,calIDX,ampIdx,Zscoring

def normalize(trace):
    # normalize to [0, 1]
    trace = trace - np.min(trace)
    trace = trace / np.max(trace)
    return trace

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y
def butter_bandpass_filter(data, cutoffs, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoffs = np.array(cutoffs) / nyq
    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoffs, btype='band', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def median_filter(signal, window_size):
    filtered_signal = []
    for i in range(len(signal)):
        # Select a window of samples centered at the current sample
        window_start = i - window_size // 2
        window_end = i + window_size // 2 + 1
        if (window_start >= 0) and (window_end <= len(signal)):
            window = signal[window_start:window_end]
            filtered_signal.append(np.median(window))
        else:
            if (window_start < 0):
                zeros = np.zeros(np.abs(window_start))
                filtered_signal.append(np.median(np.concatenate([zeros, signal[:window_end]])))
            if (window_end > len(signal)):
                zeros = np.zeros(window_end - len(signal))
                filtered_signal.append(np.median(np.concatenate([signal[window_start:], zeros])))
    return np.array(filtered_signal)

def Baisline(tracetmp):
    num_lowest_points = int(len(tracetmp) * 0.08)
    # Find the lowest 8% of values
    lowest_values = np.sort(tracetmp)[:num_lowest_points]

    # Compute the mean of these values
    mean_lowest_values = np.mean(lowest_values)
    std_lowest_value = np.std(lowest_values)
    return mean_lowest_values, std_lowest_value

def CS_detection_MR(tracetemp,burst,durationTh):
    spike_indices = LongLIST(burst)
    non_spike_indices = np.array([i for i in range(len(list(tracetemp))) if i not in spike_indices])
    non_spike_values = tracetemp[non_spike_indices]

    # Interpolate missing points
    interpolated_trace = np.copy(tracetemp)
    interpolated_trace[spike_indices] = np.interp(spike_indices, non_spike_indices, non_spike_values)

    Vmtemp = median_filter(tracetemp, 25)
    comBurst = []
    bBurst = []
    binaryL = []
    genB,std = Baisline(Vmtemp)
    StdVm = np.std(Vmtemp)
    for b in burst:
        
        baseline_left = b[0]-40
        # make sure baseline_left is within the trace
        baseline_left = max(baseline_left, 0)
        baseline = np.min(Vmtemp[baseline_left:b[0]])
        # baseVm = np.mean(tracetemp[b[0]-50:])
        eventR = Vmtemp[b[-1]:b[-1]+200]
        spikeR = Vmtemp[b[0]:b[-1]]
        LastSvm = Vmtemp[b[-1]]
        Amp = LastSvm - genB
        # if b[0]>16000 and b[0]<17000:
        #     v=3453
        if len(b)<2:
            SpikeMax = b[-1]
            FAmp = Vmtemp[SpikeMax]- genB
            platuEnd = 0
        if len(b)>1:
            Mv = np.max(spikeR)
            SpikeMax = np.where(spikeR == Mv)[0][0]+b[0]
            FAmp = Vmtemp[SpikeMax]- genB
            RetP = np.where(eventR<(LastSvm-0.2*Amp))
            if RetP[0].size > 0 and  Amp - np.min(eventR) > 3 * StdVm:
                platuEnd = RetP[0][0]
            else:
               platuEnd =  0
        
        
        VmSIkDIF = Vmtemp[b[-1]]-Vmtemp[b[0]]
        
        if (platuEnd >= durationTh and len(b) > 2 and  (VmSIkDIF > 0.5*FAmp  or FAmp  -VmSIkDIF > 0.5*FAmp) and SpikeMax >= b[-1]-4) or (VmSIkDIF > 0.5*FAmp  and b[-1] - SpikeMax + platuEnd >= durationTh and len(b) > 2):
            comBurst.append(b)
            binaryL.append(True)
        elif platuEnd < durationTh or len(b) <= 2 or (SpikeMax < b[-1]-4) or (VmSIkDIF < 0.5*FAmp and FAmp  -VmSIkDIF < 0.5*FAmp) :
            bBurst.append(b)
            binaryL.append(False)
        
    return comBurst,bBurst,binaryL,Vmtemp
    
def CS_detection(tracetemp, spktemp, axtemp, ISI_threshold=10, ADP_threshold=0.6, ADP_postwindow=50, prespike_window=5, burst_merging_threshold=14, left_cross_threshold=10, duration_threshold=5, plotflag=False):
    # normalize trace
    plotcolor = '#E76F51'
    plotcolor_simple = '#026C80'
    plotcolor_complex = '#66FF00' # bright green
    plotcolor_burst = '#EE9B00'
    spkoffset = 0.1
    linesize = 5
    linewidth = 1
    tracetemp = normalize(tracetemp)
    tracetemp = butter_highpass_filter(tracetemp, 3, 500, order=5)
    Vmtemp = median_filter(tracetemp, 25)
    bursts = get_bursts(spktemp, ISI_threshold)[0]
    burst_spikes = np.array([spk for burst in bursts for spk in burst ]).astype(int)
    single_spikes = np.array([spk for spk in spktemp if spk not in burst_spikes]).astype(int)
    if plotflag:
        axtemp.plot(tracetemp, linewidth=0.5, color=plotcolor)
        axtemp.plot(Vmtemp, linewidth=0.5, color='black')
        axtemp.axis('off')
    flag_skip = False
    complex_bursts = []
    regular_bursts = []
    CB_duration = []
    if len(bursts) == 0:
        regular_bursts = []
        complex_bursts = []
        # plot single spikes
        if plotflag:
            axtemp.scatter(single_spikes, np.ones(len(single_spikes)) * (np.max(tracetemp)+spkoffset), color=plotcolor_simple, marker='|', s=linesize, linewidth=linewidth)
    else:
        for idx_burst, burst in enumerate(bursts):
            if flag_skip:
                flag_skip = False
                print('Skipping burst', idx_burst)
                continue
            else:
                # define baseline as the minimum of the 3 points before the first spike in the burst
                baseline_left = burst[0]-prespike_window
                # make sure baseline_left is within the trace
                baseline_left = max(baseline_left, 0)
                baseline = np.min(tracetemp[baseline_left:burst[0]])
                baseline_idx = np.argmin(tracetemp[baseline_left:burst[0]]) + burst[0]-prespike_window
                # plot the baseline point
                #axes[0,0].scatter(baseline_idx, baseline, color='black', marker='o', s=5)
                spikeheight = tracetemp[burst[0]] - baseline
                # plot the spike height
                #axes[0,0].scatter(burst[0], tracetemp[burst[0]], color='black', marker='o', s=5)
                # plot the height of first spike in the burst
                #axes[0,0].plot([burst[0], burst[0]], [baseline, tracetemp[burst[0]]], '--', color='black', linewidth=0.5)
                ADP_threshold_local = baseline + ADP_threshold*spikeheight
                ADP_window_end = burst[-1]+ADP_postwindow
                # check if there's a next burst, if so, the ADP_window should not exceed the first spike of the next burst
                if idx_burst < len(bursts)-1:
                    ADP_window_end = min(ADP_window_end, bursts[idx_burst+1][0])
                # make sure the ADP_window_end is within the trace
                ADP_window_end = min(ADP_window_end, len(tracetemp))
                ADP_window = np.arange(burst[0], ADP_window_end).astype(int)
                # find during the ADP_window, the first point that is above ADP_threshold_local
                ADP_idx_left = np.where(Vmtemp[ADP_window] > ADP_threshold_local)[0]
                if len(ADP_idx_left) > 0:
                    ADP_idx_left = ADP_window[ADP_idx_left[0]]
                    ADP_idx_left = ADP_idx_left[ADP_idx_left < burst[-1]+left_cross_threshold]
                if len(ADP_idx_left) == 0:
                    ADP_idx_left = np.nan
                    ADP_idx_right = np.nan
                    # this means it never crosses the threshold, so it is a regular burst
                    regular_bursts.append(burst)
                else:
                    ADP_idx_left = ADP_idx_left[0]
                    # plot the ADP_idx_left
                    if plotflag:
                        axtemp.scatter(ADP_idx_left, Vmtemp[ADP_idx_left], color='green', marker='o', s=5)
                    # find the first point that is below ADP_threshold_local after last spike and before ADP_window_end
                    ADP_idx_right = np.where(Vmtemp[ADP_idx_left:ADP_window_end] < ADP_threshold_local)[0]
                    if len(ADP_idx_right) == 0:
                        ADP_idx_right = np.nan
                        # This means it crosses the threshold and never goes below it, so we need to consider merging with the next burst
                        # if there's a next burst, check if the first spike of the next burst is within 25ms after the last spike of the current burst
                        if idx_burst < len(bursts)-1:
                            if bursts[idx_burst+1][0] - burst[-1] < burst_merging_threshold:
                                # redefine the ADP_window to include the next burst
                                ADP_window_end = bursts[idx_burst+1][-1]+ADP_postwindow
                                ADP_window = np.arange(burst[0], ADP_window_end).astype(int)
                                # find the ADP_idx_right again
                                ADP_idx_right = np.where(Vmtemp[ADP_idx_left:ADP_window_end] < ADP_threshold_local)[0]
                                flag_skip = True
                                if len(ADP_idx_right) == 0:
                                    ADP_idx_right = np.nan
                                else:
                                    ADP_idx_right = ADP_idx_left + ADP_idx_right[0]
                                    # plot the ADP_idx_right
                                    if plotflag:
                                        axtemp.scatter(ADP_idx_right, Vmtemp[ADP_idx_right], color='purple', marker='o', s=5)
                                burst = burst + bursts[idx_burst+1]
                    else: # if there's a ADP_idx_right
                        ADP_idx_right = ADP_idx_left + ADP_idx_right[0]
                        ADP_idx_max = ADP_idx_left + np.argmax(Vmtemp[ADP_idx_left:ADP_idx_right])
                        # plot the ADP_idx_right
                        if plotflag:
                            axtemp.scatter(ADP_idx_right, Vmtemp[ADP_idx_right], color='purple', marker='o', s=5)
                    # finally, we want to check if we can merge single spikes
                    # find single spike before it crosses the right threshold
                    single_spike_within_burst = single_spikes[np.logical_and(single_spikes >= burst[-1], single_spikes <= ADP_idx_right)]
                    if len(single_spike_within_burst) > 0:
                        # merge all single_spike_within_burst to the burst
                        burst = np.append(burst, single_spike_within_burst)
                        # remove all single_spike_within_burst from single_spikes
                        single_spikes = np.array([spk for spk in single_spikes if spk not in single_spike_within_burst])
                    # extrac check if the duration is longer than duration_threshold
                    if ADP_idx_right - ADP_idx_left > duration_threshold :
                        complex_bursts.append(burst)
                        CB_duration.append(ADP_idx_right - ADP_idx_left)
                    else:
                        regular_bursts.append(burst)
                # plot the ADP window with the threshold
                if plotflag:
                    axtemp.plot(ADP_window, np.ones(len(ADP_window))*ADP_threshold_local, '--', color='gray', linewidth=linewidth)
                    # plot the complex bursts
                    for burst in complex_bursts:
                        axtemp.scatter(burst, np.ones(len(burst)) * (np.max(tracetemp)+spkoffset), color=plotcolor_complex, marker='|', s=linesize, linewidth=linewidth)
                    # plot the regular bursts
                    for burst in regular_bursts:
                        axtemp.scatter(burst, np.ones(len(burst)) * (np.max(tracetemp)+spkoffset), color=plotcolor_burst, marker='|', s=linesize, linewidth=linewidth)
                    # plot single spikes
                    axtemp.scatter(single_spikes, np.ones(len(single_spikes)) * (np.max(tracetemp)+spkoffset), color=plotcolor_simple, marker='|', s=linesize, linewidth=linewidth)
    return {'allspikes': spktemp, 'bursts': bursts, 'complex_bursts': complex_bursts, 'regular_bursts': regular_bursts, 'single_spikes': single_spikes, 'CB_duration': CB_duration, 'Vm':Vmtemp}

def LongLIST (LIST):
    lL = []
    for i in range(len(LIST)):
        lL = lL +list(np.atleast_1d(LIST[i]))
    return(lL)

def ChooseCom (COMPLiDX,ChosenSpikeIdx):
    FinalCom = []
    FinalBurstNonCom = []
    BoolianListCom = []
    longChosSpike = LongLIST(COMPLiDX)
    for b in range(len(ChosenSpikeIdx)):
        if ChosenSpikeIdx[b][0] in longChosSpike:
            #matching_sublists = [sublist for sublist in COMPLiDX if b[0] in sublist]
            FinalCom.append(ChosenSpikeIdx[b])
            BoolianListCom.append(True)
        if ChosenSpikeIdx[b][0] not in longChosSpike:
            FinalBurstNonCom.append(ChosenSpikeIdx[b])
            BoolianListCom.append(False)
        
    return FinalCom,FinalBurstNonCom,BoolianListCom

def get_bursts(spikes, burst_threshold=12):
    if len(spikes) == 0:
        return [], [], 0
    else:
        # Define burst threshold
        bursts = []
        # Initialize the first burst group
        current_burst = [spikes[0]]
        # Iterate through the burst_spike array
        for i in range(1, len(spikes)):
            # If the difference between the current and previous spike is less than or equal to 14
            if spikes[i] - spikes[i - 1] <= burst_threshold:
                # Add the current spike to the current burst group
                current_burst.append(spikes[i])
            else:
                # Otherwise, add the current burst group to Bursts and start a new group
                bursts.append(current_burst)
                current_burst = [spikes[i]]
        # Add the last burst group to Bursts if it is not empty
        if len(current_burst) > 0:
            bursts.append(current_burst)
        # Filter bursts to only include those with more than one spike
        bursts = [burst for burst in bursts if len(burst) > 1]
        burst_events = [burst[0] for burst in bursts]
        number_of_bursts = len(bursts)
        return bursts, burst_events, number_of_bursts
    
def SingleSpk(Burstidx,amp,calIdx,cal,vol,VolXax,CalXax,NonFcal):
    SSidx =[]
    burstIdx =[]
    SingAmp = []
    SingIdx = []
    SingleVolT = []
    singleCalT = []
    singleCalTzSC = []
    burstAmp = []
    bburstIdx = []
    burstVolT = []
    burstCalT = []
    burstCalTzSC = []
    AmpIdx = 0
    x = 0
    for i,b in enumerate(Burstidx):
        if len(b) ==1:
            
            SSidx.append(b)
            CalIdxS  = VolToCalIdx(b[0] - 25,VolXax,CalXax) -1
            CalIDP = VolToCalIdx(b[0],VolXax,CalXax)
            if x == 2:
                P = '0P'
            x=+1
            vEidx = min(b[0] + 125,np.size(VolXax)-1)
            CalIdxE = VolToCalIdx(b[0] + 125,VolXax,CalXax)
            if CalIdxS <0:
                CalIdxE = CalIdxE + np.abs(CalIdxS) +1
                CalIdxS = 1
            if CalIdxE - CalIdxS > 10:
                CalIdxE = CalIdxE -(CalIdxE - CalIdxS - 10)
            if CalIdxE - CalIdxS < 10:
                CalIdxE = CalIdxE - (CalIdxE - CalIdxS -10)
            # Extract baseline values
            baseline_data = cal[CalIdxS-100:CalIdxS-50]

            # Calculate the mean and standard deviation of the baseline
            mean_baseline,std_baseline = Baisline(NonFcal)
            #std_baseline = np.std(baseline_data)

            # Calculate Z-scores for each point in the trace
            # Compute mean and standard deviation
            mean_trace = np.mean(cal)
            std_trace = np.std(cal)

            # Compute Z-scores
            z_scores = (cal - mean_trace) / std_trace
            #z_scores = cal
            A = np.max([0,CalIdxS-1])
            B = np.min([CalIdxE-1,np.size(CalXax)])
            
            curr_z = z_scores[A:B]
            singleCalTzSC.append(curr_z)
            aMPzc = np.max(z_scores[CalIDP:B])- np.min(z_scores[CalIDP:B])
            
            CALr = list(cal[np.max([0,CalIdxS-1]):CalIdxE-1] - np.min(cal[np.max([0,CalIdxS-1]):CalIdxE-1]))
            CALrA = list(cal[np.max([0,CalIDP]):CalIdxE-1]) - np.min(cal[np.max([0,CalIdxS-1]):CalIdxE-1])
            Ampc = np.max(CALr)
            AmpcA = np.max(z_scores[CalIDP:B])
            AmpIdx = np.argmin(np.abs(curr_z - AmpcA))
            SingIdx.append(AmpIdx)
            singleCalT.append(curr_z)
            SingleVolT.append(vol[b[0]-25:b[0]+125])
            SingAmp.append(AmpcA)
        if len(b) > 1:
            burstIdx.append(b)
            if len(b)>2 and len(b) < 8:
                CalIdxS  = VolToCalIdx(b[0] - 25,VolXax,CalXax) -1
                CalIdxE = VolToCalIdx(b[0] + 125,VolXax,CalXax)
                if CalIdxE - CalIdxS > 10:
                    CalIdxE = CalIdxE -(CalIdxE - CalIdxS - 10)
                if CalIdxE - CalIdxS < 10:
                    CalIdxE = CalIdxE - (CalIdxE - CalIdxS -10)
                # Extract baseline values
                baseline_data = cal[CalIdxS-100:CalIdxS-50]

                # Calculate the mean and standard deviation of the baseline
                mean_baseline,std_baseline = Baisline(NonFcal)
                #std_baseline = np.std(baseline_data)

                # Calculate Z-scores for each point in the trace
                z_scores = cal
                curr_z = z_scores[np.max([0,CalIdxS-1]):CalIdxE-1]
                burstCalTzSC.append(curr_z)
                aMPzc = np.max(curr_z)- np.min(curr_z)
                
                CALr = list(cal[np.max([0,CalIdxS-1]):CalIdxE-1] - np.min(cal[np.max([0,CalIdxS-1]):CalIdxE-1]))
                Ampc = np.max(CALr)
                AmpIdx = np.argmin(np.abs(CALr - Ampc))
                bburstIdx.append(AmpIdx)
                burstCalT.append(CALr)
                burstVolT.append(vol[b[0]-25:b[0]+125])
                burstAmp.append(aMPzc)
    if burstAmp:
        burstCalT = burstCalT/np.max(burstAmp)
        burstAmp = burstAmp
        #if SingAmp:
            # singleCalT = singleCalT/np.max(burstAmp)
            #SingAmp = SingAmp/np.max(burstAmp)
    return SSidx,burstIdx,singleCalT,SingleVolT,SingAmp,SingIdx,burstCalT,burstVolT,burstAmp,bburstIdx

def linear_model(x, m, c):
    return m * x + c

# Polynomial model: y = ax^2 + bx + c
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# Exponential model: y = a * exp(b * x) + c
def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

def MeanRes(Amp,spNum,spX):
    Famp = []
    Xax = []
    for i in spX:
        currAmp = [val for idx,val in enumerate(Amp) if spNum[idx] == i]
        if currAmp:
            Famp.append(np.mean(currAmp))
            Xax.append(i)
    return Famp,Xax

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

def meanCalc (binn, tRACE):
    avgS = []
    for i in range(len(binn)):
        if i < len(binn) -1:
            currBin = tRACE[binn[i]:binn[i+1]]
            avgS.append(np.mean(currBin))
        
        elif i  == len(binn) -1 : 
            currBin = tRACE[binn[i]:-1]
            avgS.append(np.mean(currBin))
    return avgS

def calculate_firing_rate(spike_indices, trace ,window_size, step_size,VAx, caltRACE,CAx,C ='N'):
    # Initialize an array to hold the firing rates
    if C == 'y':
        x = 15
    firing_rates = []
    window_starts = range(0, len(caltRACE) - (window_size + 1), step_size)#start index of binns in calcium trace
    cBinn = []
    CtBin =[]
    vBinn =[]
    MvIDX = []

    # Calculate the firing rate for each window
    for start in window_starts:
        
        differenceS = np.abs(VAx - CAx[start])#finding relative start index of windows in voltage
        Startidx = np.argmin(differenceS)
        end = start + window_size
        differenceE = np.abs(VAx - CAx[end]) #finding relative end index of windows in voltage
        CtBin.append(np.mean(CAx[start:end+1]))
        Endidx = np.argmin(differenceE)
        MidlleIdx = Startidx  +(Endidx -Startidx)/2
        MvIDX.append(MidlleIdx)
        spikes_in_window = [spike for spike in spike_indices if Startidx <= spike < Endidx]
        wd = VAx[Endidx]-VAx[Startidx]
        firing_rate = len(spikes_in_window) / wd  # FR = number of spikes / window size/ sampling rate -> firing rate as number of spike for second, note window size is given for calcium firing rate so we devide by 30 hz
        if np.isnan(firing_rates).any():
            x= 'why'
        firing_rates.append(firing_rate)
        cBinn.append(CAx[start])
        
        vBinn.append(Startidx)
    calAvg = meanCal(window_starts,caltRACE)
    

    return firing_rates,calAvg,cBinn, vBinn,MvIDX,CtBin


def calculate_Vm_Cal(Vmtrace ,window_size, step_size,VAx, caltRACE,CAx):
    firing_rates = []
    window_starts = range(0, len(caltRACE) - (window_size + 1), step_size)#start index of binns in calcium trace
    cBinn = []
    vBinn =[]
    MvIDX = []
    MeanvM = []
    MeanCal = []
    CtBin =[]
    # Calculate the firing rate for each window
    for start in window_starts:
        differenceS = np.abs(VAx - CAx[start])#finding relative start index of windows in voltage
        Startidx = np.argmin(differenceS)
        end = start + window_size
        differenceE = np.abs(VAx - CAx[end]) #finding relative end index of windows in voltage
        Endidx = np.argmin(differenceE)
        MeanvM.append(np.mean(Vmtrace[Startidx:Endidx+1]))
        cBinn.append(CAx[start])
        CtBin.append(np.mean(CAx[start:end+1]))
        
        vBinn.append(Startidx)
    calAvg = meanCal(window_starts,caltRACE)
    return MeanvM,calAvg,cBinn, vBinn,CtBin

def dataEXTRAC(path):
    TracePathCal = os.path.join(path,'calTraceDF.csv')
    TracePathVol = os.path.join(path,'volTraceDF.csv')
    TracePathSPIKE = os.path.join(path,'SpikeIdx.csv')
    TracePathCalAX = os.path.join(path,'calTime.csv')
    TracePathVolAX = os.path.join(path,'volTime.csv')
    VolTrace = pd.read_csv(TracePathVol)
    VolTrace = np.array(VolTrace)
    VolTrace = VolTrace.flatten()
    Trace = VolTrace
    CalTrace = pd.read_csv(TracePathCal)
    CalTrace = np.array(CalTrace)
    CalTrace = CalTrace.flatten()
    spikeId = pd.read_csv(TracePathSPIKE)
    spikeId = np.array(spikeId)
    spikeId = spikeId.flatten()
    VolaX = pd.read_csv(TracePathVolAX)
    VolaX = np.array(VolaX)
    VolaX = VolaX.flatten()
    Calax = pd.read_csv(TracePathCalAX)
    Calax = np.array(Calax)
    Calax = Calax.flatten()
    return Trace,CalTrace,spikeId,VolaX,Calax
    

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



def plotFR(Path, FRlist,CalList,plotName,lag = 0):
    LagFRlist = FRlist[0:(-lag-1)]
    LagCalList = CalList[lag:-1]
    popt_linearZ, _ = curve_fit(linear_model, LagFRlist, LagCalList)
    popt_quadraticZ, _ = curve_fit(quadratic_model, LagFRlist, LagCalList)
    popt_exponentialZ, _ = curve_fit(exponential_model, LagFRlist, LagCalList,maxfev=1000000)
    y_pred_linearZ = linear_model(np.array(LagFRlist), *popt_linearZ)
    y_pred_quadraticZ = quadratic_model(np.array(LagFRlist), *popt_quadraticZ)
    y_pred_exponentialZ = exponential_model(np.array(LagFRlist), *popt_exponentialZ)
    metricsZ = {
        "Linear": (r2_score(LagCalList, y_pred_linearZ), mean_squared_error(LagCalList, y_pred_linearZ)),
        "Quadratic": (r2_score(LagCalList, y_pred_quadraticZ), mean_squared_error(LagCalList, y_pred_quadraticZ)),
        "Exponential": (r2_score(LagCalList, y_pred_exponentialZ), mean_squared_error(LagCalList, y_pred_exponentialZ))
    }

    # Print the metrics
    for model, (r2, mse) in metricsZ.items():
        print(f"{model} Model: R² = {r2:.3f}, MSE = {mse:.3f}")
    #create all relevent subplot
    correlation, p_value = pearsonr(LagFRlist, LagCalList)
    Linear_r2 = metricsZ['Linear'][0]
    predXZ = set(LagFRlist)  # R² is the first item in the tuple
    predXZ = np.array(list(predXZ))
    y_pred_PlotZ =  linear_model(predXZ, *popt_linearZ)
    y_pred_PlotZ =y_pred_PlotZ.tolist()
    predXZ = predXZ.tolist()

    fitLineZ = go.Scatter(
        x=predXZ,
        y=y_pred_PlotZ,
        mode="lines",name='Fitted line',
        line=go.scatter.Line(color="gray", dash="dash", width=3),  # Add the 'dash' attribute
        showlegend=False
    )
    svg_path = os.path.join(Path, f'lag{lag}.svg')
    html_path = os.path.join(Path, f'lag{lag}.html')
    r2 = metricsZ["Linear"][0]  # Access the first element (R²) for the linear model
    fig = go.Figure()
    # Add data for non-complex spikes
    fig.add_trace(go.Scatter(
        x=LagFRlist,
        y=LagCalList,
        mode='markers',
        name='Non-complex spikes',
        marker=dict(color='dimgrey', size=8, symbol='circle'),showlegend=False
    ))

    fig.add_trace(fitLineZ)
    # Layout customization
    fig.update_layout(
        title="Amplitude vs. Number of Spikes",
        
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
        annotations=[
            dict(
                text=f"R^2: {Linear_r2}\nPearson correlation :{correlation}\np_value:{p_value}",  # Text content
                x=1,                          # x-position (aligned with top-right corner)
                y=1,                          # y-position (aligned with top-right corner)
                xref="paper",                 # Reference to the paper coordinates
                yref="paper",                 # Reference to the paper coordinates
                showarrow=False,              # No arrow
                align="right",                # Align text to the right
                font=dict(size=8, color="black"),  # Font size and color
                bgcolor="rgba(0,0,0,0)",    # Background color with transparency
                bordercolor="rgba(0,0,0,0)",          # Border color
                borderwidth=1                 # Border width
            )
        ],


                xaxis=dict(
                title="Firing rate number of spike\second",
                showgrid=False,               # Show grid
                gridcolor='lightgray',       # Grid color for the X-axis
                zerolinecolor="rgba(0,0,0,0)",        # Zero line color
                showline=True,               # Show axis line
                linecolor='black',           # Axis line color
                ticks="outside", tickfont=dict(
                size=12  # Adjust the font size for the X-axis ticks
                )             # Ticks outside the plot
                ),
                yaxis=dict(
                    title="Avrage calcium df/f",
                    showgrid=False,
                    gridcolor='lightgray',
                    zerolinecolor="rgba(0,0,0,0)",
                    showline=True,
                    linecolor='black',
                    ticks="outside",tickfont=dict(
                    size=12  # Adjust the font size for the X-axis ticks
                ) 
                )
                
            
        )
    # Show the plot
    #fig.show()

    fig.write_image(svg_path)
        # Save as HTML
    fig.write_html(html_path)
    return LagFRlist,LagCalList

def Findx(time,timeX):
    differences = np.abs(timeX - time)
    IDX = np.argmin(differences)
    return IDX

def CalLeanFit(fr,calA):
    if len(fr) < 2 or len(calA)< 2:
        correlation,p_value = 0,1
    else:
        correlation, p_value = pearsonr(fr,calA)
    
    try:
        popt_linear, _ = curve_fit(linear_model, fr, calA)
        predX = np.linspace(np.min(fr), np.max(fr), 100)
        y_pred = linear_model(predX, *popt_linear)
        predX = set(fr)  # R² is the first item in the tuple
        predX = np.array(list(predX))
        y_pred_Plot =  linear_model(predX, *popt_linear)
        y_pred_Plot =y_pred_Plot.tolist()
        predX = predX.tolist()
        slope = (popt_linear[0])
        return correlation,p_value,predX,y_pred_Plot,slope

    except Exception as e:
        print(f"⚠️ Linear fit failed: {e}")
        return None, None, None, None, None

    
def CalCoorelation (fr,avgC,MaxL,ws,step,lag,t):
    wStart = np.arange(0,len(fr)-ws,step)
    TraceCor = []
    MeanfR = []
    MeanT = []
    for l in wStart:
        calForCoor = avgC[l:l+ws]
        fRforCorr = fr[l:l+ws]
        MeanFr = np.mean(fr[l:l+ws])
        MeanT.append(np.mean(t[l:l+ws]))
        MeanfR.append(MeanFr)
        correlation =np.corrcoef(fRforCorr, calForCoor)[0, 1]
        TraceCor.append(correlation)
        #correlation = signal.correlate(fr,avgC, mode='full', method='auto')
    if lag == 0:
            MaxL = len(TraceCor)
            
    return TraceCor,MaxL,MeanfR,MeanT

def PadCor (MaxI,CorrL):
    PcORR = []
    for l in range(len(CorrL)):
        x = CorrL[l]
        if MaxI > len(CorrL[l]):
            x = np.append([x], [np.nan] * (MaxI - len(CorrL[l]))).tolist()
        PcORR.append(x)
    return PcORR
        
def CorrWindow(c_map,FR,Idx,TIME):
    corrF = c_map[Idx]
    start = 0
    CorrW = []
    corrD = []
    crooRvAL = []
    mEANfr = []
    CURRNfR = []
    edgeFrDiff = []

    for i in range(len(corrF) - 1):
        if corrF[i]*corrF[i+1] < 0 or corrF[i]*corrF[i+1] == 0:
            CorrW.append(corrF[start:i])
            corrD.append(TIME[i]-TIME[start])
            crooRvAL.append(np.mean(corrF[start:i]))
            mEANfr.append(np.mean(CURRNfR))
            edgeFrDiff.append(FR[i+1]- FR[i])
            CURRNfR = []
            start = i+1
        else:
            CURRNfR.append(FR[i])
        if i == len(corrF) - 2:
            CorrW.append(corrF[start:i+1])
            crooRvAL.append(np.mean(corrF[start:i+1]))
            corrD.append(TIME[i+1]-TIME[start])
            
            start = i+1
    return CorrW,corrD,crooRvAL,mEANfr,edgeFrDiff
    
    
def lagOptimaizre(VolTrace,spike_indices ,window_size, step_size,VolAx, caltRACE,CalAx, LagList,path,smoothw):
    p_corr = []
    Np_corr = []
    Allcor = []
    p_val = []
    figLag = go.Figure()
    FullLag = np.arange(-1*LagList[-1],LagList[-1],0.033)
    slope = []
    correlation_map = []
    Ncorrelation_map = []
    colors = plt.cm.jet(np.linspace(0, 1, len(LagList)))  # Tab10 color map
    VolAx = VolAx - VolAx[0]
    CalAx = CalAx - CalAx[0]
    MaxL = 0
    for i,l in enumerate(LagList): 
       
        calLagId = Findx(l,CalAx)
        volLagId = (Findx(l,VolAx) +1)*-1
        LagVol = VolTrace[0:volLagId] 
        LagVolAx = VolAx[0:volLagId]
        LagCal = caltRACE[calLagId:-1]
        LagCalAx = CalAx[calLagId:-1] - CalAx[calLagId]
        lagSpike_indices = [i for i in spike_indices if i < len(LagVol)]
        if l >0:
            NcalLagId = Findx(l,CalAx) * -1
            NvolLagId = Findx(l,VolAx)
            NLagVol = VolTrace[NvolLagId :] 
            NLagVolAx = VolAx[NvolLagId:] - VolAx[NvolLagId]
            NLagCal = caltRACE[0:NcalLagId]
            NLagCalAx = CalAx[0:NcalLagId]
            Nspike_indices = spike_indices - NvolLagId
            NlagSpike_indices = [i for i in Nspike_indices if i > 0]
            Nf_r,Ncal_Avg,Nc_Binn, Nv_Binn,Nx,NTime = calculate_firing_rate(NlagSpike_indices,NLagVol,window_size,step_size,NLagVolAx,NLagCal,NLagCalAx)
            Ncor,Np,NpredX,Ny_pr,Nsl = CalLeanFit(Nf_r,Ncal_Avg)
            NcP,NMaxL,M,CorrT = CalCoorelation(Nf_r,Ncal_Avg,MaxL,6,1,l,NTime)
            Ncorrelation_map.append(NcP)
            Np_corr.append(Ncor)
        f_r,cal_Avg,c_Binn, v_Binn,x,Time = calculate_firing_rate(lagSpike_indices,LagVol,window_size,step_size,LagVolAx,LagCal,LagCalAx)
        
        cP,MaxL,MFr,CorrT = CalCoorelation(f_r,cal_Avg,MaxL,6,1,l,Time)
        if l == 0:
            WindowFR = MFr
            tIMEsc = CorrT
        correlation_map.append(cP)
       
        cor,p,predX,y_pr,sl = CalLeanFit(f_r,cal_Avg)
        
        figLag.add_trace(go.Scatter(
        x=predX,
        y=y_pr,
        mode="lines", name=f"Lag: {l}",
        line=dict(color=f"rgba({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255},1)"),
        showlegend=True
    ))
        p_corr.append(cor)
        
        slope.append(sl)
        p_val.append(p)
    
    figLag.update_layout(
        
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            xaxis=dict(
                    title="Firing rate number of spike\second",
                    showgrid=False,               # Show grid
                    gridcolor='lightgray',       # Grid color for the X-axis
                    zerolinecolor="rgba(0,0,0,0)",        # Zero line color
                    showline=True,               # Show axis line
                    linecolor='black',           # Axis line color
                    ticks="outside", tickfont=dict(
                    size=12  # Adjust the font size for the X-axis ticks
                )             # Ticks outside the plot
                ),
            yaxis=dict(
                    title="Avrage calcium df/f",
                    showgrid=False,
                    gridcolor='lightgray',
                    zerolinecolor="rgba(0,0,0,0)",
                    showline=True,
                    linecolor='black',
                    ticks="outside",tickfont=dict(
                    size=12  # Adjust the font size for the X-axis ticks
                ) 
                )
                
            
        )
    
    #figLag.show()
    svg_pathL = os.path.join(path, f'FitForDiffLag{smoothw}.svg')
    html_pathL = os.path.join(path, f'FitForDiffLag{smoothw}.html')
    figLag.write_image(svg_pathL)
    # Save as HTML
    figLag.write_html(html_pathL)  
    svg_pathS = os.path.join(path, f'SlopeDiffLag{smoothw}.svg')
    html_pathS = os.path.join(path, f'SlopeDiffLag{smoothw}.html')
    figSlope = go.Figure()

    PnCor = PadCor(MaxL, Ncorrelation_map)

    PCor = PadCor(MaxL, correlation_map)

    revnCor = list(reversed(PnCor[1:]))
    #revnCor = [PnCor[-(i+1)] for i, v in enumerate(PnCor)  if i <len(Pn)]
    fullCorr = revnCor+PCor 
    fcorrelation_map = np.vstack(fullCorr)
# Add data for non-complex spikes
    figSlope.add_trace(go.Scatter(
        y=slope,
        x=LagList,
        mode='markers',
        name='Non-complex spikes',
        marker=dict(color='dimgrey', size=8, symbol='circle'),showlegend=False
    ))
    figSlope.update_layout(
    title="slope for diffrentlag",
    
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
    xaxis=dict(
    title="Lag Time",
    showgrid=False,               # Show grid
    gridcolor='lightgray',       # Grid color for the X-axis
    zerolinecolor="rgba(0,0,0,0)",        # Zero line color
    showline=True,               # Show axis line
    linecolor='black',           # Axis line color
    ticks="outside", tickfont=dict(
    size=12  # Adjust the font size for the X-axis ticks
    )             # Ticks outside the plot
            ),
            yaxis=dict(
                title="slopeValue",
                showgrid=False,
                gridcolor='lightgray',
                zerolinecolor="rgba(0,0,0,0)",
                showline=True,
                linecolor='black',
                ticks="outside",tickfont=dict(
                size=12  # Adjust the font size for the X-axis ticks
            ) 
            )
            
        
    )
    # Show the plot
    #figSlope.show()
    FullCOr = Np_corr + p_corr

    figSlope.write_image(svg_pathS)
        # Save as HTML
    figSlope.write_html(html_pathS)

    svg_pathC = os.path.join(path, f'CorrDiffLag{smoothw}.svg')
    html_pathC = os.path.join(path, f'CorrDiffLag{smoothw}.html')
    figCorr = go.Figure()
# Add data for non-complex spikes
    figCorr.add_trace(go.Scatter(
        y=FullCOr[1:],
        x=FullLag,
        mode='lines',  # Ensures a continuous line instead of points
        line=dict(color='red', width=2),
        name='Non-complex spikes',
        showlegend=False
    ))
    figCorr.update_layout(
        title="Correlation for Different Lags",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
        xaxis=dict(
            title="Lag Time",
            showgrid=False,
            gridcolor='lightgray',
            zerolinecolor="rgba(0,0,0,0)",
            showline=True,
            linecolor='black',
            ticks="outside",
            tickfont=dict(size=12),
              # Set the x-axis range explicitly
        ),
        yaxis=dict(
            title="Correlation Value",
            showgrid=False,
            gridcolor='lightgray',
            zerolinecolor="rgba(0,0,0,0)",
            showline=True,
            linecolor='black',
            ticks="outside",
            tickfont=dict(size=12),
            range=[-0.5, 0.8]
        )
    )
    # Show the plot
    #figCorr.show()

    figCorr.write_image(svg_pathC)
        # Save as HTML
    figCorr.write_html(html_pathC)
    
    maxFitIdSlope = np.argmax(np.abs(slope))
    maxFitIdCorr = np.argmax(np.abs(p_corr))
    MaxcalLagId = Findx(LagList[maxFitIdSlope],CalAx)
    MaxvolLagId = (Findx(LagList[maxFitIdSlope],VolAx) +1)*-1
    MaxSlopeF = figLag['data'][maxFitIdSlope]
    MaxCorrF = figLag['data'][maxFitIdCorr]
    
    return LagList[maxFitIdSlope],MaxSlopeF,slope[maxFitIdSlope],MaxcalLagId,MaxvolLagId,LagList[maxFitIdCorr],MaxCorrF,p_corr[maxFitIdCorr],fcorrelation_map,WindowFR,tIMEsc

def lagOptimaizreVm(VMTrace,window_size, step_size,VolAx, caltRACE,CalAx, LagList,path,smoothw):
    p_corr = []
    p_val = []
    figLag = go.Figure()
    slope = []
    correlation_map = []
    Ncorrelation_map = []
    colors = plt.cm.jet(np.linspace(0, 1, len(LagList)))  # Tab10 color map
    VolAx = VolAx - VolAx[0]
    CalAx = CalAx - CalAx[0]
    MaxL = 0
    for i,l in enumerate(LagList): 
       
        calLagId = Findx(l,CalAx)
        volLagId = (Findx(l,VolAx) +1)*-1
        LagVol = VMTrace[0:volLagId] 
        LagVolAx = VolAx[0:volLagId]
        LagCal = caltRACE[calLagId:-1]
        LagCalAx = CalAx[calLagId:-1] - CalAx[calLagId]
        
        NcalLagId = Findx(l,CalAx) * -1
        NvolLagId = Findx(l,VolAx)
        NLagVol = VMTrace[NvolLagId :] 
        NLagVolAx = VolAx[NvolLagId:] - VolAx[NvolLagId]
        NLagCal = caltRACE[0:NcalLagId]
        NLagCalAx = CalAx[0:NcalLagId]
        #Nspike_indices = spike_indices - NvolLagId
        #NlagSpike_indices = [i for i in Nspike_indices if i > 0]
        v_m,cal_Avg,c_Binn, v_Binn,Time = calculate_Vm_Cal(LagVol,window_size,step_size,LagVolAx,LagCal,LagCalAx)
        Nv_m,Ncal_Avg,Nc_Binn, Nv_Binn,NTime= calculate_Vm_Cal(NLagVol,window_size,step_size,NLagVolAx,NLagCal,NLagCalAx)
        NcP,NMaxL,M, nCorrT = CalCoorelation(Nv_m,Ncal_Avg,MaxL,6,1,l,NTime)
        cP,MaxL,MFr,CorrT = CalCoorelation(v_m,cal_Avg,MaxL,6,1,l,Time)
        if l == 0:
            WindowFR = MFr
        correlation_map.append(cP)
        Ncorrelation_map.append(NcP)
        cor,p,predX,y_pr,sl = CalLeanFit(v_m,cal_Avg)
        figLag.add_trace(go.Scatter(
        x=predX,
        y=y_pr,
        mode="lines", name=f"Lag: {l}",
        line=dict(color=f"rgba({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255},1)"),
        showlegend=True
    ))
        p_corr.append(cor)
        slope.append(sl)
        p_val.append(p)
    
    figLag.update_layout(
        
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            xaxis=dict(
                    title="Firing rate number of spike\second",
                    showgrid=False,               # Show grid
                    gridcolor='lightgray',       # Grid color for the X-axis
                    zerolinecolor="rgba(0,0,0,0)",        # Zero line color
                    showline=True,               # Show axis line
                    linecolor='black',           # Axis line color
                    ticks="outside", tickfont=dict(
                    size=12  # Adjust the font size for the X-axis ticks
                )             # Ticks outside the plot
                ),
            yaxis=dict(
                    title="Avrage calcium df/f",
                    showgrid=False,
                    gridcolor='lightgray',
                    zerolinecolor="rgba(0,0,0,0)",
                    showline=True,
                    linecolor='black',
                    ticks="outside",tickfont=dict(
                    size=12  # Adjust the font size for the X-axis ticks
                ) 
                )
                
            
        )
    
    #figLag.show()
    svg_pathL = os.path.join(path, f'FitForDiffLag{smoothw}.svg')
    html_pathL = os.path.join(path, f'FitForDiffLag{smoothw}.html')
    figLag.write_image(svg_pathL)
    # Save as HTML
    figLag.write_html(html_pathL)  
    svg_pathS = os.path.join(path, f'SlopeDiffLag{smoothw}.svg')
    html_pathS = os.path.join(path, f'SlopeDiffLag{smoothw}.html')
    figSlope = go.Figure()

    PnCor = PadCor(MaxL, Ncorrelation_map)

    PCor = PadCor(MaxL, correlation_map)

    revnCor = list(reversed(PnCor[1:]))
    #revnCor = [PnCor[-(i+1)] for i, v in enumerate(PnCor)  if i <len(Pn)]
    fullCorr = revnCor+PCor 
    fcorrelation_map = np.vstack(fullCorr)
# Add data for non-complex spikes
    figSlope.add_trace(go.Scatter(
        y=slope,
        x=LagList,
        mode='markers',
        name='Non-complex spikes',
        marker=dict(color='dimgrey', size=8, symbol='circle'),showlegend=False
    ))
    figSlope.update_layout(
    title="slope for diffrentlag",
    
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
    xaxis=dict(
    title="Lag Time",
    showgrid=False,               # Show grid
    gridcolor='lightgray',       # Grid color for the X-axis
    zerolinecolor="rgba(0,0,0,0)",        # Zero line color
    showline=True,               # Show axis line
    linecolor='black',           # Axis line color
    ticks="outside", tickfont=dict(
    size=12  # Adjust the font size for the X-axis ticks
    )             # Ticks outside the plot
            ),
            yaxis=dict(
                title="slopeValue",
                showgrid=False,
                gridcolor='lightgray',
                zerolinecolor="rgba(0,0,0,0)",
                showline=True,
                linecolor='black',
                ticks="outside",tickfont=dict(
                size=12  # Adjust the font size for the X-axis ticks
            ) 
            )
            
        
    )
    # Show the plot
    #figSlope.show()

    figSlope.write_image(svg_pathS)
        # Save as HTML
    figSlope.write_html(html_pathS)

    svg_pathC = os.path.join(path, f'CorrDiffLag{smoothw}.svg')
    html_pathC = os.path.join(path, f'CorrDiffLag{smoothw}.html')
    figCorr = go.Figure()
# Add data for non-complex spikes
    figCorr.add_trace(go.Scatter(
        y=p_corr,
        x=LagList,
        mode='markers',
        name='Non-complex spikes',
        marker=dict(color='dimgrey', size=8, symbol='circle'),showlegend=False
    ))
    figCorr.update_layout(
        title="Corralation for diffrentlag",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
        xaxis=dict(
        title="Lag Time",
        showgrid=False,               # Show grid
        gridcolor='lightgray',       # Grid color for the X-axis
        zerolinecolor="rgba(0,0,0,0)",        # Zero line color
        showline=True,               # Show axis line
        linecolor='black',           # Axis line color
        ticks="outside", tickfont=dict(
        size=12  # Adjust the font size for the X-axis ticks
        )             # Ticks outside the plot
                ),
                yaxis=dict(
                    title="corraltion value",
                    showgrid=False,
                    gridcolor='lightgray',
                    zerolinecolor="rgba(0,0,0,0)",
                    showline=True,
                    linecolor='black',
                    ticks="outside",tickfont=dict(
                    size=12  # Adjust the font size for the X-axis ticks
                ) 
                )
                
            
        )
    # Show the plot
    #figCorr.show()

    figCorr.write_image(svg_pathC)
        # Save as HTML
    figCorr.write_html(html_pathC)
    
    maxFitIdSlope = np.argmax(np.abs(slope))
    maxFitIdCorr = np.argmax(np.abs(p_corr))
    MaxcalLagId = Findx(LagList[maxFitIdSlope],CalAx)
    MaxvolLagId = (Findx(LagList[maxFitIdSlope],VolAx) +1)*-1
    MaxSlopeF = figLag['data'][maxFitIdSlope]
    MaxCorrF = figLag['data'][maxFitIdCorr]
    
    return LagList[maxFitIdSlope],MaxSlopeF,slope[maxFitIdSlope],MaxcalLagId,MaxvolLagId,LagList[maxFitIdCorr],MaxCorrF,p_corr[maxFitIdCorr],fcorrelation_map,WindowFR
# Create figure of voltage and calciu, for each FOV
def plotVolCal(path, VolAX, CALax,CalTrace, VolTrace,Name,NameSV,smW =1):
    pathFig = os.path.join(path,Name)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=VolAX, y=VolTrace.squeeze(), name="Voltage",line=dict(color='red', width=2)),secondary_y=False,)
    fig.add_trace(go.Scatter(x= CALax, y= CalTrace.squeeze(), name="Calcium",line=dict(color='blue', width=2)),secondary_y=True,)
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
    fig.write_image(os.path.join(path, NameSV), format="svg")

def devideTr (win_time,calT,VolT,calX,volX,spike_indices):
    comp_starts = np.arange(0, len(calX)-1, win_time)
    comp_starts = comp_starts.tolist()
    AllVolA = []
    AllVolT = []
    AllCalA = []
    AllCalT = []
    AllSPin = []
    # if comp_starts[-1] >=  len(calX):
    #     comp_starts.append(len(calX))
    for start in comp_starts:
        differenceC = np.abs(calX - calX[start])
        StartidxC = np.argmin(differenceC)
        differenceV = np.abs(volX - calX[start])
        StartidxV = np.argmin(differenceV)
        endC = start + win_time
        if endC > len(calX) - 1:
            endC = len(calX) -1
        differenceEc = np.abs(calX - calX[endC])
        differenceEv = np.abs(volX - calX[endC]) 
        EndidxC = np.argmin(differenceEc)
        EndidxV = np.argmin(differenceEv)
        spikes_in_window = [spike for spike in spike_indices if StartidxV <= spike < EndidxV]- StartidxV
        AllSPin.append(spikes_in_window)
        currCalX = calX[StartidxC:EndidxC] - calX[StartidxC]
        currCalT = calT[StartidxC:EndidxC]
        currVolX = volX[StartidxV:EndidxV] - volX[StartidxV]
        currVolT = VolT[StartidxV:EndidxV]
        AllVolA.append(currVolX)
        AllVolT.append(currVolT)
        AllCalA.append(currCalX)
        AllCalT.append(currCalT)
    return AllCalA,AllCalT,AllVolA,AllVolT,AllSPin


def downsample_signal(data, factor):
    """Downsamples the signal by averaging over non-overlapping windows."""
    n = len(data) // factor  # Number of new samples
    return np.mean(data[:n * factor].reshape(n, factor), axis=1)

def OScaLC(calSig,WindowS,overlapS,fs,downSfac = 'N'):
    if downSfac.upper() == 'Y':
        dCalSig = downsample_signal(calSig,4)
        dfs = fs/4
        f, Pxx = signal.welch(dCalSig, dfs, window='hamming', nperseg=WindowS, noverlap=overlapS)
    else:
        f, Pxx = signal.welch(calSig, fs, window='hamming', nperseg=WindowS, noverlap=overlapS)
    return f,Pxx


def calc_PSD_nospike(trace,fr,nuMax=100,nuMin=0.1):
    trace = trace - np.mean(trace)
    # calculate power spectral density
    freq=np.arange(0,trace.size)*fr/trace.size
    sMax = np.where(freq > nuMax)[0][0]
    sMin = np.where(freq > nuMin)[0][0]
    psd=np.multiply(fft(trace),np.conj(fft(trace)))
    return psd[sMin:sMax],freq[sMin:sMax]


# def remove_Frame_Multi(TraceV, TraceC, SpikeIdx, TimeC, TimeV):
    
#     # 1. Setup the initial plot
#     fig, ax = plt.subplots(figsize=(10, 4))
    
#     # We use a line object (l1) so we can update it later if needed, 
#     # though usually we just update the scatter plot of spikes.
#     ax.plot(TimeV, TraceV, color='cornflowerblue', label='Trace')
    
#     # Plot spikes - we save this object 'scat' to update it inside the loop
#     scat = ax.scatter(TimeV[SpikeIdx], TraceV[SpikeIdx], 
#                       color='red', s=15, zorder=5, label='Spikes')
    
#     ax.set_title('Left-click twice to define a range to remove.\nPress ENTER (or Middle Click) when finished.')
#     ax.legend(loc='upper right')
#     plt.draw()
    
#     print("--- Interaction Mode Started ---")
#     print("1. Click START of range.")
#     print("2. Click END of range.")
#     print("3. Repeat as needed.")
#     print("4. Press ENTER key to finish.")

#     # 2. Loop for multiple selections
#     while True:
#         # Get 2 clicks from the user
#         points = plt.ginput(n=2, timeout=-1, mouse_stop=2) # mouse_stop=2 usually means middle click
        
#         # If user pressed ENTER or didn't click 2 points, break the loop
#         if len(points) < 2:
#             print("Selection finished.")
#             break
            
#         # Extract x-coordinates (Time)
#         t_start = points[0][0]
#         t_end = points[1][0]
        
#         # Ensure correct order
#         if t_start > t_end:
#             t_start, t_end = t_end, t_start
            
#         print(f"Removing spikes between {t_start:.2f}s and {t_end:.2f}s...")
        
#         # 3. Highlight the removed area visually (optional but helpful)
#         ax.axvspan(t_start, t_end, color='red', alpha=0.2)
        
#         # 4. Remove Spikes in that range
#         spike_times = TimeV[SpikeIdx]
        
#         # Keep spikes that are NOT in the range
#         valid_mask = (spike_times < t_start) | (spike_times > t_end)
#         SpikeIdx = SpikeIdx[valid_mask]
        
#         # 5. Update the plot immediately
#         # We update the data in the scatter plot so you see them disappear
#         new_spike_times = TimeV[SpikeIdx]
#         new_spike_y = TraceV[SpikeIdx]
        
#         # Set new offsets (x,y positions) for the scatter plot
#         scat.set_offsets(np.c_[new_spike_times, new_spike_y])
        
#         # Refresh the canvas
#         fig.canvas.draw()

#     plt.close(fig)
#     return TraceV, TraceC, SpikeIdx


def remove_Frame_Multi(TraceV, TraceC, SpikeIdx, TimeV, TimeC, mot):
    """
    Interactive artifact removal for Voltage (500Hz) & Calcium (30Hz)

    Controls:
    - Zoom / Pan: normal toolbar
    - Hold 'd' + click start/end → delete
    - Ctrl+Z → undo
    - ENTER or close window → finish
    """

    import numpy as np
    import matplotlib.pyplot as plt

    TraceV = np.asarray(TraceV)
    TraceC = np.asarray(TraceC)
    SpikeIdx = np.asarray(SpikeIdx, dtype=int)
    TimeV = np.asarray(TimeV)
    TimeC = np.asarray(TimeC)
    mot = np.asarray(mot[:len(TraceV)])

    # -------------------------------
    # Masks & state
    # -------------------------------
    mask_V = np.ones(len(TraceV), dtype=bool)
    mask_C = np.ones(len(TraceC), dtype=bool)

    spike_bool = np.zeros(len(TraceV), dtype=bool)
    spike_bool[SpikeIdx] = True

    deletion_stack = []
    click_pts = []
    delete_mode = False
    done = False

    # -------------------------------
    # Plot
    # -------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(TimeV, TraceV, color='red', lw=0.5, label='Voltage')
    ax.plot(TimeC, TraceC, color='blue', lw=0.5, label='calcium')
    ax.scatter(TimeV[SpikeIdx], TraceV[SpikeIdx],
               color='black', s=10, zorder=5, label='Spikes')
    ax.set_title("Zoom freely | Hold 'd' delete | Ctrl+Z undo | ENTER finish")
    ax.set_xlabel("Time (s)")
    ax.legend()

    # -------------------------------
    # Helpers
    # -------------------------------
    def redraw():
        ax.clear()
        ax.plot(TimeV[mask_V], TraceV[mask_V], color='red', lw=0.5)
        kept_spikes = SpikeIdx[mask_V[SpikeIdx]]
        ax.scatter(TimeV[kept_spikes], TraceV[kept_spikes],
                   color='black', s=10)
        ax.set_xlabel("Time (s)")
        fig.canvas.draw_idle()

    def apply_deletion(t1, t2):
        nonlocal mask_V, mask_C
        bad_V = (TimeV >= t1) & (TimeV <= t2)
        bad_C = (TimeC >= t1) & (TimeC <= t2)

        deletion_stack.append((bad_V.copy(), bad_C.copy()))
        mask_V[bad_V] = False
        mask_C[bad_C] = False

        ax.axvspan(t1, t2, color='red', alpha=0.3)
        fig.canvas.draw_idle()

    def undo_last():
        if not deletion_stack:
            print("Nothing to undo")
            return
        bad_V, bad_C = deletion_stack.pop()
        mask_V[bad_V] = True
        mask_C[bad_C] = True
        redraw()

    # -------------------------------
    # Callbacks
    # -------------------------------
    def on_key_press(event):
        nonlocal delete_mode, done
        if event.key == 'd':
            delete_mode = True
        elif event.key == 'ctrl+z':
            undo_last()
        elif event.key in ('enter', 'return'):
            done = True

    def on_key_release(event):
        nonlocal delete_mode
        if event.key == 'd':
            delete_mode = False

    def on_click(event):
        if not delete_mode or event.inaxes != ax:
            return

        click_pts.append(event.xdata)
        if len(click_pts) == 2:
            t1, t2 = sorted(click_pts)
            click_pts.clear()
            apply_deletion(t1, t2)

    def on_close(event):
        nonlocal done
        done = True

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('close_event', on_close)

    # -------------------------------
    # NON-BLOCKING LOOP (CRITICAL)
    # -------------------------------
    plt.show(block=False)
    while not done:
        plt.pause(0.05)

    plt.close(fig)

    # -------------------------------
    # Apply cuts
    # -------------------------------
    New_TraceV = TraceV[mask_V]
    New_TimeV = TimeV[mask_V]
    New_Mot = mot[mask_V]

    New_TraceC = TraceC[mask_C]
    New_TimeC = TimeC[mask_C]

    New_SpikeIdx = np.flatnonzero(spike_bool[mask_V])

    return New_TraceV,New_TraceC,New_SpikeIdx,New_TimeV,New_TimeC,New_Mot, mask_V, mask_C,
    

    

    

    

def windowed_fr_calcium_correlation(
    fr,
    calcium,
    t=None,
    fs=None,
    ws_s=5.0,
    step_s=0.1,
    sigma_s=0.0,
    sigma_fr_s=None,
    sigma_ca_s=None,
):
    """Calculate Pearson correlation (FR vs calcium) in sliding time windows.

    Returns
    -------
    corr_list : list[float]
    win_t : np.ndarray (n_windows, 2) with [start_s, end_s]
    mid_t : list[float]
    ws_s : float
    sigma_used : (sigma_fr_s, sigma_ca_s) in seconds
    """
    fr = np.asarray(fr, dtype=float).ravel()
    calcium = np.asarray(calcium, dtype=float).ravel()
    if fr.shape[0] != calcium.shape[0]:
        raise ValueError(f"fr and calcium must have same length (got {fr.shape[0]} vs {calcium.shape[0]})")

    if t is None:
        if fs is None:
            raise ValueError("Provide `t` (seconds) or `fs` (Hz).")
        fs = float(fs)
        if not (fs > 0):
            raise ValueError(f"fs must be > 0 (got {fs})")
        dt = 1.0 / fs
        t0 = 0.0
    else:
        t = np.asarray(t, dtype=float).ravel()
        if t.shape[0] != fr.shape[0]:
            raise ValueError(f"t must have same length as signals (got {t.shape[0]} vs {fr.shape[0]})")
        if t.size < 2:
            sigma_fr = float(sigma_s if sigma_fr_s is None else sigma_fr_s)
            sigma_ca = float(sigma_s if sigma_ca_s is None else sigma_ca_s)
            return [], np.zeros((0, 2), dtype=float), [], float(ws_s), (sigma_fr, sigma_ca)
        dt = float(np.median(np.diff(t)))
        if not (dt > 0):
            raise ValueError("t must be strictly increasing in seconds.")
        t0 = float(t[0])

    ws_s = float(ws_s)
    step_s = float(step_s)
    if not (ws_s > 0):
        raise ValueError(f"ws_s must be > 0 (got {ws_s})")
    if not (step_s > 0):
        raise ValueError(f"step_s must be > 0 (got {step_s})")

    sigma_fr_s = float(sigma_s if sigma_fr_s is None else sigma_fr_s)
    sigma_ca_s = float(sigma_s if sigma_ca_s is None else sigma_ca_s)

    def _smooth(x, sigma_seconds):
        if sigma_seconds <= 0:
            return x
        sigma_samples = float(sigma_seconds) / dt
        try:
            return filters.gaussian_filter1d(x, sigma=sigma_samples, mode="nearest")
        except Exception:
            radius = int(np.ceil(4 * sigma_samples))
            if radius < 1:
                return x
            tt = np.arange(-radius, radius + 1, dtype=float)
            k = np.exp(-(tt**2) / (2.0 * sigma_samples**2))
            k /= k.sum()
            return np.convolve(x, k, mode="same")

    fr = _smooth(fr, sigma_fr_s)
    calcium = _smooth(calcium, sigma_ca_s)

    ws_n = int(round(ws_s / dt))
    step_n = int(round(step_s / dt))
    if ws_n < 2:
        raise ValueError(f"Window too small for dt={dt:.6g}s (ws_s={ws_s} -> ws_n={ws_n} samples)")
    if step_n < 1:
        step_n = 1

    last_start = fr.size - ws_n
    if last_start < 0:
        return [], np.zeros((0, 2), dtype=float), [], float(ws_s), (sigma_fr_s, sigma_ca_s)

    corr_list = []
    win_t = []
    mid_t = []
    for start in range(0, last_start + 1, step_n):
        end = start + ws_n
        x = fr[start:end]
        y = calcium[start:end]

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            r = np.nan
        else:
            xf = x[finite]
            yf = y[finite]
            if np.std(xf) == 0 or np.std(yf) == 0:
                r = np.nan
            else:
                r = float(np.corrcoef(xf, yf)[0, 1])

        start_t = float(t0 + start * dt)
        end_t = float(start_t + ws_n * dt)
        win_t.append((start_t, end_t))
        mid_t.append((start_t + end_t) / 2.0)
        corr_list.append(r)

    return corr_list, np.asarray(win_t, dtype=float), mid_t, float(ws_s), (sigma_fr_s, sigma_ca_s)
