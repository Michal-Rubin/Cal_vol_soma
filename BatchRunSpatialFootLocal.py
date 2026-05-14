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
from caiman.base.movies import movie
from skimage.draw import polygon

def save_volpy_plots(home_dir, cellNum, MeanImg, estimates,):
    """
    volpy introduce some summary plots.
    we usally don't ise them but we save them anyway by this function
    """
    #mean_img_path = os.path.join(home_dir, consts.PIPELINE_DIR, consts.MC_DIR, consts.MEAN_IMAGE + '.npy')
    #mean_img = np.load(mean_img_path, allow_pickle=True)

     # Data extraction
    t = estimates['t']
    t_sub = estimates['t_sub']
    t_rec = estimates['t_rec']
    spikes = estimates['spikes']
    weight = estimates['weights']

    frame_times = np.arange(len(t))
    vmax = np.percentile(MeanImg, 99)

    # Spike y-value markers
    spike_vals = 1.05 * np.max(t) * np.ones_like(spikes)

    # Prepare overlay image
    overlay = weight.copy()
    overlay[overlay == 0] = np.nan

    # Create plotly subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "image"}, {"type": "image"}],
               [{"colspan": 2}, None]],
        subplot_titles=(f"Spatial Component {cellNum}", "ROI on Mean Image", "Temporal Traces"),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    # Plot 1: Spatial component
    fig.add_trace(go.Heatmap(
        z=weight,
        colorscale='gray',
        zmin=0,
        zmax=np.max(weight) * 0.5,
        showscale=False
    ), row=1, col=1)

    # Plot 2: Mean image + ROI overlay
    fig.add_trace(go.Heatmap(
        z=MeanImg,
        colorscale='gray',
        zmax=vmax,
        showscale=False
    ), row=1, col=2)

    fig.add_trace(go.Heatmap(
        z=overlay,
        colorscale='Hot',
        opacity=0.4,
        showscale=False
    ), row=1, col=2)

    # Plot 3: Traces
    fig.add_trace(go.Scatter(
        x=frame_times, y=t,
        name='t', line=dict(color='blue')
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=frame_times, y=t_sub,
        name='t_sub', line=dict(color='green')
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=frame_times, y=t_rec,
        name='t_rec', line=dict(color='red', dash='dot')
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=frame_times[spikes],
        y=spike_vals,
        mode='markers',
        marker=dict(color='red', symbol='circle-open'),
        name='Spikes'
    ), row=2, col=1)

    fig.update_layout(
        height=800, width=1000,
        title_text=f"VolPy ROI Summary - Cell {cellNum}",
        showlegend=True
    )

    # Save paths
    html_path = os.path.join(home_dir, f"SpatialFootPrint{cellNum}.html")
    png_path = os.path.join(home_dir, f"SpatialFootPrint{cellNum}.png")

    # Save HTML
    pio.write_html(fig, file=html_path, auto_open=False)

    # Save PNG (requires kaleido)
   
    pio.write_image(fig, file=png_path, width=1000, height=800, scale=2)
    pickle_path = os.path.join(home_dir, f"{cellNum}_FigureObject.fig.pickle")
    with open(pickle_path, 'wb') as f:
        pickle.dump(fig, f)
    
    return

def volS_wrapper(args):
    # Because ProcessPoolExecutor requires a single argument function
    numC, roiM, images = args
    return volS(numC, roiM, images)

def volS(numC, roiM, images,path, lowFr_filt, Hp_Filt, param_meth,context_s = 10,censor_s = 6):
    fr = 500
    cell_n = numC
    bw = roiM #binary roi mask   
    #weights_init = pars[4]    
    #args = pars[5]
    context_size = context_s
    censor_size = censor_s
    winLen = int(fr * 0.008) # window length for temporal templates
    output = {}
    output['rawROI'] = {}
    print(f'Now processing cell number {cell_n}')

    # load the movie in C-order mermory mapping file
    # 9.6.2022 - Yaniv add this try and exept for handling volpy input in tif format (the format that the Neural Network denoiser is saving)
    #try:
        #Yr, dims, T = cm.load_memmap(fnames)
        #if bw.shape == dims:
         #   images = np.reshape(Yr.T, [T] + list(dims), order='F')
        #else:
          #  raise Exception('Dimensions of movie and ROIs do not accord')
    #except:
        #import tifffile
        #images = tifffile.imread(fnames)
       # T = images.shape[0]
      #  dims = images.shape[1:]
        
    # extract the context region from the entire movie
    # Expand ROI to get context
    bwexp = binary_dilation(bw, structure=np.ones((context_size, context_size)))
    Xinds = np.where(np.any(bwexp > 0, axis=1))[0]
    Yinds = np.where(np.any(bwexp > 0, axis=0))[0]

    # Crop to region of interest + context
    bw = bw[Xinds[0]:Xinds[-1]+1, Yinds[0]:Yinds[-1]+1]
    # Create mask for background (excluding ROI + censor_size border)
    notbw = 1 - binary_dilation(bw, structure=disk(censor_size))
    data = images[:, Xinds[0]:Xinds[-1]+1, Yinds[0]:Yinds[-1]+1]
   
    data = np.array(images[:, Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1])
    bw = (bw > 0)
    notbw = (notbw > 0)
    ref = np.median(data[:500, :, :], axis=0)
    bwexp[Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1] = True

    # remove the photobleaching effect by high-pass filtering the signal
    output['mean_im'] = np.mean(data, axis=0)
    data = np.reshape(data, (data.shape[0], -1))
    data = data - np.mean(data, 0)
    data = data - np.mean(data, 0)   #do again because of numeric issues
    dataraw=data
    data_hp = signal_filter(data.T,lowFr_filt, fr).T  
    data_lp = data - data_hp
    fig = go.Figure()
    t0 = np.nanmean(data_hp[:, bw.ravel()], 1)
    t0 = t0 - np.mean(t0)
    # remove any variance in trace that can be predicted from the background principal components
    data_svd = data_hp[:, notbw.ravel()]
    if data_svd.shape[1] < 8 + 1:
        raise Exception(f'Too few pixels ({data_svd.shape[1]}) for background extraction (at least {8} needed);'
                        f'please decrease context_size and censor_size')
    Ub, Sb, Vb = svds(data_svd, 8)
    alpha = 8* 0.01    # square of F-norm of Ub is equal to number of principal components
    reg = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr').fit(Ub, t0)
    t0 = np.double(t0 - np.matmul(Ub, reg.coef_))
     # spike detection for the initial trace
    ts, spikes, t_rec, templates, low_spikes, thresh,pks = denoise_spikes(t0, 
                                          winLen , path,cell_n, fr , hp_freq=Hp_Filt, clip=0,
                                          threshold_method=param_meth, 
                                          min_spikes=2, pnorm=0.5, threshold=3, 
                                           do_plot=False, distance=3)
    output['rawROI']['t'] = t0.copy()
    output['rawROI']['ts'] = ts.copy()
    output['rawROI']['spikes'] = spikes.copy()
    output['rawROI']['weights'] = bw.copy()
    output['rawROI']['t'] = output['rawROI']['t'] * np.mean(t0[output['rawROI']['spikes']]) / np.mean(
        output['rawROI']['t'][output['rawROI']['spikes']])  # correct shrinkage
    output['rawROI']['templates'] = templates
    num_spikes = [spikes.shape[0]]
    # prebuild the regression matrix generate a predictor for ridge regression
    pred = np.empty_like(data_hp)
    pred[:] = data_hp
    pred = np.hstack((np.ones((data_hp.shape[0], 1), dtype=np.single), np.reshape
    (movie.gaussian_blur_2D(np.reshape(pred,
                                       (data_hp.shape[0], ref.shape[0], ref.shape[1])),
                            kernel_size_x=7, kernel_size_y=7, kernel_std_x=1.5,
                            kernel_std_y=1.5, borderType=cv2.BORDER_REPLICATE), data_hp.shape)))

    # cross-validation of regularized regression parameters
    lambdamax = np.single(np.linalg.norm(pred[:, 1:], ord='fro') ** 2)
    lambdas = lambdamax * np.logspace(-4, -2, 3)
    s_max = 1
    l_max = 2
    sigma = np.array([1, 1.5, 2])[s_max]
    recon = np.empty_like(data_hp)
    recon[:] = data_hp
    recon = np.hstack((np.ones((data_hp.shape[0], 1), dtype=np.single), np.reshape
    (movie.gaussian_blur_2D(np.reshape(recon,
                                       (data_hp.shape[0], ref.shape[0], ref.shape[1])),
                            kernel_size_x=int(2 * np.ceil(2 * sigma) + 1),
                            kernel_size_y=int(2 * np.ceil(2 * sigma) + 1),
                            kernel_std_x=sigma, kernel_std_y=sigma,
                            borderType=cv2.BORDER_REPLICATE), data_hp.shape)))
    for iteration in range(4):
        if iteration == 4 - 1:
            do_plot = True
        else:
            do_plot = False
            
        # update weights
        tr = np.single(t_rec.copy())
        
        
        Ri = Ridge(alpha=lambdas[l_max], fit_intercept=True, solver='lsqr')
        Ri.fit(recon, tr)
        weights = Ri.coef_
        weights[0] = Ri.intercept_

        # update the signal            
        t = np.matmul(recon, weights)
        t = t - np.mean(t)

        # ridge regression to remove background components
        b = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr').fit(Ub, t).coef_
        t = t - np.matmul(Ub, b)

        # correct shrinkage
        weights = weights * np.mean(t0[spikes]) / np.mean(t[spikes])
        t = np.double(t * np.mean(t0[spikes]) / np.mean(t[spikes]))
      
        # estimate spike times
        
        ts, spikes, t_rec, templates, low_spikes, thresh,pksjj = denoise_spikes(t, 
                                          winLen, path,cell_n,fr, hp_freq=1, clip=0,
                                          threshold_method=param_meth, 
                                          min_spikes=2, pnorm=0.5, threshold=3, 
                                           do_plot=do_plot, distance=3)
    
    
        num_spikes.append(spikes.shape[0])

    # compute SNR 
    if len(spikes)>0:
        t = t - np.median(t)
        selectSpikes = np.zeros(t.shape)
        selectSpikes[spikes] = 1
        sgn = np.mean(t[selectSpikes > 0])
        ff1 = -t * (t < 0)
        Ns = np.sum(ff1 > 0)
        noise = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
        snr = sgn / noise
    else:
        snr = 0

    # locality test       
    matrix = np.matmul(np.transpose(pred[:, 1:]), t_rec)
    sigmax = np.sqrt(np.sum(np.multiply(pred[:, 1:], pred[:, 1:]), axis=0))
    sigmay = np.sqrt(np.dot(t_rec, t_rec))
    IMcorr = matrix / sigmax / sigmay
    maxCorrInROI = np.max(IMcorr[bw.ravel()])
    if np.any(IMcorr[notbw.ravel()] > maxCorrInROI):
        locality = False
    else:
        locality = True

    
    # weights in the FOV
    weights = np.reshape(weights[1:],ref.shape, order='C')
    weights_FOV = np.zeros(images.shape[1:])
    weights_FOV[Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1] = weights
    
    # subthreshold activity extraction    
    t_sub = t.copy() - t_rec
    subFreq = 50
    t_sub = signal_filter(t_sub, subFreq, fr, order=5, mode='low') 

    # output
    output['cell_n'] = cell_n
    output['t'] = t
    output['traw']=np.dot(dataraw,np.reshape(weights,(-1,1)))
    output['ts'] = ts
    output['t_rec'] = t_rec        
    output['t_sub'] = t_sub
    output['spikes'] = spikes
    output['low_spikes'] = low_spikes
    output['num_spikes'] = num_spikes
    output['templates'] = templates
    output['snr'] = snr
    output['thresh'] = thresh
    output['weights'] = weights_FOV
    output['locality'] = locality    
    output['context_coord'] = np.transpose(np.vstack((Xinds[[0, -1]], Yinds[[0, -1]])))
    output['F0'] = np.abs(np.nanmean(data_lp[:, bw.flatten()] + output['mean_im'][bw][np.newaxis, :], 1))
    output['dFF'] = t / output['F0']
    output['rawROI']['dFF'] = output['rawROI']['t'] / output['F0']
    
    return output,pksjj
#fig.add_trace(go.Scatter(y=data,  mode='lines'))
   # fig.add_trace(go.Scatter(x = spikes,y=data[spikes],    mode='markers',
    #marker=dict(color='red', size=6),
    #name='Spikes'))
    #fig.add_trace(go.Scatter(y=data_hp,  mode='lines'))
    #pio.show(fig) 
    
    
    #return(data,pks)
def denoise_spikes(data, window_length, homePath,cell_number,fr=500,  hp_freq=1,  clip=100, threshold_method='adaptive_threshold', 
                   min_spikes=10, pnorm=0.5, threshold=3,  do_plot=True, distance = 3):
    """ Function for finding spikes and the temporal filter given one dimensional signals.
        Use function whitened_matched_filter to denoise spikes. Two thresholding methods can be 
        chosen, simple or 'adaptive thresholding'.

    Args:
        data: 1-d array
            one dimensional signal

        window_length: int
            length of window size for temporal filter

        fr: int
            number of samples per second in the video
            
        hp_freq: float
            high-pass cutoff frequency to filter the signal after computing the trace
            
        clip: int
            maximum number of spikes for producing templates

        threshold_method: str
            adaptive_threshold or simple method for thresholding signals
            adaptive_threshold method threshold based on estimated peak distribution
            simple method threshold based on estimated noise level 
            
        min_spikes: int
            minimal number of spikes to be detected
            
        pnorm: float
            a variable deciding the amount of spikes chosen for adaptive threshold method

        threshold: float
            threshold for spike detection in simple threshold method 
            The real threshold is the value multiply estimated noise level

        do_plot: boolean
            if Ture, will plot trace of signals and spiketimes, peak triggered
            average, histogram of heights
            
    Returns:
        datafilt: 1-d array
            signals after whitened matched filter

        spikes: 1-d array
            record of time of spikes

        t_rec: 1-d array
            recovery of original signals

        templates: 1-d array
            temporal filter which is the peak triggered average

        low_spikes: boolean
            True if number of spikes is smaller than 30
            
        thresh2: float
            real threshold in second round of spike detection 
    """
    # high-pass filter the signal for spike detection
    data = signal_filter(data, hp_freq, fr, order=5)
    data = data - np.median(data)
    pks = data[signal.find_peaks(data, height=None)[0]]

    # first round of spike detection    

    thresh, _, _, low_spikes = adaptive_thresh(pks, clip, 0.5, min_spikes)
    locs = signal.find_peaks(data, height=thresh, distance=distance)[0] 
   

    # spike template
    window_length = int(window_length)
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    locs = locs[np.logical_and(locs > (-window[0]), locs < (len(data) - window[-1]))]
    PTD = data[(locs[:, np.newaxis] + window)]
    PTA = np.median(PTD, 0)
    PTA = PTA - np.min(PTA)
    templates = PTA

    # whitened matched filtering based on spike times detected in the first round of spike detection
    datafilt = whitened_matched_filter(data, locs, window)    
    datafilt = datafilt - np.median(datafilt)

    # second round of spike detection on the whitened matched filtered trace
    pks2 = datafilt[signal.find_peaks(datafilt, height=None)[0]]
  
    thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip=0, pnorm=pnorm, min_spikes=min_spikes)  # clip=0 means no clipping
    spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    
    
    # compute reconstructed signals and adjust shrinkage
    t_rec = np.zeros(datafilt.shape)
    t_rec[spikes] = 1
    t_rec = np.convolve(t_rec, PTA, 'same')   
    factor = np.mean(data[spikes]) / np.mean(datafilt[spikes])
    datafilt = datafilt * factor
    thresh2_normalized = thresh2 * factor
        
    if do_plot:
      
        # Create an output directory or file path (replace this with your own)
        html_path = os.path.join(homePath, f"histDenoiseSpike{cell_number}.html")  # Example
        plot_list = []

        # Histogram - raw data
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=pks, nbinsx=500, name="Raw peaks"))
        fig1.add_vline(x=thresh, line_color='red', name='Threshold')
        fig1.update_layout(title='Raw data histogram')
        plot_list.append(fig1)

        # Histogram - after matched filter
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=pks2, nbinsx=500, name="Filtered peaks"))
        fig2.add_vline(x=thresh2, line_color='red', name='Threshold')
        fig2.update_layout(title='After matched filter histogram')
        plot_list.append(fig2)

        # Peak-triggered average
        fig3 = go.Figure()
        for i in range(PTD.shape[1]):
            fig3.add_trace(go.Scatter(y=PTD[:, i], line=dict(color='gray'), name=f'Trace {i}', showlegend=False))
        fig3.add_trace(go.Scatter(y=PTA, line=dict(color='black', width=2), name='Average'))
        fig3.update_layout(title='Peak-triggered average')
        plot_list.append(fig3)

        # Raw + spike markers
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=data, name='Raw data', mode='lines'))
        fig4.add_trace(go.Scatter(x=locs, y=np.max(datafilt) * 1.1 * np.ones_like(locs),
                                mode='markers', marker=dict(color='red', symbol='circle-open'),
                                name='Detected peaks'))
        fig4.add_trace(go.Scatter(x=spikes, y=np.max(datafilt) * 1 * np.ones_like(spikes),
                                mode='markers', marker=dict(color='green', symbol='circle-open'),
                                name='Spikes'))
        fig4.update_layout(title='Raw data with peak and spike markers')
        plot_list.append(fig4)

        # Filtered + spike markers
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(y=datafilt, name='Filtered', mode='lines'))
        fig5.add_trace(go.Scatter(x=locs, y=np.max(datafilt) * 1.1 * np.ones_like(locs),
                                mode='markers', marker=dict(color='red', symbol='circle-open'),
                                name='Detected peaks'))
        fig5.add_trace(go.Scatter(x=spikes, y=np.max(datafilt) * 1 * np.ones_like(spikes),
                                mode='markers', marker=dict(color='green', symbol='circle-open'),
                                name='Spikes'))
        fig5.update_layout(title='Filtered data with peak and spike markers')
        plot_list.append(fig5)

        # Save all figures into one HTML file
        with open(html_path, 'w') as f:
            for fig in plot_list:
                inner_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
                f.write(inner_html + "\n")

    return datafilt, spikes, t_rec, templates, low_spikes, thresh2_normalized,pks
def whitened_matched_filter(data, locs, window):
    """
    Function for using whitened matched filter to the original signal for better
    SNR. Use welch method to approximate the spectral density of the signal.
    Rescale the signal in frequency domain. After scaling, convolve the signal with
    peak-triggered-average to make spikes more prominent.
    
    Args:
        data: 1-d array
            input signal

        locs: 1-d array
            spike times

        window: 1-d array
            window with size of temporal filter

    Returns:
        datafilt: 1-d array
            signal processed after whitened matched filter
    
    """
    N = np.ceil(np.log2(len(data)))
    censor = np.zeros(len(data))
    censor[locs] = 1
    censor = np.int16(np.convolve(censor.flatten(), np.ones([1, len(window)]).flatten(), 'same'))
    censor = (censor < 0.5)
    noise = data[censor]

    _, pxx = signal.welch(noise, fs=2 * np.pi, window=signal.get_window('hamming', 1000), nfft=2 ** N, detrend=False,
                          nperseg=1000)
    Nf2 = np.concatenate([pxx, np.flipud(pxx[1:-1])])
    scaling_vector = 1 / np.sqrt(Nf2)

    cc = np.pad(data.copy(),(0,int(2**N-len(data))),'constant')    
    dd = (cv2.dft(cc,flags=cv2.DFT_SCALE+cv2.DFT_COMPLEX_OUTPUT)[:,0,:]*scaling_vector[:,np.newaxis])[:,np.newaxis,:]
    dataScaled = cv2.idft(dd)[:,0,0]
    PTDscaled = dataScaled[(locs[:, np.newaxis] + window)]
    PTAscaled = np.mean(PTDscaled, 0)
    datafilt = np.convolve(dataScaled, np.flipud(PTAscaled), 'same')
    datafilt = datafilt[:len(data)]
    return datafilt

def adaptive_thresh(pks, clip, pnorm=0.5, min_spikes=10):
    """ Adaptive threshold method for deciding threshold given heights of all peaks.

    Args:
        pks: 1-d array
            height of all peaks

        clip: int
            maximum number of spikes for producing templates

        pnorm: float, between 0 and 1, default is 0.5
            a variable deciding the amount of spikes chosen for adaptive threshold method
            
        min_spikes: int
            minimal number of spikes to be detected

    Returns:
        thresh: float
            threshold for choosing spikes

        falsePosRate: float
            possibility of misclassify noise as real spikes

        detectionRate: float
            possibility of real spikes being detected

        low_spikes: boolean
            true if number of spikes is smaller than minimal value
    """
    
    # find median of the kernel density estimation of peak heights
    spread = np.array([pks.min(), pks.max()])
    spread = spread + np.diff(spread) * np.array([-0.05, 0.05])
    low_spikes = False
    pts = np.linspace(spread[0], spread[1], 2001)
    kde = stats.gaussian_kde(pks)
    f = kde(pts)    
    xi = pts
    center = np.where(xi > np.median(pks))[0][0]

    fmodel = np.concatenate([f[0:center + 1], np.flipud(f[0:center])])
    if len(fmodel) < len(f):
        fmodel = np.append(fmodel, np.ones(len(f) - len(fmodel)) * min(fmodel))
    else:
        fmodel = fmodel[0:len(f)]

    # adjust the model so it doesn't exceed the data:
    csf = np.cumsum(f) / np.sum(f)
    csmodel = np.cumsum(fmodel) / np.max([np.sum(f), np.sum(fmodel)])
    lastpt = np.where(np.logical_and(csf[0:-1] > csmodel[0:-1] + np.spacing(1), csf[1:] < csmodel[1:]))[0]
    if not lastpt.size:
        lastpt = center
    else:
        lastpt = lastpt[0]
    fmodel[0:lastpt + 1] = f[0:lastpt + 1]
    fmodel[lastpt:] = np.minimum(fmodel[lastpt:], f[lastpt:])

    # find threshold
    csf = np.cumsum(f)
    csmodel = np.cumsum(fmodel)
    csf2 = csf[-1] - csf
    csmodel2 = csmodel[-1] - csmodel
    obj = csf2 ** pnorm - csmodel2 ** pnorm
    maxind = np.argmax(obj)
    thresh = xi[maxind]

    if np.sum(pks > thresh) < min_spikes:
        low_spikes = True
        logging.warning(f'Few spikes were detected. Adjusting threshold to take {min_spikes} largest spikes')
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
    elif ((np.sum(pks > thresh) > clip) & (clip > 0)):
        logging.warning(f'Selecting top {clip} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))

    ix = np.argmin(np.abs(xi - thresh))
    falsePosRate = csmodel2[ix] / csf2[ix]
    detectionRate = (csf2[ix] - csmodel2[ix]) / np.max(csf2 - csmodel2)
    return thresh, falsePosRate, detectionRate, low_spikes
def plot_roi_masks(rois):
    """
    Plots all ROI masks overlaid on a blank image.
    """
    n_rois, h, w = rois.shape
    canvas = np.zeros((h, w))
    
    for i, mask in enumerate(rois):
        canvas += mask * (i+1)  # Add intensity for each ROI

    plt.figure(figsize=(6, 6))
    #plt.imshow(canvas, cmap='nipy_spectral')
    plt.title(f'{n_rois} ROI Masks')
    plt.colorbar(label='ROI Index')
    plt.axis('off')
   # plt.show()

def _normalize_data_root_path(path):
    """
    Normalize known path typo used in some scripts:
    ...\\DataMichal_Rubin\\...  ->  ...\\Data\\Michal_Rubin\\...
    """
    norm_path = os.path.normpath(path)
    return norm_path.replace("DataMichal_Rubin", os.path.join("Data", "Michal_Rubin"))


def get_experiment_xml_path(raw_path):
    """
    return the path of experiment.xml file -
    a file that attached to each video that is taken with ThorImage in AdamLab.
    the function assumes that the path is in the same directory of the raw video
    """
    raw_path = _normalize_data_root_path(raw_path)
    for name in ("Experiment.xml", "experiment.xml"):
        candidate = os.path.join(raw_path, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Experiment XML not found under: {raw_path}")

def get_raw_video_dimensions(raw_path):
    """
    extract video's width and height fro, experiment.xml file -
    a file that attached to each video that is taken with ThorImage in AdamLab
    """
    xml = get_experiment_xml_path(raw_path)
    tree = ET.parse(xml)
    root = tree.getroot()
    width = int(root[5].attrib['width'])
    height = int(root[5].attrib['height'])
    # get number of frames in the video
    return width, height
def get_calROImask(cal_path, orig_shape=(1152,1152), img_shape=(512,512)):
    # Load XML
    cal_path = _normalize_data_root_path(cal_path)
    xml_path = os.path.join(cal_path,'cal','ROIs.xaml')
    if not os.path.exists(xml_path):
        alt_xml_path = os.path.join(cal_path, 'cal', 'ROIs.xml')
        if os.path.exists(alt_xml_path):
            xml_path = alt_xml_path
        else:
            raise FileNotFoundError(f"ROI file not found. Tried: {xml_path}, {alt_xml_path}")
    xml_dict = xmltodict.parse(open(xml_path).read())
    roi_array = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIPoly"]
    
    # Make sure we have a list
    if not isinstance(roi_array, list):
        roi_array = [roi_array]

    masks = []
    scale_y = img_shape[0] / orig_shape[0]
    scale_x = img_shape[1] / orig_shape[1]

    for roi in roi_array:
        pts_str = roi["@Points"]
        pts = pts_str.strip().split(" ")
        xs = []
        ys = []
        for p in pts:
            x, y = [float(v) for v in p.split(",")]
            xs.append(x * scale_x)
            ys.append(y * scale_y)
        rr, cc = polygon(ys, xs, shape=img_shape)
        mask = np.zeros(img_shape, dtype=bool)
        mask[rr, cc] = True
        masks.append(mask)

    return np.stack(masks)
def get_rois_mask(raw_video_path):
    """
    for given raw video path, looking for the "ROIs.xaml" file from ThorImage
    and generate binary mask as np array in the sahpe of (#ROIs, height, width)
    """
    raw_video_path = _normalize_data_root_path(raw_video_path)
    xml_path = None
    for name in ("ROIs.xaml", "ROIs.xml", "rois.xaml", "rois.xml"):
        candidate = os.path.join(raw_video_path, name)
        if os.path.exists(candidate):
            xml_path = candidate
            break

    if xml_path is None:
        raise FileNotFoundError(
            f"ROI file not found under: {raw_video_path}. "
            "Expected one of: ROIs.xaml / ROIs.xml"
        )

    xml_data = open(xml_path, "r").read()
    xml_dict = xmltodict.parse(xml_data)
    polygons_struct = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIPoly"]
    # extract polygons of ROIS.
    poly_lst = []
    for i in range(len(polygons_struct)):
        p = polygons_struct[i]['@Points']
        if p not in poly_lst:
            poly_lst.append(p)
    print("Number of ROIs found:", len( poly_lst))
   # if not  poly_lst:
     #   raise RuntimeError(f"No ROIs found for path: {raw_video_path}")
    # extract the coordinates of the rectangle ROI
    rect_data = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIRect"]
    bottom_left_x, bottom_left_y = [float(i) for i in rect_data["@BottomLeft"].split(',')]
    top_left_x, top_left_y = [float(i) for i in rect_data["@TopLeft"].split(',')]
    height = float(rect_data["@ROIHeight"])
    width = float(rect_data["@ROIWidth"])
    # generate list of polygons w.r.t the rectangle ROI
    corrected_polygons = []
    for polygon in poly_lst: # for each polygon
        corrected_points = []
        points = polygon.split(' ')
        for point in points:
            x, y = [float(i) for i in point.split(',')]
            # if the point exceeds the rectangle from above, left or right - trunc it
            x = min(max(x - bottom_left_x, 1),width) 
            y = max(1,min(y - top_left_y, height))
            corrected_points.append((x, y))
        corrected_points.append(corrected_points[0])
        corrected_polygons.append(corrected_points)
    # generate masks
    width, height = get_raw_video_dimensions(raw_video_path)
    ROIs = []
    for poly in corrected_polygons:
        flipped_poly = [(j,i) for i,j in poly]
        polygon = flipped_poly
        poly_path = Path(polygon)
        x, y = np.mgrid[:height, :width]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)
        mask = poly_path.contains_points(coors)
        mask = mask.reshape(height, width)
        if mask.sum() > 0: # fot the case that a point was signed in the slm
            ROIs.append(mask)
    ROIs = np.stack(ROIs)
    return ROIs
def trace_extraction(video, rois_mask, weights=None):
    """
    video - 3d np array represent a video.
    rois - a binary np array in the shape of (#cells, width, height).
            its represent the pixels corresponding to each cell in the video.
    weights - represent spatial components to extract the traces accordingly.
            if not supplied - just preform non weighted mean over the cell
    """
    if weights is None:
        weights = rois_mask
    df_columns = ['cell ' + str(i+1) for i in range(len(rois_mask))]
    df = pd.DataFrame(columns=df_columns)
    fig = go.Figure()
    for roi_num in range(len(rois_mask)):
        Xinds = np.where(np.any(rois_mask[roi_num] > 0, axis=1) > 0)[0]
        Yinds = np.where(np.any(rois_mask[roi_num] > 0, axis=0) > 0)[0]
        croped_video = video[:, Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1]
        
        cell_mask = weights[roi_num]
        croped_mask = cell_mask[Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1]
        masked_video = croped_video * croped_mask[np.newaxis,:,:]
        trace = masked_video.mean(axis=(1, 2))
        fig.add_trace(go.Scatter(y=trace,  mode='lines'))
        df[df.columns[roi_num]] = trace
    #fig.show()
    return df
def signal_filter(sg, freq, fr, order=3, mode='high'):
    """
    Function for high/low passing the signal with butterworth filter
    
    Args:
        sg: 1-d array
            input signal
            
        freq: float
            cutoff frequency
        
        order: int
            order of the filter
        
        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering
            
    Returns:
        sg: 1-d array
            signal after filtering            
    """
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg.astype(np.float32), padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg

# if __name__  == "__main__":
#     Homepath =[
#          #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\17-06-2025\fov1',
               
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-07-2025\fov7',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-07-2025\fov9',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\28-07-2025-motor\fov1',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\28-07-2025-motor\fov7',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\28-07-2025-motor\fov9\2',
               
               
               
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\28-07-2025-motor\fov10',
#     #           r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\16-06-2025\fov5',
#     #           r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-08-2025-anst\fov7',
#     #           r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-08-2025-anst\fov7\2',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-08-2025-anst\fov9',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\01-07-2025\fov3',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-08-2025-anst\fov8',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\01-07-2025\fov1',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\01-07-2025\fov1\2',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-07-2025\fov2',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-07-2025\fov7',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-07-2025\fov8',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\15-07-2025\fov5',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\15-07-2025\fov9',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\15-07-2025\fov10',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\30-07-2025-motor\fov5',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\30-07-2025-motor\fov5\2',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\30-07-2025-motor\fov7\2',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\30-07-2025-motor\fov10\2',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\30-07-2025-motor\fov11',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-08-2025-anst\fov10',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-08-2025-anst\fov1\2',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-08-2025-anst\fov10',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\14-08-2025-anst\fov11\2',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Xb\14-07-2025\fov1'

#     #           #r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\RUGC40\R\18-08-2025-ans\fov5',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\22-10-2025-motor\fov8',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\22-10-2025-motor\fov9',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\22-10-2025-motor\fov10',
#     #            r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\17-09-2025-rugc42-wh-s1-ans\fov5',
#        r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\21-10-2025-MOTOR\fov6',
#         r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\22-10-2025-motor\fov8'
#     ]

#     for l in range(len(Homepath)):
        
#         cP =  Homepath[l]
#         rr = get_rois_mask(cP)
#         pathMC = os.path.join(cP,r'pipeline_results\motion_corrected\motion_corrected.tiff')
#         images = tiff.imread(pathMC)
#         imagesOg = tiff.imread(pathMC)
#         T = images.shape[0]
#         dims = images.shape[1:]
#         TOg = imagesOg.shape[0]
#         dimsOg = imagesOg.shape[1:]
#         movie_obj = images
#         meanImage = np.mean(images,0)
#         # load parameters
#         cellL = range(len(rr))
#         exT = []
#         sp = []
#         # Prepare arguments
#         #args_list = [(cellL[i], rr[i], images) for i in range(len(cellL))]

#         exT = []
#         sp = []
#         VpOutput = {}
#         for i in range(len(cellL)):
          
#             os.makedirs(currP, exist_ok=True)
#             mask = rr[i].astype(float)
#             mask[mask == 0] = np.nan

#         # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#         # ax[0].imshow(meanImage, cmap='gray')
#         # ax[0].imshow(mask, cmap='autumn', alpha=0.4)
#         # ax[0].set_title("Original ROI Overlay")
#             #img_path = os.path.join(Homepath, f'roi_overlay_{i}.png')
#             #fig.savefig(img_path)
#             spPath = os.path.join(currP,'SpikeIdx.csv')
#             Output,spikeIDX = volS(cellL[i],rr[i],images,cP,lowFr_filt= 0.5, Hp_Filt= 20)
#             df = pd.DataFrame(spikeIDX)  # create df with column name
#             df.to_csv(spPath, index=False)
#             save_volpy_plots(cP,cellL[i],meanImage,Output)
#             VpOutput[cellL[i]] = Output
#         #exT.append(tr)
#         #sp.append(cSp)
#         #print(cSp.min()) 
#         #print(cSp.max())
#         tracesese = trace_extraction(movie_obj,rr)
#         for i, roi in enumerate(rr):
#             if np.sum(roi) == 0:
#                 print(f"ROI {i} is empty!")

#     v = 55
