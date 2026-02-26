import plotly.io as pio

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
#import cv2  # Make sure OpenCV is installed: pip install opencv-python
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
import sys
from matplotlib.patches import Polygon, Rectangle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#from BatchRunSpatialFootLocal import get_rois_mask,volS,trace_extraction,save_volpy_plots, plot_roi_masks
from skimage.measure import find_contours
from skimage.draw import polygon
from scipy.ndimage import center_of_mass

plt.ion()   # interactive mode ON

def get_rois_mask(raw_video_path):
    """
    for given raw video path, looking for the "ROIs.xaml" file from ThorImage
    and generate binary mask as np array in the sahpe of (#ROIs, height, width)
    """
    xml_path = os.path.join(raw_video_path,'ROIs.xaml')

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
def get_experiment_xml_path(raw_path):
    """
    return the path of experiment.xml file -
    a file that attached to each video that is taken with ThorImage in AdamLab.
    the function assumes that the path is in the same directory of the raw video
    """
    return os.path.join(raw_path,'Experiment.xml')

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
def get_calROImask(xml_path, orig_shape=(1152,1152), img_shape=(512,512)):
    # Load XML
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

# ==== Load XML ====
homeP = r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC18\L\03-07-25\fov1\cal'
homeVol = r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC18\L\03-07-25\fov1'
vol_xml_path = os.path.join(homeP,r"ROIs.xaml")
vol_Mean_path = os.path.join(homeVol,r"Mean.tif")
xml_path = os.path.join(homeP,r"ROIs.xaml")
#tree = ET.parse(xml_path)
#root = tree.getroot()

# ==== Load Image and make Min Projection ====
img_path = os.path.join(homeP,r"Image_scan_1_region_0_0.tif")
stack = tiff.imread(img_path) 
vol_Mean = tiff.imread(vol_Mean_path) 
print(np.shape(stack))  # shape: (frames, y, x) or (z, y, x)
min_img = np.mean(stack, axis=0) # min projection across frames/z

# ==== Parse ROIs ====
rois = []
rr_vol = get_rois_mask(homeVol)
#plot_roi_masks(rr_vol)
rr = get_calROImask(xml_path)

# ==== Plot ====
plt.figure()
plt.imshow(min_img,  cmap="gray", vmin=np.percentile(min_img, 1), vmax=np.percentile(min_img, 99))

# Overlay each mask
for i, mask in enumerate(rr):
    # find mask contour
    contours = find_contours(mask, 0.5)
    for contour in contours:
        plt.plot(contour[:,1], contour[:,0], linewidth=1.5, label=f'ROI {i+1}')
    # add numbers at centroid of each ROI
    
    if mask.sum() > 0:  # skip empty masks
        cy, cx = center_of_mass(mask)   # (row, col)
        plt.text(cx, cy, str(i+1),
                color="red", fontsize=15, ha="center", va="center",
                weight="bold")

plt.title("ROIs over Min Projection")
plt.axis("off")

# Ensure the layout is tight
plt.tight_layout()
out_path = os.path.join(homeP,r"ROIs_overlay.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")   # high-quality PNG
plt.show(block=True)

plt.figure()
plt.imshow(vol_Mean,  cmap="gray", vmin=np.percentile(min_img, 1), vmax=np.percentile(min_img, 99))

# Overlay each mask
for i, mask in enumerate(rr_vol):
    # find mask contour
    contours = find_contours(mask, 0.5)
    for contour in contours:
        plt.plot(contour[:,1], contour[:,0], linewidth=1.5, label=f'ROI {i+1}')
    # add numbers at centroid of each ROI
    
    if mask.sum() > 0:  # skip empty masks
        cy, cx = center_of_mass(mask)   # (row, col)
        plt.text(cx, cy, str(i+1),
                color="red", fontsize=15, ha="center", va="center",
                weight="bold")

plt.title("ROIs over Min Projection")
plt.axis("off")

# Ensure the layout is tight
plt.tight_layout()
out_path = os.path.join(homeP,r"vol_ROIs_overlay.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")   # high-quality PNG
plt.show(block=True)