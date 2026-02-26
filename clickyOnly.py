import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from roipoly import MultiRoi
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import json
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.transform import downscale_local_mean


def kif_to_raw (path):
    class VectorizedMovie:
        def __init__(self, data):
            self.data = data  # Should be a NumPy array of uint16

        def savebin(self, fname=None):
            import time
            if fname is None:
                print("No file selected. Aborting save.")
                return
            
            fullname = os.path.abspath(fname)
            print(f"saving {fullname} ...")
            tic = time.time()
            
            with open(fullname, 'wb') as fid:
                fid.write(self.data.astype(np.uint16).tobytes())
            
            print(f"saving took {time.time() - tic:.2f} s")
        def read_frame_safe(file, width, height, frame_idx=None):
            try:
                file.read(16)
                comment_len_bytes = file.read(4)
                if len(comment_len_bytes) < 4:
                    print(f"⚠️ Frame {frame_idx}: couldn't read frame comment length.")
                    return None
                
                frame_comment_length = int(struct.unpack('I', comment_len_bytes)[0])
                file.read(12 + frame_comment_length)
                
                expected_bytes = 2 * width * height
                frame_data = file.read(expected_bytes)
                
                if len(frame_data) != expected_bytes:
                    print(f"⚠️ Frame {frame_idx}: expected {expected_bytes} bytes, got {len(frame_data)} bytes. Skipping.")
                    return None
                
                frame = np.frombuffer(frame_data, dtype=np.uint16).reshape((height, width))
                return frame

            except Exception as e:
                print(f"❌ Exception while reading frame {frame_idx}: {e}")
                return None
    filenames = os.path.join(path,'check.kif')
    initialdir = path
    data_files = []
    data = [path[:-4]]
    print(f"\nReading: {path}")
    
    
    with open(path, 'rb') as file:
        file.read(6)
        num_frames = int(struct.unpack("I", file.read(4))[0])
        width = int(struct.unpack("I", file.read(4))[0])
        height = int(struct.unpack("I", file.read(4))[0])
        bpp = int(struct.unpack('H', file.read(2))[0])
    
    print(f"#Frames: {num_frames}, Width: {width}, Height: {height}, BPP: {bpp}")
    data += [num_frames, width, height, bpp]
    data_files.append(data)

    with open(filenames[0], 'rb') as file:
        file.read(46)
        comment_len = int(struct.unpack('I', file.read(4))[0])
        file.read(46)
        header = str(file.read(comment_len))[2:]
        
    i = 0
    new_header = ''
    while i < len(header) - 1:
        s1, s2 = header[i], header[i+1]
        if s1 == '\\' and s2 == 'n':
            i += 2
        elif s1 == '\\':
            new_header += s2
            i += 2
        else:
            new_header += s1
            i += 1

    wjdata = json.loads(new_header)
    binning = wjdata.get("binning")
    binningX = wjdata.get("binningX")
    binningY = wjdata.get("binningY")
    print("Binning info:")
    print("binning:", wjdata.get("binning"))
    print("binningX:", wjdata.get("binningX"))
    print("binningY:", wjdata.get("binningY"))
    with open(data_files[0][0] + '_header.txt', 'w') as f:
        for ch in new_header:
            if ch == '{':
                f.write('\n' + ch)
            else:
                f.write(ch)

    print("Parsed JSON metadata from header.")

    # Read image data
    image_data = []
    for i, path in enumerate(filenames):
        image = []
        width = data_files[i][2]
        height = data_files[i][3]
        _, num_frames, width, height, _ = data_files[i]
        with open(path, 'rb') as file:
            file.read(46)
            comment_len = int(struct.unpack('I', file.read(4))[0])
            file.read(46 + comment_len)
            for j in range(num_frames):
                frame = read_frame_safe(file, width, height, frame_idx=j)
                if frame is not None:
                    image.append(frame)
                    print(f"Read frame {j} from {os.path.basename(path)}")
    image_data.append(image)
    print(f"Loaded {len(image)} frames from {os.path.basename(path)}")

    # Flatten all images into one array for processing
    #flattened = np.array([frame for stack in image_data for frame in stack])
    #print("Combined shape:", flattened.shape)

    # Save all combined frames as test TIFF
    #tiff.imwrite(os.path.join(initialdir, 'test_output.tif'), flattened[:100], bigtiff=True)

    # Save per-file TIFFs, RAWs, and mean TIFFs
    for i, stack in enumerate(image_data):
        stack_np = np.stack(stack)
        base_dir = path
        base_name = os.path.splitext(os.path.basename(filenames[i]))[0]

        # Save TIFF
        #tiff_path = os.path.join(base_dir, f"vol.tif")
        #tiff.imwrite(tiff_path, stack_np, bigtiff=True)

        # Save RAW
        raw_path = os.path.join(base_dir, f"Image_001_001.raw")
        VectorizedMovie(stack_np).savebin(raw_path)

        # Save Mean TIFF
        mean_path = os.path.join(base_dir, f"Mean.tif")
        tiff.imwrite(mean_path, np.mean(stack_np, axis=0).astype(np.uint16), bigtiff=True)

        print(f"Saved files for {base_name}:\n-  {raw_path}\n- {mean_path}")
def clicy_raw(Path):
    # Load mean image for shape
    mean_path = os.path.join(Path, 'Mean.tif')
    mean_img = tiff.imread(mean_path)
    height, width = mean_img.shape

    # Load RAW data
    raw_path = os.path.join(Path, 'Image_001_001.raw')
    with open(raw_path, 'rb') as fid:
        data = np.fromfile(fid, dtype=np.uint16)

    num_pixels_per_frame = height * width
    num_frames = data.size // num_pixels_per_frame
    if data.size % num_pixels_per_frame != 0:
        raise ValueError("RAW file size doesn't match expected dimensions.")

    # Reshape to [frames, height, width]
    img_stack = data.reshape((num_frames, height, width))

    # Compute mean image
    mean_img = np.mean(img_stack, axis=0)

    plt.imshow(mean_img)
    
    multiroi_named = MultiRoi()
    masks=[]
    corrected_polygons=[]
    polygons = []
    for roi in multiroi_named.rois.values():
        mask=roi.get_mask(mean_img)
        masks.append(mask)
        polygons.append(np.transpose([roi.x, roi.y]))

    # Extract traces
    mov2d = img_stack.reshape(num_frames, -1)
    raw_traces = [np.mean(mov2d[:, mask.flatten()], axis=1) for mask in masks]

    return raw_traces, mean_img, polygons


def clicy(Path):
    img = tiff.imread(Path)
    img_array = np.array(img)
    mean_img = np.mean(img,0)
    plt.imshow(mean_img)
    
    multiroi_named = MultiRoi()
    masks=[]
    corrected_polygons=[]
    polygons = []
    for roi in multiroi_named.rois.values():
        mask=roi.get_mask(mean_img)
        masks.append(mask)
        polygons.append(np.transpose([roi.x, roi.y]))

    # Extract traces
    mov2d = img_array.reshape(img_array.shape[0], -1)
    raw_traces = [np.mean(mov2d[:, mask.flatten()], axis=1) for mask in masks]

    return raw_traces, mean_img, polygons



def plot_clicy_results(raw_traces, mean_img, polygons, path=None):
    # Create subplot: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Mean Image with ROIs", "Extracted Traces"))

    # --- Mean image ---
    fig.add_trace(
        go.Heatmap(
            z=mean_img,
            colorscale='gray',
            showscale=False
        ),
        row=1, col=1
    )

    # --- Add polygons to mean image ---
    for poly in polygons:
        fig.add_trace(
            go.Scatter(
                x=poly[:, 0],
                y=poly[:, 1],
                mode='lines',
                line=dict(width=2, color='red'),
                showlegend=False
            ),
            row=1, col=1
        )

    # --- Traces for each ROI ---
    for i, trace in enumerate(raw_traces):
        fig.add_trace(
            go.Scatter(
                y=trace,
                mode='lines',
                name=f'ROI {i+1}'
            ),
            row=1, col=2
        )

    fig.update_xaxes(title_text="X Pixel", row=1, col=1)
    fig.update_yaxes(title_text="Y Pixel", row=1, col=1, autorange='reversed')  # Flip Y-axis like in imshow

    fig.update_xaxes(title_text="Frame", row=1, col=2)
    fig.update_yaxes(title_text="Mean Intensity", row=1, col=2)

    fig.update_layout(
        height=600,
        width=1100,
        title_text="Calcium Imaging Summary",
        template="plotly_white"
    )

    fig.show()
    fig.write_image(os.path.join(path, "clicy.svg"), format="svg")

#pRaw = input('insert path:')
pTiff = input('insert path tiff:')
pathMC = os.path.join(r'E:\Miki\30-12-2025-awake\Hyp3\L\FOV1\z-5-x4\Image_scan_1_region_0_0.tif')
#raw_tracesNMC, mean_imgNMC, polygonsNMC = clicy_raw(pRaw)
#raw_tracesNMC, mean_imgNMC, polygonsNMC = clicy(pRaw)
#pathFig = os.path.join(pRaw,r'Combined_MC_SMALLroi.html')
raw_traces, mean_img, polygons = clicy(pathMC)
#plot_clicy_results(raw_tracesNMC, mean_imgNMC, polygonsNMC,pRaw)
plot_clicy_results(raw_traces, mean_img, polygons,pTiff)
#NoMCfORp = raw_tracesNMC[0]
MCfORp = raw_traces[0]
Xaxis = range(len(MCfORp))
Xaxis = list(Xaxis)
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=Xaxis, y=NoMCfORp, name="raw trace",line=dict(color='red', width=3)),secondary_y=False,)
fig.add_trace(go.Scatter(x= Xaxis, y= MCfORp, name="Motion correctedium",line=dict(color='blue', width=3)),secondary_y=True,)
fig.update_layout(title_text="calcium and voltage togeter")
fig.update_xaxes(title_text="Time(ms)")
fig.update_yaxes(title_text="<b>Calcium</b>", secondary_y=True)
fig.update_yaxes(title_text="<b>Voltage</b> ", secondary_y=False)
fig.show()
fig.update_layout(
title="cal-vol",
plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
paper_bgcolor="rgba(0,0,0,0)")  # Transparent paper background
fig.write_html(pathFig)
fig.update_layout(
    width=1250,  # Set the figure width in pixels
    height=375, # Set the figure height in pixels
)
fig.write_image(os.path.join(pRaw, "Combined_MC_SMALLroi.svg"), format="svg")