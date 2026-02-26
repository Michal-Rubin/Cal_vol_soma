%% Load and correct
clc;clearvars;close all
sz=get(0,'screensize');
Path=cd('Z:\Adam-Lab-Shared\Data\Michal_Rubin\SRUGC21\X\15-09-2025-awake\fov5\Sync\cal');
%% load suite2p
base = '\suite2p\plane0\ops.npy';

%info = readNPY([Path base]);
%ops = readNPY(info);
Ly = 512;
Lx = 512;
%nframes = ops.nframes;
vid = [Path '\suite2p\plane0\data.bin'];

fid = fopen(vid,'r');

% Get file size in bytes
fseek(fid, 0, 'eof');
filesize = ftell(fid);
fseek(fid, 0, 'bof');

bytes_per_pixel = 2; % int16 = 2 bytes
nframes = filesize / (Ly * Lx * bytes_per_pixel);

if mod(nframes,1) ~= 0
    error('File size not consistent with Ly and Lx');
end

% Read entire movie
data = fread(fid, Ly*Lx*nframes, 'int16=>int16');
fclose(fid);

mov = reshape(data, [Lx, Ly, nframes]);
mov = permute(mov, [2 1 3]);  % fix orientation

disp(['Loaded ', num2str(nframes), ' frames'])
%% load Tiff
info = imfinfo([Path '\Image_scan_1_region_0_0.tif']);
numFrames = numel(info);
disp(['Number of frames: ', num2str(numFrames)]);
reader = bfGetReader([Path '\Image_scan_1_region_0_0.tif']);
numFrames = reader.getImageCount();
width = reader.getSizeX();
height = reader.getSizeY();

mov = zeros(height, width, numFrames, 'uint16');  % or 'double' if needed

for k = 1:numFrames
    imgPlane = bfGetPlane(reader, k);
    mov(:,:,k) = imgPlane;
end

%% Load raw
fileName = fullfile(Path, 'Image_001_001.raw');
%figName = fullfile(Path, 'untitled.fig');
%fig = openfig(figName, 'new', 'invisible');
fid = fopen(fileName, 'rb');   % Open fil0e for reading

fileNameM = fullfile(Path, 'Mean.tif');
reader = bfGetReader(fileNameM);
width = reader.getSizeX(); 
height = reader.getSizeY();
pixPerFrame = width * height;
if fid == -1
    error('Error opening the input raw file.');
end
tmp = fread(fid, '*uint16', 'l');  % Read uint16 data with little endian format
fclose(fid);   % Close the file
% Reshape vector into 3D array (video)
numFrames = length(tmp) / (height * width);   % Calculate the number of frames
if mod(numFrames, 1) ~= 0
    error('The length of the data is not compatible with the frame dimensions.');
end
mov = reshape(tmp, [width height numFrames]);
%mov = mov(1:2:end, 1:2:end, :);  % downsample x2 spatially
mov = double(permute(mov, [2 1 3]));
%mov = permute(movC, [2 1 3]);
%mov = zeros(height, width, numFrames,'uint16');  % or 'double' if needed
%for k = 1:size(movB, 3)
  %  mov(:,:,k) = double(movB(:, :, k));  % only 1 frame in memory
    % process frame here
%end
%% thorimage RAW
fileName = fullfile(Path, 'Image_001_001.raw');
xmlFile  = fullfile(Path, 'Experiment.xml');

xDoc = xmlread(xmlFile);

cameraNodes = xDoc.getElementsByTagName('Camera');
if cameraNodes.getLength == 0
    error('No <Camera> tag found in Experiment.xml');
end

cameraNode = cameraNodes.item(0);

width  = str2double(cameraNode.getAttribute('width'));
height = str2double(cameraNode.getAttribute('height'));

if isnan(width) || isnan(height)
    error('Width/Height attributes not found or invalid');
end

fprintf('Image size from XML: %d x %d\n', width, height);

fid = fopen(fileName, 'rb');
if fid == -1
    error('Error opening the input raw file.');
end

tmp = fread(fid, '*uint16', 'l');  % little-endian uint16
fclose(fid);

pixPerFrame = width * height;
numFrames = numel(tmp) / pixPerFrame;

if mod(numFrames,1) ~= 0
    error('RAW file size not compatible with XML dimensions');
end


mov = reshape(tmp, [width height numFrames]);
mov = double(permute(mov, [2 1 3]));




%% load raw chunck

fileName = fullfile(Path, 'Image_001_001.raw');
fid = fopen(fileName, 'rb');
if fid == -1
    error('Error opening the input raw file.');
end

% Get frame size from Mean.tif
fileNameM = fullfile(Path, 'Mean.tif');
reader = bfGetReader(fileNameM);
width = reader.getSizeX();
height = reader.getSizeY();
pixPerFrame = width * height;

% Get total number of frames in RAW
fseek(fid, 0, 'eof');
fileBytes = ftell(fid);
totalFrames = fileBytes / (2 * pixPerFrame);   % 2 bytes per uint16

fprintf("Total frames: %d\n", totalFrames);

% -------- SELECT CHUNK -----------------
startFrame = 1;                        % read from first frame
endFrame   = floor(totalFrames/2);     % first half of video
% e.g. read any chunk: startFrame=100, endFrame=500;

numFramesToRead = endFrame - startFrame + 1;

% -------- SEEK TO FIRST FRAME ----------
offsetBytes = (startFrame - 1) * pixPerFrame * 2;
fseek(fid, offsetBytes, 'bof');

% -------- READ CHUNK -------------------
[tmp, count] = fread(fid, pixPerFrame * numFramesToRead, '*uint16', 'l');

fclose(fid);

if count < pixPerFrame * numFramesToRead
    warning('RAW ended earlier than expected.');
end

% -------- RESHAPE INTO MOVIE -----------
mov = reshape(tmp, [width, height, numFramesToRead]);
mov = double(permute(mov, [2 1 3]));    % convert to [height x width x T]

%% 
exposureTime = 2.000;  % in milliseconds
newFrameRate = 500;

% Load the XML struct
PathMo = cd('Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc46\R\04-02-2026-MOTOR\FOV2\cal');
xmlF = fullfile(PathMo, 'Experiment.xml');
Info = readstruct(xmlF);

% Update attributes
Info.Camera.heightAttribute = height;
Info.Camera.widthAttribute = width;
Info.Camera.OrcaFrameRateValueAttribute = newFrameRate;
Info.Camera.exposureTimeMSAttribute = exposureTime;  % as attribute
Info.Camera.exposureTimeMS = [];  % remove field if it's meant to be a tag

% Save modified struct to XML
outputXML = fullfile(['Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc46\R\04-02-2026-MOTOR\FOV2'], 'Experiment.xml');
writestruct(Info, outputXML);

% Rename root if needed
rename_root_tag(outputXML, 'ThorImageExperiment');

% === Now: Clean duplicate <exposureTimeMS> element using DOM ===
doc = xmlread(outputXML);
cameraNode = doc.getElementsByTagName('Camera').item(0);
childNodes = cameraNode.getChildNodes();
for i = childNodes.getLength()-1:-1:0
    child = childNodes.item(i);
    if strcmp(char(child.getNodeName()), 'exposureTimeMS')
        cameraNode.removeChild(child);
    end
end

% Save cleaned XML
xmlwrite(outputXML, doc);
%%
%exposureTime = 2.000;  % <-- Replace with actual exposure time in milliseconds
%PathMo = cd('Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\15-07-2025\fov5\cal');
%xmlF = fullfile(PathMo, 'Experiment.xml');
%Info = readstruct(xmlF);
%width = Info.Camera.width;
%height = Info.Camera.height;
%Info.Camera.heightAttribute = height;
%Info.Camera.widthAttribute = width;
% Set new frame rate:
%newFrameRate = 500;

% Check the Camera struct content:
%disp(Info.Camera)

% Update the attribute OrcaFrameRateValueAttribute if it exists
%if isfield(Info.Camera, 'OrcaFrameRateValueAttribute')
 %   Info.Camera.OrcaFrameRateValueAttribute = 500;
%else
    % Optionally handle nested attributes or throw error
  %  if isfield(Info.Camera, 'Attributes') && isfield(Info.Camera.Attributes, 'OrcaFrameRateValueAttribute')
   %     Info.Camera.Attributes.OrcaFrameRateValueAttribute = 500;
  %  else
    %    error('OrcaFrameRateValueAttribute not found in Camera');
 %   end
%end

% Optionally, update exposure time if needed (should be consistent with frame rate)

%Info.Camera.exposureTimeMS = 1000 / newFrameRate;  % exposure time in ms
%outputXML = fullfile('Z:\Adam-Lab-Shared\Data\Michal_Rubin\srugc17\Rb\15-07-2025\fov5', 'Experiment.xml');
%writestruct(Info, outputXML);

% your logic here

%rename_root_tag(outputXML, 'ThorImageExperiment');
%wrap_camera_with_ThorImageExperiment(xmlF);  % this is fine now
%%
% etra frame correction
ALL_ROI = {};
mov = mov(:,:,5000:37000);
%%
movc = mov(:,400:end,:);
%%
[ROI,tra] = clicky3(mov);

%ALL_ROI = [ALL_ROI, ROI]; 

%s = size(ROI,2);
%ALLr = zeros(s);
%for i = 1:size(ROI,2)
 %   ALLr(i)=ROI(:,i);
%end
%ALLr= ROI{1};
%% forigh stuff i used for mitochondrial stuff
Fs = 30;                    % Sampling frequency
[n, nTraces] = size(tra);    % n = timepoints, nTraces = number of signals
f = (0:n-1)*(Fs/n);          % Frequency axis

% Determine subplot grid size
nCols = ceil(sqrt(nTraces));   
nRows = ceil(nTraces / nCols);

figure;
for i = 1:nTraces
    trace = tra(:, i);       % Get i-th trace
    fpp = fft(trace);
    absfpp = abs(fpp);

    subplot(nRows, nCols, i);
    plot(f(1:floor(25)), absfpp(1:floor(25)));
    title(['Trace ' num2str(i)]);
    xlabel('Freq (Hz)');
    ylabel('Mag');
end

figure;
for i = 1:nTraces
    trace = tra(:, i);        % Get i-th trace
   
    trace = fillmissing(trace, 'linear');
    % Backup: replace any remaining non-finite (e.g., Infs, NaNs) with 0
    trace(~isfinite(trace)) = 0;
        % Display any bad values
    disp(['Trace ' num2str(i)]);
    disp('Any NaNs?'); disp(any(isnan(trace)));
    disp('Any Infs?'); disp(any(isinf(trace)));
    disp('Any non-finite?'); disp(any(~isfinite(trace)));

    % Compute PSD using Welch's method
    [Pxx, F] = pwelch(trace, [], [], [], Fs);

    % Plot
    subplot(nRows, nCols, i);
    plot(F, 10*log10(Pxx));
    title(['Trace ' num2str(i)]);
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB/Hz)');
    xlim([0 Fs/2]);           % Optional: zoom to Nyquist
end

%%

my_templateFile='Z:\Adam-Lab-Shared\Data\Michal_Rubin\ROIs_tamplate.xaml';
fid = fopen(my_templateFile, 'r');
%%
outputFile='Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc46\R\04-02-2026-MOTOR\FOV2\ROIs.xaml';
updateXAMLPolygonMR(my_templateFile, outputFile, ROI, width, height);


%%
function wrap_camera_with_ThorImageExperiment(xmlF)
    xDoc = xmlread(xmlF);
    root = xDoc.getDocumentElement;
    rootTag = char(root.getTagName);

    % Only proceed if the root is not already <ThorImageExperiment>
    if ~strcmp(rootTag, 'ThorImageExperiment')
        % Try to find a <Camera> element
        cameraNodes = xDoc.getElementsByTagName('Camera');

        if cameraNodes.getLength == 0
            error('No <Camera> element found to wrap.');
        end

        cameraElem = cameraNodes.item(0);

        % Create new XML document with <ThorImageExperiment> as root
        newDoc = com.mathworks.xml.XMLUtils.createDocument('ThorImageExperiment');
        newRoot = newDoc.getDocumentElement;

        % Import the <Camera> element
        importedCamera = newDoc.importNode(cameraElem, true);
        newRoot.appendChild(importedCamera);

        % Save the new XML structure
        xmlwrite(xmlF, newDoc);
        fprintf("Wrapped <Camera> inside <ThorImageExperiment>\n");
    else
        fprintf("XML already has <ThorImageExperiment> root\n");
    end
end

function unwrap_ThorImageExperiment(xmlF)
    xDoc = xmlread(xmlF);
    root = xDoc.getDocumentElement;
    rootTag = char(root.getTagName);

    % Check that the current root is <ThorImageExperiment>
    if strcmp(rootTag, 'ThorImageExperiment')
        children = root.getChildNodes();
        n = children.getLength();
        
        % Find the first element-type child node (skip text, comments, etc.)
        found = false;
        for i = 0:n-1
            node = children.item(i);
            if node.getNodeType() == node.ELEMENT_NODE
                oldRoot = node;
                found = true;
                break;
            end
        end

        if ~found
            error('No element child found inside <ThorImageExperiment>. Cannot unwrap.');
        end

        % Create a new document with the child as the root
        newDoc = com.mathworks.xml.XMLUtils.createDocument(oldRoot.getTagName());
        newRoot = newDoc.getDocumentElement;

        % Import all attributes from oldRoot to newRoot
        attrs = oldRoot.getAttributes();
        for j = 0:attrs.getLength()-1
            attr = attrs.item(j);
            newRoot.setAttribute(attr.getName(), attr.getValue());
        end

        % Import all child nodes from oldRoot to newRoot
        oldChildren = oldRoot.getChildNodes();
        for k = 0:oldChildren.getLength()-1
            child = oldChildren.item(k);
            imported = newDoc.importNode(child, true);
            newRoot.appendChild(imported);
        end

        % Save new document
        xmlwrite(xmlF, newDoc);
        fprintf("Unwrapped <ThorImageExperiment>, new root: <%s>\n", char(oldRoot.getTagName()));
    else
        fprintf("Root is <%s>, no <ThorImageExperiment> to unwrap.\n", rootTag);
    end
end

function rename_root_tag(xmlF, newRootName)
    % Read XML
    xDoc = xmlread(xmlF);
    oldRoot = xDoc.getDocumentElement;
    oldRootName = char(oldRoot.getTagName);

    % Only change if needed
    if ~strcmp(oldRootName, newRootName)
        % Create new XML document with the desired root name
        newDoc = com.mathworks.xml.XMLUtils.createDocument(newRootName);
        newRoot = newDoc.getDocumentElement;

        % Copy attributes from old root
        attrs = oldRoot.getAttributes();
        for i = 0:attrs.getLength()-1
            attr = attrs.item(i);
            newRoot.setAttribute(attr.getName(), attr.getValue());
        end

        % Copy children from old root
        children = oldRoot.getChildNodes();
        for i = 0:children.getLength()-1
            child = children.item(i);
            imported = newDoc.importNode(child, true);
            newRoot.appendChild(imported);
        end

        % Write back to file
        xmlwrite(xmlF, newDoc);
        fprintf("Renamed root <%s> → <%s>\n", oldRootName, newRootName);
    else
        fprintf("Root is already <%s>, no change made.\n", newRootName);
    end
end