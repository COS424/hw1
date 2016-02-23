%% Setup
% Clear Matlab workspace
clear all; close all;
fprintf('Loading data...\n');

% Add assignment tools
addpath(fullfile('tools'));

% Setup libraries
addpath(fullfile('lib','vlfeat','toolbox'));
vl_setup;
addpath(fullfile('lib','tsne'));

% Load music data and their category labels
[data, labels, filenames] = loadAll('.');

%% Feature Analysis
% Extract and aggregate all frame-level features
fprintf('Collecting frame-level features...\n');
feat = {};
featNames = {'MFCC','Chroma','Energy','Zero-Crossing','Spectral Flux','Roughness','Key Strength','HCDF','Inharmonicity'};
for i = 1:length(data); currFeat{i} = data{i}.mfc; end; feat{1} = currFeat; % mfcc
for i = 1:length(data); currFeat{i} = data{i}.chroma; end; feat{2} = currFeat; % chroma
for i = 1:length(data); currFeat{i} = data{i}.eng; end; feat{3} = currFeat; % energy
for i = 1:length(data); currFeat{i} = data{i}.zerocross; end; feat{4} = currFeat; % zero-crossing
for i = 1:length(data); currFeat{i} = data{i}.brightness; end; feat{5} = currFeat; % spectral flux
for i = 1:length(data); currFeat{i} = data{i}.roughness; end; feat{6} = currFeat; % roughness
for i = 1:length(data); currFeat{i} = data{i}.keystrength; end; feat{7} = currFeat; % key strength
for i = 1:length(data); currFeat{i} = data{i}.hcdf; end; feat{8} = currFeat; % hcdf
for i = 1:length(data); tmpInharmonic = data{i}.inharmonic; tmpInharmonic(find(isnan(tmpInharmonic))) = 0; currFeat{i} = tmpInharmonic; end; feat{9} = currFeat; % inharmonicity

% Generate Fisher Vectors for all frame-level features
fprintf('Generating Fisher Vectors...\n');
fv = {};
for i = 1:length(feat)
    GENDATA.data = feat{i};
    GENDATA.class = labels;
    GENDATA.classnames = {'Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'};
    fv{i} = demo_fv(GENDATA, 3, 3);
end

% Compute t-SNE embedding of features
for i = 1:length(feat)
    tsneData = tsne(fv{i}',[]);
    palette = [255/255,   0/255,   0/255;...
               255/255, 154/255,   0/255;...
               254/255, 255/255,   0/255;...
               186/255, 232/255,   6/255;...
                46/255, 206/255,  12/255;...
                12/255, 153/255, 206/255;...
                12/255,  91/255, 206/255;...
                31/255,  12/255, 206/255;...
               128/255,  12/255, 206/255;...
               206/255,  12/255, 130/255];

    classes = {};
    for j = 1:1000
        classes{j} = GENDATA.classnames{labels(j)};
    end
    hfig = figure();
    gscatter(tsneData(:,1),tsneData(:,2),classes',palette,'.',25); 
    set(hfig, 'Position', [1 1 1000 1000]);
    hold on; title(featNames(i)); hold off;
end