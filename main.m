% Clear Matlab workspace
clear all; close all;

% Add assignment tools
addpath(fullfile('tools'));

% Setup vlfeat
run('lib/vlfeat/toolbox/vl_setup');

% Load music data and their category labels
[data, labels, filenames] = loadAll('.');

% Extract MFCC features
mfcc = cell(1,length(data));
for i = 1:length(data)
    mfcc{i} = data{i}.mfc;
end

% Generate Fisher Vectors from MFCC features
GENDATA.data = mfcc;
GENDATA.class = labels;
GENDATA.classnames = {'Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'};
fv = demo_fv(GENDATA, 3, 3);

% Randomly split data ratio 7:1:2 training:validation:testing
randIDX = randsample(1:length(data),length(data));
trainingData = fv(:,randIDX(1:700));
trainingLabels = labels(:,randIDX(1:700));
validationData = fv(:,randIDX(701:800));
validationLabels = labels(:,randIDX(701:800));
testingData = fv(:,randIDX(801:1000));
testingLabels = labels(:,randIDX(801:1000));

% Train random forest classifier
model = TreeBagger(100,trainingData',trainingLabels','MinLeafSize',5);
[pred,scores] = predict(model,testingData');
pred = cellfun(@str2num,pred);
accuracy = sum(pred == testingLabels')/length(testingLabels);
fprintf('Accuracy (random forests): %f\n', accuracy);

% Train k-nearest neighbor classifer
model = fitcknn(trainingData',trainingLabels');
[pred,scores] = predict(model,testingData');
accuracy = sum(pred == testingLabels')/length(testingLabels);
fprintf('Accuracy (k-nearest neighbor): %f\n', accuracy);






