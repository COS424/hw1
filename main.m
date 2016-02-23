% %% Setup
% % Clear Matlab workspace
% clear all; close all;
% fprintf('Loading data...\n');
% 
% % Add assignment tools
% addpath(fullfile('tools'));
% 
% % Setup libraries
% addpath(fullfile('lib','vlfeat','toolbox'));
% vl_setup;
% addpath(fullfile('lib','tsne'));
% 
% % Load music data and their category labels
% [data, labels, filenames] = loadAll('.');
% 
% %% Feature Analysis
% % Extract and aggregate all frame-level features
% fprintf('Collecting frame-level features...\n');
% feat = {};
% featNames = {'MFCC','Chroma','Zero-Crossing','Spectral Flux','HCDF'};
% for i = 1:length(data); currFeat{i} = data{i}.mfc; end; feat{1} = currFeat; % mfcc
% for i = 1:length(data); currFeat{i} = data{i}.chroma; end; feat{2} = currFeat; % chroma
% for i = 1:length(data); currFeat{i} = data{i}.zerocross; end; feat{3} = currFeat; % zero-crossing
% for i = 1:length(data); currFeat{i} = data{i}.brightness; end; feat{4} = currFeat; % spectral flux
% for i = 1:length(data); currFeat{i} = data{i}.hcdf; end; feat{5} = currFeat; % hcdf
% 
% % Generate Fisher Vectors for all frame-level features
% fprintf('Generating Fisher Vectors...\n');
% fv = [];
% for i = 1:length(feat)
%     GENDATA.data = feat{i};
%     GENDATA.class = labels;
%     GENDATA.classnames = {'Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'};
%     fv = [fv; demo_fv(GENDATA, 3, 3)];
% end
% 
% 
%% Classification
% Randomly split data ratio 8:2 training:testing
% Stratified 10-fold cross validation
trainingData = {}; trainingLabels = {};
testingData = {}; testingLabels = {};
for k = 1:10
    randIDX = randsample(1:length(data),length(data));
    trainingData{k} = fv(:,randIDX(1:800));
    trainingLabels{k} = labels(:,randIDX(1:800));
    testingData{k} = fv(:,randIDX(801:1000));
    testingLabels{k} = labels(:,randIDX(801:1000));
end

% Train all classifiers
for classifierIDX = 1:1
    prec = []; rec = [];
    acc = []; conf = [];
    timeTrain = []; timeTest = [];
    
    % Stratified k-fold cross-validation
    for k = 1:10
        [pred, scores, tmpTimeTrain, tmpTimeTest] = featClassify(trainingData{k}', trainingLabels{k}', testingData{k}', classifierIDX);
        timeTrain = [timeTrain; tmpTimeTrain];
        timeTest = [timeTest; tmpTimeTest];
        tAcc = sum(pred == testingLabels{k}')/length(testingLabels{k})
        
        % Compute PR per category
        for categ = 1:10
            tGT = double(testingLabels{k}' == categ);
            tGT(find(tGT == 0)) = -1;
            [tRec,tPrec,tInfo] = vl_pr(tGT,scores(:,categ));
            
%             for conf = 0:0.001:1
%                 tPred = scores(:,categ) >= conf;
%                 tGT = testingLabels{k}' == categ;
%                 [tScores,sortIDX] = sort(scores(:,categ));
%                 tPred = tPred(sortIDX); tGT = tGT(sortIDX);
%                 if length(find(tPred)) == 0 || length(find(~tPred)) == 0
%                     continue;
%                 end
%                 TP = sum((tGT(find(tPred))-tPred(find(tPred))) == 0)/length(find(tPred));
%                 FP = sum((tGT(find(tPred))-tPred(find(tPred))) == -1)/length(find(tPred));
%                 FN = sum((tGT(find(~tPred))-tPred(find(~tPred))) == 1)/length(find(~tPred));
%                 tPrec = [tPrec;TP/(TP+FP)];
%                 tRec = [tRec;TP/(TP+FN)];
%             end
%             ap =  sum(tPrec(2:end).*(tRec(2:end)-tRec(1:(end-1))));
        end

    end
    % fprintf('Accuracy (k-nearest neighbor): %f\n', mean(accuracy));
end

% % Train multi-class linear SVM classifier
% model = fitcecoc(trainingData',trainingLabels');
% [pred,scores] = predict(model,testingData');
% accuracy = sum(pred == testingLabels')/length(testingLabels);
% fprintf('Accuracy (multi-class SVM): %f\n', accuracy);


% % Train random forest classifier
% model = TreeBagger(100,trainingData',trainingLabels','MinLeafSize',5,'CrossVal','on');
% [pred,scores] = predict(model,testingData');
% pred = cellfun(@str2num,pred);
% accuracy = sum(pred == testingLabels')/length(testingLabels);
% fprintf('Accuracy (random forests): %f\n', accuracy);


