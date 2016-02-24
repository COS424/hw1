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

%% Classification
% % Stratified 10-fold cross validation (4:1 trainin:validation ratio)
% cvTrainData = {}; cvTrainLabels = {};
% cvTestData = {}; cvTestLabels = {};
% for k = 1:10
%     randIDX = randsample(1:length(data),length(data));
%     cvTrainData{k} = fv(:,randIDX(1:800));
%     cvTrainLabels{k} = labels(:,randIDX(1:800));
%     cvTestData{k} = fv(:,randIDX(801:1000));
%     cvTestLabels{k} = labels(:,randIDX(801:1000));
% end
% 
% % Stratified k-fold cross-validation
% classifierNames = {'KNN-ef','KNN-ec','KNN-cf','KNN-cc','SVM-l','SVM-q','NB','D-l','D-q','DT','RF','BDT'};
% for classifierIDX = 1:12
%     mAcc = []; 
%     for k = 1:10
%         [pred, scores, tmpTimeTrain, tmpTimeTest] = featClassify(cvTrainData{k}', cvTrainLabels{k}', cvTestData{k}', classifierIDX);
%         tAcc = sum(pred == cvTestLabels{k}')/length(cvTestLabels{k});
%         mAcc = [mAcc, tAcc];
%     end
%     fprintf('Cross-Validation Generalization Accuracy (%s): %f\n', classifierNames{classifierIDX}, mean(mAcc));
% end

% % Final distribution: randomly split data ratio 4:1 training:testing
% randIDX = randsample(1:length(data),length(data));
% finTrainData = fv(:,randIDX(1:800));
% finTrainLabels = labels(:,randIDX(1:800));
% finTestData = fv(:,randIDX(801:1000));
% finTestLabels = labels(:,randIDX(801:1000));

for classifierIDX = [4,5,8,11]
    runtimes = [];
    [pred, scores, tmpTimeTrain, tmpTimeTest] = featClassify(finTrainData', finTrainLabels', finTestData', classifierIDX);
    acc = sum(pred == finTestLabels')/length(finTestLabels);

    % Compute PR per category
    allAP = [];
    for categ = 1:10
        tGT = double(finTestLabels' == categ);
        tGT(find(~tGT)) = -1;
        [tRec,tPrec,tInfo] = vl_pr(tGT,scores(:,categ));
        figure(categ); hold on; plot(tRec,tPrec,'LineWidth',3); axis equal; axis([0 1 0 1]); hold off;
        allAP = [allAP,tInfo.ap];
    end
    
    allAP
    mean(allAP)
    
    % Report numbers
    fprintf('Accuracy (%s): %f\n', classifierNames{classifierIDX}, acc);
    fprintf('Training runtime (%s): %f\n', classifierNames{classifierIDX}, tmpTimeTrain);
    fprintf('Testing runtime (%s): %f\n', classifierNames{classifierIDX}, tmpTimeTest);
end

classnames = {'Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'};
for categ = 1:10
    figure(categ); hold on; legend('KNN-cc','SVM-l','D-l','RF'); title(classnames{categ}); set(gca,'fontsize',14); xlabel('Recall'); ylabel('Precision'); 
    if categ ~= 11
        legend('KNN-cc','SVM-l','D-l','RF','Location','southwest');
    end
    hold off;
end

