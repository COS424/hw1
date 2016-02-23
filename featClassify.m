function [pred, scores, timeTrain, timeTest] = featClassify(trainingData, trainingLabels, testingData, method)

pred = []; scores = []; timeTrain = 0; timeTest = 0;
switch method
    case 1 % K-Nearest Neighbor
        tic;
        model = fitcknn(trainingData,trainingLabels,'NumNeighbors',10,'Standardize',1,'NSMethod','exhaustive');
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        timeTest = toc;
    case 2 % Multi-Class Linear SVM
        tic;
        model = fitcecoc(trainingData,trainingLabels);
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        timeTest = toc;
    case 3 % Random Forests
        model = TreeBagger(100,trainingData,trainingLabels,'MinLeafSize',5);
        [pred,scores] = predict(model,testingData);
        pred = cellfun(@str2num,pred);
end

end
