function [pred, scores, timeTrain, timeTest] = featClassify(trainingData, trainingLabels, testingData, method)

pred = []; scores = []; timeTrain = 0; timeTest = 0;
switch method
    case 1
        tic;
        model = fitcknn(trainingData,trainingLabels,'NumNeighbors',10,'Standardize',1,'NSMethod','exhaustive');
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        timeTest = toc;
    case 2
        fprintf('hello')
end

end

