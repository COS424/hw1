function [pred, scores, timeTrain, timeTest] = featClassify(trainingData, trainingLabels, testingData, method)

pred = []; scores = []; timeTrain = 0; timeTest = 0;
switch method
    case 1 % K-Nearest Neighbor (euclidean,fine)
        tic;
        model = fitcknn(trainingData,trainingLabels,'NumNeighbors',1,'Standardize',1,'Distance','euclidean');
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        timeTest = toc;
    case 2 % K-Nearest Neighbor (euclidean,coarse)
        tic;
        model = fitcknn(trainingData,trainingLabels,'NumNeighbors',10,'Standardize',1,'Distance','euclidean');
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        timeTest = toc;
    case 3 % K-Nearest Neighbor (cosine,fine)
        tic;
        model = fitcknn(trainingData,trainingLabels,'NumNeighbors',1,'Standardize',1,'Distance','cosine');
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        timeTest = toc;
    case 4 % K-Nearest Neighbor (cosine,coarse)
        tic;
        model = fitcknn(trainingData,trainingLabels,'NumNeighbors',10,'Standardize',1,'Distance','cosine');
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        timeTest = toc;
    case 5 % Multi-Class SVM (linear)
        tic;
        t = templateSVM('Standardize',1,'KernelFunction','linear');
        model = fitcecoc(trainingData,trainingLabels,'Learners',t);
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        timeTest = toc;
    case 6 % Multi-Class SVM (quadratic)
        tic;
        t = templateSVM('Standardize',1,'KernelFunction','polynomial','PolynomialOrder',2);
        model = fitcecoc(trainingData,trainingLabels,'Learners',t);
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        timeTest = toc;
    case 7 % Multi-Class naive bayes (normal)
        tic;
        model = fitNaiveBayes(trainingData,trainingLabels);
        timeTrain = toc;
        tic;
        pred = predict(model,testingData);
        timeTest = toc;
    case 8 % Multi-Class discriminant (linear)
        tic;
        t = templateDiscriminant('DiscrimType','linear');
        model = fitcecoc(trainingData,trainingLabels,'Learners',t);
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        timeTest = toc;
    case 9 % Multi-Class discriminant (quadratic)
        tic;
        t = templateDiscriminant('DiscrimType','pseudoQuadratic');
        model = fitcecoc(trainingData,trainingLabels,'Learners',t);
        timeTrain = toc;
        tic;
        pred = predict(model,testingData);
        timeTest = toc;
    case 10 % Decision trees
        model = fitctree(trainingData,trainingLabels);
        [pred,scores] = predict(model,testingData);
    case 11 % Random Forests
        tic;
        model = TreeBagger(100,trainingData,trainingLabels,'MinLeafSize',5);
        timeTrain = toc;
        tic;
        [pred,scores] = predict(model,testingData);
        [sortClass,sortIDX] = sort(cellfun(@str2num,model.ClassNames));
        scores = scores(:,sortIDX);
        timeTest = toc;
        pred = cellfun(@str2num,pred);
    case 12 % Boosted decision trees
        t = templateTree('Surrogate','on');
        model = fitensemble(trainingData,trainingLabels,'AdaBoostM2',100,t);
        [pred,scores] = predict(model,testingData);
end

end
