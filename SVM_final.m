load hepatitisdat.sec;
data= hepatitisdat;

x= data(:,2:20);
y=data(:,1:1);
datatrain= data(1:120,:);
datatest=data(120:end,:);

xtr=x(1:120,:);
ytr=y(1:120,:);

xt=x(120:end,:);
yt= y(120:end,:);

%% Training the model
model = fitcsvm(xtr, ytr, 'KernelFunction', 'rbf', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', true));

%% Test the model
result = predict(model, xt);
accuracy = sum(result == yt)/numel(yt)*100;
sp = sprintf("Test Accuracy = %.2f", accuracy);
disp(sp);

confusionchart(y,pedict(model,yt));
%% feature extraction:

[pcs,scrs,~,~,pctExp] = pca(data);
pareto(pctExp);
pos=find(cumsum(pctExp)>=95,1);
[idx,scores]=fscmrmr(data,y);
bar(scores(idx));
xlabel("predictor rank");
ylabel("predictor importance score");
pcaPreds= scrs(:,1:pos);
% Dividing the predictors after PCA into training and testing in the ration
% of 80:20
pcatr= pcaPreds(1:120,:);
pcatt= pcaPreds(120:end,:);
%% mdl
mdlRP=fitcsvm(pcatr, ytr, 'KernelFunction', 'rbf', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', true));
result2 = predict(mdlRP, pcatt);
accuracy2 = sum(result2 == yt)/length(yt)*100;
print = sprintf("Test Accuracy = %.2f", accuracy2);
disp(print);

loss= resubLoss(mdlRP);
disp(loss);
confusionchart(y,predict(mdllRP,y));


